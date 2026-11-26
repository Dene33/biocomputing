from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class CultureConfig:
    n_neurons: int = 2000
    frac_excitatory: float = 0.8
    radius_um: float = 1000.0
    thickness_um: float = 150.0
    dt_ms: float = 0.5

    synapses_per_neuron: float = 60.0
    length_um: float = 150.0
    w_exc: float = 2.2
    w_inh: float = -6.0
    w_spread: float = 0.5
    w_max: float = 10.0
    conduction_um_per_ms: float = 300.0
    min_delay_ms: float = 0.5
    delay_buckets: int = 8

    tau_exc_ms: float = 5.0
    tau_inh_ms: float = 10.0
    nmda_ratio: float = 0.0
    tau_nmda_ms: float = 100.0
    mg_mM: float = 1.0

    stp_u: float = 0.2
    stp_tau_rec_ms: float = 2000.0

    background_rate_hz: float = 2.0
    background_w: float = 12.0
    noise_std: float = 3.0
    noise_inh_scale: float = 0.4

    seed: int = 0
    device: str = "cpu"

    wells: int = 1
    well_spacing_um: float = 10000.0

    extra: dict = field(default_factory=dict)


PRESETS: dict[str, dict] = {
    "cpu_2k": dict(n_neurons=2000, synapses_per_neuron=60),
    "gpu_20k": dict(n_neurons=20000, synapses_per_neuron=100, radius_um=2000.0, device="cuda"),
}


def make_config(preset: str = "cpu_2k", **overrides) -> CultureConfig:
    cfg = dict(PRESETS[preset])
    cfg.update(overrides)
    return CultureConfig(**cfg)


class OrganoidCulture:
    def __init__(self, cfg: CultureConfig):
        self.cfg = cfg
        dev = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.device = dev
        parts = [self._make_well(cfg.seed + 7919 * k if k else cfg.seed, k) for k in range(cfg.wells)]
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        n_per = cfg.n_neurons
        n = n_per * cfg.wells
        self.n, self.n_exc, self.n_per_well = n, int(round(n_per * cfg.frac_excitatory)), n_per
        self.pos = torch.cat([q["pos"] for q in parts])
        self.excitatory = torch.cat([q["exc"] for q in parts])
        a, b, c, d = (torch.cat([q[k] for q in parts]) for k in "abcd")
        pre = torch.cat([q["pre"] + k * n_per for k, q in enumerate(parts)])
        post = torch.cat([q["post"] + k * n_per for k, q in enumerate(parts)])
        w = torch.cat([q["w"] for q in parts])
        delay = torch.cat([q["delay"] for q in parts])

        to = dict(device=dev)
        self.a, self.b, self.c, self.d = (x.to(**to) for x in (a, b, c, d))
        self.excitatory = self.excitatory.to(**to)
        self.pre, self.post, self.delay = pre.to(**to), post.to(**to), delay.to(**to)
        self.w = w.to(**to)
        self.w0 = self.w.clone()
        self.syn_exc = self.excitatory[self.pre]
        self.noise_scale = torch.where(self.excitatory, 1.0, cfg.noise_inh_scale).float() * cfg.noise_std
        self.alive = torch.ones(n, dtype=torch.bool, device=dev)
        self.excitability = torch.zeros(n, device=dev)
        self.plasticity = None

        self.reset_state()

    def _make_well(self, seed: int, k: int) -> dict:
        cfg = self.cfg
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        g = self.gen
        n = cfg.n_neurons
        n_exc = int(round(n * cfg.frac_excitatory))

        r = cfg.radius_um * torch.sqrt(torch.rand(n, generator=g))
        theta = 2 * math.pi * torch.rand(n, generator=g)
        z = cfg.thickness_um * torch.rand(n, generator=g)
        self.pos = torch.stack([r * torch.cos(theta) + k * cfg.well_spacing_um, r * torch.sin(theta), z], dim=1)

        re = torch.rand(n_exc, generator=g)
        ri = torch.rand(n - n_exc, generator=g)
        self.excitatory = torch.zeros(n, dtype=torch.bool)
        self.excitatory[:n_exc] = True
        a = torch.cat([torch.full((n_exc,), 0.02), 0.02 + 0.08 * ri])
        b = torch.cat([torch.full((n_exc,), 0.2), 0.25 - 0.05 * ri])
        c = torch.cat([-65 + 15 * re**2, torch.full((n - n_exc,), -65.0)])
        d = torch.cat([8 - 6 * re**2, torch.full((n - n_exc,), 2.0)])

        pre, post, w, delay = self._connect()
        return dict(pos=self.pos, exc=self.excitatory, a=a, b=b, c=c, d=d, pre=pre, post=post, w=w, delay=delay)

    def _connect(self):
        cfg, g, n = self.cfg, self.gen, self.cfg.n_neurons
        sample = torch.randint(0, n, (min(n, 200),), generator=g)
        dist_sample = torch.cdist(self.pos[sample], self.pos)
        mean_kernel = torch.exp(-dist_sample / cfg.length_um).sum(1).mean().item()
        p_max = min(1.0, cfg.synapses_per_neuron / max(mean_kernel, 1e-9))

        pres, posts = [], []
        chunk = max(1, 4_000_000 // n)
        for start in range(0, n, chunk):
            idx = torch.arange(start, min(n, start + chunk))
            dist = torch.cdist(self.pos[idx], self.pos)
            p = p_max * torch.exp(-dist / cfg.length_um)
            p[torch.arange(len(idx)), idx] = 0.0
            hit = torch.rand(p.shape, generator=g) < p
            i, j = hit.nonzero(as_tuple=True)
            pres.append(idx[i])
            posts.append(j)
        pre, post = torch.cat(pres), torch.cat(posts)

        exc = self.excitatory[pre]
        mean = torch.where(exc, torch.tensor(cfg.w_exc), torch.tensor(abs(cfg.w_inh)))
        sigma = cfg.w_spread
        w = mean * torch.exp(sigma * torch.randn(len(pre), generator=g) - sigma**2 / 2)
        w = torch.clamp(w, max=cfg.w_max)

        dist = (self.pos[pre] - self.pos[post]).norm(dim=1)
        delay_ms = cfg.min_delay_ms + dist / cfg.conduction_um_per_ms
        delay = torch.clamp((delay_ms / cfg.dt_ms).round().long(), 1, cfg.delay_buckets)
        return pre, post, w, delay

    def reset_state(self) -> None:
        n, dev = self.n, self.device
        self.v = torch.full((n,), -65.0, device=dev)
        self.u = self.b * self.v
        self.i_exc = torch.zeros(n, device=dev)
        self.i_inh = torch.zeros(n, device=dev)
        self.i_nmda = torch.zeros(n, device=dev)
        self.stp_x = torch.ones(n, device=dev)
        self.history = torch.zeros(self.cfg.delay_buckets + 1, n, device=dev)
        self.head = 0
        self.t_ms = 0.0
        self.torch_gen = torch.Generator(device=dev).manual_seed(self.cfg.seed + 1)

    def _delivery_matrix(self) -> torch.Tensor:
        version = (self.w._version, self.w.data_ptr())
        cache = getattr(self, "_A_cache", None)
        if cache is not None and cache[0] == version:
            return cache[1]
        if cache is None or cache[2] is None:
            # Built on the CPU: the sort's temporaries would crowd the GPU.
            post, pre, delay, exc = (t.cpu() for t in (self.post, self.pre, self.delay, self.syn_exc))
            rows = post + self.n * (~exc).long()
            cols = (delay - 1) * self.n + pre
            order = torch.argsort(rows * (self.cfg.delay_buckets * self.n) + cols)
            crow = torch.zeros(2 * self.n + 1, dtype=torch.long)
            crow[1:] = torch.cumsum(torch.bincount(rows, minlength=2 * self.n), 0)
            pos = torch.empty_like(order)
            pos[order] = torch.arange(len(order))
            structure = (order.to(self.device), crow.to(self.device), cols[order].to(self.device), pos.to(self.device))
            del rows, cols
        else:
            structure = cache[2]
        order, crow, col, _ = structure
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = torch.sparse_csr_tensor(crow, col, self.w[order].float(),
                                        size=(2 * self.n, self.cfg.delay_buckets * self.n), check_invariants=False)
        self._A_cache = (version, A, structure)
        if cache is None and self.device.type == "cuda":
            torch.cuda.empty_cache()
        return A

    def write_weights(self, idx: torch.Tensor, values: torch.Tensor) -> None:
        cache = getattr(self, "_A_cache", None)
        valid = cache is not None and cache[0] == (self.w._version, self.w.data_ptr())
        self.w[idx] = values
        if valid:
            _, A, structure = cache
            A.values()[structure[3][idx]] = values.to(A.dtype)
            self._A_cache = ((self.w._version, self.w.data_ptr()), A, structure)

    def excitatory_input(self) -> torch.Tensor:
        # An ordered sum, so runs repeat bit for bit; index_add_ on the GPU sums with atomics in a varying order.
        A = self._delivery_matrix()
        return (A @ torch.ones(A.shape[1], 1, device=self.device)).squeeze(1)[: self.n]

    def synapses_of(self, neurons: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_syn_lookup", None) is None:
            look = []
            for side in (self.pre, self.post):
                order = torch.argsort(side, stable=True)
                ptr = torch.zeros(self.n + 1, dtype=torch.long, device=self.device)
                ptr[1:] = torch.cumsum(torch.bincount(side, minlength=self.n), 0)
                look.append((order, ptr))
            self._syn_lookup = look
        out = []
        for order, ptr in self._syn_lookup:
            start = ptr[neurons]
            cnt = ptr[neurons + 1] - start
            total = int(cnt.sum())
            if total:
                first = torch.repeat_interleave(start - (torch.cumsum(cnt, 0) - cnt), cnt, output_size=total)
                out.append(order[first + torch.arange(total, device=self.device)])
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=self.device)

    @property
    def n_synapses(self) -> int:
        return int(self.pre.numel())

    def substep(self, i_ext: torch.Tensor | None = None) -> torch.Tensor:
        cfg, dt = self.cfg, self.cfg.dt_ms
        g = self.torch_gen

        k = self.cfg.delay_buckets + 1
        if self.device.type != "cpu":
            A = self._delivery_matrix()
            slots = [(self.head - d + 1) % k for d in range(1, self.cfg.delay_buckets + 1)]
            x = self.history[slots].reshape(-1, 1)
            out = (A @ x).squeeze(1)
            self.i_exc += out[: self.n]
            self.i_inh += out[self.n :]
            if cfg.nmda_ratio:
                self.i_nmda += cfg.nmda_ratio * out[: self.n]
        else:
            slot = (self.head - self.delay + 1) % k
            released = self.history[slot, self.pre]
            active = released > 0
            if bool(active.any()):
                contrib = self.w[active] * released[active]
                target = self.post[active]
                exc = self.syn_exc[active]
                self.i_exc.index_add_(0, target[exc], contrib[exc])
                self.i_inh.index_add_(0, target[~exc], contrib[~exc])
                if cfg.nmda_ratio:
                    self.i_nmda.index_add_(0, target[exc], cfg.nmda_ratio * contrib[exc])

        bg = torch.rand(self.n, generator=g, device=self.device) < cfg.background_rate_hz * dt * 1e-3
        current = (
            self.i_exc
            - self.i_inh
            + cfg.background_w * bg
            + self.noise_scale * torch.randn(self.n, generator=g, device=self.device)
            + self.excitability
        )
        if cfg.nmda_ratio:
            mg_block = 1.0 / (1.0 + (cfg.mg_mM / 3.57) * torch.exp(-0.062 * self.v))
            current = current + mg_block * self.i_nmda
        if i_ext is not None:
            current = current + i_ext

        v, u = self.v, self.u
        for _ in range(2):
            v = v + (dt / 2) * (0.04 * v * v + 5 * v + 140 - u + current)
        u = u + dt * self.a * (self.b * v - u)
        spikes = (v >= 30.0) & self.alive
        v = torch.where(spikes, self.c, v)
        u = torch.where(spikes, u + self.d, u)
        v = torch.where(self.alive, v, torch.full_like(v, -70.0))
        self.v, self.u = v, u

        self.i_exc *= math.exp(-dt / cfg.tau_exc_ms)
        self.i_inh *= math.exp(-dt / cfg.tau_inh_ms)
        if cfg.nmda_ratio:
            self.i_nmda *= math.exp(-dt / cfg.tau_nmda_ms)

        efficacy = cfg.stp_u * self.stp_x * spikes
        self.stp_x = self.stp_x - efficacy
        self.stp_x = self.stp_x + (1 - self.stp_x) * (dt / cfg.stp_tau_rec_ms)

        self.head = (self.head + 1) % k
        self.history[self.head] = efficacy / cfg.stp_u

        if self.plasticity is not None:
            self.plasticity.on_substep(self, spikes)
        self.t_ms += dt
        return spikes

    def run(self, ms: float, i_ext_fn=None) -> torch.Tensor:
        steps = int(round(ms / self.cfg.dt_ms))
        raster = torch.zeros(steps, self.n, dtype=torch.bool, device=self.device)
        for s in range(steps):
            raster[s] = self.substep(i_ext_fn(s) if i_ext_fn else None)
        return raster

    _STATE = ("v", "u", "i_exc", "i_inh", "i_nmda", "stp_x", "history", "w", "w0", "excitability", "alive")

    def state(self) -> dict:
        st = {k: getattr(self, k).cpu().clone() for k in self._STATE}
        return st | {"head": self.head, "t_ms": self.t_ms, "torch_gen": self.torch_gen.get_state(),
                     "gen": self.gen.get_state()}

    def load_state(self, st: dict) -> None:
        for k in self._STATE:
            getattr(self, k).copy_(st[k].to(self.device))
        self.head, self.t_ms = st["head"], st["t_ms"]
        self.torch_gen.set_state(st["torch_gen"])
        self.gen.set_state(st["gen"])

    def kill(self, fraction: float) -> None:
        alive = self.alive.nonzero(as_tuple=True)[0]
        n_kill = int(round(fraction * len(alive)))
        perm = torch.randperm(len(alive), generator=self.gen)[:n_kill].to(self.device)
        self.alive[alive[perm]] = False
