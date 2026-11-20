"""A digital organoid: a spiking network with the properties that matter when a culture sits on an MEA.

- Izhikevich neurons (Izhikevich 2003): 80% excitatory, spread between regular spiking, intrinsically bursting and
  chattering by a random ``r``; 20% inhibitory, between fast spiking and low-threshold spiking.
- Neurons placed in a flattened organoid (a disc of radius ``radius_um``, ``thickness_um`` thick) resting on the
  electrode array at z = 0.
- Sparse, distance-dependent connectivity, with axonal delays that grow with distance (in ``delay_buckets`` of 0.5 ms).
- Exponentially decaying synaptic currents.
- Tsodyks–Markram short-term depression at every presynaptic neuron: a burst uses up resources and ends, so
  spontaneous activity comes as network bursts separated by seconds of quiet, as recorded from organoids on MEAs.
- Background drive: Poisson input plus a small noise current.

Plasticity lives in ``plasticity.py`` and operates on ``self.w``. Everything runs in torch, on CPU or GPU, seeded.
"""

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

    # Connectivity: p(i -> j) = p_max * exp(-distance / length_um)
    synapses_per_neuron: float = 60.0
    length_um: float = 150.0
    w_exc: float = 2.2
    """Mean excitatory weight (pA-like units in Izhikevich's model)."""
    w_inh: float = -6.0
    w_spread: float = 0.5
    """Lognormal sigma of the weights."""
    w_max: float = 10.0
    conduction_um_per_ms: float = 300.0
    min_delay_ms: float = 0.5
    delay_buckets: int = 8

    tau_exc_ms: float = 5.0
    tau_inh_ms: float = 10.0

    # Short-term depression (Tsodyks–Markram)
    stp_u: float = 0.2
    stp_tau_rec_ms: float = 2000.0

    # Background
    background_rate_hz: float = 2.0
    background_w: float = 12.0
    noise_std: float = 3.0
    noise_inh_scale: float = 0.4
    """Inhibitory neurons get this fraction of the noise (as in Izhikevich 2003), else they fire tonically."""

    seed: int = 0
    device: str = "cpu"

    # Filled by presets; documented here for discoverability.
    extra: dict = field(default_factory=dict)


PRESETS: dict[str, dict] = {
    # Fast on a laptop CPU; the default for tests and quick runs.
    "cpu_2k": dict(n_neurons=2000, synapses_per_neuron=60),
    # Larger culture for the GPU.
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
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        n = cfg.n_neurons
        n_exc = int(round(n * cfg.frac_excitatory))
        self.n, self.n_exc = n, n_exc
        g = self.gen

        # --- Placement: a disc lying on the array.
        r = cfg.radius_um * torch.sqrt(torch.rand(n, generator=g))
        theta = 2 * math.pi * torch.rand(n, generator=g)
        z = cfg.thickness_um * torch.rand(n, generator=g)
        self.pos = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1)  # (n, 3) µm, CPU

        # --- Izhikevich parameters (Izhikevich 2003, heterogeneous).
        re = torch.rand(n_exc, generator=g)
        ri = torch.rand(n - n_exc, generator=g)
        self.excitatory = torch.zeros(n, dtype=torch.bool)
        self.excitatory[:n_exc] = True
        a = torch.cat([torch.full((n_exc,), 0.02), 0.02 + 0.08 * ri])
        b = torch.cat([torch.full((n_exc,), 0.2), 0.25 - 0.05 * ri])
        c = torch.cat([-65 + 15 * re**2, torch.full((n - n_exc,), -65.0)])
        d = torch.cat([8 - 6 * re**2, torch.full((n - n_exc,), 2.0)])

        # --- Connectivity.
        pre, post, w, delay = self._connect()

        to = dict(device=dev)
        self.a, self.b, self.c, self.d = (x.to(**to) for x in (a, b, c, d))
        self.excitatory = self.excitatory.to(**to)
        self.pre, self.post, self.delay = pre.to(**to), post.to(**to), delay.to(**to)
        self.w = w.to(**to)
        self.w0 = self.w.clone()
        """Baseline weights, towards which plasticity decays (forgetting)."""
        self.syn_exc = self.excitatory[self.pre]
        self.syn_exc_f = self.syn_exc.float()
        self.syn_inh_f = 1.0 - self.syn_exc_f
        self.noise_scale = torch.where(self.excitatory, 1.0, cfg.noise_inh_scale).float() * cfg.noise_std
        self.alive = torch.ones(n, dtype=torch.bool, device=dev)
        self.excitability = torch.zeros(n, device=dev)
        """Extra constant current per neuron (drift, set by aging)."""
        self.plasticity = None
        """An object with ``on_substep(culture, spikes)``, see plasticity.py."""

        self.reset_state()

    # ------------------------------------------------------------------------------------------------------------
    def _connect(self):
        cfg, g, n = self.cfg, self.gen, self.cfg.n_neurons
        # Expected synapses per neuron = p_max * sum_j exp(-d_ij / L); estimate the sum from a sample to set p_max.
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
            p[torch.arange(len(idx)), idx] = 0.0  # no autapses
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
        self.stp_x = torch.ones(n, device=dev)
        self.history = torch.zeros(self.cfg.delay_buckets + 1, n, device=dev)
        """Ring of the last spikes × release efficacy, newest at ``self.head``."""
        self.head = 0
        self.t_ms = 0.0
        self.torch_gen = torch.Generator(device=dev).manual_seed(self.cfg.seed + 1)

    @property
    def n_synapses(self) -> int:
        return int(self.pre.numel())

    # ------------------------------------------------------------------------------------------------------------
    def substep(self, i_ext: torch.Tensor | None = None) -> torch.Tensor:
        """Advance ``dt_ms``. ``i_ext``: external current per neuron (stimulation). Returns spikes (bool, n)."""
        cfg, dt = self.cfg, self.cfg.dt_ms
        g = self.torch_gen

        # Synaptic input from spikes that left their neuron ``delay`` substeps ago.
        k = self.cfg.delay_buckets + 1
        slot = (self.head - self.delay + 1) % k  # delay 1 = the spikes of the previous substep, at head
        released = self.history[slot, self.pre]  # efficacy × spike, per synapse
        if self.device.type != "cpu":
            # On the GPU, deliver densely: no host sync to find the active synapses (zeros add nothing).
            contrib = self.w * released
            self.i_exc.index_add_(0, self.post, contrib * self.syn_exc_f)
            self.i_inh.index_add_(0, self.post, contrib * self.syn_inh_f)
            active = None
        else:
            active = released > 0
        if active is not None and bool(active.any()):
            contrib = self.w[active] * released[active]
            target = self.post[active]
            exc = self.syn_exc[active]
            self.i_exc.index_add_(0, target[exc], contrib[exc])
            self.i_inh.index_add_(0, target[~exc], contrib[~exc])

        # Background Poisson drive and noise.
        bg = torch.rand(self.n, generator=g, device=self.device) < cfg.background_rate_hz * dt * 1e-3
        current = (
            self.i_exc
            - self.i_inh
            + cfg.background_w * bg
            + self.noise_scale * torch.randn(self.n, generator=g, device=self.device)
            + self.excitability
        )
        if i_ext is not None:
            current = current + i_ext

        # Izhikevich, integrated in two halves of dt for stability.
        v, u = self.v, self.u
        for _ in range(2):
            v = v + (dt / 2) * (0.04 * v * v + 5 * v + 140 - u + current)
        u = u + dt * self.a * (self.b * v - u)
        spikes = (v >= 30.0) & self.alive
        v = torch.where(spikes, self.c, v)
        u = torch.where(spikes, u + self.d, u)
        v = torch.where(self.alive, v, torch.full_like(v, -70.0))
        self.v, self.u = v, u

        # Synaptic decay.
        self.i_exc *= math.exp(-dt / cfg.tau_exc_ms)
        self.i_inh *= math.exp(-dt / cfg.tau_inh_ms)

        # Short-term depression: release U * x on a spike, recover towards 1.
        efficacy = cfg.stp_u * self.stp_x * spikes
        self.stp_x = self.stp_x - efficacy
        self.stp_x = self.stp_x + (1 - self.stp_x) * (dt / cfg.stp_tau_rec_ms)

        self.head = (self.head + 1) % k
        # Scale so a fully recovered synapse releases with efficacy 1.
        self.history[self.head] = efficacy / cfg.stp_u

        if self.plasticity is not None:
            self.plasticity.on_substep(self, spikes)
        self.t_ms += dt
        return spikes

    def run(self, ms: float, i_ext_fn=None) -> torch.Tensor:
        """Advance ``ms``; returns a (substeps, n) bool raster. ``i_ext_fn(substep) -> current or None``."""
        steps = int(round(ms / self.cfg.dt_ms))
        raster = torch.zeros(steps, self.n, dtype=torch.bool, device=self.device)
        for s in range(steps):
            raster[s] = self.substep(i_ext_fn(s) if i_ext_fn else None)
        return raster

    def kill(self, fraction: float) -> None:
        """Cell loss: a random ``fraction`` of the living neurons stops firing for good."""
        alive = self.alive.nonzero(as_tuple=True)[0]
        n_kill = int(round(fraction * len(alive)))
        perm = torch.randperm(len(alive), generator=self.gen)[:n_kill].to(self.device)
        self.alive[alive[perm]] = False
