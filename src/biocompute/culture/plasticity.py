from __future__ import annotations

import math

import torch


class STDP:
    def __init__(
        self,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        tau_plus_ms: float = 20.0,
        tau_minus_ms: float = 20.0,
        tau_forget_s: float = 2700.0,
        enabled: bool = True,
        multiplicative: bool = False,
        istdp_eta: float = 0.0,
        target_hz: float = 5.0,
        w_max_inh: float | None = None,
        sparse: bool = True,
        forget_every: int = 20,
    ):
        self.sparse, self.forget_every = sparse, forget_every
        self.a_plus, self.a_minus = a_plus, a_minus
        self.multiplicative = multiplicative
        self.istdp_eta, self.target_hz, self.w_max_inh = istdp_eta, target_hz, w_max_inh
        self._targets = None
        self.tau_plus_ms, self.tau_minus_ms = tau_plus_ms, tau_minus_ms
        self.tau_forget_s = tau_forget_s
        self.enabled = enabled
        self._ready = False

    def _setup(self, culture) -> None:
        dev = culture.device
        self.x_pre = torch.zeros(culture.n, device=dev)
        self.y_post = torch.zeros(culture.n, device=dev)
        self.plastic = culture.syn_exc.clone()
        dt = culture.cfg.dt_ms
        self.dec_plus = math.exp(-dt / self.tau_plus_ms)
        self.dec_minus = math.exp(-dt / self.tau_minus_ms)
        self.forget = dt / (self.tau_forget_s * 1000.0)
        self.w_ref = float(culture.w0[culture.syn_exc].mean())
        self.inh_idx = (~culture.syn_exc).nonzero(as_tuple=True)[0]
        self.inh_pre, self.inh_post = culture.pre[self.inh_idx], culture.post[self.inh_idx]
        w_max = culture.cfg.w_max
        self.w_cap = torch.where(culture.syn_exc, w_max, w_max if self.w_max_inh is None else self.w_max_inh)
        if self._targets is None:
            self._targets = torch.full((culture.n,), float(self.target_hz), device=dev)
        self.alpha = 2.0 * self._targets * 1e-3 * self.tau_minus_ms
        self._ready = True

    def set_targets(self, culture, hz) -> None:
        self._targets = torch.as_tensor(hz, dtype=torch.float32, device=culture.device).clone()
        if self._ready:
            self.alpha = 2.0 * self._targets * 1e-3 * self.tau_minus_ms

    def timing_update(self, culture, spikes: torch.Tensor) -> torch.Tensor:
        if not self._ready:
            self._setup(culture)
        self.x_pre *= self.dec_plus
        self.y_post *= self.dec_minus
        dw = torch.zeros_like(culture.w)
        if bool(spikes.any()):
            s = spikes.float()
            dep = self.a_minus * self.y_post[culture.post] * s[culture.pre]
            if self.multiplicative:
                dep = dep * (culture.w / self.w_ref)
            dw = self.a_plus * self.x_pre[culture.pre] * s[culture.post] - dep
            dw = dw * self.plastic
            if self.istdp_eta:
                pre, post = self.inh_pre, self.inh_post
                dw_i = s[pre] * (self.y_post[post] - self.alpha[post]) + s[post] * self.x_pre[pre]
                dw[self.inh_idx] += self.istdp_eta * dw_i
            self.x_pre += s
            self.y_post += s
        return dw

    def _finish(self, culture) -> None:
        w = culture.w
        w += (culture.w0 - w) * self.forget
        if self.w_max_inh is None:
            w.clamp_(0.0, culture.cfg.w_max)
        else:
            w.clamp_(min=0.0)
            torch.minimum(w, self.w_cap, out=w)

    def state(self) -> dict:
        st = {"ready": self._ready, "targets": None if self._targets is None else self._targets.cpu().clone()}
        if self._ready:
            st |= {"x_pre": self.x_pre.cpu().clone(), "y_post": self.y_post.cpu().clone()}
        return st

    def load_state(self, culture, st: dict) -> None:
        if st["targets"] is not None:
            self._targets = st["targets"].to(culture.device)
        self._ready = False
        if st["ready"]:
            self._setup(culture)
            self.x_pre.copy_(st["x_pre"].to(culture.device))
            self.y_post.copy_(st["y_post"].to(culture.device))

    def on_substep(self, culture, spikes: torch.Tensor) -> None:
        if not self.enabled:
            return
        if not self.sparse:
            dw = self.timing_update(culture, spikes)
            culture.w += dw * culture.cfg.w_max
            self._finish(culture)
            return
        self._sparse_update(culture, spikes)

    def _sparse_update(self, culture, spikes: torch.Tensor) -> None:
        if not self._ready:
            self._setup(culture)
        self.x_pre *= self.dec_plus
        self.y_post *= self.dec_minus
        fired = spikes.nonzero(as_tuple=True)[0]
        if fired.numel():
            syn = culture.synapses_of(fired)
            s = spikes.float()
            pre, post = culture.pre[syn], culture.post[syn]
            w = culture.w[syn]
            dep = self.a_minus * self.y_post[post] * s[pre]
            if self.multiplicative:
                dep = dep * (w / self.w_ref)
            dw = (self.a_plus * self.x_pre[pre] * s[post] - dep) * self.plastic[syn]
            if self.istdp_eta:
                dw_i = s[pre] * (self.y_post[post] - self.alpha[post]) + s[post] * self.x_pre[pre]
                dw = torch.where(culture.syn_exc[syn], dw, dw + self.istdp_eta * dw_i)
            w = w + dw * culture.cfg.w_max
            w = w.clamp(min=0.0)
            w = torch.minimum(w, self.w_cap[syn]) if self.w_max_inh is not None else w.clamp(max=culture.cfg.w_max)
            culture.write_weights(syn, w)
            self.x_pre += s
            self.y_post += s
        k = self.forget_every
        if self.forget and int(round(culture.t_ms / culture.cfg.dt_ms)) % k == k - 1:
            culture.w += (culture.w0 - culture.w) * (1.0 - (1.0 - self.forget) ** k)


class RewardSTDP(STDP):
    def __init__(self, tau_eligibility_ms: float = 1000.0, learning_rate: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tau_eligibility_ms = tau_eligibility_ms
        self.learning_rate = learning_rate
        self.eligibility: torch.Tensor | None = None

    def on_substep(self, culture, spikes: torch.Tensor) -> None:
        if not self.enabled:
            return
        dw = self.timing_update(culture, spikes)
        if self.eligibility is None:
            self.eligibility = torch.zeros_like(culture.w)
        self.eligibility *= math.exp(-culture.cfg.dt_ms / self.tau_eligibility_ms)
        self.eligibility += dw
        self._finish(culture)

    def reward(self, culture, value: float) -> None:
        if self.eligibility is None or not self.enabled:
            return
        culture.w += self.learning_rate * value * self.eligibility * culture.cfg.w_max
        culture.w.clamp_(0.0, culture.cfg.w_max)


class SynapticScaling:
    def __init__(self, eta: float = 1e-5, tau_rate_ms: float = 10000.0, every_ms: float = 200.0,
                 target_hz: float = 5.0):
        self.eta, self.tau_rate_ms, self.every_ms, self.target_hz = eta, tau_rate_ms, every_ms, target_hz
        self._targets = None
        self._ready = False

    def _setup(self, culture) -> None:
        dev = culture.device
        if self._targets is None:
            self._targets = torch.full((culture.n,), float(self.target_hz), device=dev)
        self.rate = self._targets.clone()
        self.ee = (culture.syn_exc & culture.excitatory[culture.post]).nonzero(as_tuple=True)[0]
        self.ee_post = culture.post[self.ee]
        dt = culture.cfg.dt_ms
        self.decay = math.exp(-dt / self.tau_rate_ms)
        self.hz_per_spike = 1000.0 / dt
        self.every = max(int(round(self.every_ms / dt)), 1)
        self.step_ms = self.every * dt
        self.count = 0
        self._ready = True

    def set_targets(self, culture, hz) -> None:
        self._targets = torch.as_tensor(hz, dtype=torch.float32, device=culture.device).clone()
        if self._ready:
            self.rate = self._targets.clone()

    def factor(self, culture) -> torch.Tensor:
        f = torch.exp(self.eta * self.step_ms * (self._targets - self.rate))
        return torch.where(culture.alive, f, torch.ones_like(f))

    def on_substep(self, culture, spikes: torch.Tensor) -> None:
        if not self._ready:
            self._setup(culture)
        self.rate.mul_(self.decay).add_(spikes.float(), alpha=self.hz_per_spike * (1.0 - self.decay))
        self.count += 1
        if self.count % self.every == 0:
            w = culture.w[self.ee] * self.factor(culture)[self.ee_post]
            culture.write_weights(self.ee, w.clamp(0.0, culture.cfg.w_max))

    def state(self) -> dict:
        st = {"ready": self._ready, "targets": None if self._targets is None else self._targets.cpu().clone()}
        if self._ready:
            st |= {"rate": self.rate.cpu().clone(), "count": self.count}
        return st

    def load_state(self, culture, st: dict) -> None:
        if st["targets"] is not None:
            self._targets = st["targets"].to(culture.device)
        self._ready = False
        if st["ready"]:
            self._setup(culture)
            self.rate.copy_(st["rate"].to(culture.device))
            self.count = st["count"]


class Chain:
    def __init__(self, *rules):
        self.rules = list(rules)

    def on_substep(self, culture, spikes: torch.Tensor) -> None:
        for r in self.rules:
            r.on_substep(culture, spikes)

    def state(self) -> list:
        return [r.state() for r in self.rules]

    def load_state(self, culture, st: list) -> None:
        for r, s in zip(self.rules, st, strict=True):  # a state saved from other rules must not load partly
            r.load_state(culture, s)


def rest(culture, seconds: float, plasticity: STDP | None = None) -> None:
    tau = (plasticity.tau_forget_s if plasticity else 2700.0)
    keep = math.exp(-seconds / tau)
    culture.w.copy_(culture.w0 + (culture.w - culture.w0) * keep)
