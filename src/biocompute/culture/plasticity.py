"""Long-term plasticity of the culture's excitatory synapses.

- ``STDP``: pair-based spike-timing-dependent plasticity with exponential traces, applied as it happens. This is
  what a real culture has, and what DishBrain-style feedback shapes: predictable stimulation produces consistent
  timing and so consistent change; unpredictable stimulation scrambles it.
- ``RewardSTDP``: the same timing rule fills an eligibility trace, and weights change only when a reward arrives
  (a dopamine-like global signal, Izhikevich 2007). A real dish has no such signal we can deliver through
  electrodes: it is the in-silico upper bound, not a protocol for hardware.

Both decay towards the culture's baseline weights with ``tau_forget_s``, the slow loss of what was learned during
rest.

``multiplicative=False`` (default) is additive STDP with hard bounds: every pairing moves a weight by the same
amount. In this culture it is unstable under repeated stimulation: the input with the stronger response is
potentiated until it fires the whole network (scripts/cartpole_replication.py --drift). ``multiplicative=True``
scales depression with the weight, relative to the culture's mean initial excitatory weight (van Rossum, Bi &
Turrigiano 2000): strong synapses depress harder, so weights settle around that mean instead of running to the bound.
"""

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
    ):
        self.a_plus, self.a_minus = a_plus, a_minus
        self.multiplicative = multiplicative
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
        self._ready = True

    def timing_update(self, culture, spikes: torch.Tensor) -> torch.Tensor:
        """The STDP change for every synapse this substep (zeros where not plastic)."""
        if not self._ready:
            self._setup(culture)
        self.x_pre *= self.dec_plus
        self.y_post *= self.dec_minus
        dw = torch.zeros_like(culture.w)
        if bool(spikes.any()):
            s = spikes.float()
            # Potentiation: post fires after pre (pre trace still up). Depression: pre fires after post.
            dep = self.a_minus * self.y_post[culture.post] * s[culture.pre]
            if self.multiplicative:
                dep = dep * (culture.w / self.w_ref)
            dw = self.a_plus * self.x_pre[culture.pre] * s[culture.post] - dep
            dw = dw * self.plastic
            self.x_pre += s
            self.y_post += s
        return dw

    def _finish(self, culture) -> None:
        w = culture.w
        w += (culture.w0 - w) * self.forget
        w.clamp_(0.0, culture.cfg.w_max)

    def on_substep(self, culture, spikes: torch.Tensor) -> None:
        if not self.enabled:
            return
        dw = self.timing_update(culture, spikes)
        culture.w += dw * culture.cfg.w_max  # a_plus/a_minus are fractions of the maximum weight
        self._finish(culture)


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
        """Deliver a reward (positive or negative): weights move along the eligibility trace."""
        if self.eligibility is None or not self.enabled:
            return
        culture.w += self.learning_rate * value * self.eligibility * culture.cfg.w_max
        culture.w.clamp_(0.0, culture.cfg.w_max)


def rest(culture, seconds: float, plasticity: STDP | None = None) -> None:
    """A rest period without stimulation, fast-forwarded: only forgetting acts (spontaneous activity during rest is
    not simulated, so this is the decay-only limit)."""
    tau = (plasticity.tau_forget_s if plasticity else 2700.0)
    keep = math.exp(-seconds / tau)
    culture.w.copy_(culture.w0 + (culture.w - culture.w0) * keep)
