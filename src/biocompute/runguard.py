"""Guards for long runs: catch errors while a run is going, not after it.

Past runs lost hours to errors found only at the end (an iSTDP sweep ran 90 minutes with inhibitory weights clamped
to w_max by an unrelated scaling step; a short run crashed while writing results). Three layers:

- ``Guard`` (here), inside each job: ``beat`` writes a heartbeat file and a progress line; ``check`` and ``culture``
  test invariants and, on failure, print ``GUARD FAIL`` and raise ``GuardFail``, which ends the job.
- ``scripts/launch.py``: smoke test (the whole pipeline at tiny scale, result writing included), detached launch,
  start-up check.
- ``scripts/watch.py``: stops the unit at the first Traceback / GUARD FAIL / stall.

Standard culture checks (``Guard.culture``): finite state; weights within [0, cap]; changed plastic weights not piled
up at a value that is not their configured cap (something else is bounding them); electrode rate plausible.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


class GuardFail(RuntimeError):
    pass


class Guard:
    def __init__(self, name: str, out: str | Path, every_s: float = 30.0):
        self.name = name
        self.path = Path(out) / "heartbeat" / f"{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every_s, self.t_last, self.t0 = every_s, 0.0, time.time()

    def fail(self, what: str) -> None:
        msg = f"GUARD FAIL [{self.name}]: {what}"
        print(msg, flush=True)
        raise GuardFail(msg)

    def check(self, ok: bool, what: str) -> None:
        if not ok:
            self.fail(what)

    def due(self) -> bool:
        """True once every ``every_s``: time for a heartbeat and the (costlier) invariant checks."""
        return time.time() - self.t_last >= self.every_s

    def beat(self, done: float, total: float, force: bool = False, **metrics) -> None:
        """Heartbeat file + progress line, at most every ``every_s`` unless ``force``. Non-finite metrics fail."""
        now = time.time()
        if not force and now - self.t_last < self.every_s:
            return
        self.t_last = now
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isfinite(v):
                self.fail(f"metric {k} = {v}")
        frac = done / total if total else 0.0
        el = now - self.t0
        eta = el / frac * (1 - frac) if frac > 0 else None
        row = {"job": self.name, "done": done, "total": total, "elapsed_s": round(el),
               "eta_s": None if eta is None else round(eta), "time": now, **metrics}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(row))
        os.replace(tmp, self.path)
        m = " ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"[beat] {self.name} {done}/{total} ({frac:.0%}, {el / 60:.1f} min, eta "
              f"{'?' if eta is None else f'{eta / 60:.1f} min'}) {m}", flush=True)

    def culture(self, c, stdp=None, rate_hz: float | None = None, max_rate_hz: float = 300.0) -> None:
        """Standard invariants of a culture (and its plasticity) mid-run."""
        import torch

        for k in ("v", "u", "w"):
            self.check(bool(torch.isfinite(getattr(c, k)).all()), f"culture.{k} has NaN/Inf")
        w, w0 = c.w, c.w0
        self.check(float(w.min()) >= 0.0, f"negative weight {float(w.min()):.4g}")
        inh_cap = c.cfg.w_max if stdp is None or getattr(stdp, "w_max_inh", None) is None else stdp.w_max_inh
        for cls, mask, cap in (("excitatory", c.syn_exc, c.cfg.w_max), ("inhibitory", ~c.syn_exc, inh_cap)):
            wc = w[mask]
            top = float(wc.max())
            self.check(top <= cap + 1e-3, f"{cls} weight {top:.4g} above its cap {cap}")
            moved = wc[wc != w0[mask]]
            if moved.numel() > 1000:
                top_m = float(moved.max())
                share = float((moved >= top_m - 1e-4).float().mean())
                self.check(share < 0.3 or abs(top_m - cap) < 1e-3,
                           f"{share:.0%} of the changed {cls} weights sit at {top_m:.4g}, not at their cap {cap}: "
                           f"something else bounds them")
        if rate_hz is not None:
            self.check(0.0 < rate_hz < max_rate_hz, f"rate {rate_hz:.4g} Hz per electrode outside (0, {max_rate_hz})")
