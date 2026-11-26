from __future__ import annotations

import numpy as np


def robot_metrics(ep, seconds: float, dt: float) -> dict:
    pos = np.array(ep.root_pos) if len(ep.root_pos) else np.zeros((1, 3))
    survived = ep.steps * dt if ep.fell_at < 0 else ep.fell_at
    out = {
        "survived_s": float(survived),
        "fell": bool(ep.fell_at >= 0),
        "distance_x_m": float(pos[-1, 0] - pos[0, 0]),
        "mean_speed_x": float((pos[-1, 0] - pos[0, 0]) / max(survived, 1e-9)),
    }
    if len(ep.teacher):
        y, p = np.array(ep.teacher), np.array(ep.student)
        out["action_mse"] = float(((y - p) ** 2).mean())
    return out


def culture_stats(raster: np.ndarray, dt_ms: float, burst_bin_ms: float = 10.0, burst_frac: float = 0.05) -> dict:
    steps, n = raster.shape
    seconds = steps * dt_ms / 1000.0
    rates = raster.sum(0) / seconds
    per_bin = max(1, int(burst_bin_ms / dt_ms))
    nb = steps // per_bin
    pop = raster[: nb * per_bin].reshape(nb, per_bin, n).any(1).sum(1) / n
    active = pop > burst_frac
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ibi = np.diff(starts) * burst_bin_ms / 1000.0
    per100 = max(1, int(100 / dt_ms))
    n100 = steps // per100
    counts = raster[: n100 * per100].reshape(n100, per100, n).sum((1, 2))
    return {
        "seconds": seconds,
        "mean_rate_hz": float(rates.mean()),
        "median_rate_hz": float(np.median(rates)),
        "frac_silent": float((rates < 0.01).mean()),
        "bursts": int(len(starts)),
        "burst_rate_per_min": float(len(starts) / seconds * 60.0),
        "mean_ibi_s": float(ibi.mean()) if len(ibi) else float("nan"),
        "cv_ibi": float(ibi.std() / ibi.mean()) if len(ibi) > 1 else float("nan"),
        "frac_spikes_in_bursts": float(
            raster[: nb * per_bin].reshape(nb, per_bin, n)[active].sum() / max(raster.sum(), 1)
        ),
        "fano_100ms": float(counts.var() / max(counts.mean(), 1e-9)),
    }
