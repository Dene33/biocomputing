"""Spikes → joint actions.

``SpikeFeatures`` turns each step's detections into features: the counts in a few time bins of the step, plus
exponentially filtered rates at several time scales (the readout's short memory), square-root transformed to steady
the Poisson-like variance.

``LinearReadout`` maps features to the 23 actions by ridge regression (the trained decoder of the reservoir
approach). ``MotorMapDecoder`` is the fixed decoder of the DishBrain approach: each joint has two opposing groups of
recording channels and its action is their rate difference, so any learning must happen inside the culture.
"""

from __future__ import annotations

import numpy as np

from biocompute.device import SpikeBlock


class SpikeFeatures:
    def __init__(self, channels: np.ndarray, bins: int = 2, taus_ms: tuple[float, ...] = (40.0, 160.0), step_ms=20.0):
        self.channels = np.asarray(channels, int)
        self.bins, self.taus_ms, self.step_ms = bins, tuple(taus_ms), step_ms
        self.decay = np.exp(-step_ms / np.asarray(self.taus_ms))
        self.reset()

    @property
    def dim(self) -> int:
        return len(self.channels) * (self.bins + len(self.taus_ms))

    def reset(self) -> None:
        self.traces = np.zeros((len(self.taus_ms), len(self.channels)), dtype=np.float32)

    def __call__(self, block: SpikeBlock) -> np.ndarray:
        counts = block.counts(self.bins)[:, self.channels]
        return self.from_counts(counts)

    def from_counts(self, counts: np.ndarray) -> np.ndarray:
        """``counts``: (bins, channels) for one step."""
        total = counts.sum(0)
        self.traces = self.traces * self.decay[:, None] + total[None]
        return np.sqrt(np.concatenate([counts.reshape(-1), self.traces.reshape(-1)])).astype(np.float32)


class LinearReadout:
    def __init__(self, in_dim: int, out_dim: int = 23):
        self.W = np.zeros((in_dim + 1, out_dim), dtype=np.float32)
        self.mean = np.zeros(in_dim, dtype=np.float32)
        self.std = np.ones(in_dim, dtype=np.float32)

    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None
    """Per-output target standardization (set when fitted with ``standardize_y``): each joint is fitted in units
    of its own spread, so joints that move a lot (or are pinned far past a limit) don't swamp the balance joints."""

    def _y_in(self, Y: np.ndarray) -> np.ndarray:
        return Y if self.y_mean is None else (Y - self.y_mean) / self.y_std

    def _y_out(self, P: np.ndarray) -> np.ndarray:
        return P if self.y_mean is None else P * self.y_std + self.y_mean

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: float = 1.0, weights: np.ndarray | None = None,
            standardize_y: bool | None = None) -> None:
        if standardize_y is not None:
            self.y_mean, self.y_std = (Y.mean(0), np.maximum(Y.std(0), 1e-2)) if standardize_y else (None, None)
        Y = self._y_in(Y)
        self.mean = X.mean(0)
        self.std = np.maximum(X.std(0), 1e-3)
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        if weights is not None:
            sw = np.sqrt(weights)[:, None]
            Z, Y = Z * sw, Y * sw
        reg = alpha * len(X) / 1000.0 * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0
        self.W = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y).astype(np.float32)

    def fit_cv(self, X: np.ndarray, Y: np.ndarray, alphas=None, folds: int = 5, standardize_y: bool = False,
               weights: np.ndarray | None = None) -> dict:
        """Choose the ridge penalty by cross-validation over contiguous blocks (neighbouring steps are correlated,
        so shuffled folds would leak), refit on everything, and return the validation R² per alpha."""
        alphas = np.logspace(-2, 5, 15) if alphas is None else np.asarray(alphas)
        Y_raw = Y
        if standardize_y:
            Y = (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-2)
        n = len(X)
        edges = np.linspace(0, n, folds + 1).astype(int)
        scores = np.zeros((len(alphas), folds))
        for f in range(folds):
            val = np.zeros(n, bool)
            val[edges[f] : edges[f + 1]] = True
            mean, std = X[~val].mean(0), np.maximum(X[~val].std(0), 1e-3)
            Zt = (X[~val] - mean) / std
            Zv = (X[val] - mean) / std
            ym = Y[~val].mean(0)
            # Ridge for every alpha from one eigendecomposition (features centred, so the bias is the mean).
            evals, evecs = np.linalg.eigh(Zt.T @ Zt)
            proj = evecs.T @ (Zt.T @ (Y[~val] - ym))
            for i, a in enumerate(alphas):
                W = evecs @ (proj / (evals + a * len(Zt) / 1000.0)[:, None])
                scores[i, f] = r2_pooled(Y[val], Zv @ W + ym)
        mean_scores = scores.mean(1)
        best = int(np.argmax(mean_scores))
        self.fit(X, Y_raw, alpha=float(alphas[best]), standardize_y=standardize_y, weights=weights)
        self.alpha = float(alphas[best])
        return {"alpha": self.alpha, "val_r2": float(mean_scores[best]),
                "val_r2_by_alpha": dict(zip(map(float, alphas), map(float, mean_scores)))}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._y_out(np.r_[(x - self.mean) / self.std, 1.0] @ self.W).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._y_out(np.c_[(X - self.mean) / self.std, np.ones(len(X))] @ self.W).astype(np.float32)


def r2_pooled(Y: np.ndarray, P: np.ndarray) -> float:
    """Variance-weighted R² over all outputs: 1 - total squared error / total variance. Joints that barely move
    (their R² alone is meaningless) count in proportion to how much they move."""
    return float(1.0 - ((Y - P) ** 2).sum() / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12))


def r2_per_output(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


class MotorMapDecoder:
    """Fixed: action_j = gain * (rate(group_j+) - rate(group_j-)) / rate_scale, filtered over ``tau_ms``."""

    def __init__(self, channels: np.ndarray, n_out: int = 23, gain: float = 1.0, rate_scale: float = 2.0,
                 tau_ms: float = 40.0, step_ms: float = 20.0, seed: int = 0):
        channels = np.asarray(channels, int)
        rng = np.random.default_rng(seed)
        channels = rng.permutation(channels)
        groups = np.array_split(channels, 2 * n_out)
        self.plus, self.minus = groups[0::2], groups[1::2]
        self.gain, self.rate_scale = gain, rate_scale
        self.decay = float(np.exp(-step_ms / tau_ms))
        self.n_out = n_out
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(self.n_out, dtype=np.float32)

    def __call__(self, block: SpikeBlock) -> np.ndarray:
        counts = block.counts(1)[0]
        diff = np.array([counts[p].sum() - counts[m].sum() for p, m in zip(self.plus, self.minus)], np.float32)
        self.state = self.decay * self.state + (1 - self.decay) * diff
        return np.tanh(self.gain * self.state / self.rate_scale).astype(np.float32)
