"""Robot observation → stimulation pulses.

Pulses are delivered only in fixed time slots of each control step (default: one slot at 0 ms). Every pulse blanks
recording on all channels for a couple of milliseconds, so stimulating all channels in the same slot leaves the
rest of the step free to record, as closed-loop MEA experiments do.

- ``amplitude`` coding: every sensory channel pulses in every slot; the amplitude carries the value.
- ``rate`` coding: fixed amplitude; a channel pulses in a slot when its phase accumulator crosses 1, so the pulse rate
  carries the value (at most one pulse per slot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObsNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, obs: np.ndarray, min_std: float = 1e-2) -> ObsNormalizer:
        return cls(obs.mean(0), np.maximum(obs.std(0), min_std))

    @classmethod
    def identity(cls, dim: int) -> ObsNormalizer:
        return cls(np.zeros(dim), np.ones(dim))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / self.std, -4.0, 4.0)


class Encoder:
    def __init__(
        self,
        channels: np.ndarray,
        obs_dim: int = 78,
        mode: str = "amplitude",
        projection: str = "random",
        slots_ms: tuple[float, ...] = (0.0,),
        gain: float = 1.5,
        min_amplitude: float = 0.1,
        rate_amplitude: float = 0.8,
        normalizer: ObsNormalizer | None = None,
        seed: int = 0,
    ):
        self.channels = np.asarray(channels, int)
        self.seed = seed
        self.mode, self.projection, self.slots_ms = mode, projection, tuple(slots_ms)
        self.gain, self.min_amplitude, self.rate_amplitude = gain, min_amplitude, rate_amplitude
        self.normalizer = normalizer or ObsNormalizer.identity(obs_dim)
        k = len(self.channels)
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((k, obs_dim)) / np.sqrt(obs_dim)
        self.b = np.zeros(k)
        self.phase = np.zeros(k)

    def fit_pca(self, obs: np.ndarray) -> None:
        """Project onto the observation's principal components (each channel one signed component), instead of a
        random projection: the most variance per electrode."""
        z = self.normalizer(obs)
        centred = z - z.mean(0)
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        k = min(len(self.channels), vt.shape[0])
        sd = s[:k] / np.sqrt(len(z))  # each component's standard deviation
        comps = vt[:k] / np.maximum(sd, 1e-6)[:, None]  # whitened: each projection has unit variance
        W = np.zeros((len(self.channels), z.shape[1]))
        W[:k] = comps  # one component per channel; the sigmoid centres it on 0.5, so both signs are coded
        self.W = W
        self.b = -self.W @ z.mean(0)
        self.projection = "pca"

    def fit_identity(self, obs: np.ndarray, skip=(6, 7, 8)) -> None:
        """One channel per observation dimension, z-scored (the normalizer's job), centred on the data mean. No
        whitening: PCA whitening blows near-constant directions up to unit variance, and any random mixing of the
        channels (by neurons, or by the next wells' rotations) is then dominated by them. ``skip``: dimensions not to
        encode (default: the velocity command, constant when standing)."""
        dims = [i for i in range(obs.shape[1]) if i not in set(skip)]
        if len(dims) != len(self.channels):
            raise ValueError(f"identity encoding needs {len(dims)} channels, got {len(self.channels)}")
        self.W = np.eye(obs.shape[1])[dims]
        self.b = -self.W @ self.normalizer(obs).mean(0)
        self.projection = "identity"

    def fit_supervised(self, obs: np.ndarray, actions: np.ndarray, ridge: float = 1.0) -> None:
        """Project onto the observation directions that predict the actions (reduced-rank regression), one per
        channel, whitened. PCA spends electrodes on whatever varies most; this spends them on what the task needs."""
        z = self.normalizer(obs)
        zc = z - z.mean(0)
        y = (actions - actions.mean(0)) / np.maximum(actions.std(0), 1e-2)
        B = np.linalg.solve(zc.T @ zc + ridge * len(z) * np.eye(z.shape[1]) / 1000.0, zc.T @ y)  # (obs, act)
        _, _, vt = np.linalg.svd(zc @ B, full_matrices=False)
        k = min(len(self.channels), vt.shape[0])
        dirs = B @ vt[:k].T  # (obs, k): obs directions whose projections give the top fitted-action components
        # Fill any remaining channels with leading principal components of what is left.
        if k < len(self.channels):
            resid = zc - zc @ dirs @ np.linalg.pinv(dirs)
            _, _, vr = np.linalg.svd(resid, full_matrices=False)
            dirs = np.c_[dirs, vr[: len(self.channels) - k].T]
        proj = zc @ dirs
        dirs = dirs / np.maximum(proj.std(0), 1e-6)
        self.W = dirs.T
        self.b = -self.W @ z.mean(0)
        self.projection = "supervised"

    def fit_select(self, obs: np.ndarray, dims, spread: float = 1.0) -> None:
        """Population code of a few observation dimensions: each channel is tuned to a random direction in the
        (normalized) subspace of ``dims``, with a random threshold, so the channels together tile that subspace
        (like sensory neurons with different preferred stimuli and thresholds). Every dimension gets redundant,
        nonlinear coverage instead of one electrode each."""
        dims = np.asarray(dims, int)
        rng = np.random.default_rng(self.seed + 7)
        z = self.normalizer(obs)[:, dims]
        mean = z.mean(0)
        k = len(self.channels)
        dirs = rng.standard_normal((k, len(dims)))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        W = np.zeros((k, self.W.shape[1]))
        W[:, dims] = dirs
        self.W = W
        self.b = -dirs @ mean + spread * rng.uniform(-1.0, 1.0, k)
        self.projection = "select"
        self.dims = dims

    def fit_axis(self, obs: np.ndarray, dims, spread: float = 1.5) -> None:
        """Population code with each channel tuned to one of ``dims``: channel i codes dimension ``dims[i % n]``,
        alternating sign, with thresholds staggered evenly over +-``spread`` (normalized units) among the channels of
        that dimension. Unlike ``fit_select``, a dimension can be read from its own electrodes without unmixing."""
        dims = np.asarray(dims, int)
        z = self.normalizer(obs)[:, dims]
        mean = z.mean(0)
        k, n = len(self.channels), len(dims)
        W = np.zeros((k, self.W.shape[1]))
        b = np.zeros(k)
        for j in range(n):
            mine = np.arange(j, k, n)
            sign = np.where(np.arange(len(mine)) % 2 == 0, 1.0, -1.0)
            thr = np.linspace(-spread, spread, len(mine)) if len(mine) > 1 else np.zeros(1)
            W[mine, dims[j]] = sign
            b[mine] = -sign * (mean[j] + thr)
        self.W, self.b = W, b
        self.projection = "axis"
        self.dims = dims

    def reset(self) -> None:
        self.phase[:] = 0.0

    def drive(self, obs: np.ndarray) -> np.ndarray:
        """Per channel, a value in (0, 1)."""
        z = self.normalizer(obs)
        return 1.0 / (1.0 + np.exp(-self.gain * (self.W @ z + self.b)))

    def pulses(self, obs: np.ndarray, step_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(channels, times_ms, amplitudes) for one control step."""
        x = self.drive(obs)
        chans, times, amps = [], [], []
        if self.mode == "amplitude":
            a = self.min_amplitude + (1 - self.min_amplitude) * x
            for t in self.slots_ms:
                chans.append(self.channels)
                times.append(np.full(len(self.channels), t))
                amps.append(a)
        elif self.mode == "rate":
            per_slot = x  # probability-like: at x = 1, a pulse in every slot
            for t in self.slots_ms:
                self.phase += per_slot
                fire = self.phase >= 1.0
                self.phase[fire] -= 1.0
                chans.append(self.channels[fire])
                times.append(np.full(int(fire.sum()), t))
                amps.append(np.full(int(fire.sum()), self.rate_amplitude))
        else:
            raise ValueError(f"Unknown encoding {self.mode!r}.")
        return np.concatenate(chans), np.concatenate(times), np.concatenate(amps)
