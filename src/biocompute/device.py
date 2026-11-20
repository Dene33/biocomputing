"""The biocomputer as the robot sees it: channels to stimulate and spikes to read, nothing else.

``Biocomputer`` is the contract a real device adapter (Cortical Labs CL1, FinalSpark Neuroplatform, a MaxWell
system) would implement. ``SimulatedBiocomputer`` implements it with a digital organoid on a simulated MEA.
Only the simulator offers ``oracle_counts`` (every neuron's spikes), for the oracle-readout control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from biocompute.culture.network import CultureConfig, OrganoidCulture
from biocompute.mea.electrodes import MEA, MEAConfig


@dataclass
class SpikeBlock:
    """Detected spikes during one ``step``: channel and time (ms from the step's start) of each."""

    channels: np.ndarray
    times_ms: np.ndarray
    duration_ms: float
    n_channels: int

    def counts(self, bins: int = 1) -> np.ndarray:
        """(bins, channels) spike counts in ``bins`` equal time bins."""
        edges = np.minimum((self.times_ms / self.duration_ms * bins).astype(int), bins - 1)
        out = np.zeros((bins, self.n_channels), dtype=np.float32)
        np.add.at(out, (edges, self.channels), 1.0)
        return out


class Biocomputer(Protocol):
    n_channels: int

    def stimulate(self, channels: np.ndarray, times_ms: np.ndarray, amplitudes: np.ndarray) -> None:
        """Queue biphasic pulses for the next ``step``: channel, time from the step's start (ms), amplitude (0–1)."""

    def step(self, ms: float) -> SpikeBlock:
        """Let the culture run ``ms`` with the queued pulses; return what the electrodes detected."""


class SimulatedBiocomputer:
    def __init__(self, culture_cfg: CultureConfig, mea_cfg: MEAConfig, keep_raster: bool = False):
        self.culture = OrganoidCulture(culture_cfg)
        self.mea = MEA(mea_cfg, self.culture.pos, self.culture.device)
        self.n_channels = self.mea.n_channels
        self.dt = culture_cfg.dt_ms
        self.keep_raster = keep_raster
        self._queue: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._last_pulse_ms = np.full(self.n_channels, -np.inf)
        self.last_raster: torch.Tensor | None = None
        self.oracle_counts: np.ndarray | None = None
        self.pulses_delivered = 0
        self.pulses_dropped = 0

    @property
    def device(self) -> torch.device:
        return self.culture.device

    def stimulate(self, channels, times_ms, amplitudes) -> None:
        self._queue.append((np.asarray(channels, int), np.asarray(times_ms, float), np.asarray(amplitudes, float)))

    def step(self, ms: float) -> SpikeBlock:
        steps = int(round(ms / self.dt))
        t0 = self.culture.t_ms
        amps = np.zeros((steps, self.n_channels), dtype=np.float32)
        if self._queue:
            ch = np.concatenate([q[0] for q in self._queue])
            tt = np.concatenate([q[1] for q in self._queue])
            aa = np.concatenate([q[2] for q in self._queue])
            order = np.argsort(tt, kind="stable")
            min_gap = 1000.0 / self.mea.cfg.max_rate_hz
            for c, t, a in zip(ch[order], tt[order], aa[order]):
                s = int(t / self.dt)
                if not 0 <= s < steps or self.mea.dead[c] or a <= 0:
                    self.pulses_dropped += 1
                    continue
                if t0 + t - self._last_pulse_ms[c] < min_gap - 1e-9:
                    self.pulses_dropped += 1  # over the electrode's rate limit
                    continue
                self._last_pulse_ms[c] = t0 + t
                amps[s, c] = max(amps[s, c], min(a, self.mea.cfg.max_amplitude))
                self.pulses_delivered += 1
            self._queue.clear()

        pulse_steps = np.flatnonzero(amps.any(1))
        blank = np.zeros(steps, dtype=bool)
        blank_len = int(np.ceil(self.mea.cfg.blank_ms / self.dt))
        for s in pulse_steps:
            blank[s : s + blank_len] = True

        amps_t = torch.from_numpy(amps).to(self.device)
        channels, times = [], []
        raster = torch.zeros(steps, self.culture.n, dtype=torch.bool, device=self.device)
        pulsed = set(pulse_steps.tolist())
        for s in range(steps):
            i_ext = self.mea.stim_current(amps_t[s]) if s in pulsed else None
            raster[s] = self.culture.substep(i_ext)
        ch, st = self.mea.record_step(raster, torch.from_numpy(blank).to(self.device), self.dt)
        order = np.argsort(st, kind="stable")
        channels.append(ch[order])
        times.append(st[order] * self.dt)
        self.oracle_counts = raster.sum(0).float().cpu().numpy()
        if self.keep_raster:
            self.last_raster = raster
        return SpikeBlock(
            np.concatenate(channels) if channels else np.zeros(0, int),
            np.concatenate(times) if times else np.zeros(0),
            ms,
            self.n_channels,
        )
