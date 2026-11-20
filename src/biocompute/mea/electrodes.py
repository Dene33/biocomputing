"""A multi-electrode array under the culture: the only way in and out.

Recording: each electrode detects a spike from a neuron with a probability that falls with distance (extracellular
amplitude ~ 1 / (1 + (d/d0)^2) against a detection threshold in noise), plus false positives. Detection is
multi-unit: an electrode reports spike times, not which neuron fired.

Stimulation: a biphasic pulse at an electrode depolarizes neurons near it, falling off with distance, for one
substep. Pulses are limited in amplitude and rate per electrode. Every pulse blanks recording on all electrodes for
``blank_ms`` (the stimulation artifact), as on real systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MEAConfig:
    rows: int = 8
    cols: int = 8
    pitch_um: float = 200.0
    skip_corners: bool = True
    """A 60-channel layout like Multi Channel Systems' 60MEA: an 8 × 8 grid without its corners."""

    # Recording
    amp_d0_um: float = 30.0
    """Distance at which a spike's amplitude has halved."""
    detect_at_um: float = 50.0
    """Distance at which detection probability is 50%."""
    detect_softness: float = 0.15
    false_positive_hz: float = 0.5
    blank_ms: float = 2.0

    # Stimulation
    stim_current: float = 100.0
    """Current (model units) injected for one substep into a neuron at the electrode, at amplitude 1."""
    stim_length_um: float = 60.0
    max_amplitude: float = 1.0
    max_rate_hz: float = 200.0

    dead_fraction: float = 0.0
    seed: int = 0


MEA_PRESETS: dict[str, dict] = {
    "mcs60": dict(rows=8, cols=8, pitch_um=200.0, skip_corners=True),
    "grid256": dict(rows=16, cols=16, pitch_um=100.0, skip_corners=False),
    "hd1024": dict(rows=32, cols=32, pitch_um=50.0, skip_corners=False),
}


def make_mea_config(preset: str = "mcs60", **overrides) -> MEAConfig:
    cfg = dict(MEA_PRESETS[preset])
    cfg.update(overrides)
    return MEAConfig(**cfg)


class MEA:
    def __init__(self, cfg: MEAConfig, neuron_pos: torch.Tensor, device: torch.device):
        self.cfg, self.device = cfg, device
        xs = (np.arange(cfg.cols) - (cfg.cols - 1) / 2) * cfg.pitch_um
        ys = (np.arange(cfg.rows) - (cfg.rows - 1) / 2) * cfg.pitch_um
        grid = [(x, y) for y in ys for x in xs]
        if cfg.skip_corners:
            corners = {(xs[0], ys[0]), (xs[-1], ys[0]), (xs[0], ys[-1]), (xs[-1], ys[-1])}
            grid = [p for p in grid if p not in corners]
        self.xy = np.array(grid, dtype=np.float32)
        self.n_channels = len(self.xy)
        rng = np.random.default_rng(cfg.seed + 1000)
        self.dead = np.zeros(self.n_channels, dtype=bool)
        n_dead = int(round(cfg.dead_fraction * self.n_channels))
        self.dead[rng.choice(self.n_channels, n_dead, replace=False)] = True

        # Electrode–neuron distances (electrodes at z = 0).
        e = torch.tensor(np.c_[self.xy, np.zeros(self.n_channels)], dtype=torch.float32)
        dist = torch.cdist(e, neuron_pos.float())  # (channels, neurons)
        p = torch.sigmoid((cfg.detect_at_um - dist) / (cfg.detect_softness * cfg.detect_at_um))
        p[p < 1e-3] = 0.0
        p[torch.from_numpy(self.dead)] = 0.0
        self.detect = p.to(device)  # dense; rows are mostly zero but channels × neurons stays small
        drive = cfg.stim_current * torch.exp(-dist / cfg.stim_length_um)
        drive[drive < 0.05 * cfg.stim_current] = 0.0
        drive[torch.from_numpy(self.dead)] = 0.0
        self.drive = drive.to(device)
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed + 2000)

    @property
    def alive_channels(self) -> np.ndarray:
        return np.flatnonzero(~self.dead)

    def stim_current(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Current per neuron for pulses of ``amplitudes`` (channels,) delivered this substep."""
        amps = amplitudes.clamp(0.0, self.cfg.max_amplitude)
        return amps @ self.drive

    def record(self, spikes: torch.Tensor, dt_ms: float) -> torch.Tensor:
        """Multi-unit detections per channel (int counts) for one substep's spikes (bool, neurons)."""
        if bool(spikes.any()):
            p = self.detect[:, spikes]  # (channels, spiking)
            hits = (torch.rand(p.shape, generator=self.gen, device=self.device) < p).sum(1)
        else:
            hits = torch.zeros(self.n_channels, dtype=torch.long, device=self.device)
        fp = torch.rand(self.n_channels, generator=self.gen, device=self.device) < self.cfg.false_positive_hz * dt_ms * 1e-3
        hits = hits + fp.long()
        hits[torch.from_numpy(self.dead).to(self.device)] = 0
        return hits

    def neurons_near(self, channel: int, radius_um: float) -> torch.Tensor:
        """Indices of neurons that a channel's stimulation reaches above ``exp(-radius/length)``."""
        threshold = self.cfg.stim_current * math.exp(-radius_um / self.cfg.stim_length_um)
        return (self.drive[channel] >= threshold).nonzero(as_tuple=True)[0]
