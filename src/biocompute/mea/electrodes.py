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

    wells: int = 1
    """One grid per well of a multi-well plate, ``well_spacing_um`` apart along x (matching the culture's wells).
    Channels are numbered well by well."""
    well_spacing_um: float = 10000.0


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
        grid = [(x + k * cfg.well_spacing_um, y) for k in range(cfg.wells) for x, y in grid]
        self.xy = np.array(grid, dtype=np.float32)
        self.channels_per_well = len(grid) // cfg.wells
        self.n_channels = len(self.xy)
        rng = np.random.default_rng(cfg.seed + 1000)
        self.dead = np.zeros(self.n_channels, dtype=bool)
        n_dead = int(round(cfg.dead_fraction * self.n_channels))
        self.dead[rng.choice(self.n_channels, n_dead, replace=False)] = True

        # Electrode–neuron distances (electrodes at z = 0), one block per well: electrodes only see their own well.
        K = cfg.wells
        C, N = self.channels_per_well, neuron_pos.shape[0] // K
        self.wells, self.neurons_per_well = K, N
        e = torch.tensor(np.c_[self.xy, np.zeros(self.n_channels)], dtype=torch.float32).reshape(K, C, 3)
        dist = torch.cdist(e, neuron_pos.float().reshape(K, N, 3))  # (wells, channels, neurons)
        dead = torch.from_numpy(self.dead).reshape(K, C)
        p = torch.sigmoid((cfg.detect_at_um - dist) / (cfg.detect_softness * cfg.detect_at_um))
        p[p < 1e-3] = 0.0
        p[dead] = 0.0
        self.detect3 = p.to(device)
        drive = cfg.stim_current * torch.exp(-dist / cfg.stim_length_um)
        drive[drive < 0.05 * cfg.stim_current] = 0.0
        drive[dead] = 0.0
        self.drive3 = drive.to(device)
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed + 2000)

    @property
    def detect(self) -> torch.Tensor:
        """Detection probability, (channels, neurons) — dense, for inspection (zero across wells)."""
        return torch.block_diag(*self.detect3)

    @property
    def drive(self) -> torch.Tensor:
        return torch.block_diag(*self.drive3)

    @property
    def alive_channels(self) -> np.ndarray:
        return np.flatnonzero(~self.dead)

    def stim_current(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Current per neuron for pulses of ``amplitudes`` (channels,) delivered this substep."""
        amps = amplitudes.clamp(0.0, self.cfg.max_amplitude).reshape(self.wells, 1, -1)
        return torch.bmm(amps, self.drive3).reshape(-1)

    def record(self, spikes: torch.Tensor, dt_ms: float) -> torch.Tensor:
        """Multi-unit detections per channel (int counts) for one substep's spikes (bool, neurons)."""
        hits = torch.zeros(self.n_channels, dtype=torch.long, device=self.device)
        n_idx = spikes.nonzero(as_tuple=True)[0]
        if len(n_idx):
            p = self.detect3[n_idx // self.neurons_per_well, :, n_idx % self.neurons_per_well]  # (spiking, C)
            s, c = (torch.rand(p.shape, generator=self.gen, device=self.device) < p).nonzero(as_tuple=True)
            hits.index_add_(0, (n_idx[s] // self.neurons_per_well) * self.channels_per_well + c,
                            torch.ones_like(c))
        fp = torch.rand(self.n_channels, generator=self.gen, device=self.device) < self.cfg.false_positive_hz * dt_ms * 1e-3
        hits = hits + fp.long()
        hits[torch.from_numpy(self.dead).to(self.device)] = 0
        return hits

    def record_step(self, raster: torch.Tensor, blank: torch.Tensor, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
        """Detections for a whole control step at once: ``raster`` (substeps, neurons) bool, ``blank`` (substeps,)
        bool. Same statistics as calling ``record`` per unblanked substep, with one host sync instead of one per
        substep. Returns (channel, substep) per detection."""
        live = raster & ~blank[:, None]
        s_idx, n_idx = live.nonzero(as_tuple=True)
        ch_out, st_out = [], []
        for i in range(0, len(n_idx), 32768):  # chunks: a network burst can put 10⁵ spikes in one step
            ni, si = n_idx[i : i + 32768], s_idx[i : i + 32768]
            well = ni // self.neurons_per_well
            p = self.detect3[well, :, ni % self.neurons_per_well]  # (spikes, channels of that well)
            hit_k, hit_c = (torch.rand(p.shape, generator=self.gen, device=self.device) < p).nonzero(as_tuple=True)
            ch_out.append(well[hit_k] * self.channels_per_well + hit_c)
            st_out.append(si[hit_k])
            del p
        fp = torch.rand((raster.shape[0], self.n_channels), generator=self.gen, device=self.device)
        fp = (fp < self.cfg.false_positive_hz * dt_ms * 1e-3) & ~blank[:, None]
        fp_s, fp_c = fp.nonzero(as_tuple=True)
        ch_out.append(fp_c)
        st_out.append(fp_s)
        ch = torch.cat(ch_out).cpu().numpy()
        st = torch.cat(st_out).cpu().numpy()
        keep = ~self.dead[ch]
        return ch[keep], st[keep]

    def neurons_near(self, channel: int, radius_um: float) -> torch.Tensor:
        """Indices of neurons that a channel's stimulation reaches above ``exp(-radius/length)``."""
        threshold = self.cfg.stim_current * math.exp(-radius_um / self.cfg.stim_length_um)
        w, c = divmod(int(channel), self.channels_per_well)
        return (self.drive3[w, c] >= threshold).nonzero(as_tuple=True)[0] + w * self.neurons_per_well
