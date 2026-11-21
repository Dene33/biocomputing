"""A model of the vestibular dish as a channel: what the robot side receives instead of the true IMU (obs[0:6]).

The true signal is delayed, sampled at a reduced rate (sample and hold), corrupted by noise (white or AR(1), in units of
each dimension's spread) and optionally quantised, in that order, as scripts/blind_teacher.py's ``chain``. Parameters
come from the Phase 1 spec (docs/next-steps.md) and, once measured, from the Phase 3 probe of the simulated dish.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

STANDING_IMU = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
# Spread of obs[0:6] over teacher stand_push rollouts (outputs/blind_teacher/results.json).
IMU_SPREAD = np.array([0.0368, 0.0277, 0.0368, 0.0186, 0.0256, 0.0014])


@dataclass
class DishModel:
    delay: int = 0
    """Control steps (20 ms) of added latency."""
    hold: int = 1
    """A new value every ``hold`` steps (50 / hold Hz)."""
    noise: float | list = 0.0
    """Noise std in units of each dimension's spread (one value, or one per dimension)."""
    tau: float | list = 0.0
    """Noise correlation time in steps (0: white; one value, or one per dimension)."""
    gain: float | list = 1.0
    """Gain around the standing values (a decoder shrinks large values: gain < 1)."""
    renorm_gravity: bool = False
    """Rebuild grav_z from grav_x and grav_y (unit vector), as a controller would: grav_z barely moves, so its decoded
    value is mostly noise."""
    quant: int = 0
    """Levels per dimension over standing +- 3 spreads (0: graded)."""
    silenced: bool = False
    """The dish is off: the robot side receives the standing values."""

    def to_dict(self) -> dict:
        return asdict(self)


class DishChannel:
    def __init__(self, model: DishModel, rng: np.random.Generator, spread: np.ndarray = IMU_SPREAD):
        self.m, self.rng, self.spread = model, rng, np.asarray(spread, float)
        self.buf: deque = deque()
        self.held = STANDING_IMU.copy()
        self.ar = None
        self.step = 0

    def __call__(self, imu: np.ndarray) -> np.ndarray:
        m = self.m
        if m.silenced:
            return STANDING_IMU.copy()
        x = np.asarray(imu, float).copy()
        if m.delay:
            self.buf.append(x)
            x = self.buf.popleft() if len(self.buf) > m.delay else STANDING_IMU.copy()
        if self.step % max(m.hold, 1) == 0:
            self.held = x
        x = self.held.copy()
        self.step += 1
        x = STANDING_IMU + np.asarray(m.gain, float) * (x - STANDING_IMU)
        if np.any(np.asarray(m.noise) != 0):
            e = self.rng.standard_normal(6)
            tau = np.broadcast_to(np.asarray(m.tau, float), (6,))
            if np.any(tau > 0):
                a = np.where(tau > 0, np.exp(-1.0 / np.maximum(tau, 1e-9)), 0.0)
                self.ar = e if self.ar is None else a * self.ar + np.sqrt(1 - a * a) * e
                e = self.ar
            x = x + np.asarray(m.noise, float) * self.spread * e
        if m.renorm_gravity:
            gxy = np.clip(x[3:5], -0.99, 0.99)
            x[5] = -np.sqrt(max(1.0 - gxy @ gxy, 1e-4))
        if m.quant:
            lo, hi = STANDING_IMU - 3 * self.spread, STANDING_IMU + 3 * self.spread
            u = np.clip((x - lo) / (hi - lo), 0, 1 - 1e-9)
            x = lo + (np.floor(u * m.quant) + 0.5) / m.quant * (hi - lo)
        return x
