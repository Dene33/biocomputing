from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

STANDING_IMU = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
IMU_SPREAD = np.array([0.0368, 0.0277, 0.0368, 0.0186, 0.0256, 0.0014])


@dataclass
class DishModel:
    delay: int = 0
    hold: int = 1
    noise: float | list = 0.0
    tau: float | list = 0.0
    gain: float | list = 1.0
    renorm_gravity: bool = False
    quant: int = 0
    silenced: bool = False

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
