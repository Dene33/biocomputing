from __future__ import annotations

import numpy as np


class Feedback:
    def __init__(
        self,
        channels: np.ndarray,
        mode: str = "dishbrain",
        predictable_hz: float = 100.0,
        predictable_ms: float = 100.0,
        predictable_amp: float = 0.6,
        unpredictable_hz: float = 5.0,
        unpredictable_s: float = 4.0,
        unpredictable_amp: float = 1.0,
        slot_ms: float = 10.0,
        seed: int = 0,
    ):
        self.channels = np.asarray(channels, int)
        self.mode = mode
        self.predictable_hz, self.predictable_ms, self.predictable_amp = predictable_hz, predictable_ms, predictable_amp
        self.unpredictable_hz, self.unpredictable_s = unpredictable_hz, unpredictable_s
        self.unpredictable_amp, self.slot_ms = unpredictable_amp, slot_ms
        self.rng = np.random.default_rng(seed)
        self.predictable_left_ms = 0.0
        self.unpredictable_left_ms = 0.0
        self.counts = {"predictable": 0, "unpredictable": 0}

    def good(self) -> None:
        if self.mode == "off":
            return
        if self.mode == "random":
            self._random_event()
            return
        self.predictable_left_ms = self.predictable_ms
        self.counts["predictable"] += 1

    def bad(self) -> None:
        if self.mode == "off":
            return
        if self.mode == "random":
            self._random_event()
            return
        self.unpredictable_left_ms = self.unpredictable_s * 1000.0
        self.counts["unpredictable"] += 1

    def _random_event(self) -> None:
        if self.rng.random() < 0.5:
            self.predictable_left_ms = self.predictable_ms
            self.counts["predictable"] += 1
        else:
            self.unpredictable_left_ms = self.unpredictable_s * 1000.0
            self.counts["unpredictable"] += 1

    @property
    def busy(self) -> bool:
        return self.unpredictable_left_ms > 0

    def pulses(self, step_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chans, times, amps = [], [], []
        if self.predictable_left_ms > 0:
            period = 1000.0 / self.predictable_hz
            t = 0.0
            while t < min(step_ms, self.predictable_left_ms):
                chans.append(self.channels)
                times.append(np.full(len(self.channels), t))
                amps.append(np.full(len(self.channels), self.predictable_amp))
                t += period
            self.predictable_left_ms -= step_ms
        if self.unpredictable_left_ms > 0:
            p = self.unpredictable_hz * step_ms / 1000.0
            fire = self.rng.random(len(self.channels)) < p
            slots = int(step_ms // self.slot_ms)
            chans.append(self.channels[fire])
            times.append(self.rng.integers(0, slots, fire.sum()) * self.slot_ms)
            amps.append(np.full(fire.sum(), self.unpredictable_amp))
            self.unpredictable_left_ms -= step_ms
        if not chans:
            return np.zeros(0, int), np.zeros(0), np.zeros(0)
        return np.concatenate(chans), np.concatenate(times), np.concatenate(amps)
