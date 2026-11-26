from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from biocompute.device import SimulatedBiocomputer, SpikeBlock
from biocompute.interface.decoders import LinearReadout, SpikeFeatures
from biocompute.interface.encoders import Encoder
from biocompute.interface.feedback import Feedback
from biocompute.robot.asimov import N_JOINTS, AsimovEnv


class CultureController:
    def __init__(
        self,
        device: SimulatedBiocomputer | None,
        encoder: Encoder,
        features: SpikeFeatures | None,
        readout: Callable[[np.ndarray], np.ndarray] | None = None,
        feedback: Feedback | None = None,
        source: str = "mea",
        step_ms: float = 20.0,
        efference: bool = False,
    ):
        self.device, self.encoder, self.features = device, encoder, features
        self.readout, self.feedback, self.source, self.step_ms = readout, feedback, source, step_ms
        self.efference = efference
        self.stimulate_inputs = True
        self.last_block: SpikeBlock | None = None
        if source == "oracle":
            n = device.culture.n
            self.oracle = SpikeFeatures(np.arange(n), bins=1, taus_ms=(40.0, 160.0), step_ms=step_ms)

    EFFERENCE = slice(55, 78)

    @property
    def feature_dim(self) -> int:
        extra = 23 if self.efference else 0
        if self.source == "none":
            return len(self.encoder.channels) + extra
        if self.source == "oracle":
            return self.oracle.dim + extra
        return self.features.dim + extra

    def reset_readout_state(self) -> None:
        self.encoder.reset()
        if self.features is not None:
            self.features.reset()
        if self.source == "oracle":
            self.oracle.reset()

    def sense(self, obs: np.ndarray, stimulate: bool = True) -> np.ndarray:
        if self.source == "none":
            return self.encoder.drive(obs).astype(np.float32)
        if stimulate:
            self.device.stimulate(*self.encoder.pulses(obs, self.step_ms))
        if self.feedback is not None:
            self.device.stimulate(*self.feedback.pulses(self.step_ms))
        block = self.device.step(self.step_ms)
        self.last_block = block
        if self.source == "oracle":
            return self.oracle.from_counts(self.device.oracle_counts[None])
        return self.features(block)

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.sense(obs, stimulate=self.stimulate_inputs)
        if self.efference:
            x = np.r_[x, obs[self.EFFERENCE]].astype(np.float32)
        a = self.readout(x) if self.readout is not None else np.zeros(N_JOINTS, np.float32)
        return a, x


@dataclass
class Episode:
    obs: list = field(default_factory=list)
    features: list = field(default_factory=list)
    student: list = field(default_factory=list)
    teacher: list = field(default_factory=list)
    root_pos: list = field(default_factory=list)
    fell_at: float = -1.0
    steps: int = 0

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array(getattr(self, k)) for k in ("obs", "features", "student", "teacher", "root_pos")} | {
            "fell_at": np.array(self.fell_at)
        }


def run_episode(
    env: AsimovEnv,
    controller: CultureController,
    seconds: float,
    command_fn: Callable[[float], tuple[float, float, float]] = lambda t: (0.0, 0.0, 0.0),
    teacher: Callable[[np.ndarray], np.ndarray] | None = None,
    beta: float = 0.0,
    latency_steps: int = 1,
    clip: float = 6.0,
    push_fn: Callable[[float], np.ndarray | None] | None = None,
    stop_on_fall: bool = True,
    rng: np.random.Generator | None = None,
    on_step: Callable[[int, AsimovEnv, CultureController], None] | None = None,
) -> Episode:
    rng = rng or np.random.default_rng()
    env.reset(command_fn(0.0))
    controller.reset_readout_state()
    ep = Episode()
    pending = deque([np.zeros(N_JOINTS, np.float32)] * latency_steps)
    for step in range(int(round(seconds / env.dt))):
        t = step * env.dt
        env.set_command(command_fn(t))
        if push_fn is not None:
            force = push_fn(t)
            if force is not None:
                env.push(force)
        obs = env.observe()
        student, feats = controller.act(obs)
        student = np.clip(student, -clip, clip)
        label = teacher(obs) if teacher is not None else None
        chosen = label if (label is not None and rng.random() < beta) else student
        pending.append(chosen)
        result = env.step(pending.popleft())
        ep.obs.append(obs)
        ep.features.append(feats)
        ep.student.append(student)
        if label is not None:
            ep.teacher.append(label)
        ep.root_pos.append(result.root_pos)
        ep.steps = step + 1
        if on_step is not None:
            on_step(step, env, controller)
        if result.fell:
            ep.fell_at = env.fell_at
            if stop_on_fall:
                break
    return ep
