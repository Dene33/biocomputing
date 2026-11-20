"""Asimov 1 in MuJoCo, stepped one control step (20 ms) at a time.

Built on cyclotron's sim2sim (``PolicyRun``, ``_MujocoRobot``, ``velocity_observation``), so the robot, its PD
control, torque limits, firmware ankle clamp and the policy's observation are exactly those of
``cyclotron.sim2sim.play_velocity``. The difference is that the caller supplies each action, so anything (the
teacher policy, a culture's readout) can drive the joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from biocompute import paths

sim2sim = paths.import_sim2sim()

N_JOINTS = 23
OBS_DIM = 78


@dataclass
class Perturbation:
    """Changes to the robot, for robustness and adaptation experiments."""

    foot_friction: float | None = None
    """Sliding friction of the feet (MJCF: 0.6)."""
    effort_scale: dict[str, float] = field(default_factory=dict)
    """Joint name regex → multiplier on its torque limit, e.g. ``{"left_knee_joint": 0.5}`` for a weak knee."""
    motor_speed_ratio: float | None = None
    delay: int = 0
    """Physics steps (5 ms) before a joint target is used."""


@dataclass
class StepResult:
    fell: bool
    root_pos: np.ndarray
    root_quat: np.ndarray


class AsimovEnv:
    """One Asimov 1, starting standing in the default pose, driven by policy actions (23, normalized)."""

    def __init__(
        self,
        policy_dir: str | Path = paths.POLICY_DIR,
        xml_path: str | Path = paths.ASIMOV_XML,
        perturbation: Perturbation | None = None,
    ):
        self.run = sim2sim.PolicyRun.load(policy_dir)
        if self.run.interface != "velocity":
            raise ValueError("Expected a velocity-interface policy (the walking policy).")
        self.xml_path = Path(xml_path)
        self.perturbation = perturbation or Perturbation()
        self.dt = self.run.policy_dt
        self.reset()

    @property
    def joint_names(self) -> list[str]:
        return self.run.joint_names

    def reset(self, command=(0.0, 0.0, 0.0)) -> None:
        p = self.perturbation
        self.robot = sim2sim._MujocoRobot(
            self.run, self.xml_path, p.motor_speed_ratio, p.delay, p.foot_friction, firmware_clamp=True
        )
        if p.effort_scale:
            import re

            effort = self.run.effort_limit.copy()
            for pattern, scale in p.effort_scale.items():
                for i, name in enumerate(self.run.joint_names):
                    if re.fullmatch(pattern, name):
                        effort[i] *= scale
            # _MujocoRobot reads the limits from its run, so give it a copy with the weakened limits.
            from dataclasses import replace

            self.robot.run = replace(self.run, effort_limit=effort)
        self.data = self.robot.data
        self.pelvis = self.robot.model.body("pelvis_link").id
        self.robot.place(
            np.array([0.0, 0.0, self.run.root_height]), np.array([1.0, 0.0, 0.0, 0.0]), self.run.default_joint_pos
        )
        self.command = np.asarray(command, dtype=np.float64)
        self.action = np.zeros(N_JOINTS, dtype=np.float32)
        self.history = sim2sim.ObservationHistory(self.run.observations)
        self.steps = 0
        self.fell_at = -1.0
        self.push_steps_left = 0

    def observe(self) -> np.ndarray:
        """The policy observation (78) now, with the current command. Call once per control step, before ``step``:
        a policy that sees past steps keeps its history here."""
        d = self.data
        return sim2sim.velocity_observation(
            self.run,
            d.qvel[3:6],  # free joint angular velocity, in the pelvis frame
            sim2sim.projected_gravity(d.qpos[3:7]),
            self.command,
            d.qpos[self.robot.qpos],
            d.qvel[self.robot.qvel],
            self.action,
            self.history,
        )

    def set_command(self, command) -> None:
        self.command = np.asarray(command, dtype=np.float64)

    def push(self, force_xy, seconds: float = 0.1) -> None:
        """Push the pelvis with a horizontal force (N) for ``seconds``."""
        self.data.xfrc_applied[self.pelvis, :2] = force_xy
        self.push_steps_left = max(1, int(round(seconds / self.dt)))

    def step(self, action: np.ndarray) -> StepResult:
        self.action = np.asarray(action, dtype=np.float32)
        self.robot.step(self.robot.run.joint_targets(self.action))
        if self.push_steps_left:
            self.push_steps_left -= 1
            if not self.push_steps_left:
                self.data.xfrc_applied[self.pelvis] = 0.0
        self.steps += 1
        fell = self.is_fallen()
        if fell and self.fell_at < 0:
            self.fell_at = self.steps * self.dt
        return StepResult(fell, self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy())

    def is_fallen(self) -> bool:
        d = self.data
        return bool(
            sim2sim.projected_gravity_z(d.qpos[3:7]) > sim2sim.FALL_GRAVITY_Z or d.qpos[2] < sim2sim.MIN_PELVIS_HEIGHT
        )

    @property
    def root_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()
