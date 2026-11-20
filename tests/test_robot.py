"""The stepwise AsimovEnv driven by the teacher reproduces cyclotron's play_velocity exactly."""

import numpy as np

from biocompute import paths
from biocompute.robot.asimov import AsimovEnv, sim2sim
from biocompute.robot.teacher import Teacher

SCHEDULE = "0:1.5,0.3:2,0:2"


def test_teacher_parity_with_play_velocity():
    teacher = Teacher()
    schedule = sim2sim.parse_command_schedule(SCHEDULE)
    reference = sim2sim.play_velocity(sim2sim.PolicyRun.load(paths.POLICY_DIR), teacher, paths.ASIMOV_XML, schedule)

    env = AsimovEnv()
    commands = sim2sim.schedule_commands(schedule, env.dt)
    positions = [env.root_pos]
    for command in commands:
        env.set_command(command)
        result = env.step(teacher(env.observe()))
        positions.append(result.root_pos)
    assert float(reference["fell_at"]) == -1.0 and env.fell_at == -1.0
    np.testing.assert_allclose(np.array(positions), reference["root_pos"], atol=1e-9)
    assert reference["root_pos"][-1, 0] > 0.3  # it walked forward
