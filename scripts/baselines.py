"""Reference points for every task: the teacher policy (the ceiling) and zero action, joints held at the default
pose by PD (the floor a culture must beat). Writes outputs/baselines.json.

    uv run scripts/baselines.py
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

from biocompute.robot.asimov import AsimovEnv
from biocompute.robot.teacher import Teacher
from biocompute.train.tasks import make_task

env, teacher = AsimovEnv(), Teacher()
policies = {"teacher": teacher, "zero": lambda obs: np.zeros(23, np.float32)}
out = []
for task_name in ("stand", "stand_push", "walk", "walk_schedule"):
    for name, policy in policies.items():
        survived, dist = [], []
        for seed in range(5):
            task = make_task(task_name, 20.0, seed=100 + seed)
            env.reset()
            q = deque([np.zeros(23, np.float32)])  # the same one-step latency as the culture loop
            start = env.root_pos[0]
            for s in range(1000):
                t = s * env.dt
                env.set_command(task.command_fn(t))
                if task.push_fn and (f := task.push_fn(t)) is not None:
                    env.push(f)
                q.append(policy(env.observe()))
                if env.step(q.popleft()).fell:
                    break
            survived.append(env.fell_at if env.fell_at >= 0 else 20.0)
            dist.append(env.root_pos[0] - start)
        row = {"task": task_name, "policy": name, "survived_s": float(np.mean(survived)),
               "full": f"{sum(s >= 20.0 for s in survived)}/5", "distance_m": float(np.mean(dist))}
        out.append(row)
        print(row)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/baselines.json").write_text(json.dumps(out, indent=2))
