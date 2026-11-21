"""Score a robot-side controller on stand_push with its vestibular input coming through the dish model.

A controller is a Python file defining

    write_weights() -> dict          # all parameters, written in code (no gradient training)
    act(W, jpos, jvel, last_action, vest) -> np.ndarray (23,)
    reset(W)                         # optional: clear filter state at the start of each episode

with inputs in the teacher's observation units: jpos = obs[9:32], jvel = obs[32:55], last_action = obs[55:78] (the
robot's last executed command), vest = the dish's estimate of obs[0:6] (gyro x0.25, projected gravity).

    uv run python wsp/controller/eval_controller.py wsp/controller/track_a.py --note "what changed"

Prints ``score: <mean survival s over the validation seeds>`` and appends a row to results.csv. Also reports
survival with the dish silenced (the controller must fall: the dish has to be necessary). ``--test`` scores the
hidden test seeds; use it only once per track, at the end.
"""

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import time
from collections import deque
from pathlib import Path

import numpy as np

from biocompute.eval.dish_channel import DishChannel, DishModel

HERE = Path(__file__).parent
VAL_SEEDS = list(range(7000, 7060))
TEST_SEEDS = list(range(8000, 8040))
SECONDS, STEPS = 20.0, 1000
_G = {}


def load(path):
    spec = importlib.util.spec_from_file_location("controller", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init():
    from biocompute.robot.asimov import AsimovEnv

    _G["env"] = AsimovEnv()


def episode(job):
    from biocompute.train.tasks import make_task

    seed, model = job
    env, mod, W = _G["env"], _G["mod"], _G["W"]
    dish = DishChannel(model, np.random.default_rng(seed + 1))
    task = make_task("stand_push", SECONDS, seed=seed)
    env.reset()
    if hasattr(mod, "reset"):
        mod.reset(W)
    q = deque([np.zeros(23, np.float32)])
    for s in range(STEPS):
        if (f := task.push_fn(s * env.dt)) is not None:
            env.push(f)
        o = env.observe()
        a = mod.act(W, o[9:32], o[32:55], o[55:78], dish(o[0:6]))
        q.append(np.asarray(a, np.float32))
        if env.step(q.popleft()).fell:
            break
    return env.fell_at if env.fell_at >= 0 else SECONDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("controller")
    p.add_argument("--dish", default=str(HERE / "dish_model.json"))
    p.add_argument("--note", default="")
    p.add_argument("--test", action="store_true", help="Hidden test seeds: once per track, at the end")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--workers", type=int, default=14)
    a = p.parse_args()

    t0 = time.time()
    model = DishModel(**{k: v for k, v in json.loads(Path(a.dish).read_text()).items() if not k.startswith("_")})
    _G["mod"] = load(a.controller)
    _G["W"] = _G["mod"].write_weights()
    n_params = int(sum(np.size(v) for v in _G["W"].values()))
    seeds = TEST_SEEDS if a.test else VAL_SEEDS
    silenced = DishModel(**(model.to_dict() | {"silenced": True}))
    with mp.get_context("fork").Pool(a.workers, _init) as pool:
        r = np.array(pool.map(episode, [(s, model) for s in seeds]))
        rs = np.array(pool.map(episode, [(s, silenced) for s in seeds[:10]]))
    score = float(r.mean())
    print(f"score: {score:.3f}")
    print(f"full: {int((r >= SECONDS).sum())}/{len(r)}  silenced: {rs.mean():.2f} s  params: {n_params}  "
          f"seeds: {'TEST' if a.test else 'val'}  {time.time() - t0:.0f} s")
    if not a.no_log:
        f = HERE / "results.csv"
        new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "controller", "seeds", "score", "full", "silenced_s", "params", "dish", "note"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), Path(a.controller).name, "test" if a.test else "val",
                        round(score, 3), f"{int((r >= SECONDS).sum())}/{len(r)}", round(float(rs.mean()), 2),
                        n_params, json.dumps(model.to_dict()), a.note])


if __name__ == "__main__":
    main()
