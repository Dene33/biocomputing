"""Open-loop comparison of encodings: replay teacher episodes through the culture, fit the ridge readout, report
cross-validated R² of the standardized (clipped) teacher actions.

    uv run python outputs/probe/encoding.py collect
    uv run python outputs/probe/encoding.py run <name> [key=value ...]
"""
import json
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "teacher_data.pkl"


def collect():
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train.tasks import make_task

    env, teacher = AsimovEnv(), Teacher()
    lo = (env.run.action_clip[:, 0] - env.run.default_joint_pos) / env.run.action_scale
    hi = (env.run.action_clip[:, 1] - env.run.default_joint_pos) / env.run.action_scale
    sd = None
    rng = np.random.default_rng(0)
    O, Y = [], []
    for ep in range(5):
        task = make_task("stand_push", 20, seed=500 + ep)
        env.reset()
        q = deque([np.zeros(23, np.float32)])
        for s in range(1000):
            if (f := task.push_fn(s * env.dt)) is not None:
                env.push(f)
            o = env.observe()
            y = np.clip(teacher(o), lo, hi)
            # Executed action gets noise (DART), so the data covers states off the teacher's own path.
            noise = 0.0 if sd is None else 0.3 * sd * rng.standard_normal(23)
            O.append(o)
            Y.append(y)
            q.append((y + noise).astype(np.float32))
            if env.step(q.popleft()).fell:
                break
        sd = np.array(Y).std(0)
    pickle.dump({"obs": np.array(O), "y": np.array(Y)}, open(DATA, "wb"))
    print(len(O), "samples")


def run(name, **kw):
    import torch

    from biocompute.interface.decoders import LinearReadout
    from biocompute.train.setup import Setup, build

    d = pickle.load(open(DATA, "rb"))
    O, Y = d["obs"], d["y"]
    proj = kw.pop("projection", "pca")
    setup = Setup(**{"culture": "gpu_20k", "mea": "hd1024"} | kw)
    if proj == "supervised":
        setup.projection = "random"
    ctrl = build(setup, O)
    if proj == "supervised":
        ctrl.encoder.fit_supervised(O, Y)
    if ctrl.device is not None:
        ctrl.device.step(1000.0)
    t = time.time()
    F = np.array([ctrl.sense(o) for o in O])
    ms = (time.time() - t) / len(O) * 1000
    D = np.array([ctrl.encoder.drive(o) for o in O])
    res = {"name": name, "ms_per_step": round(ms, 1), "dim": F.shape[1]}
    res["r2_features"] = LinearReadout(F.shape[1]).fit_cv(F, Y, standardize_y=True)["val_r2"]
    res["r2_drives_linear"] = LinearReadout(D.shape[1]).fit_cv(D, Y, standardize_y=True)["val_r2"]
    res["r2_decode_drives"] = LinearReadout(F.shape[1]).fit_cv(F, D, standardize_y=True)["val_r2"]
    print(json.dumps(res), flush=True)
    with open(HERE / "encoding_results.jsonl", "a") as f:
        f.write(json.dumps(res | {"setup": setup.to_dict(), "projection": proj}) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect()
    else:
        kw = {}
        for a in sys.argv[3:]:
            k, v = a.split("=")
            kw[k] = json.loads(v) if v[0] in "[{0123456789-" or v in ("true", "false") else v
            if isinstance(kw[k], list):
                kw[k] = tuple(kw[k])
        run(sys.argv[2], **kw)
