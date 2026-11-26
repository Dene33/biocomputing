import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from biocompute.train import parallel


class ZeroReadout:
    def __call__(self, x):
        return np.zeros(23, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--control", default="none", choices=["none", "silent", "dead", "zero"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--task", default=None)
    a = p.parse_args()

    run = Path(a.run)
    summary = json.loads((run / "summary.json").read_text())
    setup = dict(summary["setup"])
    task = a.task or summary["args"]["task"]
    if a.control == "silent":
        setup["silent_inputs"] = True
    if a.control == "dead":
        setup["source"] = "dead"
    with open(run / "readout.pkl", "rb") as f:
        readout = pickle.load(f)
    if a.control == "zero":
        readout = ZeroReadout()

    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train import dagger
    from biocompute.train.tasks import make_task

    env, teacher = AsimovEnv(), Teacher()
    seconds = summary["args"]["seconds"]
    seed = summary["args"]["seed"]
    obs = dagger.teacher_observations(env, teacher, [make_task(task, seconds, seed=seed)], seconds=seconds,
                                      episodes=6, noise=0.3, seed=seed)
    lo = (env.run.action_clip[:, 0] - env.run.default_joint_pos) / env.run.action_scale
    hi = (env.run.action_clip[:, 1] - env.run.default_joint_pos) / env.run.action_scale
    acts = np.clip(np.array([teacher(o) for o in obs]), lo, hi)
    pool = parallel.Pool(setup, obs, a.workers, threads=1, action_samples=acts)
    res = pool.map([dict(readout=readout, task=task, seconds=seconds, seed=2000 + s, beta=0.0,
                         latency=summary["args"]["latency"], label=False) for s in range(a.seeds)])
    pool.close()
    surv = [r["survived_s"] for r in res]
    out = {"run": str(run), "control": a.control, "task": task, "seeds": a.seeds,
           "survived_s_mean": float(np.mean(surv)), "full": f"{sum(s >= seconds for s in surv)}/{len(surv)}",
           "survived_s": surv, "distance_x_m": [r["distance_x_m"] for r in res]}
    print(json.dumps({k: out[k] for k in ("control", "survived_s_mean", "full")}), flush=True)
    (run / f"eval_{a.control}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
