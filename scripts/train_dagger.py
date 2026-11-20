"""Reservoir readout by parallel DAgger, with labels clipped to the joint range and standardized per joint.

    uv run scripts/train_dagger.py --culture gpu_20k --mea hd1024 --workers 6
    uv run scripts/train_dagger.py --source none        # control: no culture (linear policy of the encoded obs)
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from biocompute.train import parallel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="stand_push")
    p.add_argument("--source", default="mea", choices=["mea", "oracle", "none", "dead"])
    p.add_argument("--culture", default="gpu_20k")
    p.add_argument("--mea", default="hd1024")
    p.add_argument("--n-stim", type=int, default=32)
    p.add_argument("--slots", type=float, nargs="+", default=[0.0])
    p.add_argument("--projection", default="pca")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--beta0", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--latency", type=int, default=1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="outputs/dagger")
    a = p.parse_args()

    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train import dagger
    from biocompute.train.setup import Setup, build
    from biocompute.train.tasks import make_task

    name = f"{a.task}_{a.source}_{a.culture}_{a.mea}_s{a.seed}{a.tag}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    env, teacher = AsimovEnv(), Teacher()
    obs = dagger.teacher_observations(env, teacher, [make_task(a.task, a.seconds, seed=a.seed)], seconds=a.seconds)
    setup = Setup(culture=a.culture, mea=a.mea, n_stim=a.n_stim, source=a.source, seed=a.seed,
                  slots_ms=tuple(a.slots), projection=a.projection)
    # Feature dimension without building a culture here: the MEA's alive channels × (bins + filters).
    if a.source == "none":
        dim = a.n_stim
    elif a.source == "oracle":
        from biocompute.culture.network import make_config

        dim = make_config(a.culture).n_neurons * 3
    else:
        from biocompute.mea.electrodes import make_mea_config

        cfg = make_mea_config(a.mea, seed=a.seed)
        n_ch = cfg.rows * cfg.cols - (4 if cfg.skip_corners else 0)
        n_ch -= int(round(cfg.dead_fraction * n_ch))
        dim = n_ch * (setup.bins + len(setup.taus_ms))
    pool = parallel.Pool(setup.to_dict(), obs, a.workers, threads=a.threads)
    print(f"{name}: {dim} features, {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    res = parallel.train(pool, dim, a.task, rounds=a.rounds, episodes_per_round=a.episodes, beta0=a.beta0,
                         seconds=a.seconds, latency=a.latency, eval_seeds=a.eval_seeds, seed=a.seed,
                         log=lambda r: (rows.append(r), print(json.dumps(r), flush=True)))
    ro = res["best_readout"]
    ev = pool.map([dict(readout=ro, task=a.task, seconds=a.seconds, seed=1000 + s, beta=0.0, latency=a.latency,
                        label=False) for s in range(10)])
    pool.close()
    surv = [e["survived_s"] for e in ev]
    summary = {"name": name, "setup": setup.to_dict(), "args": vars(a), "train_seconds": time.time() - t0,
               "history": rows, "final_eval": ev, "final_survived_s_mean": float(np.mean(surv)),
               "final_full": f"{sum(s >= a.seconds for s in surv)}/{len(surv)}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "readout.pkl", "wb") as f:
        pickle.dump(ro, f)
    print(json.dumps({k: summary[k] for k in ("final_survived_s_mean", "final_full")}), flush=True)


if __name__ == "__main__":
    main()
