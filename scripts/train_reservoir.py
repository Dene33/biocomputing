import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from biocompute.robot.asimov import AsimovEnv
from biocompute.robot.teacher import Teacher
from biocompute.train import dagger
from biocompute.train.setup import Setup, build
from biocompute.train.tasks import make_task

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="stand", choices=["stand", "stand_push", "walk", "walk_schedule"])
parser.add_argument("--source", default="mea", choices=["mea", "oracle", "none", "dead"])
parser.add_argument("--culture", default="cpu_2k")
parser.add_argument("--mea", default="mcs60")
parser.add_argument("--n-stim", type=int, default=32)
parser.add_argument("--encoding", default="amplitude", choices=["amplitude", "rate"])
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--episodes", type=int, default=2, help="Episodes per DAgger round")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--alpha", type=float, default=None, help="Ridge penalty; default: cross-validated")
parser.add_argument("--latency", type=int, default=1)
parser.add_argument("--eval-seeds", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--out", default="outputs/reservoir")
args = parser.parse_args()
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)

name = f"{args.task}_{args.source}_{args.mea}_s{args.seed}"
out = Path(args.out) / name
out.mkdir(parents=True, exist_ok=True)

env, teacher = AsimovEnv(), Teacher()
task = make_task(args.task, args.seconds, seed=args.seed)
obs = dagger.teacher_observations(env, teacher, [task], seconds=args.seconds)
setup = Setup(culture=args.culture, mea=args.mea, n_stim=args.n_stim, encoding=args.encoding, source=args.source,
              seed=args.seed)
ctrl = build(setup, obs)
dagger.warm_up(ctrl)
print(f"{name}: {ctrl.feature_dim} features")

t0 = time.time()
log_rows = []
result = dagger.train(env, ctrl, teacher, [task], rounds=args.rounds, episodes_per_round=args.episodes,
                      alpha=args.alpha, latency_steps=args.latency,
                      log=lambda row: (log_rows.append(row), print(json.dumps(row), flush=True)))
ctrl.readout = result["readout"]
evals = [
    dagger.evaluate(env, ctrl, [make_task(args.task, args.seconds, seed=100 + s)], args.latency, seeds=[s])[0]
    for s in range(args.eval_seeds)
]
summary = {
    "name": name,
    "setup": setup.to_dict(),
    "args": vars(args),
    "train_seconds": time.time() - t0,
    "history": log_rows,
    "eval": evals,
    "eval_survived_s_mean": float(np.mean([e["survived_s"] for e in evals])),
    "eval_full_episodes": int(sum(not e["fell"] for e in evals)),
    "eval_distance_x_m_mean": float(np.mean([e["distance_x_m"] for e in evals])),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
with open(out / "readout.pkl", "wb") as f:
    pickle.dump(result["readout"], f)
print(json.dumps({k: v for k, v in summary.items() if k.startswith("eval_")}, indent=2))
