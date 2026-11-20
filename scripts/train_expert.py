"""Train a balance expert with Augmented Random Search (Mania et al. 2018): a policy linear in the culture's own
stimulation signals, optimised directly for surviving pushes.

The walking teacher cannot be imitated through a reservoir: students with R² ≈ 0.9 still fall within seconds,
because their errors are systematic and feed back through the robot. So the culture imitates this expert instead.
It reads exactly what the culture is told (the encoder's per-channel drive, 32 values in (0, 1)) and maps it
linearly to all 23 joint actions, so a reservoir readout can represent it in principle. It is trained with the
culture's handicaps: one step of latency, noise on its inputs, and noise on its actions.

    uv run scripts/train_expert.py --iters 300
"""

import argparse
import json
import multiprocessing as mp
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=300)
parser.add_argument("--directions", type=int, default=32)
parser.add_argument("--top", type=int, default=16)
parser.add_argument("--step", type=float, default=0.02)
parser.add_argument("--noise", type=float, default=0.03)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--feature-noise", type=float, default=0.05, help="Std of noise added to the 32 drives")
parser.add_argument("--action-noise", type=float, default=0.05, help="Std of noise added to actions")
parser.add_argument("--latency", type=int, default=1)
parser.add_argument("--force", type=float, nargs=2, default=(120.0, 200.0))
parser.add_argument("--workers", type=int, default=15)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--init", default=None, help="Start from this expert.pkl")
parser.add_argument("--out", default="outputs/expert")
args = parser.parse_args()

_W = {}


def _init(encoder, a):
    from biocompute.robot.asimov import AsimovEnv

    _W["env"], _W["enc"], _W["args"] = AsimovEnv(), encoder, a


def rollout(job):
    """job = (M, seed, train). Returns (score, survived_s)."""
    from biocompute.train.tasks import _pushes

    M, seed, train = job
    env, enc, a = _W["env"], _W["enc"], _W["args"]
    rng = np.random.default_rng(seed)
    push = _pushes(seed, force_n=tuple(a.force))
    env.reset()
    q = deque([np.zeros(23, np.float32)] * a.latency)
    steps = int(a.seconds / env.dt)
    score = 0.0
    for s in range(steps):
        t = s * env.dt
        if (f := push(t)) is not None:
            env.push(f)
        x = enc.drive(env.observe())
        if train:
            x = x + a.feature_noise * rng.standard_normal(x.shape)
        act = M @ np.r_[x - 0.5, 1.0]
        if train:
            act = act + a.action_noise * rng.standard_normal(23)
        q.append(np.clip(act, -6, 6).astype(np.float32))
        res = env.step(q.popleft())
        # Alive bonus, minus tilt and drift (stand in place, upright).
        tilt = np.linalg.norm(env.data.qpos[3:7][1:3])
        drift = np.linalg.norm(env.data.qvel[0:2])
        score += 1.0 - 0.5 * tilt - 0.05 * drift
        if res.fell:
            break
    survived = env.fell_at if env.fell_at >= 0 else a.seconds
    return score / steps, survived


def main():
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train import dagger
    from biocompute.train.setup import Setup, build
    from biocompute.train.tasks import make_task

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env, teacher = AsimovEnv(), Teacher()
    obs = dagger.teacher_observations(env, teacher, [make_task("stand_push", 20.0, seed=0)], seconds=20.0)
    setup = Setup(source="none", seed=args.seed)
    encoder = build(setup, obs).encoder
    dim = len(encoder.channels) + 1
    M = np.zeros((23, dim))
    if args.init:
        M = pickle.load(open(args.init, "rb"))["M"]
    rng = np.random.default_rng(args.seed)
    pool = mp.get_context("fork").Pool(args.workers, _init, (encoder, args))
    log = []
    best = (-1.0, M)
    for it in range(args.iters):
        t0 = time.time()
        deltas = rng.standard_normal((args.directions, *M.shape))
        seeds = rng.integers(0, 2**31, args.directions)
        jobs = [(M + args.noise * d, int(s), True) for d, s in zip(deltas, seeds)]
        jobs += [(M - args.noise * d, int(s), True) for d, s in zip(deltas, seeds)]
        res = pool.map(rollout, jobs)
        r = np.array([x[0] for x in res]).reshape(2, -1)
        order = np.argsort(-np.maximum(r[0], r[1]))[: args.top]
        sigma = r[:, order].std() + 1e-8
        M = M + args.step / (args.top * sigma) * np.einsum("k,kij->ij", r[0, order] - r[1, order], deltas[order])
        row = {"iter": it, "train_score": float(r.mean()), "train_survived": float(np.mean([x[1] for x in res])),
               "seconds": time.time() - t0}
        if it % 10 == 9 or it == args.iters - 1:
            ev = pool.map(rollout, [(M, 100 + s, False) for s in range(10)])
            row["eval_survived"] = float(np.mean([e[1] for e in ev]))
            row["eval_full"] = int(sum(e[1] >= args.seconds for e in ev))
            if row["eval_survived"] >= best[0]:
                best = (row["eval_survived"], M.copy())
                with open(out / "expert.pkl", "wb") as f:
                    pickle.dump({"M": M, "encoder": encoder, "setup": setup.to_dict(), "args": vars(args),
                                 "eval": row}, f)
        log.append(row)
        print(json.dumps(row), flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
