"""Parallel DAgger: several worker processes, each with its own copy of the same culture (same seed) and robot.

On hardware this is one dish run for longer; here it only saves wall-clock time. Each worker keeps its culture
running between episodes (a real culture can't be reset). The readout is fitted in the parent on everything the
workers collected and sent back out each round.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from biocompute.interface.decoders import LinearReadout, r2_pooled
from biocompute.robot.asimov import N_JOINTS

_W: dict = {}


def _init(setup_dict: dict, obs_samples: np.ndarray, threads: int, warm_s: float) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train.setup import Setup, build

    _W["env"], _W["teacher"] = AsimovEnv(), Teacher()
    _W["ctrl"] = build(Setup(**setup_dict), obs_samples)
    if _W["ctrl"].device is not None:
        _W["ctrl"].device.step(warm_s * 1000.0)
    run = _W["env"].run
    _W["lo"] = (run.action_clip[:, 0] - run.default_joint_pos) / run.action_scale
    _W["hi"] = (run.action_clip[:, 1] - run.default_joint_pos) / run.action_scale


def _episode(job: dict) -> dict:
    from biocompute.loop import run_episode
    from biocompute.train.tasks import make_task

    env, ctrl = _W["env"], _W["ctrl"]
    ctrl.readout = job["readout"]
    task = make_task(job["task"], job["seconds"], seed=job["seed"])
    teacher = _W["teacher"] if job["label"] else None
    ep = run_episode(env, ctrl, task.seconds, task.command_fn, teacher=teacher, beta=job["beta"],
                     latency_steps=job["latency"], push_fn=task.push_fn, rng=np.random.default_rng(job["seed"]))
    out = {"steps": ep.steps, "fell_at": ep.fell_at, "survived_s": ep.fell_at if ep.fell_at >= 0 else task.seconds,
           "distance_x_m": float(ep.root_pos[-1][0] - ep.root_pos[0][0]) if ep.root_pos else 0.0}
    if job["label"]:
        out["X"] = np.array(ep.features, np.float32)
        out["Y"] = np.clip(np.array(ep.teacher, np.float32), _W["lo"], _W["hi"])
    return out


class Pool:
    def __init__(self, setup_dict: dict, obs_samples: np.ndarray, workers: int, threads: int = 2,
                 warm_s: float = 1.0):
        self.pool = mp.get_context("spawn").Pool(workers, _init, (setup_dict, obs_samples, threads, warm_s))
        self.workers = workers

    def map(self, jobs: list[dict]) -> list[dict]:
        return self.pool.map(_episode, jobs, chunksize=1)

    def close(self) -> None:
        self.pool.terminate()


def train(pool: Pool, feature_dim: int, task: str, rounds: int = 10, episodes_per_round: int = 8,
          beta0: float = 0.5, beta_decay: float = 0.6, seconds: float = 20.0, latency: int = 1,
          eval_seeds: int = 4, log=print, seed: int = 0) -> dict:
    X, Y, history = [], [], []
    readout = None
    rng = np.random.default_rng(seed)
    best = (-1.0, None)
    for r in range(rounds):
        t0 = time.time()
        beta = 1.0 if r == 0 else beta0 * beta_decay ** (r - 1)
        jobs = [dict(readout=readout, task=task, seconds=seconds, seed=int(rng.integers(1, 10**6)), beta=beta,
                     latency=latency, label=True) for _ in range(episodes_per_round)]
        res = pool.map(jobs)
        X += [x["X"] for x in res]
        Y += [x["Y"] for x in res]
        Xa, Ya = np.concatenate(X), np.concatenate(Y)
        ro = LinearReadout(feature_dim)
        cv = ro.fit_cv(Xa, Ya, standardize_y=True)
        readout = ro
        ev = pool.map([dict(readout=ro, task=task, seconds=seconds, seed=100 + s, beta=0.0, latency=latency,
                            label=False) for s in range(eval_seeds)])
        surv = [e["survived_s"] for e in ev]
        row = {"round": r, "beta": beta, "samples": len(Xa), "val_r2_std": cv["val_r2"], "alpha": cv["alpha"],
               "collect_survived_s": float(np.mean([x["survived_s"] for x in res])),
               "eval_survived_s": float(np.mean(surv)), "eval_full": int(sum(s >= seconds for s in surv)),
               "seconds": time.time() - t0}
        history.append(row)
        log(row)
        if row["eval_survived_s"] >= best[0]:
            best = (row["eval_survived_s"], ro)
    return {"readout": readout, "best_readout": best[1], "history": history}
