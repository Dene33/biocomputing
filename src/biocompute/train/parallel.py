from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from biocompute.interface.decoders import LinearReadout
from biocompute.robot.asimov import N_JOINTS

_W: dict = {}


def _init(setup_dict: dict, obs_samples: np.ndarray, action_samples, threads: int, warm_s: float) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train.setup import Setup, build

    _W["env"], _W["teacher"] = AsimovEnv(), Teacher()
    setup_dict = dict(setup_dict, slots_ms=tuple(setup_dict["slots_ms"]), taus_ms=tuple(setup_dict["taus_ms"]))
    _W["ctrl"] = build(Setup(**setup_dict), obs_samples, action_samples)
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
        out["O"] = np.array(ep.obs, np.float32)
        out["Y"] = np.clip(np.array(ep.teacher, np.float32), _W["lo"], _W["hi"])
    return out


class Pool:
    def __init__(self, setup_dict: dict, obs_samples: np.ndarray, workers: int, threads: int = 2,
                 warm_s: float = 1.0, action_samples: np.ndarray | None = None):
        self.pool = mp.get_context("spawn").Pool(workers, _init,
                                                 (setup_dict, obs_samples, action_samples, threads, warm_s))
        self.workers = workers

    def map(self, jobs: list[dict]) -> list[dict]:
        return self.pool.map(_episode, jobs, chunksize=1)

    def close(self) -> None:
        self.pool.terminate()


def train(pool: Pool, feature_dim: int, task: str, rounds: int = 10, episodes_per_round: int = 8,
          beta0: float = 0.5, beta_decay: float = 0.6, seconds: float = 20.0, latency: int = 1,
          eval_seeds: int = 4, log=print, seed: int = 0, recovery_weight: float = 4.0, readout_kind: str = "ridge",
          skip_dim: int = 0, ckpt_dir=None) -> dict:
    import pickle
    from pathlib import Path

    X, Y, O, history, readouts = [], [], [], [], []
    readout = None
    rng = np.random.default_rng(seed)
    best = (-1.0, None)
    start = 0
    ckpt = Path(ckpt_dir) if ckpt_dir else None
    if ckpt is not None and (ckpt / "ckpt_state.pkl").exists():
        with open(ckpt / "ckpt_state.pkl", "rb") as f:
            st = pickle.load(f)
        d = np.load(ckpt / "ckpt_data.npz")
        X, Y, O = [d["X"].astype(np.float32)], [d["Y"]], [d["O"]]
        history, readouts, best, rng = st["history"], st["readouts"], st["best"], st["rng"]
        readout, start = readouts[-1], st["round"] + 1
        log({"resumed_after_round": st["round"], "samples": len(X[0])})
    for r in range(start, rounds):
        t0 = time.time()
        beta = 1.0 if r == 0 else beta0 * beta_decay ** (r - 1)
        jobs = [dict(readout=readout, task=task, seconds=seconds, seed=int(rng.integers(1, 10**6)), beta=beta,
                     latency=latency, label=True) for _ in range(episodes_per_round)]
        res = pool.map(jobs)
        X += [x["X"] for x in res]
        Y += [x["Y"] for x in res]
        O += [x["O"] for x in res]
        Xa, Ya, Oa = np.concatenate(X), np.concatenate(Y), np.concatenate(O)
        gyro = np.abs(Oa[:, 0:2]).sum(1)
        w = 1.0 + recovery_weight * (gyro > np.percentile(gyro, 80))
        if readout_kind == "mlp":
            from biocompute.interface.decoders import MLPReadout

            ro = MLPReadout(feature_dim, skip_dim=skip_dim)
            cv = ro.fit(Xa, Ya, weights=w, seed=seed + r) | {"alpha": float("nan")}
        else:
            ro = LinearReadout(feature_dim)
            cv = ro.fit_cv(Xa, Ya, standardize_y=True, weights=w)
        readout = ro
        ev = pool.map([dict(readout=ro, task=task, seconds=seconds, seed=100 + s, beta=0.0, latency=latency,
                            label=False) for s in range(eval_seeds)])
        surv = [e["survived_s"] for e in ev]
        row = {"round": r, "beta": beta, "samples": len(Xa), "val_r2_std": cv["val_r2"], "alpha": cv["alpha"],
               "collect_survived_s": float(np.mean([x["survived_s"] for x in res])),
               "eval_survived_s": float(np.mean(surv)), "eval_full": int(sum(s >= seconds for s in surv)),
               "seconds": time.time() - t0}
        history.append(row)
        readouts.append(ro)
        log(row)
        if row["eval_survived_s"] >= best[0]:
            best = (row["eval_survived_s"], ro)
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            np.savez(ckpt / "ckpt_data.tmp.npz", X=Xa.astype(np.float16), Y=Ya, O=Oa)
            with open(ckpt / "ckpt_state.tmp.pkl", "wb") as f:
                pickle.dump({"round": r, "history": history, "readouts": readouts, "best": best, "rng": rng}, f)
            (ckpt / "ckpt_data.tmp.npz").replace(ckpt / "ckpt_data.npz")
            (ckpt / "ckpt_state.tmp.pkl").replace(ckpt / "ckpt_state.pkl")
            X, Y, O = [Xa], [Ya], [Oa]
    return {"readout": readout, "best_readout": best[1], "history": history, "readouts": readouts,
            "data": (np.concatenate(X), np.concatenate(Y), np.concatenate(O))}
