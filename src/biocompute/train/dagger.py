from __future__ import annotations

import time

import numpy as np

from biocompute.eval.metrics import robot_metrics
from biocompute.interface.decoders import LinearReadout, r2_pooled
from biocompute.loop import CultureController, run_episode
from biocompute.robot.asimov import AsimovEnv
from biocompute.train.tasks import Task


def teacher_observations(env: AsimovEnv, teacher, tasks: list[Task], seconds: float = 10.0, episodes: int = 1,
                         noise: float = 0.0, seed: int = 0) -> np.ndarray:
    obs, acts = [], []
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        for task in tasks:
            sd = np.array(acts).std(0) if acts else np.zeros(23)
            env.reset(task.command_fn(0.0))
            for step in range(int(seconds / env.dt)):
                t = step * env.dt
                env.set_command(task.command_fn(t))
                if task.push_fn is not None and (f := task.push_fn(t)) is not None:
                    env.push(f)
                o = env.observe()
                a = teacher(o)
                obs.append(o)
                acts.append(a)
                executed = a + (noise * np.minimum(sd, 1.0) * rng.standard_normal(23) if ep > 0 else 0.0)
                if env.step(executed.astype(np.float32)).fell:
                    break
    return np.array(obs)


def warm_up(ctrl: CultureController, seconds: float = 1.0) -> None:
    if ctrl.device is not None:
        ctrl.device.step(seconds * 1000.0)


def evaluate(env, ctrl, tasks: list[Task], latency_steps: int = 1, seeds=(0,)) -> list[dict]:
    out = []
    for task in tasks:
        for seed in seeds:
            ep = run_episode(env, ctrl, task.seconds, task.command_fn, push_fn=task.push_fn,
                             latency_steps=latency_steps, rng=np.random.default_rng(seed))
            out.append({"task": task.name, "seed": seed} | robot_metrics(ep, task.seconds, env.dt))
    return out


def train(
    env: AsimovEnv,
    ctrl: CultureController,
    teacher,
    tasks: list[Task],
    rounds: int = 8,
    episodes_per_round: int = 2,
    beta0: float = 1.0,
    beta_decay: float = 0.6,
    alpha: float | None = None,
    latency_steps: int = 1,
    log=print,
) -> dict:
    X, Y = [], []
    readout = LinearReadout(ctrl.feature_dim)
    history = []
    rng = np.random.default_rng(0)
    for r in range(rounds):
        beta = beta0 * beta_decay**r if r > 0 else 1.0
        ctrl.readout = readout if r > 0 else None
        t0 = time.time()
        steps = 0
        for e in range(episodes_per_round):
            task = tasks[(r * episodes_per_round + e) % len(tasks)]
            ep = run_episode(env, ctrl, task.seconds, task.command_fn, teacher=teacher, beta=beta,
                             latency_steps=latency_steps, push_fn=task.push_fn, rng=rng)
            X.append(np.array(ep.features))
            Y.append(np.array(ep.teacher))
            steps += ep.steps
        Xa, Ya = np.concatenate(X), np.concatenate(Y)
        if alpha is None:
            cv = readout.fit_cv(Xa, Ya)
        else:
            readout.fit(Xa, Ya, alpha=alpha)
            cv = {"alpha": alpha, "val_r2": float("nan")}
        r2 = r2_pooled(Ya, readout.predict(Xa))
        ctrl.readout = readout
        ev = evaluate(env, ctrl, tasks[:1], latency_steps)
        row = {"round": r, "beta": beta, "samples": len(Xa), "train_r2": r2, "val_r2": cv["val_r2"], "alpha": cv["alpha"], "eval_survived_s": ev[0]["survived_s"],
               "eval_distance_x_m": ev[0]["distance_x_m"], "seconds": time.time() - t0}
        history.append(row)
        log(row)
    return {"readout": readout, "history": history, "X": np.concatenate(X), "Y": np.concatenate(Y)}
