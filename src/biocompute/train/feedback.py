from __future__ import annotations

import time

import numpy as np

from biocompute.culture.plasticity import RewardSTDP
from biocompute.eval.metrics import robot_metrics
from biocompute.interface.feedback import Feedback
from biocompute.loop import CultureController, run_episode
from biocompute.robot.asimov import AsimovEnv
from biocompute.train.tasks import Task


def session(
    env: AsimovEnv,
    ctrl: CultureController,
    feedback: Feedback,
    task: Task,
    minutes: float,
    good_every_s: float = 1.0,
    min_progress_m: float = 0.1,
    walking: bool = False,
    latency_steps: int = 1,
    log=print,
) -> list[dict]:
    ctrl.feedback = feedback
    plasticity = ctrl.device.culture.plasticity
    rows = []
    sim_s = 0.0
    t_wall = time.time()
    episode = 0
    while sim_s < minutes * 60.0:
        state = {"last_x": 0.0}
        every = int(round(good_every_s / env.dt))

        def on_step(step, env_, ctrl_):
            if (step + 1) % every:
                return
            x = env_.root_pos[0]
            good = (x - state["last_x"] >= min_progress_m) if walking else True
            state["last_x"] = x
            if good and not env_.is_fallen():
                feedback.good()
                if isinstance(plasticity, RewardSTDP):
                    plasticity.reward(ctrl_.device.culture, +1.0)

        ep = run_episode(env, ctrl, task.seconds, task.command_fn, push_fn=task.push_fn, latency_steps=latency_steps,
                         on_step=on_step)
        m = robot_metrics(ep, task.seconds, env.dt)
        sim_s += ep.steps * env.dt
        if m["fell"]:
            feedback.bad()
            if isinstance(plasticity, RewardSTDP):
                plasticity.reward(ctrl.device.culture, -1.0)
            while feedback.busy:
                ctrl.device.stimulate(*feedback.pulses(ctrl.step_ms))
                ctrl.device.step(ctrl.step_ms)
                sim_s += ctrl.step_ms / 1000.0
        w = ctrl.device.culture.w
        row = {"episode": episode, "sim_minutes": sim_s / 60.0, "wall_s": time.time() - t_wall,
               "mean_w_exc": float(w[ctrl.device.culture.syn_exc].mean())} | m
        rows.append(row)
        log(row)
        episode += 1
    return rows


def blocks(rows: list[dict], n_blocks: int = 4) -> list[dict]:
    t = np.array([r["sim_minutes"] for r in rows])
    edges = np.linspace(0, t.max() + 1e-9, n_blocks + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, tt in zip(rows, t) if lo <= tt < hi]
        if sel:
            out.append({"from_min": float(lo), "to_min": float(hi), "episodes": len(sel),
                        "survived_s": float(np.mean([r["survived_s"] for r in sel])),
                        "distance_x_m": float(np.mean([r["distance_x_m"] for r in sel]))})
    return out
