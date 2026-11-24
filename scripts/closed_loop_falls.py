"""Why does the 1 x 12 relay still fall in A1b (scripts/closed_loop_vestibular.py, --edit affine)? A diagnostic of
the finished run; it changes nothing in it.

1. Push table (CPU, from the episode files): stand_push pushes every 3 s from a random direction. For each arm, how
   many pushes of each direction the robot met, and how many were followed by a fall within 3 s. Directions: front
   (the force points backward, |angle| > 135 deg from +x), back (|angle| < 45 deg), side (the rest).
2. Traces (GPU, ``--traces``): replays chosen seeds step by step from the A1b setups (same warm state and noise
   seeds, so the outcomes match the run's), recording the true IMU, the decoded IMU and what the edited teacher sees,
   (decoded - a) / g. Modes: ``closed`` (the A1b culture_edit arm), ``shadow`` (the teacher acts on the true IMU
   while the dish decodes alongside, so decoding errors can't feed back), ``dead`` (the dead_edit arm). For each
   trace: the peak forward lean after the push, when grav_x first passes the encoder's input limit (the normalizer
   clips every input at 4 standard deviations of the replay), and the teacher's view there.

Run (A1b must be finished; about 3 minutes on the GPU):

    uv run scripts/closed_loop_falls.py --traces

Output: <a1>/falls.json (and falls_traces.npz, not committed).

Results (24 Nov 2026, A1b, seeds 9100-9139). Pushes followed by a fall within 3 s, 1 x 12:

| Arm          | front  | side   | back |
|--------------|--------|--------|------|
| culture_edit | 12/41  | 5/96   | 0/39 |
| dead_edit    | 2/52   | 6/119  | 0/53 |

Ideal IMU and 1 x 32 culture_edit: no falls. Traces reproduce the run's outcomes (falls at 8.02, 11.60, 8.00 s).
In the three falls the relay reads the pitch rate 2.4-2.5 spreads too high 0.5-1.0 s after the push (survivors
0.1-1.4, dead culture 1.1-1.9, shadow decoding -0.3 to +0.6). The forward lean passes the input limit (7.9 spreads)
1.0-1.1 s after the push and reaches 33-42 spreads while the teacher sees at most 9.3-9.7. With the true IMU the same
pushes peak at 2.5-4.5 spreads. The dead culture passed the limit once (seed 9112, 12.4 spreads) and recovered, so
the limit makes recovery harder rather than impossible.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_vestibular as clv  # noqa: E402

TRACE_SEEDS = {"closed": [9128, 9112, 9101, 9103, 9115], "shadow": [9128, 9112, 9101], "dead": [9128, 9112]}


def pushes(seed: int, seconds: float = 20.0, dt: float = 0.02):
    from biocompute.train.tasks import make_task

    task, out = make_task("stand_push", seconds, seed=seed), []
    for step in range(int(round(seconds / dt))):
        f = task.push_fn(step * dt)
        if f is not None:
            out.append((round(step * dt, 2), float(np.hypot(*f)), float(np.degrees(np.arctan2(f[1], f[0])))))
    return out


def direction(angle: float) -> str:
    return "front" if abs(angle) > 135 else ("back" if abs(angle) < 45 else "side")


def push_table(a1: Path) -> dict:
    res = {}
    for d in sorted((a1 / "jobs").glob("*/*")):
        rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
        if not rows:
            continue
        tab = {k: {"pushes": 0, "falls_within_3s": 0} for k in ("front", "side", "back")}
        falls = []
        for r in rows:
            last = None
            for t, F, ang in pushes(r["seed"], r["seconds"]):
                if t > r["survived_s"]:
                    break
                k = direction(ang)
                tab[k]["pushes"] += 1
                tab[k]["falls_within_3s"] += int(not r["full"] and r["survived_s"] < t + 3)
                last = (t, round(F), round(ang))
            if not r["full"] and last:
                falls.append({"seed": r["seed"], "fell_s": r["survived_s"], "last_push": last,
                              "direction": direction(last[2])})
        res[f"{d.parent.name}/{d.name}"] = {"by_direction": tab, "falls": falls}
    return res


def traces(a1: Path, data: str) -> dict:
    from biocompute.eval.dish_channel import IMU_SPREAD
    from biocompute.interface.decoders import LinearReadout
    from biocompute.loop import run_episode
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.train.setup import build
    from biocompute.train.tasks import make_task

    O = clv.load_replay(data, 0)
    env, cache, out, arrays = AsimovEnv(), {}, {}, {}

    class Traced(clv.VestibularController):
        shadow = False

        def act(self, obs):
            v = self.vest(obs)
            self.true.append(np.asarray(obs[0:6], float))
            self.decoded.append(v)
            imu = np.asarray(obs[0:6], float) if self.shadow else v
            return (np.asarray(clv.track_a().act(self.W, obs[9:32], obs[32:55], obs[55:78], imu), np.float32),
                    np.zeros(0, np.float32))

    for mode, seeds in TRACE_SEEDS.items():
        dish_name = "dead" if mode == "dead" else "live"
        if dish_name not in cache:
            blob = pickle.loads((a1 / "setups" / f"s12_{dish_name}.pkl").read_bytes())
            dish = build(clv.make_setup(12, dish_name), O)
            ro = LinearReadout(blob["cols"], 6)
            for k, v in blob["readout"].items():
                setattr(ro, k, v)
            cache[dish_name] = (blob, dish, ro)
        blob, dish, ro = cache[dish_name]
        g, a = np.array(blob["row"]["affine_gain"]), np.array(blob["row"]["affine_icpt"])
        stand = np.array(blob["row"]["stand_imu"])
        nm = dish.encoder.normalizer
        lo = float(nm.mean[3] - 4 * nm.std[3])  # grav_x input limit on the forward side
        W = clv.teacher_weights(None) if mode == "shadow" else clv.teacher_weights(g, a)
        for seed in seeds:
            dish.device.load_state(blob["warm"])
            dish.device.culture.torch_gen.manual_seed(7 * seed + 1)
            dish.device.mea.gen.manual_seed(7 * seed + 2)
            ctl = Traced(dish, ro, blob["cols"], W, True, blob["obs0"])
            ctl.shadow = mode == "shadow"
            task = make_task("stand_push", 20.0, seed=seed)
            ep = run_episode(env, ctl, 20.0, task.command_fn, push_fn=task.push_fn, clip=1e6)
            T, P = np.array(ctl.true), np.array(ctl.decoded)
            view = (P[:, :5] - a) / g
            front = [p for p in pushes(seed) if direction(p[2]) == "front"]
            p0 = int(round(front[0][0] / 0.02))
            w = slice(p0, min(p0 + 125, len(T)))
            sp = lambda x: round(float((x - stand[3]) / IMU_SPREAD[3]), 1)  # noqa: E731
            beyond = np.flatnonzero(T[w, 3] < lo)
            out[f"{mode}_{seed}"] = {
                "fell_at": round(float(ep.fell_at), 2), "front_push": front[0],
                "peak_forward_lean_spreads": sp(T[w, 3].min()),
                "input_limit_spreads": sp(lo),
                "passes_limit_after_s": None if not len(beyond) else round(float(beyond[0] * 0.02), 2),
                "teacher_view_min_spreads": sp(view[w, 3].min()),
                "gyro_y_view_minus_true_spreads_0.5_1.0s": round(float(
                    (view[p0 + 25:p0 + 50, 1] - T[p0 + 25:p0 + 50, 1]).mean() / IMU_SPREAD[1]), 2)}
            arrays[f"{mode}_{seed}"] = np.c_[T, P, view]
            print(json.dumps({f"{mode}_{seed}": out[f"{mode}_{seed}"]}), flush=True)
    np.savez(a1 / "falls_traces.npz", **arrays)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a1", default="outputs/closed_loop_vestibular_affine")
    ap.add_argument("--data", default="outputs/probe/teacher_data_20k.pkl")
    ap.add_argument("--traces", action="store_true")
    a = ap.parse_args()
    a1 = Path(a.a1)
    res = {"push_table": push_table(a1)}
    for k, v in res["push_table"].items():
        print(k, json.dumps(v["by_direction"]), flush=True)
    if a.traces:
        res["traces"] = traces(a1, a.data)
    clv.write_json(a1 / "falls.json", res)


if __name__ == "__main__":
    main()
