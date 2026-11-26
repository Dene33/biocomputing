import argparse
import itertools
import json
import multiprocessing as mp
from collections import deque
from pathlib import Path

import numpy as np

LEGS = np.arange(12)
KNEES = np.array([3, 9])
KNEE_SIGN = np.array([1.0, -1.0])
SECONDS = 20.0
_E = {}

CONDITIONS = {
    "nominal": {},
    "knees_0.6": {"effort": {"(left|right)_knee_joint": 0.6}},
    "knees_0.45": {"effort": {"(left|right)_knee_joint": 0.45}},
    "knees_0.3": {"effort": {"(left|right)_knee_joint": 0.3}},
    "left_knee_0.3": {"effort": {"left_knee_joint": 0.3}},
    "ankles_0.5": {"effort": {"(left|right)_ankle_.*": 0.5}},
    "ankles_0.3": {"effort": {"(left|right)_ankle_.*": 0.3}},
    "payload_3kg": {"payload": 3.0},
    "payload_6kg": {"payload": 6.0},
    "payload_10kg": {"payload": 10.0},
    "payload_20kg": {"payload": 20.0},
    "friction_0.3": {"friction": 0.3},
    "friction_0.15": {"friction": 0.15},
    "delay_10ms": {"delay": 2},
    "delay_20ms": {"delay": 4},
    "motor_speed_0.6": {"speed": 0.6},
    "motor_speed_0.4": {"speed": 0.4},
}


def _init():
    import onnxruntime as ort

    from biocompute import paths
    from biocompute.robot.teacher import Teacher

    t = Teacher()
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    t.session = ort.InferenceSession(str(Path(paths.POLICY_DIR) / "policy.onnx"), so, providers=["CPUExecutionProvider"])
    _E["teacher"], _E["envs"] = t, {}


def env_for(cond):
    from biocompute.robot.asimov import AsimovEnv, Perturbation

    if cond not in _E["envs"]:
        c = CONDITIONS[cond]
        _E["envs"] = {cond: AsimovEnv(perturbation=Perturbation(
            foot_friction=c.get("friction"), effort_scale=c.get("effort", {}), motor_speed_ratio=c.get("speed"),
            delay=c.get("delay", 0)))}
    return _E["envs"][cond]


def episode(job):
    from biocompute.train.tasks import make_task

    cond, corr, seed = job
    env, teacher = env_for(cond), _E["teacher"]
    task = make_task("stand_push", SECONDS, seed=seed)
    env.reset()
    if (kg := CONDITIONS[cond].get("payload")):
        env.robot.model.body_mass[env.pelvis] += kg
    gain, bias, imu = corr
    q = deque([np.zeros(23, np.float32)])
    for s in range(int(SECONDS / env.dt)):
        if (f := task.push_fn(s * env.dt)) is not None:
            env.push(f)
        o = env.observe()
        if imu != 1.0:
            o = o.copy()
            o[0:5] *= imu
        a = teacher(o)
        a[LEGS] *= gain
        a[KNEES] += bias * KNEE_SIGN
        q.append(a.astype(np.float32))
        if env.step(q.popleft()).fell:
            break
    return env.fell_at if env.fell_at >= 0 else SECONDS


def score(pool, cond, corr, seeds):
    r = np.array(pool.map(episode, [(cond, corr, s) for s in seeds], chunksize=max(1, len(seeds) // 10)))
    return {"survived_s": round(float(r.mean()), 2), "full": int((r >= SECONDS).sum()), "n": len(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="outputs/adaptation_slot")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(6000, 6000 + a.seeds))
    base = (1.0, 0.0, 1.0)
    grid = [c for c in itertools.product((0.8, 1.0, 1.25, 1.5), (-0.6, -0.3, 0.0, 0.3, 0.6), (1.0, 1.5)) if c != base]
    doc = {"seeds": seeds, "conditions": {}, "best": {}, "cross": {}}
    with mp.get_context("fork").Pool(a.workers, _init) as pool:
        for cond in CONDITIONS:
            doc["conditions"][cond] = score(pool, cond, base, seeds)
            print(cond, doc["conditions"][cond], flush=True)
        failing = [c for c, v in doc["conditions"].items() if v["full"] <= 0.8 * a.seeds and c != "nominal"]
        for cond in failing:
            res = {str(c): score(pool, cond, c, seeds) for c in grid}
            best = max(res, key=lambda k: (res[k]["full"], res[k]["survived_s"]))
            doc["best"][cond] = {"correction": best, **res[best], "uncorrected": doc["conditions"][cond],
                                 "all": res}
            print("best for", cond, best, res[best], flush=True)
            (out / "results.json").write_text(json.dumps(doc, indent=1))
        bests = sorted({doc["best"][c]["correction"] for c in failing})
        for b in bests:
            corr = tuple(json.loads(b.replace("(", "[").replace(")", "]")))
            doc["cross"][b] = {cond: score(pool, cond, corr, seeds) for cond in ["nominal"] + failing}
            print("cross", b, {k: v["full"] for k, v in doc["cross"][b].items()}, flush=True)
            (out / "results.json").write_text(json.dumps(doc, indent=1))
    (out / "results.json").write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
