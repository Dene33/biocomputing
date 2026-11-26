import importlib.util
import json
import multiprocessing as mp
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("adaptation_slot", HERE / "adaptation_slot.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

INV = np.argsort([0, 1, 6, 7, 12, 13, 14, 18, 19, 2, 3, 8, 9, 15, 16, 20, 21, 4, 5, 10, 11, 17, 22])
CORR = {"none": (1.0, 0.0, 1.0), "knee_fix": (1.0, -0.6, 1.0), "payload_fix": (1.0, 0.0, 1.5)}
FORCES = {"gentle_40-80N": (40.0, 80.0), "medium_80-120N": (80.0, 120.0), "task_120-200N": (120.0, 200.0)}


def episode(job):
    from biocompute.train.tasks import _pushes

    cond, corr, force, seed = job
    env, teacher = A.env_for(cond), A._E["teacher"]
    push = _pushes(seed, force_n=FORCES[force])
    env.reset()
    if (kg := A.CONDITIONS[cond].get("payload")):
        env.robot.model.body_mass[env.pelvis] += kg
    gain, bias, imu = CORR[corr]
    q = deque([np.zeros(23, np.float32)])
    peaks, cur, since, base, hist, track = [], 0.0, None, np.zeros(2), deque(maxlen=10), []
    for s in range(1000):
        if (f := push(s * env.dt)) is not None:
            env.push(f)
            if since is not None:
                peaks.append(cur)
            cur, since = 0.0, s
            base = np.mean(hist, 0) if hist else np.zeros(2)
        o = env.observe()
        hist.append(o[3:5].copy())
        track.append(float(np.abs(0.25 * o[55:78] - o[9:32][INV])[[3, 9]].mean()))
        if since is not None and s - since < 50:
            cur = max(cur, float(np.linalg.norm(o[3:5] - base)))
        o2 = o.copy()
        o2[0:5] *= imu
        a = teacher(o2)
        a[A.LEGS] *= gain
        a[A.KNEES] += bias * A.KNEE_SIGN
        q.append(a.astype(np.float32))
        if env.step(q.popleft()).fell:
            break
    survived = env.fell_at < 0
    if since is not None and survived:
        peaks.append(cur)
    return (env.fell_at if env.fell_at >= 0 else 20.0), peaks, float(np.mean(track))


def main():
    out = Path("outputs/struggle_signal")
    forces = ["gentle_40-80N", "medium_80-120N"]
    out.mkdir(parents=True, exist_ok=True)
    doc = {}
    with mp.get_context("fork").Pool(10, A._init) as pool:
        for cond in ("knees_0.45", "payload_20kg", "nominal"):
            for force in forces:
                for corr in CORR:
                    res = pool.map(episode, [(cond, corr, force, 6000 + s) for s in range(20)], chunksize=2)
                    t = np.array([r[0] for r in res])
                    pk = [p for r in res for p in r[1]]
                    row = {"survived_s": round(float(t.mean()), 2), "full": int((t >= 20).sum()),
                           "peak_tilt_mean": round(float(np.mean(pk)), 4) if pk else None,
                           "peak_tilt_p90": round(float(np.percentile(pk, 90)), 4) if pk else None, "pushes": len(pk),
                           "knee_tracking_error": round(float(np.mean([r[2] for r in res])), 4)}
                    doc[f"{cond}|{force}|{corr}"] = row
                    print(cond, force, corr, row, flush=True)
    (out / "results.json").write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
