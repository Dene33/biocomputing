"""Weight-space programming of a balance expert: write its 23 × 78 weight matrix from physics, with a handful of
interpretable gains, then tune those gains by a cross-entropy search on stand_push.

The matrix maps the policy observation linearly to joint actions. Only the pelvis tilt (projected gravity) and
rate (gyro) enter, into the ankle, hip and knee joints: an ankle/hip balance strategy. Pitch joints of the two legs
have mirrored axes (their default angles are mirrored); whether roll joints are mirrored is a searched sign.

    uv run scripts/program_expert.py --iters 15
"""

import argparse
import json
import multiprocessing as mp
from collections import deque
from pathlib import Path

import numpy as np

JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
          'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
          'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
          'waist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
          'right_elbow_joint', 'right_wrist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
          'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_yaw_joint']
J = {n: i for i, n in enumerate(JOINTS)}
WX, WY, GX, GY = 0, 1, 3, 4  # observation: gyro × 0.25 at 0:3, projected gravity at 3:6
GAINS = ["ankle_pitch_p", "ankle_pitch_d", "hip_pitch_p", "hip_pitch_d", "ankle_roll_p", "ankle_roll_d",
         "hip_roll_p", "hip_roll_d", "knee_p"]


def program(gains: np.ndarray, roll_mirror: float) -> np.ndarray:
    """The 23 × 78 weight matrix for these gains."""
    kpa, kda, kph, kdh, kra, krda, krh, krdh, kk = gains
    K = np.zeros((23, 78))
    for side, s in (("left", 1.0), ("right", -1.0)):
        K[J[f"{side}_ankle_pitch_joint"], [GX, WY]] = s * kpa, s * kda
        K[J[f"{side}_hip_pitch_joint"], [GX, WY]] = s * kph, s * kdh
        K[J[f"{side}_knee_joint"], GX] = s * kk
        r = 1.0 if side == "left" else roll_mirror
        K[J[f"{side}_ankle_roll_joint"], [GY, WX]] = r * kra, r * krda
        K[J[f"{side}_hip_roll_joint"], [GY, WX]] = r * krh, r * krdh
    return K


_E = {}


def _init():
    from biocompute.robot.asimov import AsimovEnv

    _E["env"] = AsimovEnv()


def rollout(job) -> float:
    """job = (K, push seed). Seconds survived on stand_push (20 s), with one step of latency."""
    from biocompute.train.tasks import _pushes

    K, seed = job
    env = _E["env"]
    push = _pushes(seed)
    env.reset()
    q = deque([np.zeros(23, np.float32)])
    for s in range(1000):
        if (f := push(s * env.dt)) is not None:
            env.push(f)
        q.append(np.clip(K @ env.observe(), -6, 6).astype(np.float32))
        if env.step(q.popleft()).fell:
            break
    return env.fell_at if env.fell_at >= 0 else 20.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--out", default="outputs/expert")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("fork").Pool(a.workers, _init)
    rng = np.random.default_rng(0)
    results = []
    for mirror in (1.0, -1.0):
        mu, sd = np.zeros(len(GAINS)), np.full(len(GAINS), 3.0)
        best = (0.0, mu.copy())
        for it in range(a.iters):
            P = mu + sd * rng.standard_normal((a.pop, len(GAINS)))
            seeds = rng.integers(0, 10**6, 3)
            S = np.array(pool.map(rollout, [(program(g, mirror), int(s)) for g in P for s in seeds]))
            S = S.reshape(a.pop, 3).mean(1)
            elite = P[np.argsort(-S)[: a.pop // 5]]
            mu, sd = elite.mean(0), elite.std(0) + 0.1
            ev = pool.map(rollout, [(program(mu, mirror), 100 + s) for s in range(15)])
            row = {"roll_mirror": mirror, "iter": it, "pop_best": float(S.max()), "eval_survived": float(np.mean(ev)),
                   "eval_full": int(sum(e >= 20 for e in ev)), "gains": dict(zip(GAINS, mu.round(3).tolist()))}
            print(json.dumps(row), flush=True)
            if row["eval_survived"] > best[0]:
                best = (row["eval_survived"], mu.copy())
        results.append({"roll_mirror": mirror, "eval_survived": best[0], "gains": best[1].tolist()})
    win = max(results, key=lambda r: r["eval_survived"])
    np.save(out / "programmed_K.npy", program(np.array(win["gains"]), win["roll_mirror"]))
    (out / "programmed.json").write_text(json.dumps({"results": results, "gain_names": GAINS}, indent=1))
    print(json.dumps(win))


if __name__ == "__main__":
    main()
