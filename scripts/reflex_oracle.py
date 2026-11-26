import argparse
import json
import multiprocessing as mp
from collections import deque
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--directions", type=int, default=8)
p.add_argument("--forces", type=float, nargs="+", default=[140.0, 180.0])
p.add_argument("--length", type=int, default=75, help="Primitive length in 20 ms steps")
p.add_argument("--seeds", type=int, default=20)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--out", default="outputs/reflex")
args = p.parse_args()

_E = {}


def _init():
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher

    _E["env"], _E["teacher"] = AsimovEnv(), Teacher()


def record(job):
    theta, force, length = job
    env, teacher = _E["env"], _E["teacher"]
    env.reset()
    q = deque([np.zeros(23, np.float32)])
    A = []
    for s in range(50 + length + 100):
        if s == 50:
            env.push(force * np.array([np.cos(theta), np.sin(theta)]))
        o = env.observe()
        if 50 <= s < 50 + length:
            a = teacher(o)
            A.append(a)
        else:
            a = np.zeros(23, np.float32)
        q.append(a)
        if env.step(q.popleft()).fell:
            break
    return {"theta": theta, "force": force, "A": np.array(A), "stood": env.fell_at < 0}


def episode(job):
    from biocompute.train.tasks import make_task

    mode, seed, delay, err, lib = job
    env, teacher = _E["env"], _E["teacher"]
    rng = np.random.default_rng(seed + 7)
    task = make_task("stand_push", 20.0, seed=seed)
    env.reset()
    q = deque([np.zeros(23, np.float32)])
    pending, active = None, None
    for s in range(1000):
        f = task.push_fn(s * env.dt)
        if f is not None:
            env.push(f)
            if mode in ("replay", "gated_teacher"):
                if mode == "gated_teacher":
                    choice = "teacher"
                else:
                    th = np.arctan2(f[1], f[0])
                    mag = np.linalg.norm(f)
                    d = np.abs(np.angle(np.exp(1j * (lib["theta"] - th))))
                    score = d + 0.002 * np.abs(lib["force"] - mag)
                    choice = int(np.argmin(score))
                    if rng.random() < err:
                        wrong = np.flatnonzero(np.abs(lib["theta"] - lib["theta"][choice]) > 1e-6)
                        choice = int(rng.choice(wrong))
                pending = (s + delay, choice)
        if pending is not None and s >= pending[0]:
            active, pending = (pending[1], 0), None
        o = env.observe()
        if mode == "teacher":
            a = teacher(o)
        elif active is not None:
            k, i = active
            if k == "teacher":
                a = teacher(o)
                done = i + 1 >= lib["length"]
            else:
                a = lib["A"][k][i]
                done = i + 1 >= len(lib["A"][k])
            active = None if done else (k, i + 1)
        else:
            a = np.zeros(23, np.float32)
        q.append(np.asarray(a, np.float32))
        if env.step(q.popleft()).fell:
            break
    return env.fell_at if env.fell_at >= 0 else 20.0


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("fork").Pool(args.workers, _init)
    thetas = np.arange(args.directions) * 2 * np.pi / args.directions
    jobs = [(t, f, args.length) for t in thetas for f in args.forces]
    rec = pool.map(record, jobs)
    print(f"library: {sum(r['stood'] for r in rec)}/{len(rec)} recordings stood after the primitive ended", flush=True)
    lib = {"theta": np.array([r["theta"] for r in rec]), "force": np.array([r["force"] for r in rec]),
           "A": [r["A"] for r in rec], "length": args.length}
    np.savez(out / "library.npz", theta=lib["theta"], force=lib["force"], A=np.array(lib["A"], dtype=object),
             allow_pickle=True)

    def run(mode, delay=0, err=0.0):
        res = pool.map(episode, [(mode, 100 + s, delay, err, lib) for s in range(args.seeds)])
        row = {"mode": mode, "delay_steps": delay, "error_rate": err, "survived_s": float(np.mean(res)),
               "full": f"{sum(r >= 20 for r in res)}/{len(res)}"}
        print(json.dumps(row), flush=True)
        return row

    rows = [run("stance"), run("teacher"), run("gated_teacher", 0), run("gated_teacher", 10)]
    for delay in (0, 5, 10, 15):
        rows.append(run("replay", delay))
    for err in (0.1, 0.25, 1.0):
        rows.append(run("replay", 5, err))
    (out / "oracle.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
