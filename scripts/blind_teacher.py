"""Phase 1.1 (docs/next-steps.md), blind-teacher sweep: how good must a vestibular organoid be?

The teacher runs stand_push (20 s, 1-step action latency as in the culture loop) while one group of its observations
is degraded and everything else stays exact:

- ``imu``    indices 0-5: gyro (x0.25 in the observation) and projected gravity. The signals the dish would carry.
- ``gyro`` / ``grav``  each half of the IMU alone.
- ``joints`` indices 9-54: joint positions and velocities (the reverse test, IMU exact).
- ``jpos`` / ``jvel``  each half alone.

The last action (55-77, the efference copy) and the command (6-8) are always exact.

Degradations (``spread`` = each dimension's std over teacher stand_push rollouts on separate seeds):

- removed: set to the standing value (gyro 0, gravity (0, 0, -1), joints at the default pose, velocities 0)
- noise:   + level x spread, independent per step (``tau`` 0) or AR(1) with time constant ``tau`` steps
- delay:   the value from k control steps ago (20 ms each)
- hold:    sampled every n steps and held (50/n Hz)
- quant:   n uniform levels over each dimension's [0.5, 99.5] percentile range, clipped
- chain:   several of the above in order (delay, hold, noise, quant), as a dish would deliver them

    uv run scripts/blind_teacher.py
"""

import argparse
import json
import multiprocessing as mp
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=40)
p.add_argument("--seed-base", type=int, default=5000, help="Fresh push seeds: seed-base .. seed-base + seeds - 1")
p.add_argument("--spread-seeds", type=int, default=10, help="Seeds (9000+) for the observation spread")
p.add_argument("--workers", type=int, default=14)
p.add_argument("--out", default="outputs/blind_teacher")
args = p.parse_args()

GROUPS = {
    "imu": np.arange(0, 6), "gyro": np.arange(0, 3), "grav": np.arange(3, 6),
    "joints": np.arange(9, 55), "jpos": np.arange(9, 32), "jvel": np.arange(32, 55),
}
STEPS, SECONDS = 1000, 20.0
_E = {}


def _init():
    import onnxruntime as ort

    from biocompute import paths
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher

    t = Teacher()
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1  # one thread per worker
    t.session = ort.InferenceSession(str(Path(paths.POLICY_DIR) / "policy.onnx"), so, providers=["CPUExecutionProvider"])
    _E["env"], _E["teacher"] = AsimovEnv(), t


class Degrade:
    """Stateful corruption of obs[idx]; call once per control step."""

    def __init__(self, cond, stats, rng):
        self.kind, self.level = cond["kind"], cond.get("level", 0)
        self.idx = GROUPS[cond["group"]] if cond["group"] else None
        self.tau = cond.get("tau", 0)
        self.rng = rng
        if self.idx is not None:
            self.sd = stats["spread"][self.idx]
            self.stand = stats["standing"][self.idx]
            self.lo, self.hi = stats["p_lo"][self.idx], stats["p_hi"][self.idx]
        self.buf = deque()
        self.ar = None
        self.held = None
        self.step = 0
        if self.kind == "chain":
            self.parts = [Degrade({"group": cond["group"], **pc}, stats, rng) for pc in cond["parts"]]

    def __call__(self, o):
        if self.kind == "chain":  # a dish: late, at its own rate, noisy, quantised (parts applied in order)
            for part in self.parts:
                o = part(o)
            return o
        if self.idx is None:
            return o
        o = o.copy()
        x = o[self.idx]
        k = self.kind
        if k == "removed":
            y = self.stand
        elif k == "noise":
            e = self.rng.standard_normal(len(self.idx))
            if self.tau:
                a = np.exp(-1.0 / self.tau)
                self.ar = e if self.ar is None else a * self.ar + np.sqrt(1 - a * a) * e  # unit stationary std
                e = self.ar
            y = x + self.level * self.sd * e
        elif k == "delay":
            self.buf.append(x.copy())
            y = self.buf.popleft() if len(self.buf) > self.level else self.stand  # standing value until filled
        elif k == "hold":
            if self.step % self.level == 0:
                self.held = x.copy()
            y = self.held
        elif k == "quant":
            n = self.level
            u = np.clip((x - self.lo) / np.maximum(self.hi - self.lo, 1e-9), 0, 1 - 1e-9)
            y = self.lo + (np.floor(u * n) + 0.5) / n * (self.hi - self.lo)
        else:
            raise ValueError(k)
        self.step += 1
        o[self.idx] = y
        return o


def episode(job):
    from biocompute.train.tasks import make_task

    cond, seed, stats, record = job
    env, teacher = _E["env"], _E["teacher"]
    deg = Degrade(cond, stats, np.random.default_rng(seed * 7919 + zlib.crc32(json.dumps(cond, sort_keys=True).encode()) % 10007))
    task = make_task("stand_push", SECONDS, seed=seed)
    env.reset()
    q = deque([np.zeros(23, np.float32)])
    O = []
    for s in range(STEPS):
        if (f := task.push_fn(s * env.dt)) is not None:
            env.push(f)
        o = env.observe()
        if record:
            O.append(o)
        a = np.zeros(23, np.float32) if cond["policy"] == "zero" else teacher(deg(o))
        q.append(a)
        if env.step(q.popleft()).fell:
            break
    t = env.fell_at if env.fell_at >= 0 else SECONDS
    return (t, np.array(O)) if record else t


def conditions():
    C = [{"policy": "teacher", "group": None, "kind": "exact"}, {"policy": "zero", "group": None, "kind": "exact"}]
    T = lambda **kw: C.append({"policy": "teacher", **kw})  # noqa: E731
    for g in ("imu", "gyro", "grav", "joints", "jpos", "jvel"):
        T(group=g, kind="removed")
    for g in ("imu", "joints"):
        for tau in (0, 10, 50):
            for lv in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
                T(group=g, kind="noise", level=lv, tau=tau)
        for k in (1, 2, 3, 4, 5, 7, 10, 15):
            T(group=g, kind="delay", level=k)
        for n in (2, 3, 5, 10):
            T(group=g, kind="hold", level=n)
        for n in (2, 4, 8, 16):
            T(group=g, kind="quant", level=n)
    for g in ("gyro", "grav"):  # which half of the IMU carries the tolerance limits
        for lv in (0.3, 0.5, 1.0, 2.0):
            T(group=g, kind="noise", level=lv, tau=0)
        for k in (2, 5, 10):
            T(group=g, kind="delay", level=k)
    for g in ("imu",):  # combined dish-like corruption
        for parts in ([("delay", 2), ("noise", 0.5, 10)], [("delay", 2), ("hold", 2), ("noise", 0.5, 10)],
                      [("delay", 2), ("hold", 2), ("noise", 1.0, 10)], [("delay", 3), ("hold", 2), ("noise", 0.5, 10)],
                      [("delay", 2), ("hold", 2), ("noise", 0.5, 10), ("quant", 8)],
                      [("delay", 1), ("hold", 3), ("noise", 0.7, 10), ("quant", 8)]):
            T(group=g, kind="chain", parts=[dict(zip(("kind", "level", "tau"), x)) for x in parts])
    return C


def label(c):
    if c["group"] is None:
        return c["policy"]
    if c["kind"] == "chain":
        return f"{c['group']}:" + "+".join(f"{x['kind']}={x['level']}" for x in c["parts"])
    s = f"{c['group']}:{c['kind']}"
    if c["kind"] != "removed":
        s += f"={c['level']}"
    if c["kind"] == "noise":
        s += f",tau={c['tau']}"
    return s


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("fork").Pool(args.workers, _init)
    t0 = time.time()

    # Observation statistics under the exact teacher on separate seeds.
    exact = {"policy": "teacher", "group": None, "kind": "exact"}
    rec = pool.map(episode, [(exact, 9000 + s, None, True) for s in range(args.spread_seeds)])
    O = np.concatenate([r[1] for r in rec])
    env0 = __import__("biocompute.robot.asimov", fromlist=["AsimovEnv"]).AsimovEnv()
    env0.reset()
    stats = {"spread": O.std(0), "standing": env0.observe(), "p_lo": np.percentile(O, 0.5, 0),
             "p_hi": np.percentile(O, 99.5, 0)}
    print(f"spread from {len(O)} steps; teacher stood {sum(r[0] >= SECONDS for r in rec)}/{len(rec)}", flush=True)

    C = conditions()
    seeds = [args.seed_base + s for s in range(args.seeds)]
    jobs = [(c, s, stats, False) for c in C for s in seeds]
    res = pool.map(episode, jobs, chunksize=4)
    rows = []
    for i, c in enumerate(C):
        r = np.array(res[i * len(seeds):(i + 1) * len(seeds)])
        rows.append({"label": label(c), **c, "survived_s": float(r.mean()), "full": int((r >= SECONDS).sum()),
                     "n": len(r), "per_seed": r.round(2).tolist()})
        print(f"{label(c):32s} {r.mean():5.1f} s  {int((r >= SECONDS).sum()):2d}/{len(r)}", flush=True)
    names = {"gyro": ["gyro_x", "gyro_y", "gyro_z"], "grav": ["grav_x", "grav_y", "grav_z"]}
    doc = {
        "task": "stand_push", "seconds": SECONDS, "seeds": seeds, "action_latency_steps": 1,
        "spread": {"seeds": [9000 + s for s in range(args.spread_seeds)], "steps": len(O),
                   "imu_obs_units": dict(zip(names["gyro"] + names["grav"], stats["spread"][:6].round(4).tolist())),
                   "gyro_rad_s": (stats["spread"][:3] / 0.25).round(4).tolist(),
                   "joint_pos_mean": float(stats["spread"][9:32].mean()),
                   "joint_vel_mean": float(stats["spread"][32:55].mean())},
        "rows": rows, "wall_s": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote {out / 'results.json'} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
