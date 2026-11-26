import argparse
import json
import multiprocessing as mp
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--variants", nargs="+", default=["full", "noimu", "noimu_novel", "noimu_hist", "imu_novel"])
p.add_argument("--rounds", type=int, default=16)
p.add_argument("--episodes", type=int, default=16)
p.add_argument("--beta0", type=float, default=0.5)
p.add_argument("--beta-decay", type=float, default=0.6)
p.add_argument("--recovery-weight", type=float, default=4.0)
p.add_argument("--ensemble", type=int, default=4)
p.add_argument("--eval-seeds", type=int, default=10, help="Per round, seeds 100+")
p.add_argument("--test-seeds", type=int, default=40, help="Final, seeds 5000+ (the blind-teacher sweep's)")
p.add_argument("--workers", type=int, default=14)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="outputs/no_imu_student")
args = p.parse_args()

IMU, JPOS, JVEL, LAST = np.arange(0, 6), np.arange(9, 32), np.arange(32, 55), np.arange(55, 78)
SECONDS, STEPS, HIST = 20.0, 1000, 4


def features(variant):
    cur = {"full": [IMU, JPOS, JVEL], "noimu": [JPOS, JVEL], "noimu_novel": [JPOS], "noimu_hist": [JPOS, JVEL],
           "imu_novel": [IMU, JPOS]}[variant]
    return np.concatenate(cur + [LAST]), variant == "noimu_hist"


class Featurizer:
    def __init__(self, variant):
        self.idx, self.hist = features(variant)
        self.past = deque(maxlen=HIST)

    def __call__(self, o):
        x = o[self.idx]
        if not self.hist:
            return x
        js = o[9:55]
        past = list(self.past) + [js] * (HIST - len(self.past))
        self.past.appendleft(js.copy())
        return np.concatenate([x[:-23]] + [js - q for q in past] + [x[-23:]])

    @property
    def dim(self):
        return len(self.idx) + (HIST * 46 if self.hist else 0)


_E = {}


def _init():
    import onnxruntime as ort

    from biocompute import paths
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher

    t = Teacher()
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    t.session = ort.InferenceSession(str(Path(paths.POLICY_DIR) / "policy.onnx"), so, providers=["CPUExecutionProvider"])
    env = AsimovEnv()
    run = env.run
    _E.update(env=env, teacher=t, lo=(run.action_clip[:, 0] - run.default_joint_pos) / run.action_scale,
              hi=(run.action_clip[:, 1] - run.default_joint_pos) / run.action_scale)


def episode(job):
    from biocompute.train.tasks import make_task

    env, teacher = _E["env"], _E["teacher"]
    student, variant, seed, beta, label = job["student"], job["variant"], job["seed"], job["beta"], job["label"]
    blind_imu = job.get("blind_imu", False)
    rng = np.random.default_rng(seed)
    feat = Featurizer(variant)
    task = make_task("stand_push", SECONDS, seed=seed)
    env.reset()
    q = deque([np.zeros(23, np.float32)])
    X, Y, O = [], [], []
    for s in range(STEPS):
        if (f := task.push_fn(s * env.dt)) is not None:
            env.push(f)
        o = env.observe()
        if blind_imu:
            o = o.copy()
            o[0:6] = [0, 0, 0, 0, 0, -1]
        x = feat(o)
        y = np.clip(teacher(o), _E["lo"], _E["hi"]) if label else None
        if student is None or (label and rng.random() < beta):
            a = teacher(o) if y is None else y
        else:
            a = student(x)
        if label:
            X.append(x)
            Y.append(y)
            O.append(o)
        q.append(np.asarray(a, np.float32))
        if env.step(q.popleft()).fell:
            break
    out = {"survived_s": env.fell_at if env.fell_at >= 0 else SECONDS}
    if label:
        out.update(X=np.array(X, np.float32), Y=np.array(Y, np.float32), O=np.array(O, np.float32))
    return out


def train_variant(pool, variant, log):
    from biocompute.interface.decoders import EnsembleReadout, MLPReadout

    dim = Featurizer(variant).dim
    rng = np.random.default_rng(args.seed)
    X, Y, O, history, readouts = [], [], [], [], []
    student = None
    for r in range(args.rounds):
        t0 = time.time()
        beta = 1.0 if r == 0 else args.beta0 * args.beta_decay ** (r - 1)
        res = pool.map(episode, [dict(student=student, variant=variant, seed=int(rng.integers(1, 10**6)), beta=beta,
                                      label=True) for _ in range(args.episodes)])
        X += [x["X"] for x in res]
        Y += [x["Y"] for x in res]
        O += [x["O"] for x in res]
        Xa, Ya, Oa = np.concatenate(X), np.concatenate(Y), np.concatenate(O)
        gyro = np.abs(Oa[:, 0:2]).sum(1)
        w = 1.0 + args.recovery_weight * (gyro > np.percentile(gyro, 80))
        ro = MLPReadout(dim, skip_dim=23)
        fit = ro.fit(Xa, Ya, weights=w, seed=args.seed + r)
        readouts.append(ro)
        student = EnsembleReadout(readouts[-args.ensemble:]) if r >= args.ensemble else ro
        ev = pool.map(episode, [dict(student=student, variant=variant, seed=100 + s, beta=0.0, label=False)
                                for s in range(args.eval_seeds)])
        surv = [e["survived_s"] for e in ev]
        row = {"variant": variant, "round": r, "beta": round(beta, 4), "samples": len(Xa), "val_r2_std": fit["val_r2"],
               "collect_survived_s": float(np.mean([x["survived_s"] for x in res])),
               "eval_survived_s": float(np.mean(surv)), "eval_full": int(sum(s >= SECONDS for s in surv)),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log(row)
    return student, history


def test(pool, student, variant, **kw):
    seeds = [5000 + s for s in range(args.test_seeds)]
    res = pool.map(episode, [dict(student=student, variant=variant, seed=s, beta=0.0, label=False, **kw)
                             for s in seeds])
    r = np.array([x["survived_s"] for x in res])
    return {"survived_s": float(r.mean()), "full": int((r >= SECONDS).sum()), "n": len(r),
            "per_seed": r.round(2).tolist()}


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("fork").Pool(args.workers, _init)
    res_path = out / "results.json"
    doc = json.loads(res_path.read_text()) if res_path.exists() else {"args": vars(args), "variants": {}}
    with open(out / "log.jsonl", "a") as logf:
        def log(row):
            print(json.dumps(row), flush=True)
            logf.write(json.dumps(row) + "\n")
            logf.flush()

        for v in args.variants:
            if v in doc["variants"]:
                log({"skip": v, "reason": "already in results.json"})
                continue
            student, history = train_variant(pool, v, log)
            entry = {"features": Featurizer(v).dim, "history": history, "test": test(pool, student, v)}
            if v in ("full", "imu_novel"):
                entry["test_imu_removed"] = test(pool, student, v, blind_imu=True)
            log({"variant": v, "test": {k: x for k, x in entry["test"].items() if k != "per_seed"},
                 "test_imu_removed": {k: x for k, x in entry.get("test_imu_removed", {}).items() if k != "per_seed"}})
            doc["variants"][v] = entry
            res_path.write_text(json.dumps(doc, indent=1))
            with open(out / f"student_{v}.pkl", "wb") as f:
                pickle.dump(student, f)
    pool.terminate()


if __name__ == "__main__":
    main()
