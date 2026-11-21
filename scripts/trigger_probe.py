"""Phase 3.1 add-on: can a simple readout of the vestibular dish detect a push ("step now") better than the same
readout of a linear filter of the raw IMU? The first test of the dish *computing* (docs/next-steps.md, Phase 4):
the dish gets only the IMU, so it wins only if its own dynamics (memory, thresholds) make the trigger easier to read.

1. ``collect`` (CPU): teacher stand_push episodes with DART action noise; stores observations, push onsets, episode
   starts.
2. ``run`` (GPU): replays the IMU through the dish (as scripts/vestibular_probe.py), and scores push detection.

Label: 1 in the ``--window`` steps after each push onset (default 10 = 200 ms; a recovery triggered within 200 ms
survives, scripts/reflex_oracle.py). Readouts: cross-validated ridge on
- ``dish``        the dish's spike features (counts + 40 and 160 ms traces), one config
- ``imu_now``     the current IMU (6)
- ``imu_filter``  the IMU over the last ``--lags`` steps (a linear filter with memory as long as the dish's traces)
- ``imu_filter_sq`` the same plus squares and |values| (a cheap nonlinearity, so the dish isn't compared only
                  against a purely linear baseline)
- ``drives_filter`` the stimulation pattern itself (the encoder's sigmoid outputs, one well) over the same lags:
                  separates what the encoder's thresholds give from what the neurons add

Scores: frame-wise AUC; and per push, the detection latency at a threshold giving 1 false alarm per 10 s outside
push windows, plus the miss rate (no detection within 300 ms).

    uv run scripts/trigger_probe.py collect
    uv run scripts/trigger_probe.py run --n-stim 32 --wells 6 --encoding axis
"""

import argparse
import json
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

OUT = Path("outputs/trigger_probe")


def collect(a):
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train.tasks import make_task

    env, teacher = AsimovEnv(), Teacher()
    lo = (env.run.action_clip[:, 0] - env.run.default_joint_pos) / env.run.action_scale
    hi = (env.run.action_clip[:, 1] - env.run.default_joint_pos) / env.run.action_scale
    rng = np.random.default_rng(1)
    O, starts, onsets, sd = [], [], [], None
    Y = []
    for ep in range(a.episodes):
        task = make_task("stand_push", 20.0, seed=600 + ep)
        env.reset()
        starts.append(len(O))
        q = deque([np.zeros(23, np.float32)])
        for s in range(1000):
            if (f := task.push_fn(s * env.dt)) is not None:
                env.push(f)
                onsets.append(len(O))
            o = env.observe()
            y = np.clip(teacher(o), lo, hi)
            noise = 0.0 if sd is None else a.dart * sd * rng.standard_normal(23)
            O.append(o)
            Y.append(y)
            q.append((y + noise).astype(np.float32))
            if env.step(q.popleft()).fell:
                break
        sd = np.array(Y).std(0)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "data.pkl", "wb") as f:
        pickle.dump({"obs": np.array(O, np.float32), "starts": starts, "onsets": onsets}, f)
    print(f"{len(O)} steps, {len(starts)} episodes, {len(onsets)} pushes")


def cv_ridge(X, y, folds=5, alphas=np.logspace(-2, 5, 15)):
    """Out-of-fold ridge scores over contiguous blocks; alpha by out-of-fold squared error."""
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    n = len(X)
    edges = np.linspace(0, n, folds + 1).astype(int)
    P = torch.zeros(len(alphas), n, device=dev)
    for f in range(folds):
        va = torch.zeros(n, dtype=torch.bool, device=dev)
        va[edges[f]:edges[f + 1]] = True
        Xt, ytr = X[~va], yt[~va]
        mu, sd = Xt.mean(0), Xt.std(0).clamp(min=1e-3)
        Z = ((Xt - mu) / sd).double()
        ev, V = torch.linalg.eigh(Z.T @ Z)
        proj = V.T @ (Z.T @ (ytr - ytr.mean()).double())
        Zv = ((X[va] - mu) / sd).double()
        for i, al in enumerate(alphas):
            w = V @ (proj / (ev.clamp(min=0) + al * len(Xt) / 1000.0))
            P[i, va] = (Zv @ w).float() + ytr.mean()
    best = int(torch.argmin(((P - yt[None]) ** 2).sum(1)))
    return P[best].cpu().numpy()


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y > 0.5
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def detection(score, onsets, starts, n, window, fa_per_10s=1.0, max_lat=15):
    """Threshold at a false-alarm rate outside push windows; per push, steps until the score first crosses it."""
    inside = np.zeros(n, bool)
    for o in onsets:
        inside[o:o + max_lat + 10] = True
    first = np.zeros(n, bool)
    first[starts] = True
    for k in range(1, 50):  # skip each episode's first second (settling)
        first[np.clip(np.array(starts) + k, 0, n - 1)] = True
    out = ~inside & ~first
    # rising crossings outside windows, as a function of threshold: pick the lowest threshold within budget
    budget = fa_per_10s * out.sum() / 500.0
    thr_grid = np.quantile(score[out], np.linspace(0.9, 0.99999, 400))
    thr = thr_grid[-1]
    for t in thr_grid:
        above = score > t
        rises = above & ~np.r_[False, above[:-1]] & out
        if rises.sum() <= budget:
            thr = t
            break
    lats = []
    for o in onsets:
        hit = np.flatnonzero(score[o:min(o + max_lat, n)] > thr)
        lats.append(int(hit[0]) if len(hit) else None)
    found = [x for x in lats if x is not None]
    return {"threshold": float(thr), "median_latency_ms": float(np.median(found) * 20) if found else None,
            "p90_latency_ms": float(np.percentile(found, 90) * 20) if found else None,
            "miss_rate": round(1 - len(found) / len(onsets), 3), "within_200ms": round(sum(x < 10 for x in found) / len(onsets), 3)}


def lagged(X, lags, starts):
    """[x_t, x_{t-1}, ..., x_{t-lags+1}], padded with each episode's first value."""
    n = len(X)
    ep = np.zeros(n, int)
    ep[starts] = 1
    ep = np.cumsum(ep) - 1
    st = np.array(starts)[ep]
    cols = []
    for k in range(lags):
        idx = np.maximum(np.arange(n) - k, st)
        cols.append(X[idx])
    return np.concatenate(cols, 1)


def score_all(feats, d, window):
    O, starts, onsets = d["obs"], d["starts"], d["onsets"]
    n = len(O)
    y = np.zeros(n, np.float32)
    for o in onsets:
        y[o:o + window] = 1.0
    res = {}
    for name, X in feats.items():
        s = cv_ridge(X, y)
        res[name] = {"dim": int(X.shape[1]), "auc": round(auc(s, y), 4)} | detection(s, onsets, starts, n, window)
        print(name, json.dumps(res[name]), flush=True)
    return res


def run(a):
    d = pickle.load(open(OUT / "data.pkl", "rb"))
    O = d["obs"]
    imu = O[:, 0:6]
    feats = {"imu_now": imu, "imu_filter": lagged(imu, a.lags, d["starts"])}
    feats["imu_filter_sq"] = np.c_[feats["imu_filter"], feats["imu_filter"] ** 2, np.abs(feats["imu_filter"])]
    from biocompute.train.setup import Setup, build

    enc_setup = Setup(n_stim=a.n_stim, encoder_gain=0.5, projection=a.encoding, select_dims=(0, 1, 2, 3, 4, 5),
                      select_spread=1.0, source="none", seed=0)
    enc = build(enc_setup, O).encoder
    feats["drives_filter"] = lagged(np.array([enc.drive(o) for o in O], np.float32), a.lags, d["starts"])
    name = f"baselines_{a.encoding}_s{a.n_stim}"
    meta = {}
    if not a.no_dish:
        import torch

        name = f"{a.encoding}_s{a.n_stim}_w{a.wells}" + ("_dead" if a.dead else "")
        setup = Setup(culture="gpu_20k", mea="grid256", mea_overrides={"stim_length_um": 120.0}, n_stim=a.n_stim,
                      n_feedback=0, encoding="amplitude", encoder_gain=0.5, projection=a.encoding,
                      select_dims=(0, 1, 2, 3, 4, 5), select_spread=1.0, slots_ms=(0.0, 5.0, 10.0, 15.0), bins=2,
                      taus_ms=(40.0, 160.0), wells=a.wells, source="dead" if a.dead else "mea", seed=0)
        ctrl = build(setup, O)
        ctrl.device.step(1000.0)
        t0 = time.time()
        F = np.array([ctrl.sense(o) for o in O], np.float32)  # one continuous session, as on a real dish
        meta = {"ms_per_step": round((time.time() - t0) / len(O) * 1000, 1), "setup": setup.to_dict()}
        feats["dish"] = F
        np.savez_compressed(OUT / f"feats_{name}.npz", F=F.astype(np.float16))
        del ctrl
        torch.cuda.empty_cache()
    res = score_all(feats, d, a.window)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "window": a.window, "lags": a.lags, "steps": len(O),
                            "pushes": len(d["onsets"]), **meta, "scores": res}) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["collect", "run"])
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--dart", type=float, default=0.3)
    p.add_argument("--encoding", default="axis")
    p.add_argument("--n-stim", type=int, default=32)
    p.add_argument("--wells", type=int, default=6)
    p.add_argument("--dead", action="store_true", help="Control: recurrent synapses removed")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--lags", type=int, default=8, help="Linear filter memory in steps (8 = 160 ms)")
    p.add_argument("--no-dish", action="store_true", help="Baselines only (CPU)")
    a = p.parse_args()
    collect(a) if a.cmd == "collect" else run(a)
