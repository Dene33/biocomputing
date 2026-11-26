import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

LAGS = (0, 3, 6)
SOURCES = ("input", "amp50_live", "amp50_dead", "amp200_live", "amp200_dead")


def load():
    from biocompute.interface.encoders import Encoder, ObsNormalizer

    d = pickle.load(open("outputs/trigger_probe/data.pkl", "rb"))
    O, starts = d["obs"], np.array(d["starts"])
    push = np.zeros(len(O), int)
    for o in d["onsets"]:
        push[o:o + 10] = 1
    enc = Encoder(np.arange(32), mode="amplitude", normalizer=ObsNormalizer.fit(O), gain=0.5, seed=0)
    enc.fit_axis(O, (0, 1, 2, 3, 4, 5), spread=1.0)
    src = {"input": np.array([enc.drive(o) for o in O], np.float32)}
    for name in SOURCES[1:]:
        F = np.load(f"outputs/dish_regimes/feats_{name}_s32_w1.npz")["F"]
        src[name] = F[:, : F.shape[1] // 2].astype(np.float32)
    return O[:, :5].astype(np.float32), starts, push, src


def target(T, starts, push, n, name):
    if name == "push":
        return push, np.ones(n, bool)
    k = int(name[3:])
    ep_start = starts[np.searchsorted(starts, np.arange(n), side="right") - 1]
    idx = np.arange(n) - k
    return T[np.maximum(idx, 0)], idx >= ep_start


def score(pred, y, name):
    from sklearn.metrics import r2_score, roc_auc_score

    if name == "push":
        return round(float(roc_auc_score(y, pred)), 4)
    return round(float(np.mean([r2_score(y[:, j], pred[:, j]) for j in range(y.shape[1])])), 4)


def knn(Etr, ytr, Ete):
    from sklearn.neighbors import KNeighborsRegressor

    return KNeighborsRegressor(n_neighbors=25, metric="cosine").fit(Etr, ytr).predict(Ete)


def ridge(Xtr, ytr, Xte):
    from sklearn.linear_model import RidgeCV

    return RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Xtr, ytr).predict(Xte)


def cebra_embed(Xtr, Xte, label, iters, seed):
    import torch
    from cebra import CEBRA

    torch.manual_seed(seed)
    np.random.seed(seed)
    m = CEBRA(model_architecture="offset1-model", batch_size=512, learning_rate=3e-4, temperature=1.0,
              output_dimension=8, max_iterations=iters, distance="cosine",
              conditional="time" if label is None else "time_delta", time_offsets=10, device="cuda_if_available",
              verbose=False)
    m.fit(Xtr) if label is None else m.fit(Xtr, label)
    return m.transform(Xtr), m.transform(Xte)


TASKS = ("ridge", "time") + tuple(f"behavior:{t}" for t in ("lag0", "lag3", "lag6", "push"))
SHUFFLED = tuple(f"shuffled:{t}" for t in ("lag0", "lag3", "lag6", "push"))
DATA = {}


def _init():
    import torch

    torch.set_num_threads(1)
    DATA["d"] = load()


def task(job):
    from biocompute.runguard import Guard

    fold, s, kind, iters, out = job
    T, starts, push, src = DATA["d"]
    n = len(T)
    targets = [f"lag{k}" for k in LAGS] + ["push"]
    ep = np.searchsorted(starts, np.arange(n), side="right") - 1
    test = (ep >= 8 * fold) & (ep < 8 * fold + 8)
    X = src[s]
    Z = (X - X[~test].mean(0)) / (X[~test].std(0) + 1e-3)
    g = Guard(f"cebra_{fold}_{s}_{kind.replace(':', '_')}", out, every_s=1e9)
    base = {"fold": fold, "source": s}
    t0 = time.time()
    rows = []
    if kind == "ridge":
        for t in targets:
            y, ok = target(T, starts, push, n, t)
            tr, te = ~test & ok, test & ok
            rows.append(base | {"method": "ridge", "target": t, "score": score(ridge(Z[tr], y[tr], Z[te]), y[te], t)})
    elif kind == "time":
        Etr, Ete = cebra_embed(Z[~test], Z[test], None, iters, seed=fold)
        g.check(bool(np.isfinite(Etr).all() and np.isfinite(Ete).all()), "CEBRA-Time embedding not finite")
        E = np.zeros((n, Etr.shape[1]), np.float32)
        E[~test], E[test] = Etr, Ete
        for t in targets:
            y, ok = target(T, starts, push, n, t)
            tr, te = ~test & ok, test & ok
            rows.append(base | {"method": "cebra_time", "target": t, "wall_s": round(time.time() - t0),
                                "score": score(knn(E[tr], y[tr], E[te]), y[te], t)})
    else:
        mode, t = kind.split(":")
        y, ok = target(T, starts, push, n, t)
        tr, te = ~test & ok, test & ok
        lab = y[tr]
        if t != "push":
            lab = (lab - lab.mean(0)) / (lab.std(0) + 1e-9)
        if mode == "shuffled":
            lab = lab[np.random.default_rng(fold).permutation(len(lab))]
        Etr, Ete = cebra_embed(Z[tr], Z[te], lab, iters, seed=fold)
        g.check(bool(np.isfinite(Etr).all() and np.isfinite(Ete).all()), f"CEBRA embedding not finite ({kind})")
        rows.append(base | {"method": "cebra_behavior" if mode == "behavior" else "cebra_shuffled", "target": t,
                            "wall_s": round(time.time() - t0), "score": score(knn(Etr, y[tr], Ete), y[te], t)})
    return job, rows


def main():
    import multiprocessing as mp

    from biocompute.runguard import Guard

    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="outputs/cebra_probe")
    ap.add_argument("--smoke", action="store_true", help="whole pipeline at tiny scale into <out>/smoke")
    a = ap.parse_args()
    out = Path(a.out)
    floor = 0.5
    if a.smoke:
        out = out / "smoke"
        a.iters, a.folds, a.sources, a.workers, floor = 300, [0], ["input", "amp50_live"], 2, -1e9
        for f in out.glob("*.json*"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {}
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done[(r["fold"], r["source"], r["method"], r["target"])] = r["score"]

    def is_done(fold, s, kind):
        if kind == "ridge":
            return all((fold, s, "ridge", t) in done for t in ("lag0", "lag3", "lag6", "push"))
        if kind == "time":
            return all((fold, s, "cebra_time", t) in done for t in ("lag0", "lag3", "lag6", "push"))
        mode, t = kind.split(":")
        return (fold, s, "cebra_behavior" if mode == "behavior" else "cebra_shuffled", t) in done

    jobs = [(f, s, k, a.iters, out) for f in a.folds for s in a.sources
            for k in TASKS + (SHUFFLED if s == "amp50_live" else ())]
    control = (a.folds[0], "input", "behavior:lag0", a.iters, out)
    jobs.sort(key=lambda j: j[:3] != control[:3])
    todo = [j for j in jobs if not is_done(*j[:3])]
    g = Guard("cebra_probe", out, every_s=0.0)
    g.beat(len(jobs) - len(todo), len(jobs), force=True)

    def emit(rows):
        with open(res_path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)

    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        if todo and todo[0][:3] == control[:3]:
            _, rows = pool.apply(task, (todo.pop(0),))
            emit(rows)
            r2 = rows[0]["score"]
        else:
            r2 = done[(control[0], "input", "cebra_behavior", "lag0")]
        g.check(r2 >= floor, f"positive control failed: CEBRA-Behavior on the stimulation drive decodes the current "
                             f"IMU at R^2 {r2} < {floor}")
        for k, (job, rows) in enumerate(pool.imap_unordered(task, todo)):
            emit(rows)
            g.beat(len(jobs) - len(todo) + k + 1, len(jobs), force=True,
                   last=f"{job[1]}/{job[2]}", last_wall_s=rows[-1].get("wall_s", 0))

    rows = [json.loads(line) for line in res_path.read_text().splitlines()]
    summ = {}
    for r in rows:
        summ.setdefault(f"{r['source']} {r['method']}", {}).setdefault(r["target"], []).append(r["score"])
    table = {k: {t: round(float(np.mean(v)), 3) for t, v in d.items()} for k, d in summ.items()}
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    for k, d in table.items():
        print(k, d, flush=True)


if __name__ == "__main__":
    main()
