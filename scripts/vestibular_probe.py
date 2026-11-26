import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--encodings", nargs="+", default=["axis", "select"])
p.add_argument("--n-stim", type=int, nargs="+", default=[12, 32])
p.add_argument("--wells", type=int, nargs="+", default=[1, 2, 6])
p.add_argument("--spread", type=float, default=1.0, help="Threshold spread (normalized units)")
p.add_argument("--gain", type=float, default=0.5)
p.add_argument("--steps", type=int, default=0, help="Use only the first N steps (0: all)")
p.add_argument("--data", default="outputs/probe/teacher_data_20k.pkl")
p.add_argument("--out", default="outputs/vestibular_probe")
p.add_argument("--tag", default="")
args = p.parse_args()

IMU = np.arange(6)
NAMES = ["gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z"]


def cv_ridge(F, Y, folds=5, alphas=np.logspace(-1, 5, 13)):
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = torch.as_tensor(F, dtype=torch.float32, device=dev)
    ym, ys = Y.mean(0), Y.std(0)
    Yz = torch.as_tensor((Y - ym) / ys, dtype=torch.float32, device=dev)
    n = len(F)
    edges = np.linspace(0, n, folds + 1).astype(int)
    P = torch.zeros(len(alphas), *Yz.shape, device=dev)
    for f in range(folds):
        va = torch.zeros(n, dtype=torch.bool, device=dev)
        va[edges[f]:edges[f + 1]] = True
        Xt, Yt = F[~va], Yz[~va]
        mu, sd = Xt.mean(0), Xt.std(0).clamp(min=1e-3)
        Z = (Xt - mu) / sd
        G, C = Z.T @ Z, Z.T @ (Yt - Yt.mean(0))
        ev, V = torch.linalg.eigh(G.double())
        proj = V.T @ C.double()
        Zv = ((F[va] - mu) / sd).double()
        for i, a in enumerate(alphas):
            W = V @ (proj / (ev.clamp(min=0) + a * len(Xt) / 1000.0)[:, None])
            P[i, va] = (Zv @ W).float() + Yt.mean(0)
    err = ((P - Yz[None]) ** 2).sum((1, 2))
    best = int(torch.argmin(err))
    return P[best].cpu().numpy() * ys + ym, float(alphas[best])


def score(Ptrue, Pred, spread, ep_start):
    out = {}
    r2 = 1 - ((Ptrue - Pred) ** 2).mean(0) / Ptrue.var(0)
    rmse_spread = np.sqrt(((Ptrue - Pred) ** 2).mean(0)) / spread
    lags = {}
    for k in range(0, 4):
        ok = np.ones(len(Ptrue), bool)
        ok[:k] = False
        for s in ep_start:
            ok[s:s + k] = False
        idx = np.flatnonzero(ok)
        e = ((Ptrue[idx - k] - Pred[idx]) ** 2).mean(0)
        lags[k] = (1 - e / Ptrue.var(0)).round(3).tolist()
    out["r2"] = dict(zip(NAMES, r2.round(3).tolist()))
    out["r2_mean"] = round(float(r2.mean()), 3)
    out["rmse_over_spread"] = dict(zip(NAMES, rmse_spread.round(3).tolist()))
    out["rmse_over_spread_xy"] = round(float(rmse_spread[[0, 1, 2, 3, 4]].max()), 3)
    out["r2_by_lag"] = lags
    out["best_lag"] = [int(np.argmax([lags[k][j] for k in range(4)])) for j in range(6)]
    stand = np.array([0, 0, 0, 0, 0, -1.0])
    m = (np.abs(Ptrue[:, :5] - stand[:5]) <= 3 * spread[:5]).all(1)
    e_in = np.sqrt(((Ptrue[m] - Pred[m]) ** 2).mean(0)) / spread
    out["in_range_frac"] = round(float(m.mean()), 3)
    out["rmse_over_spread_in_range"] = dict(zip(NAMES, e_in.round(3).tolist()))
    out["rmse_over_spread_in_range_xy"] = round(float(e_in[:5].max()), 3)
    return out


def main():
    import torch  # noqa: F401  (GPU culture)

    from biocompute.train.setup import Setup, build

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {json.loads(l)["name"] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
    d = pickle.load(open(args.data, "rb"))
    O = d["obs"][: args.steps or None]
    ep_start = [0] + [i for i in range(1, len(O)) if np.abs(O[i, 9:55] - O[i - 1, 9:55]).max() > 0.5]
    spread = np.array(list(json.loads(Path("outputs/blind_teacher/results.json").read_text())
                           ["spread"]["imu_obs_units"].values()))
    T = O[:, IMU]
    for enc in args.encodings:
        for n_stim in args.n_stim:
            for wells in args.wells:
                name = f"{enc}_s{n_stim}_w{wells}{args.tag}"
                if name in done:
                    print("skip", name, flush=True)
                    continue
                setup = Setup(culture="gpu_20k", mea="grid256", mea_overrides={"stim_length_um": 120.0},
                              n_stim=n_stim, n_feedback=0, encoding="amplitude", encoder_gain=args.gain,
                              projection=enc, select_dims=tuple(IMU.tolist()), select_spread=args.spread,
                              slots_ms=(0.0, 5.0, 10.0, 15.0), bins=2, taus_ms=(40.0, 160.0), wells=wells,
                              diverse_wells=False, source="mea", seed=0)
                ctrl = build(setup, O)
                ctrl.device.step(1000.0)
                t0 = time.time()
                F = np.array([ctrl.sense(o) for o in O], np.float32)
                ms = (time.time() - t0) / len(O) * 1000
                D = np.array([ctrl.encoder.drive(o) for o in O], np.float32)
                C = F.shape[1] // 4
                row = {"name": name, "encoding": enc, "n_stim": n_stim, "wells": wells, "stim_total": n_stim * wells,
                       "ms_per_step": round(ms, 1), "steps": len(O), "channels_recorded": C,
                       "mean_rate_hz": round(float((F[:, :2 * C] ** 2).sum(1).mean() / C / 0.02), 2)}
                preds = {}
                for fs, cols in (("bins", slice(0, 2 * C)), ("+40", slice(0, 3 * C)), ("all", slice(0, 4 * C))):
                    P, alpha = cv_ridge(F[:, cols], T)
                    row[fs] = score(T, P, spread, ep_start) | {"alpha": alpha}
                    preds[fs] = P.astype(np.float32)
                P, _ = cv_ridge(D, T)
                row["drives_ceiling"] = score(T, P, spread, ep_start)
                preds["drives_ceiling"] = P.astype(np.float32)
                np.savez_compressed(out / f"preds_{name}.npz", truth=T.astype(np.float32), **preds)
                print(json.dumps({k: (v if not isinstance(v, dict) else
                                      {kk: v[kk] for kk in ("r2_mean", "rmse_over_spread_in_range_xy", "best_lag")})
                                  for k, v in row.items()}), flush=True)
                with open(res_path, "a") as f:
                    f.write(json.dumps(row | {"setup": setup.to_dict()}) + "\n")
                del ctrl, F
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
