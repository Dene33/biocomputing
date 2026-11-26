import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("trigger_probe", HERE / "trigger_probe.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

REGIMES = {
    "amp200": dict(encoding="amplitude", slots_ms=(0.0, 5.0, 10.0, 15.0), rate_scale=1.0),
    "amp50": dict(encoding="amplitude", slots_ms=(0.0,), rate_scale=1.0),
    "rate50": dict(encoding="rate", slots_ms=(0.0,), rate_scale=1.0),
    "rate20": dict(encoding="rate", slots_ms=(0.0,), rate_scale=0.4),
}
SPREAD = np.array([0.0368, 0.0277, 0.0368, 0.0186, 0.0256, 0.0014])
STAND = np.array([0, 0, 0, 0, 0, -1.0])


def lag_r2(X, T, starts, lags=13):
    out = []
    n = len(X)
    ep = np.cumsum(np.isin(np.arange(n), starts)) - 1
    st = np.array(starts)[ep]
    for k in range(lags):
        idx = np.arange(n) - k
        ok = idx >= st
        Y = T[np.maximum(idx, 0)][:, :5]
        r = []
        for j in range(5):
            p = tp.cv_ridge(X[ok], Y[ok, j])
            y = Y[ok, j]
            r.append(1 - ((p - y) ** 2).mean() / y.var())
        out.append(round(float(np.mean(r)), 4))
    return out


def channel_error(X, T):
    m = (np.abs(T[:, :5] - STAND[:5]) <= 3 * SPREAD[:5]).all(1)
    e = []
    for j in range(5):
        p = tp.cv_ridge(X, T[:, j])
        e.append(float(np.sqrt(((p[m] - T[m, j]) ** 2).mean()) / SPREAD[j]))
    return [round(x, 3) for x in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--n-stim", type=int, default=32)
    ap.add_argument("--wells", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--nmda", type=float, default=0.0, help="Slow NMDA-type excitation (culture nmda_ratio)")
    ap.add_argument("--live-only", action="store_true",
                    help="skip the dead culture (its weights are zero, so NMDA changes nothing there)")
    ap.add_argument("--out", default="outputs/dish_regimes")
    a = ap.parse_args()

    import torch

    from biocompute.train.setup import Setup, build

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {json.loads(l)["name"] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
    d = pickle.load(open("outputs/trigger_probe/data.pkl", "rb"))
    if a.steps:
        keep = [s for s in d["starts"] if s < a.steps]
        d = {"obs": d["obs"][: a.steps], "starts": keep, "onsets": [o for o in d["onsets"] if o < a.steps - 20]}
    O, starts = d["obs"], d["starts"]
    T = O[:, 0:6]
    y = np.zeros(len(O), np.float32)
    for o in d["onsets"]:
        y[o:o + 10] = 1.0

    for regime in a.regimes:
        for source in ("mea",) if a.live_only else ("mea", "dead"):
            name = f"{regime}_{'live' if source == 'mea' else 'dead'}_s{a.n_stim}_w{a.wells}"
            name += f"_nmda{a.nmda}" if a.nmda else ""
            if name in done:
                print("skip", name, flush=True)
                continue
            r = REGIMES[regime]
            setup = Setup(culture="gpu_20k", culture_overrides={"nmda_ratio": a.nmda} if a.nmda else {}, mea="grid256", mea_overrides={"stim_length_um": 120.0}, n_stim=a.n_stim,
                          n_feedback=0, encoding=r["encoding"], encoder_gain=0.5, projection="axis",
                          select_dims=(0, 1, 2, 3, 4, 5), select_spread=1.0, slots_ms=r["slots_ms"], bins=2,
                          taus_ms=(40.0, 160.0), wells=a.wells, source=source, seed=0)
            ctrl = build(setup, O)
            ctrl.encoder.rate_scale = r["rate_scale"]
            ctrl.device.step(1000.0)
            cult = ctrl.device.culture
            p0 = ctrl.device.pulses_delivered
            t0 = time.time()
            F, res_x = [], []
            for i, o in enumerate(O):
                F.append(ctrl.sense(o))
                if i % 50 == 0:
                    res_x.append(float(cult.stp_x[cult.excitatory].mean()))
            F = np.array(F, np.float32)
            ms = (time.time() - t0) / len(O) * 1000
            pulses = ctrl.device.pulses_delivered - p0
            C = F.shape[1] // 4
            bins = F[:, :2 * C]
            D = np.array([ctrl.encoder.drive(o) for o in O], np.float32)[:, : a.n_stim]
            row = {"name": name, "regime": regime, "nmda_ratio": a.nmda, "culture": "live" if source == "mea" else "dead",
                   "n_stim": a.n_stim, "wells": a.wells, "steps": len(O), "ms_per_step": round(ms, 1),
                   "pulses_per_electrode_hz": round(pulses / (a.n_stim * a.wells) / (len(O) * 0.02), 1),
                   "spike_rate_hz_per_channel": round(float((bins ** 2).sum(1).mean() / C / 0.02), 2),
                   "synaptic_resources_exc": round(float(np.mean(res_x)), 3)}
            for fs, X in (("all", F), ("bins", bins)):
                s = tp.cv_ridge(X, y)
                row[f"trigger_{fs}"] = {"auc": round(tp.auc(s, y), 4)} | tp.detection(s, d["onsets"], starts, len(O), 10)
            row["memory_dish_bins"] = lag_r2(bins, T, starts)
            row["memory_input"] = lag_r2(D, T, starts)
            row["channel_error_40ms"] = channel_error(F[:, :3 * C], T)
            print(json.dumps({k: v for k, v in row.items() if not k.startswith("memory")} |
                             {"mem_dish": row["memory_dish_bins"][:6], "mem_in": row["memory_input"][:6]}), flush=True)
            with open(res_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            np.savez_compressed(out / f"feats_{name}.npz", F=F.astype(np.float16))
            del ctrl, cult, F
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
