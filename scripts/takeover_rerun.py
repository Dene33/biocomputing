from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartpole_replication as CP  # noqa: E402

from biocompute.runguard import job_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFS = {"additive": ROOT / "outputs/drift_scaling/drift_scaling.json",
        "multiplicative": ROOT / "outputs/drift_scaling/drift_multiplicative_scaling.json"}
TAKEOVER = {0: ("additive", 136, 155, 139), 1: ("multiplicative", 129, 34, 35), 2: ("additive", 100, 217, 233)}
ETA = 1e-5
SECONDS = 600.0
COS = 0.5
NULL_MIN_UM = 500.0
FAR_UM = 200.0
WIN_MS = 50.0
ANSWER = 0.3
AREA_ACC = 0.75


def arm_specs(seconds: float, rest_s: float):
    out = []
    for seed, (rule, *_) in TAKEOVER.items():
        for x in (1.0, 0.0):
            sp = {"scale": x, "rule": rule if x else "additive", "eta": 0.0, "w_max_inh": 30.0, "floor": 0.5,
                  "homeo": ETA} | ({"seed": seed} if seed else {})
            name = f"x{x:g}" + CP.spec_tag({**sp, "scale": 1.0}) + (f"_s{seed}" if seed else "")
            key = job_key(f"drift_{name}", {**sp, "seconds": seconds, "rest_s": rest_s, "dump": True})
            out.append((name, key, sp))
    return out


def stored_row(name: str, rule: str) -> dict:
    rows = json.loads(REFS[rule].read_text())["rows"]
    hit = [r for r in rows if r["arm"] == name]
    if len(hit) != 1:
        raise SystemExit(f"{len(hit)} stored rows named {name} in {REFS[rule]}")
    return hit[0]


def load_dump(p: Path) -> dict:
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def ee_synapses(d: dict) -> dict:
    ee = d["syn_exc"] & d["excitatory"][d["post"]]
    pre, post = d["pre"][ee], d["post"][ee]
    xy = d["pos"][:, :2].astype(np.float64)
    return {"pre": pre, "pre_xy": xy[pre], "vec": xy[post] - xy[pre],
            "dw": d["w_end"][ee].astype(np.float64) - d["w_start"][ee].astype(np.float64), "n": len(d["pos"])}


def direction_score(s: dict, centre, cut: float = COS) -> dict:
    r = s["pre_xy"] - np.asarray(centre, np.float64)
    den = np.linalg.norm(s["vec"], axis=1) * np.linalg.norm(r, axis=1)
    ok = den > 0
    cos = np.zeros(len(den))
    cos[ok] = (s["vec"][ok] * r[ok]).sum(1) / den[ok]
    da, dt = s["dw"][ok & (cos > cut)], s["dw"][ok & (cos < -cut)]
    score = float(da.mean() - dt.mean())
    se = float(np.sqrt(da.var(ddof=1) / len(da) + dt.var(ddof=1) / len(dt)))
    return {"score": score, "se": se, "z": score / se if se > 0 else float("nan"), "n_away": int(len(da)),
            "n_toward": int(len(dt)), "dw_away": float(da.mean()), "dw_toward": float(dt.mean())}


def within_score(s: dict, centre, cut: float = COS) -> tuple[float, float, int]:
    r = s["pre_xy"] - np.asarray(centre, np.float64)
    den = np.linalg.norm(s["vec"], axis=1) * np.linalg.norm(r, axis=1)
    cos = np.zeros(len(den))
    ok = den > 0
    cos[ok] = (s["vec"][ok] * r[ok]).sum(1) / den[ok]
    a, t = cos > cut, cos < -cut
    na = np.bincount(s["pre"][a], minlength=s["n"])
    nt = np.bincount(s["pre"][t], minlength=s["n"])
    sa = np.bincount(s["pre"][a], weights=s["dw"][a], minlength=s["n"])
    st = np.bincount(s["pre"][t], weights=s["dw"][t], minlength=s["n"])
    both = (na > 0) & (nt > 0)
    d = sa[both] / na[both] - st[both] / nt[both]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int(both.sum())


def score_maps(s: dict, exy: np.ndarray, winner: int, loser: int, min_um: float = NULL_MIN_UM) -> dict:
    dist = np.linalg.norm(exy - exy[winner], axis=1)
    far = (np.linalg.norm(exy[:, None] - exy[[winner, loser]][None], axis=2) >= min_um).all(1)
    out = {}
    for name, fn in (("plain", lambda c: direction_score(s, c)["score"]), ("within", lambda c: within_score(s, c)[0])):
        m = np.array([fn(c) for c in exy])
        out[name] = {"winner": round(float(m[winner]), 4), "loser": round(float(m[loser]), 4),
                     "rank_of_winner": int((m > m[winner]).sum()) + 1, "n_positions": len(m),
                     "peak_dist_to_winner_um": round(float(dist[int(np.argmax(m))]), 1),
                     "corr_with_dist_to_winner": round(float(np.corrcoef(m, dist)[0, 1]), 3),
                     "null_median": round(float(np.median(m[far])), 4),
                     "null_p95": round(float(np.percentile(m[far], 95)), 4),
                     "null_min": round(float(m[far].min()), 4)}
    w = within_score(s, exy[winner])
    out["within"] |= {"winner_se": round(w[1], 4), "neurons": w[2]}
    return out


def null_scores(s: dict, exy: np.ndarray, enc_xy: np.ndarray, min_um: float = NULL_MIN_UM) -> np.ndarray:
    far = (np.linalg.norm(exy[:, None] - enc_xy[None], axis=2) >= min_um).all(1)
    return np.array([direction_score(s, c)["score"] for c in exy[far]])


def hubs(s: dict, exc: np.ndarray, xy: np.ndarray, centre, top: float = 0.01) -> dict:
    gain = np.bincount(s["pre"], weights=np.maximum(s["dw"], 0.0), minlength=s["n"])[exc]
    pos = xy[exc]
    k = max(1, int(round(top * len(gain))))
    idx = np.argsort(gain, kind="stable")[::-1][:k]
    d_all = np.linalg.norm(pos - np.asarray(centre), axis=1)
    return {"top_share": float(gain[idx].sum() / max(gain.sum(), 1e-12)), "n_top": k,
            "top_median_dist_um": float(np.median(d_all[idx])), "all_median_dist_um": float(np.median(d_all)),
            "top_within_300um": float((d_all[idx] <= 300).mean()), "all_within_300um": float((d_all <= 300).mean())}


def grid_index(exy: np.ndarray):
    xs = np.unique(np.round(exy[:, 0], 3))
    pitch = float(np.min(np.diff(xs)))
    ix = np.round((exy[:, 0] - exy[:, 0].min()) / pitch).astype(int)
    iy = np.round((exy[:, 1] - exy[:, 1].min()) / pitch).astype(int)
    return iy, ix, pitch


def pulse_responses(d: dict, win_ms: float = WIN_MS):
    o = np.argsort(d["spike_t_ms"], kind="stable")
    t, ch = d["spike_t_ms"][o], d["spike_ch"][o].astype(int)
    o = np.argsort(d["pulse_t_ms"], kind="stable")
    pt, pc = d["pulse_t_ms"][o], d["pulse_ch"][o].astype(int)
    gap = np.diff(pt)
    iso = (np.r_[np.inf, gap] > win_ms) & (np.r_[gap, np.inf] > win_ms)
    pt, pc = pt[iso], pc[iso]
    nch = len(d["electrode_xy"])
    counts = np.zeros((len(pt), nch), np.float32)
    lat = np.full((len(pt), nch), np.nan, np.float32)
    lo, hi = np.searchsorted(t, pt, side="right"), np.searchsorted(t, pt + win_ms, side="right")
    for k in range(len(pt)):
        c, dt = ch[lo[k]:hi[k]], t[lo[k]:hi[k]] - pt[k]
        counts[k] = np.bincount(c, minlength=nch)
        u, first = np.unique(c, return_index=True)
        lat[k, u] = dt[first]
    return pt, pc, counts, lat


def _ridge_sign(Xtr, ytr, Xte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    b = ytr.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - b))
    return np.sign(B @ w + b)


def decode_across_time(minute, y, F, cols) -> list:
    X = F[:, cols]
    tr = np.flatnonzero(minute == 0)
    acc = []
    for m in range(int(minute.max()) + 1):
        te = np.flatnonzero(minute == m)
        if m == 0:
            halves = (tr[::2], tr[1::2])
            hits = sum(int((_ridge_sign(X[a], y[a], X[b]) == y[b]).sum()) for a, b in (halves, halves[::-1]))
            acc.append(round(hits / max(len(tr), 1), 3))
        elif len(te) and len(np.unique(y[tr])) == 2:
            acc.append(round(float((_ridge_sign(X[tr], y[tr], X[te]) == y[te]).mean()), 3))
        else:
            acc.append(None)
    return acc


def latency_fields(lat, minute, label, inputs, iy, ix, size, answer=ANSWER):
    nm = int(minute.max()) + 1
    out = np.full((len(inputs), nm, size, size), np.nan)
    for a, e in enumerate(inputs):
        for m in range(nm):
            L = lat[(label == e) & (minute == m)]
            if len(L) == 0:
                continue
            share = np.isfinite(L).mean(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(L, 0)
            mean[share < answer] = np.nan
            out[a, m, iy, ix] = mean
    return out


def weight_field(s: dict, iy, ix, exy, pitch, size) -> np.ndarray:
    x0, y0 = exy[:, 0].min(), exy[:, 1].min()
    gx = np.round((s["pre_xy"][:, 0] - x0) / pitch).astype(int)
    gy = np.round((s["pre_xy"][:, 1] - y0) / pitch).astype(int)
    ok = (gx >= 0) & (gx < size) & (gy >= 0) & (gy < size)
    u = s["vec"][ok] / np.maximum(np.linalg.norm(s["vec"][ok], axis=1, keepdims=True), 1e-12)
    V = np.zeros((size, size, 2))
    for k in (0, 1):
        np.add.at(V[..., k], (gy[ok], gx[ok]), s["dw"][ok] * u[:, k])
    return V


def field_cosine(L: np.ndarray, V: np.ndarray, pitch: float):
    gy, gx = np.gradient(L, pitch)
    G = np.stack([gx, gy], -1)
    gn, vn = np.linalg.norm(G, axis=-1), np.linalg.norm(V, axis=-1)
    ok = np.isfinite(gn) & (gn > 0) & (vn > 0)
    if ok.sum() < 5:
        return None, int(ok.sum())
    cos = (G[ok] * V[ok]).sum(-1) / (gn[ok] * vn[ok])
    return round(float(cos.mean()), 3), int(ok.sum())


def local_area(lat, minute, label, inputs, iy, ix, size, win_ms=WIN_MS, acc_min=AREA_ACC) -> list:
    nm = int(minute.max()) + 1
    out = []
    for m in range(nm):
        sel = np.flatnonzero((minute == m) & np.isin(label, inputs))
        y = (label[sel] == inputs[0]).astype(int)
        if len(sel) < 8 or y.min() == y.max():
            out.append(None)
            continue
        G = np.full((len(sel), size + 2, size + 2), win_ms + 10.0)
        L = np.where(np.isfinite(lat[sel]), lat[sel], win_ms + 10.0)
        G[:, iy + 1, ix + 1] = L
        Fe = np.stack([G[:, 1 + dy:1 + dy + size, 1 + dx:1 + dx + size] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], -1)
        hits = np.zeros((size, size))
        for a, b in ((np.arange(0, len(sel), 2), np.arange(1, len(sel), 2)),
                     (np.arange(1, len(sel), 2), np.arange(0, len(sel), 2))):
            if len(np.unique(y[a])) < 2:
                continue
            sd = Fe[a].std(0) + 1e-3
            mu = [Fe[a][y[a] == k].mean(0) for k in (0, 1)]
            dist = [(((Fe[b] - mu[k]) / sd) ** 2).sum(-1) for k in (0, 1)]
            pred = (dist[1] < dist[0]).astype(int)
            hits += (pred == y[b][:, None, None]).sum(0)
        acc = hits / len(sel)
        out.append(round(float((acc[iy, ix] >= acc_min).mean()), 3))
    return out


def analyse_arm(d: dict, winner: int, loser: int, minute_s: float) -> tuple[dict, dict]:
    exy = d["electrode_xy"].astype(np.float64)
    s = ee_synapses(d)
    cw, cl = exy[winner], exy[loser]
    res = {"winner": direction_score(s, cw), "loser": direction_score(s, cl)}
    nl = null_scores(s, exy, np.stack([cw, cl]))
    res["null"] = {"n": int(len(nl)), "p95": float(np.percentile(nl, 95)), "median": float(np.median(nl)),
                   "winner_rank": float((nl < res["winner"]["score"]).mean())}
    res["hubs"] = hubs(s, d["excitatory"].astype(bool), d["pos"][:, :2].astype(np.float64), cw)
    res["dw_ee_mean"] = float(s["dw"].mean())
    pt, pc, F, lat = pulse_responses(d)
    minute = np.floor((pt - float(d["t_start_ms"])) / (minute_s * 1000.0)).astype(int)
    keep = minute < int(round((float(d["t_end_ms"]) - float(d["t_start_ms"])) / (minute_s * 1000.0)))
    pt, pc, F, lat, minute = pt[keep], pc[keep], F[keep], lat[keep], minute[keep]
    y = np.where(pc == winner, 1.0, -1.0)
    far = (np.linalg.norm(exy[:, None] - np.stack([cw, cl])[None], axis=2) > FAR_UM).all(1)
    res["pulses_isolated"] = {"winner": int((pc == winner).sum()), "loser": int((pc == loser).sum())}
    res["decode_all"] = decode_across_time(minute, y, F, np.arange(len(exy)))
    res["decode_far"] = decode_across_time(minute, y, F, np.flatnonzero(far))
    iy, ix, pitch = grid_index(exy)
    size = int(max(iy.max(), ix.max())) + 1
    Lf = latency_fields(lat, minute, pc, (winner, loser), iy, ix, size)
    V = weight_field(s, iy, ix, exy, pitch, size)
    res["latency_vs_weights"] = {f"{who}_minute{m + 1}": field_cosine(Lf[a, m], V, pitch)
                                 for a, who in enumerate(("winner", "loser")) for m in (0, Lf.shape[1] - 1)}
    res["latency_vs_weights_per_minute"] = {who: [field_cosine(Lf[a, m], V, pitch)[0] for m in range(Lf.shape[1])]
                                            for a, who in enumerate(("winner", "loser"))}
    res["local_area"] = local_area(lat, minute, pc, (winner, loser), iy, ix, size)
    res["after_run"] = score_maps(s, exy, winner, loser)
    return res, {"latency": Lf, "weights": V}


def reading(per: dict) -> dict:
    out, ok_all, lit_all = {}, True, True
    for seed in TAKEOVER:
        t, k = per.get(f"takeover_{seed}"), per.get(f"control_{seed}")
        if t is None or k is None:
            out[str(seed)] = None
            ok_all = lit_all = False
            continue
        tw, kw = t["winner"], k["winner"]
        r1 = tw["z"] > 2
        r2 = tw["score"] - kw["score"] > 2 * np.hypot(tw["se"], kw["se"])
        r3 = tw["score"] > t["loser"]["score"] and tw["score"] > t["null"]["p95"]
        literal = bool(tw["z"] > 2 and not kw["z"] > 2)
        out[str(seed)] = {"score": tw["score"], "z": round(tw["z"], 2), "control_score": kw["score"],
                          "control_z": round(kw["z"], 2), "loser_score": t["loser"]["score"],
                          "null_p95": t["null"]["p95"], "rule1_z_over_2": bool(r1),
                          "rule2_beats_control": bool(r2), "rule3_beats_loser_and_null": bool(r3),
                          "confirmed": bool(r1 and r2 and r3), "plan_literal": literal}
        ok_all &= bool(r1 and r2 and r3)
        lit_all &= literal
    return {"per_culture": out, "confirmed": ok_all, "plan_literal_confirmed": lit_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/takeover_rerun")
    ap.add_argument("--analyse-only", action="store_true", help="only the analysis, on the dumps already made")
    ap.add_argument("--smoke", action="store_true",
                    help="culture 0's two arms for 4 s into <out>/smoke (no stored-row check), analysis with 1 s "
                         "'minutes'")
    a = ap.parse_args()
    out = Path(a.out)
    seconds, rest_s, minute_s = SECONDS, 20.0, 60.0
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
        seconds, rest_s, minute_s, a.workers = 4.0, 2.0, 1.0, 2
    out.mkdir(parents=True, exist_ok=True)
    specs = arm_specs(seconds, rest_s)
    if a.smoke:
        specs = [sp for sp in specs if sp[2].get("seed", 0) == 0]
    print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": [n for n, _, _ in specs],
                      "seconds": seconds, "workers": a.workers, "smoke": a.smoke}), flush=True)
    jobs = []
    for name, key, sp in specs:
        ref = None if a.smoke else stored_row(name, TAKEOVER[sp.get("seed", 0)][0] if sp["scale"] else "additive")
        jobs.append((name, key, sp, seconds, out, {"rest_s": rest_s, "dump": True, "ref": ref}))
    todo = [j for j in jobs if not (out / "jobs" / f"{j[1]}.json").exists()]
    if todo and not a.analyse_only:
        CP.pool_map(CP._drift_safe, todo, min(a.workers, len(todo)))
    per, maps = {}, {}
    for name, key, sp, *_ in jobs:
        p = out / "jobs" / f"{key}.json"
        if not p.exists():
            print(json.dumps({"missing": name}), flush=True)
            continue
        row = json.loads(p.read_text())
        seed = sp.get("seed", 0)
        _, win, lose, _ = TAKEOVER[seed]
        tag = f"{'takeover' if sp['scale'] else 'control'}_{seed}"
        t0 = time.time()
        res, mp_ = analyse_arm(load_dump(Path(row["dump"])), win, lose, minute_s)
        res |= {"arm": name, "equals_stored_row": row.get("equals_stored_row", False), "rate_ratio": row["rate_ratio"],
                "gain_end": row["gain_60s"][-1], "w_ee_mean_vs_start": row["w_ee_mean_vs_start"]}
        per[tag] = res
        maps |= {f"{tag}_{k}": v for k, v in mp_.items()}
        print(json.dumps({"analysed": tag, "s": round(time.time() - t0, 1), "winner": res["winner"],
                          "control_or_loser": res["loser"]["score"], "null_p95": res["null"]["p95"],
                          "hubs": res["hubs"], "decode_all": res["decode_all"], "decode_far": res["decode_far"],
                          "local_area": res["local_area"], "latency_vs_weights": res["latency_vs_weights"]}),
              flush=True)
    rd = reading(per)
    if not a.smoke:
        for tag, r in per.items():
            if not r["equals_stored_row"]:
                print(f"GUARD FAIL [takeover]: {tag} did not repeat its stored row", flush=True)
    res = {"seconds": seconds, "minute_s": minute_s, "reading": rd, "arms": per,
           "settings": {"cos": COS, "null_min_um": NULL_MIN_UM, "far_um": FAR_UM, "win_ms": WIN_MS,
                        "answer": ANSWER, "area_acc": AREA_ACC, "eta": ETA, "takeover": TAKEOVER}}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez(out / "maps.npz", **maps)
    print(json.dumps({"reading": rd}), flush=True)


if __name__ == "__main__":
    main()
