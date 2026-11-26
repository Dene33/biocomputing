from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("public_data_targets", Path(__file__).resolve().parent /
                                               "public_data_targets.py")
PDT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PDT)

WAGENAAR_MAP = [[None, 23, 25, 28, 31, 34, 36, None],
                [20, 21, 24, 29, 30, 35, 38, 39],
                [18, 19, 22, 27, 32, 37, 40, 41],
                [15, 16, 17, 26, 33, 42, 43, 44],
                [None, 13, 12, 3, 56, 47, 46, 45],
                [11, 10, 7, 2, 57, 52, 49, 48],
                [9, 8, 5, 0, 59, 54, 51, 50],
                [None, 6, 4, 1, 58, 55, 53, None]]
PITCH_UM = 200.0
MIN_EL = 8
SIGMAS_MS = (0, 5, 10, 20, 50)
N_NULL = 200
N_ANGLES = 36
SQUARE_UM = 100.0
GATE = {"site_um": 200.0, "class_agree": 0.70, "speed_ratio": (0.67, 1.5)}


def wagenaar_xy() -> np.ndarray:
    xy = np.full((60, 2), np.nan)
    for r, row in enumerate(WAGENAAR_MAP):
        for c, ch in enumerate(row):
            if ch is not None:
                xy[ch] = (c * PITCH_UM, r * PITCH_UM)
    return xy


def onsets(t, ch, burst, burstlets, sigma_ms: float) -> dict:
    s, e, chans = burst
    out = {}
    if sigma_ms == 0:
        for bs, be, c in burstlets:
            if c in chans and bs <= e and be >= s and (c not in out or bs < out[c]):
                out[c] = bs
        return {c: (v - s) * 1000.0 for c, v in out.items()}
    lo, hi = s - 0.1, e
    grid = np.arange(lo, hi + 1e-9, 0.001)
    sig = sigma_ms / 1000.0
    for c in chans:
        ts = t[(ch == c) & (t >= lo - 4 * sig) & (t <= hi)]
        if len(ts) == 0:
            continue
        r = np.exp(-0.5 * ((grid[:, None] - ts[None]) / sig) ** 2).sum(1)
        k = int(np.argmax(r >= 0.5 * r.max()))
        out[c] = (grid[k] - s) * 1000.0
    return out


def _r2_slope(x, y):
    xm, ym = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    sxx, syy, sxy = (xm * xm).sum(-1), (ym * ym).sum(-1), (xm * ym).sum(-1)
    with np.errstate(all="ignore"):
        return np.where((sxx > 0) & (syy > 0), sxy**2 / (sxx * syy), 0.0), np.where(sxx > 0, sxy / sxx, 0.0)


def fit_patterns(xy: np.ndarray, on: np.ndarray, centre: np.ndarray) -> dict:
    order = np.argsort(on, kind="stable")[:3]
    site = xy[..., order, :].mean(-2)
    d = np.linalg.norm(xy - site[..., None, :], axis=-1)
    on_b = np.broadcast_to(on, d.shape)
    r2_src, slope = _r2_slope(d, on_b)
    A = np.concatenate([xy - xy.mean(-2, keepdims=True), np.ones(xy.shape[:-1] + (1,))], -1)
    y = on_b - on_b.mean(-1, keepdims=True)
    AtA = np.einsum("...ni,...nj->...ij", A, A)
    Aty = np.einsum("...ni,...n->...i", A, y)
    coef = np.linalg.solve(AtA + 1e-9 * np.eye(3), Aty[..., None])[..., 0]
    res = y - np.einsum("...ni,...i->...n", A, coef)
    r2_plane = 1 - (res**2).sum(-1) / np.maximum((y**2).sum(-1), 1e-12)
    grad = coef[..., :2]
    th = np.arctan2(xy[..., 1] - centre[1], xy[..., 0] - centre[0])
    t0 = np.linspace(-np.pi, np.pi, N_ANGLES, endpoint=False)
    xa = np.mod(th[..., None, :] - t0[:, None], 2 * np.pi)
    r2_rot = _r2_slope(xa, np.broadcast_to(on, xa.shape))[0].max(-1)
    with np.errstate(divide="ignore"):
        speed_src = np.where(slope > 0, 1.0 / np.where(slope > 0, slope, 1), np.nan)
        gn = np.linalg.norm(grad, axis=-1)
        speed_plane = np.where(gn > 0, 1.0 / np.where(gn > 0, gn, 1), np.nan)
    return {"source": r2_src, "plane": r2_plane, "rotation": r2_rot, "speed_source": speed_src,
            "speed_plane": speed_plane, "site": site}


def classify(xy: np.ndarray, on: np.ndarray, centre, rng) -> dict:
    obs = fit_patterns(xy, on, centre)
    perm = np.stack([xy[rng.permutation(len(xy))] for _ in range(N_NULL)])
    nul = fit_patterns(perm, on, centre)
    excess = {k: float(obs[k] - np.percentile(nul[k], 95)) for k in ("source", "plane", "rotation")}
    best = max(excess, key=excess.get)
    return {"class": best if excess[best] > 0 else "complex", "r2": {k: round(float(obs[k]), 3) for k in excess},
            "excess": {k: round(v, 3) for k, v in excess.items()},
            "speed_source_mm_s": None if not np.isfinite(obs["speed_source"]) else round(float(obs["speed_source"]), 1),
            "speed_plane_mm_s": None if not np.isfinite(obs["speed_plane"]) else round(float(obs["speed_plane"]), 1),
            "site": [round(float(v), 1) for v in obs["site"]]}


def analyse_recording(t, ch, duration_s, exy, sigmas=SIGMAS_MS, seed=0, max_bursts=None) -> dict:
    n_ch = len(exy)
    ok = np.isfinite(exy).all(1)
    keep = ok[np.clip(ch, 0, n_ch - 1)] & (ch >= 0) & (ch < n_ch)
    t, ch = np.asarray(t, float)[keep], np.asarray(ch, int)[keep]
    o = np.argsort(t, kind="stable")
    t, ch = t[o], ch[o]
    _, _, bursts, bl = PDT.network_bursts(t, ch, duration_s, n_ch)
    centre = exy[ok].mean(0)
    rng = np.random.default_rng(seed)
    out = []
    for bi, b in enumerate(bursts):
        if len(b[2]) < MIN_EL:
            continue
        if max_bursts is not None and len(out) >= max_bursts:
            break
        row = {"i": bi, "start_s": round(float(b[0]), 4), "end_s": round(float(b[1]), 4), "n_el": len(b[2])}
        for sg in sigmas:
            on = onsets(t, ch, b, bl, sg)
            cs = sorted(on)
            if len(cs) < MIN_EL:
                continue
            v = np.array([on[c] for c in cs])
            r = classify(exy[cs], v, centre, rng)
            r["start_electrode"] = int(cs[int(np.argmin(v))])
            row[f"sigma{sg}"] = r
        out.append(row)
    return {"n_bursts": len(bursts), "n_analysed": len(out), "bursts": out}


def summary(bursts: list, sigma: int) -> dict:
    rs = [b[f"sigma{sigma}"] for b in bursts if f"sigma{sigma}" in b]
    if not rs:
        return {"n": 0}
    cls = [r["class"] for r in rs]
    starts = np.array([r["start_electrode"] for r in rs])
    _, cnt = np.unique(starts, return_counts=True)
    cnt = np.sort(cnt)[::-1]
    sp = [r["speed_source_mm_s"] for r in rs if r["class"] != "complex" and r["speed_source_mm_s"] is not None]
    return {"n": len(rs), "class_share": {k: round(cls.count(k) / len(rs), 3)
                                          for k in ("source", "plane", "rotation", "complex")},
            "speed_source_mm_s_median": None if not sp else round(float(np.median(sp)), 1),
            "top_start_share": round(float(cnt[0] / len(rs)), 3),
            "starts_for_80pct": int(np.searchsorted(np.cumsum(cnt) / len(rs), 0.8) + 1)}


def truth(z: dict, bursts: list, exy: np.ndarray, seed: int = 0) -> list:
    dt = float(z["dt_ms"]) / 1000.0
    nt = z["n_step"].astype(np.int64)
    tn = nt * dt
    pos = z["pos"][:, :2].astype(np.float64)
    lo_xy, hi_xy = exy.min(0), exy.max(0)
    sq = np.floor((pos - pos.min(0)) / SQUARE_UM).astype(int)
    ns = sq.max(0) + 1
    sq_id = sq[:, 0] * ns[1] + sq[:, 1]
    centres = (np.stack(np.divmod(np.arange(ns[0] * ns[1]), ns[1]), 1) + 0.5) * SQUARE_UM + pos.min(0)
    inside_sq = ((centres >= lo_xy) & (centres <= hi_xy)).all(1)
    centre = exy.mean(0)
    rng = np.random.default_rng(seed)
    out = []
    for b in bursts:
        s, e = b["start_s"], b["end_s"]
        a, z_ = np.searchsorted(tn, s - 0.2), np.searchsorted(tn, e, side="right")
        ids, ts = sq_id[z["n_id"][a:z_]], tn[a:z_]
        tot = np.bincount(ids, minlength=len(centres))
        need = np.maximum(3, np.ceil(0.1 * tot)).astype(int)
        o = np.argsort(ids, kind="stable")  # ts is time-ordered, so within a square the order stays by time
        ids_o, ts_o = ids[o], ts[o]
        first = np.searchsorted(ids_o, np.arange(len(centres)))
        sel = np.flatnonzero(tot >= 10)
        on = (ts_o[first[sel] + need[sel] - 1] - s) * 1000.0
        if len(sel) < 3:
            out.append(None)
            continue
        site = centres[sel[np.argsort(on, kind="stable")[:3]]].mean(0)
        row = {"site": [round(float(v), 1) for v in site],
               "inside": bool(((site >= lo_xy) & (site <= hi_xy)).all())}
        m = inside_sq[sel]
        if m.sum() >= MIN_EL:
            r = classify(centres[sel][m], on[m], centre, rng)
            row |= {"class": r["class"], "speed_source_mm_s": r["speed_source_mm_s"], "r2": r["r2"]}
        out.append(row)
    return out


TRUTH_V2_SIGMA_MS = 10.0
TRUTH_V2_EARLIEST = 0.05


def truth_v2(z: dict, bursts: list, exy: np.ndarray, seed: int = 0) -> list:
    dt = float(z["dt_ms"]) / 1000.0
    tn = z["n_step"].astype(np.int64) * dt
    pos = z["pos"][:, :2].astype(np.float64)
    lo_xy, hi_xy = exy.min(0), exy.max(0)
    sq = np.floor((pos - pos.min(0)) / SQUARE_UM).astype(int)
    ns = sq.max(0) + 1
    sq_id = sq[:, 0] * ns[1] + sq[:, 1]
    n_sq = ns[0] * ns[1]
    centres = (np.stack(np.divmod(np.arange(n_sq), ns[1]), 1) + 0.5) * SQUARE_UM + pos.min(0)
    inside_sq = ((centres >= lo_xy) & (centres <= hi_xy)).all(1)
    centre = exy.mean(0)
    rng = np.random.default_rng(seed)
    out = []
    for b in bursts:
        s, e = b["start_s"], b["end_s"]
        lo = s - 0.1
        nb = int(np.floor((e - lo) / 0.001)) + 1
        a, z_ = np.searchsorted(tn, lo), np.searchsorted(tn, e, side="right")
        ids = sq_id[z["n_id"][a:z_]]
        tb = np.minimum(((tn[a:z_] - lo) / 0.001).astype(np.int64), nb - 1)
        H = np.bincount(ids * nb + tb, minlength=n_sq * nb).reshape(n_sq, nb).astype(np.float64)
        sel = np.flatnonzero(H.sum(1) >= 10)
        if len(sel) < 3:
            out.append(None)
            continue
        k = np.arange(nb)
        K = np.exp(-0.5 * ((k[:, None] - k[None]) / TRUTH_V2_SIGMA_MS) ** 2)
        Sm = H[sel] @ K
        first = np.argmax(Sm >= 0.5 * Sm.max(1, keepdims=True), axis=1)
        on = (lo + first * 0.001 - s) * 1000.0
        n_early = max(3, int(np.ceil(TRUTH_V2_EARLIEST * len(sel))))
        site = centres[sel[np.argsort(on, kind="stable")[:n_early]]].mean(0)
        row = {"site": [round(float(v), 1) for v in site],
               "inside": bool(((site >= lo_xy) & (site <= hi_xy)).all())}
        m = inside_sq[sel]
        if m.sum() >= MIN_EL:
            r = classify(centres[sel][m], on[m], centre, rng)
            row |= {"class": r["class"], "speed_source_mm_s": r["speed_source_mm_s"], "r2": r["r2"]}
        out.append(row)
    return out


def recovery(model: dict) -> dict:
    res = {}
    for sg in SIGMAS_MS:
        err, agree, ratio = [], [], []
        for rec in model.values():
            for b, tr in zip(rec["bursts"], rec["truth"]):
                e = b.get(f"sigma{sg}")
                if e is None or tr is None or not tr["inside"]:
                    continue
                err.append(float(np.hypot(*(np.array(e["site"]) - np.array(tr["site"])))))
                if "class" in tr:
                    agree.append(e["class"] == tr["class"])
                    if e["speed_source_mm_s"] and tr["speed_source_mm_s"]:
                        ratio.append(e["speed_source_mm_s"] / tr["speed_source_mm_s"])
        res[sg] = {"n": len(err), "site_err_um_median": None if not err else round(float(np.median(err)), 1),
                   "class_agree": None if not agree else round(float(np.mean(agree)), 3),
                   "speed_ratio_median": None if not ratio else round(float(np.median(ratio)), 3)}
    ok = {k: v for k, v in res.items() if v["site_err_um_median"] is not None}
    best = min(ok, key=lambda k: (ok[k]["site_err_um_median"], k)) if ok else None
    b = res.get(best, {})
    lo, hi = GATE["speed_ratio"]
    passed = bool(best is not None and b["site_err_um_median"] <= GATE["site_um"]
                  and (b["class_agree"] or 0) >= GATE["class_agree"]
                  and b["speed_ratio_median"] is not None and lo <= b["speed_ratio_median"] <= hi)
    return {"per_sigma": res, "sigma": best, "gate": GATE, "pass": passed}


def run_model(rec_dir: Path, out: Path, max_bursts=None, truth_fn=None):
    truth_fn = truth_fn or truth
    model = {}
    for p in sorted(rec_dir.glob("live_*.npz")):
        with np.load(p) as zz:
            z = {k: zz[k] for k in zz.files}
        exy = z["electrode_xy"].astype(np.float64)
        r = analyse_recording(z["e_t_s"], z["e_ch"].astype(int), float(z["record_s"]), exy, max_bursts=max_bursts)
        r["truth"] = truth_fn(z, r["bursts"], exy)
        inside = [tr["inside"] for tr in r["truth"] if tr is not None]
        r["true_start_inside_share"] = round(float(np.mean(inside)), 3) if inside else None
        model[p.stem] = r
        print(json.dumps({"rec": p.stem, "n_bursts": r["n_bursts"], "n_analysed": r["n_analysed"],
                          "true_inside": r["true_start_inside_share"]}), flush=True)
    rc = recovery(model)
    res = {"recovery": rc, "per_recording": {k: {"n_bursts": v["n_bursts"], "n_analysed": v["n_analysed"],
                                                 "true_start_inside_share": v["true_start_inside_share"],
                                                 "summary": {sg: summary(v["bursts"], sg) for sg in SIGMAS_MS}}
                                             for k, v in model.items()}}
    out.write_text(json.dumps(res, indent=1))
    (out.with_suffix(".bursts.json")).write_text(json.dumps(model))
    print(json.dumps({"recovery": rc}), flush=True)
    return res


def per_culture(per: dict) -> dict:
    cult = {}
    for v in per.values():
        if v["summary"]["n"]:
            cult.setdefault(v["culture"], []).append(v["summary"])
    keys = ("source", "plane", "rotation", "complex")
    out = {}
    for c, ss in cult.items():
        sp = [s["speed_source_mm_s_median"] for s in ss if s["speed_source_mm_s_median"] is not None]
        out[c] = {**{k: float(np.median([s["class_share"][k] for s in ss])) for k in keys},
                  "speed": float(np.median(sp)) if sp else float("nan"),
                  "top_start_share": float(np.median([s["top_start_share"] for s in ss]))}
    return out


def run_wagenaar(raw: Path, manifest: dict, sigma: int, out: Path, validated: bool, limit=None):
    exy = wagenaar_xy()
    per = {}
    for name in sorted(manifest["wagenaar"])[:limit]:
        t, ch = PDT.load_spk(raw / "wagenaar_dense" / name)
        dur = float(np.ceil(t.max() / 60.0) * 60.0) if len(t) else 0.0
        r = analyse_recording(t, ch, dur, exy, sigmas=(sigma,))
        info = manifest["wagenaar"][name]
        per[name] = {"culture": f"{info['batch']}-{info['culture']}", "div": info["div"],
                     "n_bursts": r["n_bursts"], "summary": summary(r["bursts"], sigma)}
        print(json.dumps({"file": name, **per[name]["summary"]}), flush=True)
    keys = ("source", "plane", "rotation", "complex")
    per_c = per_culture(per)
    across = {k: PDT.spread([v[k] for v in per_c.values()]) for k in (*keys, "speed", "top_start_share")}
    res = {"validated_on_model": validated, "sigma": sigma, "map_source": "dataset help page (see docstring)",
           "across_cultures": across, "per_culture": per_c, "per_recording": per}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"across_cultures": across}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "wagenaar"])
    ap.add_argument("--rec", default=str(ROOT / "outputs/realism_scorecard/rec"))
    ap.add_argument("--raw", default=str(ROOT / "outputs/public_data/raw"))
    ap.add_argument("--out", default=str(ROOT / "outputs/realism_scorecard"))
    ap.add_argument("--smoke", action="store_true", help="20 bursts per model recording, or 2 Wagenaar recordings")
    ap.add_argument("--truth", choices=["v1", "v2"], default="v1",
                    help="v2: the labelled rerun's truth (model), or the scale and validation it gives (wagenaar)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.truth == "v1" else "_v2"
    if a.what == "model":
        run_model(Path(a.rec), out / f"burst_spread_model{tag}{'_smoke' if a.smoke else ''}.json",
                  max_bursts=20 if a.smoke else None, truth_fn=truth if a.truth == "v1" else truth_v2)
        return
    mp = out / f"burst_spread_model{tag}.json"
    if not mp.exists():
        sys.exit("run the model first: the smoothing scale and the validation come from it")
    rc = json.loads(mp.read_text())["recovery"]
    manifest = json.loads((Path(a.raw).parent / "manifest.json").read_text())
    run_wagenaar(Path(a.raw), manifest, int(rc["sigma"]), out / f"burst_spread_wagenaar{tag}"
                 f"{'_smoke' if a.smoke else ''}.json", rc["pass"], limit=2 if a.smoke else None)


if __name__ == "__main__":
    main()
