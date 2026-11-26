"""Step 3, second half (docs/route-to-computation.md, "Order" item 4.3; docs/literature-notes.md, proposal 4): how
network bursts spread across the array (start sites, speed, pattern), from electrode spike times only. Model first,
where the truth is known; Wagenaar's recordings only after the method recovers the model's spread.

Pre-registered 25 Nov 2026, before the model recordings exist.

    uv run python -I scripts/burst_spread.py model      # outputs/realism_scorecard/rec/*.npz -> burst_spread_model.json
    uv run python -I scripts/burst_spread.py wagenaar   # outputs/public_data/raw/wagenaar_dense -> ..._wagenaar.json

Standalone numpy (``python -I``: downloaded files are untrusted data). The burst detector is B0's
(``public_data_targets.network_bursts``, the one behind ``spike_stats``), loaded by path.

Method, per network burst on >= 8 electrodes:

- **Onset** per electrode, from spike times. ``sigma 0``: the start of its first burstlet in the burst (>= 4 spikes
  with short intervals, B0's rule). ``sigma > 0`` (5, 10, 20, 50 ms): its spikes from 100 ms before the burst to
  its end, smoothed by a Gaussian of that width; onset where the smoothed rate first reaches half its peak. The scale
  is chosen on the model (below) and then fixed.
- **Start site**: the centroid of the three earliest electrodes.
- **Fits** of onset against position: *source* (onset against the distance from the start site; speed = 1 / slope),
  *plane* (onset against x and y; speed = 1 / |gradient|), *rotation* (onset against the angle around the array's
  centre, the best of 36 starting angles).
- **Null that keeps each electrode's spikes**: electrode positions permuted among the burst's electrodes (200
  permutations), each fitted the same way, the start site included. A pattern counts when its R^2 exceeds the null's
  95th percentile. The class is the counting pattern with the largest excess (R^2 - null 95th percentile); none:
  *complex*.
- **Start-site concentration**: share of bursts starting at the most common start electrode (the earliest one), and
  how many start electrodes cover 80% of bursts.

Model truth (from every neuron's spikes): neurons binned in 100 um squares. A square's onset is when its count since
200 ms before the burst first reaches max(3, 10% of its count up to the burst's end); squares with >= 10 spikes.
True start site: the centroid of the three earliest squares over the whole culture. True pattern and speed: the same
fits and null on the squares inside the array's area.

Recovery gate, fixed now: with the chosen scale, over the five cultures' bursts whose true start lies inside the
array's area (its bounding box), (1) the median start-site error is <= 200 um (one pitch), (2) the electrode pattern
class matches the true class in >= 70% of bursts, and (3) the median ratio of electrode to true speed (the source
fit) lies within 0.67-1.5. The scale chosen is the one with the smallest median site error. If the gate fails, the
Wagenaar run is reported as unvalidated, with the failure.

Wagenaar's electrode map (``WAGENAAR_MAP``) comes from the dataset's own documentation: the help page of the
dataset site (https://potterlab.bme.gatech.edu/development-data/html/help.html, Internet Archive capture
20241102155336, sha256 1dca0abdf449ea30074776f86c76aaa5baa2caf90caf59b60e9e17c12583dc39), section "Channel
numbers": channel numbers 0-59 by position in the 8 x 8 MEA, rows top to bottom. Its table leaves the four corners
and the first column of row 5 blank, and channel 14 doesn't appear in it (the reference electrode, as the 59
electrodes in the article). The page gives the map for the "full info" files; the ".spk.txt" files we use carry the
same channel numbers (0..59). The pitch is 200 um (MCS 8 x 8 arrays; the article). Spikes on channel 14 are dropped.

Labelled rerun, truth v2 (25 Nov 2026, after the gate above failed; fixed before running it). The run with the truth
above failed the recovery gate: median start-site error 578-639 um, class agreement 0.44-0.51. A look at 60 bursts
of culture 0 showed the truth, not the electrodes, to be ill-defined. Its three earliest squares (of ~1,000) lie a
median 1.16 mm apart and fire 85-190 ms before the burst: background spikes reaching the 3-spike threshold. Onset
hardly grows with distance from them (median r = 0.15), against r = 0.61 from the centroid of the earliest 5% of
squares, which lies a median 0.8 mm from the truth used. ``--truth v2`` changes only the truth; the electrode method,
the scales and the gate stay as they are:

- square onset: its spikes from 100 ms before the burst to its end, binned per 1 ms and smoothed by a Gaussian of
  10 ms; onset where the smoothed count first reaches half its peak (the electrodes' sigma > 0 rule; squares with
  >= 10 spikes in that window);
- true start site: the centroid of the earliest 5% of squares (at least 3);
- true pattern and speed: as before, on the squares inside the array's area.

Output burst_spread_model_v2.json. Whatever it gives, the pre-registered verdict above stands: unvalidated. If v2
passes, the Wagenaar run is also reported as validated against truth v2, labelled as such.

Results (25 Nov 2026; five model cultures, 2,197 bursts; electrode side identical in both runs).

| Truth | Bursts starting under the array | Best scale | Site error (gate 200 um) | Class agreement (0.70) | Speed ratio |
|-------|---------------------------------|------------|--------------------------|------------------------|-------------|
| v1    | 33% (734)                       | 0 ms       | 578 um                   | 0.51                   | 0.99        |
| v2    | 18% (392)                       | 20 ms      | 239 um                   | 0.52                   | 0.95        |

The gate fails with both truths: the method is **not validated**. Against v2, the electrodes get the speed right and
the start site to about 1.2 pitches, but they confuse a wave from a point with a front sweeping across (v2: 116 of
175 true sources called source, 67 of 147 planes called plane, 19 of 70 rotations called rotation; chance ~0.34).
Reasoning, not tested: over a 1.4 mm window a wave from a point near or beyond the edge looks like a front, and in
the model 70-90% of bursts start outside the array (the culture is 4 mm across). Two truths tried; no more variants.

Wagenaar, as pre-registered (scale 0 ms, unvalidated; ``burst_spread_wagenaar.json``), against the model measured the
same way. Medians over 29 real cultures [IQR], and the model's five cultures (range):

| Measure                                      | Real              | Model         |
|----------------------------------------------|-------------------|---------------|
| no simple pattern ("complex")                | 41% [21-63%]      | 2-15%         |
| front sweeping across ("plane")              | 23% [12-37%]      | 50-58%        |
| from a point ("source")                      | 10% [6-18%]       | 25-34%        |
| rotating                                     | 11% [7-14%]       | 6-9%          |
| speed, source fit (mm/s)                     | 30 [21-45]        | 31-41         |
| bursts starting at the commonest electrode   | 15% [13-20%]      | 4-9%          |
| start electrodes covering 80% of bursts      | 13 (per recording)| 31-36         |

Read with the method's limits: same yardstick on both sides, but the class and start site aren't validated. The
speed, which the method does recover on the model, matches. Real bursts are messier (41% fit no simple pattern
against 2-15%) and start at fewer favoured sites; model bursts are mostly orderly fronts from anywhere.
"""

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

# Rows top to bottom, columns left to right; None: no electrode (corners, reference).
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
    """(60, 2) electrode positions in um by channel number; NaN for channel 14 (the reference)."""
    xy = np.full((60, 2), np.nan)
    for r, row in enumerate(WAGENAAR_MAP):
        for c, ch in enumerate(row):
            if ch is not None:
                xy[ch] = (c * PITCH_UM, r * PITCH_UM)
    return xy


# --- Onsets -------------------------------------------------------------------------------------------------------------
def onsets(t, ch, burst, burstlets, sigma_ms: float) -> dict:
    """Per electrode of ``burst`` (start s, end s, channels): its onset in ms after the burst's start."""
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


# --- Fits (vectorised over permutations) --------------------------------------------------------------------------------
def _r2_slope(x, y):
    """Least-squares line of y on x along the last axis: R^2 and slope."""
    xm, ym = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    sxx, syy, sxy = (xm * xm).sum(-1), (ym * ym).sum(-1), (xm * ym).sum(-1)
    with np.errstate(all="ignore"):
        return np.where((sxx > 0) & (syy > 0), sxy**2 / (sxx * syy), 0.0), np.where(sxx > 0, sxy / sxx, 0.0)


def fit_patterns(xy: np.ndarray, on: np.ndarray, centre: np.ndarray) -> dict:
    """``xy`` (..., n, 2) positions, ``on`` (n,) onsets in ms: R^2 and speed of the source, plane and rotation fits,
    the start site, for every leading index of ``xy`` (the permutations)."""
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
        speed_src = np.where(slope > 0, 1.0 / np.where(slope > 0, slope, 1), np.nan)  # um/ms = mm/s
        gn = np.linalg.norm(grad, axis=-1)
        speed_plane = np.where(gn > 0, 1.0 / np.where(gn > 0, gn, 1), np.nan)
    return {"source": r2_src, "plane": r2_plane, "rotation": r2_rot, "speed_source": speed_src,
            "speed_plane": speed_plane, "site": site}


def classify(xy: np.ndarray, on: np.ndarray, centre, rng) -> dict:
    """The burst's fits, their nulls (positions permuted) and its pattern class."""
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


# --- Per recording ------------------------------------------------------------------------------------------------------
def analyse_recording(t, ch, duration_s, exy, sigmas=SIGMAS_MS, seed=0, max_bursts=None) -> dict:
    """Every burst on >= MIN_EL electrodes, per smoothing scale: onsets, class, speed, start site and electrode."""
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


# --- Model truth --------------------------------------------------------------------------------------------------------
def truth(z: dict, bursts: list, exy: np.ndarray, seed: int = 0) -> list:
    """Per analysed burst: the true start site over the whole culture, whether it lies in the array's area, and the
    true pattern and speed from the 100 um squares inside that area."""
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
    """Truth v2 (labelled rerun, see the docstring): smoothed half-peak onsets per 100 um square, start site = the
    centroid of the earliest 5% of squares."""
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
        K = np.exp(-0.5 * ((k[:, None] - k[None]) / TRUTH_V2_SIGMA_MS) ** 2)  # 1 ms bins: sigma in bins
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
    """The gate: per scale, start-site error, class agreement and speed ratio over bursts that truly start inside."""
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
    """Per culture: the median over its recordings of each class share, speed and top start share. Recordings with
    no analysed burst are left out, and cultures with none at all."""
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
