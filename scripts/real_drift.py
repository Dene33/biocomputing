import argparse
import importlib.util
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from biocompute.runguard import Guard

ROOT = Path(__file__).resolve().parent.parent
PKL = ROOT / "outputs/public_data/raw/robbins_zenodo/aggregated_experiment_data.pkl"
MODEL_ROWS = [ROOT / "outputs/drift_scaling/drift_scaling.json",
              ROOT / "outputs/drift_scaling/drift_multiplicative_scaling.json"]
MIN_MINUTES, MIN_STEPS_PER_MIN, MIN_PULSES = 8, 150, 10
STABLE_MOVED, DRIFT_MOVED, BAND = 0.10, 0.25, (2 / 3, 1.5)


def load_pickle(path):
    spec = importlib.util.spec_from_file_location("pdt", Path(__file__).resolve().parent / "public_data_targets.py")
    pdt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdt)
    with open(path, "rb") as f:
        return pdt.SafeUnpickler(f).load()


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def ratio(a, b):
    return float(a) / float(b) if b > 0 else float("nan")


def hours_apart(a, b):
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 3600


def fit_gains(P, Y):
    if (P.sum(0) < MIN_PULSES).any() or (P.std(0) == 0).any():
        return np.full((2, 2), np.nan)
    A = np.column_stack([np.ones(len(P)), P])
    return np.linalg.lstsq(A, Y, rcond=None)[0][1:].T


def window_stats(g_min, rate_first30, rate_min):
    first, last = g_min[:3], g_min[-3:]
    if np.isnan(first).any() or np.isnan(last).any() or len(g_min) < MIN_MINUTES:
        return None
    sd = np.nanstd(g_min[1:], axis=0, ddof=1)
    se = np.sqrt(2 * sd ** 2 / 3)
    f, l = first.mean(0), last.mean(0)
    diff = l - f
    moved = np.abs(diff) > 2 * se
    pref_s, pref_e = f.argmax(1), l.argmax(1)
    pref_gain = np.array([f[k, pref_s[k]] for k in (0, 1)])
    r_last3 = float(np.mean(rate_min[-3:]))
    return {"moved_own": int(moved[0, 0] + moved[1, 1]), "moved_cross": int(moved[0, 1] + moved[1, 0]),
            "moved": int(moved.sum()), "diff": diff, "se2": 2 * se, "gain_first3": f, "gain_last3": l,
            "pref_gain_first3": pref_gain,
            "detectable_vs_pref": ratio(np.median(2 * se), np.mean(pref_gain)),
            "specific_start": bool(pref_s[0] != pref_s[1]),
            "takeover": bool(pref_s[0] != pref_s[1] and pref_e[0] == pref_e[1]),
            "rate_ratio_first30": ratio(r_last3, rate_first30),
            "rate_ratio_first3": ratio(r_last3, np.mean(rate_min[:3]))}


def analyze_run(run, md, guard):
    g, s = run["game_log"], run["stim_log"]
    t = np.asarray(g["time"], float)
    Y = np.column_stack([np.asarray(g["spike_count_l"], float), np.asarray(g["spike_count_r"], float)])
    dt = float(np.median(np.diff(t)))
    guard.check(0.1 < dt < 0.5, f"step interval {dt:.3f} s")
    tags = list(s.get("tag", []))
    elecs = [[int(x) for x in re.findall(r"\d+", e)] if isinstance(e, str) else [int(x) for x in np.atleast_1d(e)]
             for e in s.get("stim_electrodes", [])]
    guard.check(len(elecs) == len(tags), "stim log columns differ in length")
    st = np.array([float(x) if re.fullmatch(r"[-+0-9.eE]+", str(x)) else np.nan for x in s.get("time", [])])
    sens = np.array([tg == "sensory" for tg in tags], bool) & np.isfinite(st)
    per_e = Counter(x for k in np.flatnonzero(sens) for x in elecs[k])
    enc = sorted(e for e, _ in per_e.most_common(2))
    other = sum(per_e.values()) - sum(per_e[e] for e in enc)
    mixed = len(enc) < 2 or other > 0.05 * max(sum(per_e.values()), 1)
    P, P_after = np.zeros((len(t), 2)), np.zeros((len(t), 2))
    for j, e in enumerate(enc):
        tj = np.sort(st[[k for k in np.flatnonzero(sens) if e in elecs[k]]])
        P[:, j] = np.searchsorted(tj, t, "right") - np.searchsorted(tj, t - dt, "right")
        P_after[:, j] = np.searchsorted(tj, t + dt, "right") - np.searchsorted(tj, t, "right")
    rel = t - t[0]
    minute = np.floor(rel / 60).astype(int)
    full = [m for m in range(minute.max() + 1) if (minute == m).sum() >= MIN_STEPS_PER_MIN]
    g_min = np.array([fit_gains(P[minute == m], Y[minute == m]) for m in full]) if full else np.zeros((0, 2, 2))
    rate_min = np.array([Y[minute == m].mean() / dt for m in full])
    first30 = rel < 30
    row = {"exp_type": run["exp_type"], "basename": run["basename"], "start_time": run["start_time"],
           "minutes": len(full), "steps": len(t), "train_pulses": int(sum(tg == "train" for tg in tags)),
           "sensory_pulses": int(sens.sum()), "encoders": enc, "mixed_encoders": bool(mixed),
           "metadata_mismatch": bool(set(enc) != set(md["sensory_electrodes"])), "rate_hz": float(Y.mean() / dt),
           "rate_hz_per_minute": rate_min, "gain_60s": g_min, "gain_run": fit_gains(P, Y),
           "gain_run_pulses_after": fit_gains(P_after, Y),
           "end_offset_s": float(t[-1])}
    if len(full) >= MIN_MINUTES and first30.sum() >= 50 and not mixed:
        row["within"] = window_stats(g_min, float(Y[first30].mean() / dt), rate_min)
    return row


def model_rows():
    out = []
    for p in MODEL_ROWS:
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            dec = np.asarray(r["dec_hz_10s"], float).mean(1)
            rate_min = dec[: len(dec) // 6 * 6].reshape(-1, 6).mean(1)
            w = window_stats(np.asarray(r["gain_60s"], float), float(dec[:3].mean()), rate_min)
            if r["scale"] == 0:
                kind = f"scaling alone {r['homeo']:g}" if r.get("homeo") else "no plasticity"
            else:
                kind = f"{r['rule']} STDP + " + (f"scaling {r['homeo']:g}" if r.get("homeo") else "old 5 s step")
            out.append({"arm": r["arm"], "seed": r.get("seed", 0), "kind": kind, "within": w})
    uniq = {(o["kind"], o["seed"]): o for o in out}
    return list(uniq.values())


def spread(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {"n": len(v), "median": float(np.median(v)), "iqr": np.percentile(v, [25, 75]),
            "p5_p95": np.percentile(v, [5, 95])}


def summarize(rows):
    w = [r["within"] for r in rows if r.get("within")]
    if not w:
        return {"runs": len(rows), "runs_measured": 0}
    n_path = 4 * len(w)
    return {"runs": len(rows), "runs_measured": len(w),
            "moved_share": sum(x["moved"] for x in w) / n_path,
            "moved_own_share": sum(x["moved_own"] for x in w) / (2 * len(w)),
            "moved_cross_share": sum(x["moved_cross"] for x in w) / (2 * len(w)),
            "takeover_share": float(np.mean([x["takeover"] for x in w])),
            "specific_start_share": float(np.mean([x["specific_start"] for x in w])),
            "rate_ratio_first30": spread([x["rate_ratio_first30"] for x in w]),
            "rate_ratio_first3": spread([x["rate_ratio_first3"] for x in w]),
            "detectable_2se": spread([float(np.median(x["se2"])) for x in w]),
            "pref_gain_first3": spread([float(np.mean(x["pref_gain_first3"])) for x in w]),
            "detectable_vs_pref": spread([x["detectable_vs_pref"] for x in w]),
            "abs_change": spread([float(np.mean(np.abs(x["diff"]))) for x in w])}


def reading(s):
    if not s.get("runs_measured"):
        return "no data"
    if s["rate_ratio_first30"] is None or s["detectable_2se"] is None:
        return "no data"
    ratio = s["rate_ratio_first30"]["median"]
    power = s["detectable_2se"]["median"] <= s["pref_gain_first3"]["median"]
    if s["moved_share"] >= DRIFT_MOVED or ratio > BAND[1]:
        return "drifts"
    if not power:
        return "inconclusive (no power)"
    if s["moved_share"] <= STABLE_MOVED and BAND[0] <= ratio <= BAND[1]:
        return "holds steady"
    return "in between"


def sessions(rows_by_exp):
    hours, rests, silent = [], [], []
    by_cfg = defaultdict(list)
    for exp, rows in rows_by_exp.items():
        for r in rows:
            if not r["mixed_encoders"]:
                by_cfg[(exp, tuple(r["encoders"]))].append(r)
    for (exp, enc), rows in by_cfg.items():
        rows = sorted(rows, key=lambda r: r["start_time"])
        nulls = [r for r in rows
                 if r["exp_type"] == "none" and not r["train_pulses"] and np.isfinite(r["gain_run"]).all()]
        h = np.array([hours_apart(rows[0]["start_time"], r["start_time"]) for r in nulls])
        if len(nulls) >= 3 and h.max() - h.min() >= 1.0 and nulls[0]["rate_hz"] <= 0:
            silent.append(exp)
        elif len(nulls) >= 3 and h.max() - h.min() >= 1.0:
            G = np.array([r["gain_run"] for r in nulls])
            rate = np.array([r["rate_hz"] for r in nulls])
            slope = np.array([[np.polyfit(h, G[:, i, j], 1)[0] for j in (0, 1)] for i in (0, 1)])
            hours.append({"exp": exp, "encoders": enc, "null_runs": len(nulls), "span_h": float(h.max() - h.min()),
                          "gain_slope_per_h": slope,
                          "gain_slope_per_h_vs_pref": ratio(np.abs(slope).max(), np.abs(G.mean(0)).max()),
                          "gain_range": G.max(0) - G.min(0), "rate_vs_first": rate / rate[0],
                          "pref_flips": int(len({tuple(g.argmax(1)) for g in G}) > 1)})
        for a, b in zip(rows, rows[1:]):
            gap = hours_apart(a["start_time"], b["start_time"]) * 60 - a["end_offset_s"] / 60
            wa, wb = a.get("within"), b.get("within")
            if gap >= 30 and wa and wb:
                across = wb["gain_first3"] - wa["gain_last3"]
                rests.append({"exp": exp, "gap_min": gap, "types": [a["exp_type"], b["exp_type"]],
                              "change_across_rest": across, "change_within_before": wa["diff"],
                              "rate_ratio_across": ratio(np.mean(b["rate_hz_per_minute"][:3]),
                                                         np.mean(a["rate_hz_per_minute"][-3:]))})
    return hours, rests, silent


def hours_summary(hours, silent):
    later = [x for h in hours for x in h["rate_vs_first"][1:]]
    span = [h["span_h"] for h in hours]
    return {"sessions": len(hours), "left_out_silent_first_run": silent,
            "span_h": [min(span, default=None), max(span, default=None)],
            "rate_vs_first": spread(later), "rate_vs_first_max": max(later, default=None),
            "sessions_in_band_every_run": sum(all(BAND[0] <= x <= BAND[1] for x in h["rate_vs_first"]) for h in hours)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/real_drift"))
    ap.add_argument("--smoke", action="store_true", help="first 3 experiments into <out>/smoke")
    a = ap.parse_args()
    out = Path(a.out)
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    guard = Guard("real_drift", out, every_s=10.0)
    data = load_pickle(PKL)
    exps = list(data)[:3] if a.smoke else list(data)
    rows_by_exp, allrows = defaultdict(list), []
    for i, exp in enumerate(exps):
        e = data[exp]
        for run in e["cartpole"]["runs"]:
            if not run["logs_available"] or not run["game_log"] or len(run["game_log"]["time"]) < 50:
                continue
            row = analyze_run(run, e["metadata"], guard) | {"exp": exp, "organoid": e["metadata"]["organoid"]}
            rows_by_exp[exp].append(row)
            allrows.append(row)
        guard.beat(i + 1, len(exps), runs=len(allrows))
    null = [r for r in allrows if r["exp_type"] == "none" and not r["train_pulses"]]
    summ = {"none": summarize(null), "random": summarize([r for r in allrows if r["exp_type"] == "random"]),
            "adaptive": summarize([r for r in allrows if r["exp_type"] == "rl"]),
            "excluded_null_with_train_pulses": sum(1 for r in allrows if r["exp_type"] == "none" and r["train_pulses"])}
    summ["organoids_in_none"] = len({r["organoid"] for r in null if r.get("within")})
    mx = lambda key: spread([float(np.abs(r[key]).max()) for r in null if np.isfinite(r[key]).all()])  # noqa: E731
    summ["timing_control_none"] = {"max_abs_gain_pulses_before_read": mx("gain_run"),
                                   "max_abs_gain_pulses_after_read": mx("gain_run_pulses_after")}
    read = reading(summ["none"])
    mrows = model_rows()
    by_kind = defaultdict(list)
    for m in mrows:
        by_kind[m["kind"]].append(m)
    model = {k: summarize(v) for k, v in sorted(by_kind.items())}
    hours, rests, silent = sessions(rows_by_exp)
    absmean = lambda key: spread([float(np.mean(np.abs(r[key]))) for r in rests])  # noqa: E731
    res = {"reading": read, "summary": summ, "model": model, "hours": hours,
           "hours_summary": hours_summary(hours, silent), "rests": rests,
           "rests_summary": {"n": len(rests), "abs_change_across_rest": absmean("change_across_rest"),
                             "abs_change_within_before": absmean("change_within_before"),
                             "rate_ratio_across": spread([r["rate_ratio_across"] for r in rests])},
           "runs": allrows}
    res = clean(res)
    for r in res["runs"]:
        if r.get("within"):
            guard.check(r["within"]["rate_ratio_first3"] is None or r["within"]["rate_ratio_first3"] > 0, "rate ratio")
    (out / "real_drift.json").write_text(json.dumps(res, indent=1))
    guard.beat(len(exps), len(exps), force=True, runs=len(allrows))
    print(json.dumps({"reading": read, "none": res["summary"]["none"], "random": res["summary"]["random"],
                      "adaptive": res["summary"]["adaptive"]}), flush=True)


if __name__ == "__main__":
    main()
