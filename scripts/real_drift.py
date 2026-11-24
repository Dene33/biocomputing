"""The drift test on real tissue: do real organoids hold rates and pathways under the cart-pole input pulses?

Data: the cart-pole organoid record (Robbins et al. 2026, Cell Reports 45:116984; Zenodo 10.5281/zenodo.17684862,
CC-BY 4.0), already fetched by scripts/public_data_targets.py into outputs/public_data/raw/robbins_zenodo/ and loaded
with its restricted unpickler. 38 experiments on 16 mouse cortical organoids (MaxOne HD-MEA), 426 game runs, most
15 min: 130 with no training ("none": the input pulses go on, no training stimulation), 67 random, 229 adaptive.
Per game step (every ~0.2 s): spike counts on the two output (motor) electrodes; per pulse: time and electrode(s).
A "none" run is our drift test (scripts/cartpole_replication.py --drift) done on real tissue: two encoders pulsed at
the encoding rates, two decoders read, no training signal.

Measures, the same as our drift test's, computed per run (runs with >= 8 full minutes):
- Gains: per minute, each decoder's count per step regressed on both encoders' sensory pulse counts in the step's
  read window (the 0.2 s before the step): a 2x2 matrix [decoder][encoder], spikes per pulse.
- Rate ratio: the decoders' mean rate over the last 3 full minutes against the first 30 s (the model runs away within
  ~40 s, so the first 30 s is the nearest a run gets to its start), and against the first 3 minutes.
- Pathways moved: each of the 4 gains, last 3 minutes against the first 3, beyond 2 SE, SE from the run's per-minute
  spread over minutes 2 to the end (sqrt(2 sd^2 / 3)); the 2 SE is also reported as the smallest change the run can
  see, in spikes per pulse and against the run's own gains.
- Takeover: each decoder's preferred encoder differs in the first 3 minutes and is the same one in the last 3.
Encoders per run are the run's two most-pulsed sensory electrodes in its stim log (the metadata is stale in 7
experiments, found by the guard on the first launch); runs with > 5% of sensory pulses elsewhere are left out, and
the across-hours and across-rest comparisons pair only runs of one experiment with the same two encoders (a changed
decoder pair can't be seen in the logs, so those two stay a caveat).
The same measures are computed from our model's drift rows (outputs/drift_scaling: gain_60s, dec_hz_10s; 10 minutes,
first 3 vs last 3 minutes), so the comparison is like for like. Reported, not part of the reading: the same per
condition for random and adaptive runs; across hours, each null run's whole-run gains and rate against the session's
first; across rests of >= 30 min between runs, the change from the last 3 minutes of one run to the first 3 of the
next.

Timing control, added after the first run: the same whole-run gains with the pulses of the 0.2 s *after* each read,
which can't cause its spikes. If they come out as large as the real ones, the gains measure slow covariation with
the game (pulse rates and decoder activity both follow the pole), not evoked spikes.

Reading, fixed before the first run on the data (24 Nov 2026), on the "none" runs (one run with training pulses
among them is excluded):
- Real tissue holds steady (our gate is realistic, the model's takeover is not) if the share of pathways moved is
  <= 10%, the median rate ratio (last 3 min / first 30 s) is within 2/3-1.5, and the test has the power to see a
  change: the median 2 SE is no larger than the median preferred-encoder gain.
- Real tissue drifts too (our gate is too strict; the model should match the real spread instead) if the share moved
  is >= 25% or the median rate ratio is above 1.5.
- Otherwise: in between, reported as is. Without the power, inconclusive.

Result (24 Nov 2026): the reading is "inconclusive (no power)", and the timing control shows the pathway measure
doesn't work on these logs at all. 129 null runs (1 excluded), 98 with >= 8 full minutes, from 12 organoids.
- Pathways: preferred-encoder gains median 0.08 spikes per pulse against a smallest visible change (2 SE) of 0.43;
  14.5% of pathways "moved". The pulses of the 0.2 s after each read give the same gains (median largest |gain|
  0.34 against 0.34), so 200 ms counts don't isolate evoked spikes, and the logs keep no spike times. The pathway
  half of the drift test can't be answered from this record.
- Rates (no regression involved): over a 15-min run of input pulses alone, the decoders' rate in the last 3 minutes
  is 0.77x the first 30 s (IQR 0.61-0.96, p95 1.18) and 0.94x the first 3 minutes (0.87-1.04): real tissue settles
  slightly and never runs away. Our model on the same measure: no plasticity 0.98, additive STDP 1.45-3.1,
  weight-dependent STDP with scaling at 1e-4 1.02. Over hours (15 sessions of 4-21 h, each null run against the
  session's first): median 1.2x (IQR 0.93-1.48, p5-p95 0.52-1.8); 7 of 15 sessions stay within 1.5x in every run,
  one reaches 6.3x. Across rests of >= 30 min: 1.07x (0.91-1.24).
- So the rate half of our drift gate (<= 1.5x over 10 minutes) is realistic, and over hours real rates wander by up
  to ~1.5-2x.

    uv run scripts/launch.py --unit bio-realdrift --log outputs/real_drift/run.log -- python -I scripts/real_drift.py
Output: outputs/real_drift/real_drift.json (summary, reading, model comparison, per-run rows).
"""

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
    """JSON-safe: NaN/inf -> None, numpy -> python, rounded."""
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


def fit_gains(P, Y):
    """[decoder, encoder] spikes per pulse, or NaNs if an encoder has too few pulses or no variance."""
    if (P.sum(0) < MIN_PULSES).any() or (P.std(0) == 0).any():
        return np.full((2, 2), np.nan)
    A = np.column_stack([np.ones(len(P)), P])
    return np.linalg.lstsq(A, Y, rcond=None)[0][1:].T


def window_stats(g_min, rate_first30, rate_min):
    """The within-run measures from per-minute gains [m, dec, enc] and rates [m]; same for model and real runs."""
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
            "detectable_vs_pref": float(np.median(2 * se) / max(np.mean(pref_gain), 1e-9)),
            "specific_start": bool(pref_s[0] != pref_s[1]),
            "takeover": bool(pref_s[0] != pref_s[1] and pref_e[0] == pref_e[1]),
            "rate_ratio_first30": r_last3 / rate_first30 if rate_first30 > 0 else float("nan"),
            "rate_ratio_first3": r_last3 / float(np.mean(rate_min[:3])) if np.mean(rate_min[:3]) > 0 else float("nan")}


def analyze_run(run, md, guard):
    g, s = run["game_log"], run["stim_log"]
    t = np.asarray(g["time"], float)
    Y = np.column_stack([np.asarray(g["spike_count_l"], float), np.asarray(g["spike_count_r"], float)])
    dt = float(np.median(np.diff(t)))
    guard.check(0.1 < dt < 0.5, f"step interval {dt:.3f} s")
    tags = list(s.get("tag", []))
    # Electrode lists are strings like "[18422, 20846]"; a few logs repeat their header line as a row ("tag",
    # "stim_electrodes"), which gives no electrodes and isn't a sensory pulse.
    elecs = [[int(x) for x in re.findall(r"\d+", e)] if isinstance(e, str) else [int(x) for x in np.atleast_1d(e)]
             for e in s.get("stim_electrodes", [])]
    guard.check(len(elecs) == len(tags), "stim log columns differ in length")
    st = np.array([float(x) if re.fullmatch(r"[-+0-9.eE]+", str(x)) else np.nan for x in s.get("time", [])])
    sens = np.array([tg == "sensory" for tg in tags], bool) & np.isfinite(st)
    # The encoders are the run's own two most-pulsed sensory electrodes: the metadata is stale where an experiment
    # was reconfigured (7 of 38 experiments, two with no electrodes listed).
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
    """Our model's drift arms, through the same within-run measures."""
    out = []
    for p in MODEL_ROWS:
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            dec = np.asarray(r["dec_hz_10s"], float).mean(1)  # [10 s window], mean of both decoders
            rate_min = dec[: len(dec) // 6 * 6].reshape(-1, 6).mean(1)
            w = window_stats(np.asarray(r["gain_60s"], float), float(dec[:3].mean()), rate_min)
            if r["scale"] == 0:
                kind = f"scaling alone {r['homeo']:g}" if r.get("homeo") else "no plasticity"
            else:
                kind = f"{r['rule']} STDP + " + (f"scaling {r['homeo']:g}" if r.get("homeo") else "old 5 s step")
            out.append({"arm": r["arm"], "seed": r.get("seed", 0), "kind": kind, "within": w})
    uniq = {(o["kind"], o["seed"]): o for o in out}  # the reused no-plasticity arms appear in both files
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
    """Across hours (null runs) and across rests (any consecutive runs >= 30 min apart)."""
    hours, rests = [], []
    by_cfg = defaultdict(list)  # one configuration = one experiment with one encoder pair
    for exp, rows in rows_by_exp.items():
        for r in rows:
            if not r["mixed_encoders"]:
                by_cfg[(exp, tuple(r["encoders"]))].append(r)
    for (exp, enc), rows in by_cfg.items():
        rows = sorted(rows, key=lambda r: r["start_time"])
        t0 = datetime.fromisoformat(rows[0]["start_time"])
        nulls = [r for r in rows
                 if r["exp_type"] == "none" and not r["train_pulses"] and np.isfinite(r["gain_run"]).all()]
        if len(nulls) >= 3:
            h = np.array([(datetime.fromisoformat(r["start_time"]) - t0).total_seconds() / 3600 for r in nulls])
            if h.max() - h.min() >= 1.0:
                G = np.array([r["gain_run"] for r in nulls])
                rate = np.array([r["rate_hz"] for r in nulls])
                slope = np.array([[np.polyfit(h, G[:, i, j], 1)[0] for j in (0, 1)] for i in (0, 1)])
                pref = np.abs(G.mean(0)).max()
                hours.append({"exp": exp, "encoders": enc, "null_runs": len(nulls), "span_h": float(h.max() - h.min()),
                              "gain_slope_per_h": slope, "gain_slope_per_h_vs_pref": np.abs(slope).max() / pref,
                              "gain_range": G.max(0) - G.min(0), "rate_vs_first": rate / rate[0],
                              "pref_flips": int(len({tuple(g.argmax(1)) for g in G}) > 1)})
        for a, b in zip(rows, rows[1:]):
            end_a = datetime.fromisoformat(a["start_time"]).timestamp() + a["end_offset_s"]
            gap = (datetime.fromisoformat(b["start_time"]).timestamp() - end_a) / 60
            wa, wb = a.get("within"), b.get("within")
            if gap >= 30 and wa and wb:
                across = wb["gain_first3"] - wa["gain_last3"]
                rests.append({"exp": exp, "gap_min": gap, "types": [a["exp_type"], b["exp_type"]],
                              "change_across_rest": across, "change_within_before": wa["diff"],
                              "rate_ratio_across": float(np.mean(b["rate_hz_per_minute"][:3])
                                                         / np.mean(a["rate_hz_per_minute"][-3:]))})
    return hours, rests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/real_drift")
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
    hours, rests = sessions(rows_by_exp)
    absmean = lambda key: spread([float(np.mean(np.abs(r[key]))) for r in rests])  # noqa: E731
    res = {"reading": read, "summary": summ, "model": model, "hours": hours, "rests": rests,
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
