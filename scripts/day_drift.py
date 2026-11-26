"""Track A, step A2d, part 1 (pre-registered 26 Nov 2026, before any number below was computed): how much does a real
culture change from one day to the next, and does our aging model change as much?

Why. A2 and A2c age the culture with ``biocompute.culture.aging.age``: each neuron's excitability drifts on its own
(0.5 x sqrt(days) current units) and 2% of the neurons die per day. Over 7 days that moved A2's replay rate only from
66.0 to 58.5 Hz. A2c's decision ("a session needs two numbers per IMU channel") and A2's "aging is harmless" were shown
under that drift. FinalSpark's whole-life recordings show organoids changing more: before this was written, fs369's
6-hour table (one organoid group's mean rate per electrode) was seen moving 1.5-4x between days with no stimulation.
That table mixes in nothing the model saw, so here both sides are measured the same way.

Real data (FinalSpark whole-life exports, outputs/public_data/raw/finalspark/, gitignored; read with h5py only, no
pandas, no unpickling: the pandas attributes are pickles). fs369 (21.7 days, 4 organoid groups of 8 electrodes, mostly
spontaneous) is the main record; fs437 (6 days, dying, stimulated by many clients) is reported, not used to decide.
Electrode ids are taken mod 32 (fs369 sits on 64-95); a group is ``electrode // 8``.

- Spontaneous rate per electrode per day. Day d = [first event + 24 h d, + 24 h). Counted: events with |amplitude|
  <= 300 uV (spontaneous p99.99 is 282 uV; stimulation artifacts reach mV) in minutes that are recording (any event on
  any electrode) and not near stimulation: each stimulation session (stimulations less than 5 min apart) is excluded
  from 10 min before its first to 60 min after its last stimulation. A day counts if >= 12 h remain. Rate = count /
  usable seconds.
- Day-to-day change at lags L = 1, 3, 7 days: for every day d with d and d + L both counted, |log2(rate(d + L) /
  rate(d))|, the later rate floored at 0.01 Hz. Electrode level: electrodes >= 0.1 Hz on day d. Group level: the
  group's mean over its 8 electrodes, groups >= 0.1 Hz per electrode on day d. Main: every such pair; also reported
  without pairs whose later day falls in the record's last 5 days (decline before death).
- Evoked response across days (fs369 only). Whole-array sweeps: stimulation sessions where every stimulation drives
  all 32 electrodes at once, >= 5 s apart, >= 100 stimulations and >= 12 settings (in fs369: four sweeps of 32
  settings of both phases' amplitude and width and the polarity, 10-20 stimulations each; the fourth has 22 of
  them). A setting is every stimulation parameter of the log. A sweep counts only if the stimulation reached the
  array: the event rate over all electrodes in the 10 ms after a stimulation is >= 2x the rate in the 100 ms before
  (the hour-227 session in fs369 shows no artifact at all). Response of an electrode to a setting: the mean over its
  stimulations of the clean count in [14, 60] ms after the stimulation minus the count in [-46, 0] ms (artifacts end
  before 14 ms: silent groups read 0.0-0.1 Hz from 14 ms on). For a pair of sweeps, each electrode's response is the
  mean over the settings both share (>= 12 needed), and the change is |log2(later / earlier)|, the later floored at
  0.05 spikes per stimulation, over electrodes >= 0.5 spikes per stimulation in the earlier sweep (groups: the sum
  over the group's electrodes, >= 2). Lag classes: ~1 day (0.5-1.5 d) and ~7 days (5-9 d; pooled over the pairs in
  it). Noise floor: the same change between odd and even stimulations of each setting within one sweep.
  The medium change of fs369 (28 Feb, ~9.1 days in) falls between the second and third sweeps; it stays in (dishes
  get medium changes), and is reported.

Model. A2's own sessions on s32 (scripts/closed_loop_sessions.py, the same code): A1b's warm day-0 state, ``age(cu,
days, seed)`` for days 1, 3, 7 with aging seeds 11 (A2's) to 15, A2's noise seeding (by session name), 1 s rest, A2's
60 s replay (3000 steps of 20 ms) and its 50 standing steps. Then, new here, 5 s with no input and 120 s of
spontaneous activity (no stimulation). ``age`` draws the dead neurons from the culture's own generator, restored
with the warm state, so every aging seed kills the same neurons; only the excitability drift differs (aging.py as
shipped). Plus ``day0`` and ``day0_rep`` (day 0 again with other noise seeds: the measurement's noise floor).
- Spontaneous change per electrode (the 256 recording electrodes, >= 0.1 Hz at day 0) and for the whole culture
  (the counterpart of one organoid group): day L against day 0, as the real data's lag L.
- Evoked change: each electrode's rate during the replay (the decoder's input; spikes per s), day L against day 0;
  the model's day 1 is compared with the real ~1 day class, day 7 with the ~7 day class.
Positive controls (GUARD FAIL otherwise): with aging seed 11 and at day 0, the device state after the replay and
standing steps equals A2's stored session bit for bit, and the replay rate equals its stored rate; aging on its
nominal dose (alive share 0.98^days within 0.01; excitability spread 0.5 sqrt(days) within 10%); no weight moves;
rates plausible.

Reading (fixed now). For each measure, kind and lag: the model is too gentle at that lag if its median |log2 change|
(electrodes: pooled over channels and aging seeds; culture/group: over aging seeds) is below the real 25th
percentile, too strong if above the real 75th, else in range. Verdicts:
- spontaneous, electrode level and group level (lags 1, 3, 7): too gentle (or too strong) at >= 2 of 3 lags;
- evoked, electrode level (~1 day, ~7 days): too gentle (or too strong) at both;
- evoked, group level: reported, not used (fs369 has at most 3 live groups per sweep pair).
Decision: if any of the three verdicts is "too gentle" or "too strong", part 2 fits a realistic day-to-day change
(its form chosen from these results and pre-registered before it runs) and A2c is rerun under it as a labelled run.
If all three are "in range", aging.py's drift is realistic in size, A2c's verdict stands and no rerun is needed.
Reported, not decided on: how the real change grows with the lag (a random walk grows like sqrt(L); a bounded wander
doesn't), signed medians (direction), the fs437 record, the noise floors.

Result of part 1 (run 26 Nov 2026, 12:06-12:34; day_drift.json). Controls held: aging seed 11 and day 0 reproduce
A2's stored sessions bit for bit (replay 65.99 / 64.63 / 62.82 / 58.51 Hz at days 0 / 1 / 3 / 7), aging on its
nominal dose, no weight moved. Median |log2 change| (real 25-75% in brackets; model over 5 aging seeds):

| Measure                        | 1 day                     | 3 days                  | 7 days                  |
|--------------------------------|---------------------------|-------------------------|-------------------------|
| spontaneous, electrode         | 0.26 [0.18-0.96] in range | 0.36 [0.46-2.2] gentle  | 0.43 [0.73-4.8] gentle  |
| spontaneous, culture / group   | 0.27 [0.14-0.73] in range | 0.39 [0.45-1.6] gentle  | 0.40 [0.68-4.0] gentle  |
| evoked, electrode (~1 / ~7 d)  | 0.05 [0.07-0.31] gentle   |                         | 0.22 [0.34-2.0] gentle  |

Verdicts: spontaneous electrode, spontaneous group and evoked electrode all too gentle (evoked group too, not used).
Decision: part 2. The model's change stops growing after a day and goes the other way: its spontaneous rate rises
(+0.27 to +0.40 log2 per culture, every seed) while real groups mostly decline (signed group medians -0.16 / -0.72 /
-1.56; two of fs369's four groups die out). Its evoked rate falls a little (-0.03 / -0.07 / -0.17). Noise floors
(day 0 again, other noise seeds): electrode 0.05 spontaneous, 0.02 evoked. The aging seeds barely differ (culture
change 0.24-0.29 at day 1, 0.32-0.46 at day 7): with 20,000 neurons, independent per-neuron drift averages out.

Part 2 (pre-registered 26 Nov 2026, before any calibration session): a realistic day-to-day change for A2c.
- Form. The real change is mostly coherent (the group's signed medians are close to its absolute ones at 3 and 7
  days), and the evoked change per electrode is mixed in sign across groups (after the medium change two groups
  respond more, one collapses). So the knob is a shift x of every neuron's excitability (current units), added to
  aging.py's day-L state (seed 11, as A2). It sizes the coherent change the electrodes see; no mechanism (cell loss,
  medium) is claimed. aging.py's per-neuron drift and death stay as they are.
- Calibration: x in {-3, -2, -1, -0.5, +0.5, +1, +2} at days 1, 3, 7 (21 sessions, part 1's code; x = 0 is part 1's
  seed-11 row), measured as part 1 against day 0.
- Targets, on the decoder's input (evoked per electrode, median |log2 change|): day 1 = the real ~1-day median
  (0.174), day 7 = the real ~7-day median (0.659), day 3 = linear in the lag between them (0.336); a stress dose at
  day 7 = the real ~7-day 75th percentile (1.99), downward.
- Doses: per lag and direction (down: x < 0, up: x > 0), the smallest |x| at which the model reaches the target,
  linear between calibration points; if no point reaches it, the largest tried, flagged. Seven doses: days 1, 3, 7
  down and up, and the day-7 stress dose.
- Reported, not fitted: each dose's spontaneous change against the real group's signed medians.
- Then A2c is rerun on those seven sessions as a labelled run (A2d); its pre-registration and gate (planned:
  refit60 >= 34/40 full runs in every session; stale reported) go into scripts/closed_loop_sessions.py with the doses,
  before its first episode.

Result of part 2 (run 26 Nov 2026, 12:38-13:13; calibration.json). The evoked change grows with |x| in both
directions from x = 0's (day 1: 0.055, day 7: 0.215). Doses: days 1 / 3 / 7 down -0.597 / -1.003 / -1.583, up
+0.734 / +1.626 / +2.0, stress -3.0. Not reached: day 7 up (+2.0 gives 0.353 of 0.659; an upward shift first undoes
the 7-day state's own fall) and the stress dose (-3.0 gives 1.217 of 1.99; the culture is then almost silent on its
own, -3.6 log2 spontaneous, yet still answers the replay at x0.43). The down doses also move the spontaneous rate in
the real direction, somewhat too far (-0.85 / -1.08 / -2.09 against the real groups' -0.16 / -0.72 / -1.56).
Part 3 (A2d, scripts/closed_loop_sessions.py) passed: the 60 s edit refit keeps the relay at 35-40/40 in all seven
sessions; the stale relay drops only at day 7 up (19/40).

Run (guarded, resumable; the real part runs first, on the CPU, ~1 min, cached in <out>/real_<fs>_<key>.json):

    uv run scripts/launch.py --unit bio-a2d --log outputs/day_drift/run.log -- \\
        python -I scripts/day_drift.py --workers 3
    uv run scripts/watch.py --unit bio-a2d --log outputs/day_drift/run.log
    # part 2 (part 1's sessions are reused):
    uv run scripts/launch.py --unit bio-a2d-cal --log outputs/day_drift/run_calibrate.log -- \\
        python -I scripts/day_drift.py --calibrate --workers 3

Outputs (outputs/day_drift): calibration.json (part 2), real_<fs>_<key>.json, jobs/<name>_<key>.json (one per model session), rec/<name>_<key>.npz
(the spontaneous electrode spikes), day_drift.json (both sides, the comparison and the decision).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_sessions as CLS  # noqa: E402
import closed_loop_vestibular as clv  # noqa: E402
import finalspark_data as FD  # noqa: E402

from biocompute.runguard import job_key  # noqa: E402

ROOT = clv.ROOT
FS = FD.FS
MAIN_FS = "fs369"
S, MS = FD.S, FD.MS  # ns per s, per ms
REAL = dict(art_uv=300.0, live_hz=0.1, floor_hz=0.01, min_day_h=12.0, excl_before_s=600.0, excl_after_s=3600.0,
            session_gap_s=300.0, lags=(1, 3, 7), last_days=5, post_ms=(14.0, 60.0), pre_ms=(-46.0, 0.0),
            resp_el=0.5, resp_group=2.0, floor_resp=0.05, sweep_min_gap_s=5.0, sweep_min_events=100,
            sweep_min_settings=12, delivered_ratio=2.0, lag_classes={"1d": (0.5, 1.5), "7d": (5.0, 9.0)})
DAYS = (1, 3, 7)
AGING_SEEDS = (11, 12, 13, 14, 15)
SHIFTS = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0)  # part 2's calibration grid (excitability, current units)
CFG = "s32"
MODEL_KEYS = ("data", "a1", "a2", "recal_steps", "settle_s", "spont_s")


# --- Real data ------------------------------------------------------------------------------------------------------
def day_rates(t, e, amp, stim_t, p=REAL) -> dict:
    """Spontaneous rate per electrode per day (Hz) and the usable seconds per day (see the module docstring)."""
    t0 = int(t[0])
    mins = ((t - t0) // (60 * S)).astype(np.int64)
    nmin = int(mins[-1]) + 1
    on = np.zeros(nmin, bool)
    on[mins] = True
    excl = np.zeros(nmin, bool)
    ut = np.unique(stim_t)
    for a, b in FD.stim_sessions(ut, p["session_gap_s"]):
        lo = int((ut[a] - p["excl_before_s"] * S - t0) // (60 * S))
        hi = int((ut[b] + p["excl_after_s"] * S - t0) // (60 * S)) + 1
        excl[max(lo, 0):max(min(hi, nmin), 0)] = True
    use = on & ~excl
    day = np.arange(nmin) // 1440
    nd = int(day[-1]) + 1
    use_s = np.bincount(day[use], minlength=nd) * 60.0
    keep = (np.abs(amp) <= p["art_uv"]) & use[mins]
    counts = np.bincount((mins[keep] // 1440) * 32 + e[keep], minlength=nd * 32).reshape(nd, 32)
    rate = counts / np.maximum(use_s, 1.0)[:, None]
    return {"rate": rate, "use_s": use_s, "valid": use_s >= p["min_day_h"] * 3600, "n_days": nd}


def summary(v) -> dict:
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"n": 0}
    q = np.percentile(v, [25, 50, 75, 90])
    return {"n": int(len(v)), "p25": round(float(q[0]), 3), "median": round(float(q[1]), 3),
            "p75": round(float(q[2]), 3), "p90": round(float(q[3]), 3)}


def lag_changes(rate, valid, lags, p=REAL, last_days: int | None = None) -> dict:
    """|log2| day-to-day changes per lag, electrode and group level, plus signed medians (later over earlier)."""
    nd = len(rate)
    out = {}
    for L in lags:
        el, gr, el_s, gr_s = [], [], [], []
        for d in range(nd - L):
            if not (valid[d] and valid[d + L]) or (last_days is not None and d + L >= nd - last_days):
                continue
            r0, r1 = rate[d], rate[d + L]
            live = r0 >= p["live_hz"]
            x = np.log2(np.maximum(r1[live], p["floor_hz"]) / r0[live])
            el += np.abs(x).tolist()
            el_s += x.tolist()
            for g in range(4):
                g0, g1 = r0[8 * g:8 * g + 8].mean(), r1[8 * g:8 * g + 8].mean()
                if g0 >= p["live_hz"]:
                    y = float(np.log2(max(g1, p["floor_hz"]) / g0))
                    gr.append(abs(y))
                    gr_s.append(y)
        out[str(L)] = {"electrode": summary(el) | {"signed_median": _med(el_s)},
                       "group": summary(gr) | {"signed_median": _med(gr_s)}}
    return out


def _med(v):
    return None if len(v) == 0 else round(float(np.median(v)), 3)


def sweeps(rec: dict, p=REAL) -> list[dict]:
    """Whole-array sweeps with per-setting, per-electrode responses (see the module docstring)."""
    t, e, amp, st = rec["t"], rec["e"], rec["amp"], rec["st"]
    keys = sorted(k for k in st if k not in ("time_of_stim", "electrode"))
    P = np.round(np.stack([st[k] for k in keys], 1), 6)
    ut, start, nel = np.unique(st["time_of_stim"], return_index=True, return_counts=True)
    bounds = np.r_[start, len(st["time_of_stim"])]
    found = []
    for a, b in FD.stim_sessions(ut, p["session_gap_s"]):
        idx = np.arange(a, b + 1)
        if len(idx) < p["sweep_min_events"] or not (nel[idx] == 32).all():
            continue
        if np.median(np.diff(ut[idx])) < p["sweep_min_gap_s"] * S:
            continue
        uniform = all(len(np.unique(P[bounds[i]:bounds[i + 1]], axis=0)) == 1 for i in idx)
        setting = [tuple(P[start[i]].tolist()) for i in idx]
        if not uniform or len(set(setting)) < p["sweep_min_settings"]:
            continue
        times = ut[idx]
        # delivery: all events (any amplitude) in the 10 ms after vs the 100 ms before, all electrodes
        n_after = sum(np.searchsorted(t, u + 10 * MS) - np.searchsorted(t, u) for u in times) / 10.0
        n_before = sum(np.searchsorted(t, u) - np.searchsorted(t, u - 100 * MS) for u in times) / 100.0
        ratio = n_after / max(n_before, 1e-9)
        d = (FD.window_counts(t, e, amp, times, *p["post_ms"], p["art_uv"])
             - FD.window_counts(t, e, amp, times, *p["pre_ms"], p["art_uv"]))
        per = {}
        for s in sorted(set(setting)):
            k = np.array([x == s for x in setting])
            rows = d[k]
            per[s] = {"all": rows.mean(0), "odd": rows[0::2].mean(0), "even": rows[1::2].mean(0)}
        found.append({"t_first": int(times[0]), "t_last": int(times[-1]), "n": int(len(idx)), "settings": per,
                      "delivery_ratio": round(float(ratio), 2), "delivered": bool(ratio >= p["delivered_ratio"])})
    return found


def _resp(sw: dict, shared, part: str) -> np.ndarray:
    return np.mean([sw["settings"][s][part] for s in shared], 0)


def evoked_changes(sws: list[dict], t_first_event: int, p=REAL) -> dict:
    """Pairs of delivered sweeps sharing >= sweep_min_settings settings: |log2| changes per electrode and per group;
    lag classes pooled; noise floor (odd vs even stimulations within a sweep)."""
    sws = [s for s in sws if s["delivered"]]
    pairs, classes = [], {k: {"electrode": [], "group": []} for k in p["lag_classes"]}
    for i in range(len(sws)):
        for j in range(i + 1, len(sws)):
            shared = sorted(set(sws[i]["settings"]) & set(sws[j]["settings"]))
            if len(shared) < p["sweep_min_settings"]:
                continue
            r0, r1 = _resp(sws[i], shared, "all"), _resp(sws[j], shared, "all")
            lag = (sws[j]["t_first"] - sws[i]["t_first"]) / S / 86400
            live = r0 >= p["resp_el"]
            el = np.abs(np.log2(np.maximum(r1[live], p["floor_resp"]) / r0[live]))
            g0 = np.array([r0[8 * g:8 * g + 8].sum() for g in range(4)])
            g1 = np.array([r1[8 * g:8 * g + 8].sum() for g in range(4)])
            gl = g0 >= p["resp_group"]
            gr = np.abs(np.log2(np.maximum(g1[gl], p["floor_resp"]) / g0[gl]))
            pairs.append({"days": [round((sws[k]["t_first"] - t_first_event) / S / 86400, 2) for k in (i, j)],
                          "lag_days": round(lag, 2), "shared_settings": len(shared),
                          "electrode": summary(el), "group": summary(gr),
                          "group_change_signed": np.round(np.log2(np.maximum(g1[gl], p["floor_resp"]) / g0[gl]),
                                                          3).tolist()})
            for c, (lo, hi) in p["lag_classes"].items():
                if lo <= lag <= hi:
                    classes[c]["electrode"] += el.tolist()
                    classes[c]["group"] += gr.tolist()
    floor = []
    for s in sws:
        shared = sorted(s["settings"])
        full, odd, even = (_resp(s, shared, k) for k in ("all", "odd", "even"))
        live = full >= p["resp_el"]
        floor += np.abs(np.log2(np.maximum(odd[live], p["floor_resp"]) / np.maximum(even[live], p["floor_resp"]))
                        ).tolist()
    return {"pairs": pairs, "classes": {c: {k: summary(v) for k, v in d.items()} for c, d in classes.items()},
            "noise_floor_electrode": summary(floor)}


def real_part(fs: str, out: Path, smoke: bool) -> dict:
    stem = job_key(f"real_{fs}", {k: (list(v) if isinstance(v, tuple) else v) for k, v in REAL.items()})
    path = out / f"{stem}.json"
    if path.exists():
        return json.loads(path.read_text())
    t_start = time.time()
    rec = FD.load_record(FS[fs], fs)
    dr = day_rates(rec["t"], rec["e"], rec["amp"], rec["st"]["time_of_stim"])
    res = {"fs": fs, "file": str(FS[fs].relative_to(ROOT)), "n_events": int(len(rec["t"])),
           "n_stim_rows": int(len(rec["st"]["time_of_stim"])), "n_days": dr["n_days"],
           "usable_hours_per_day": np.round(dr["use_s"] / 3600, 1).tolist(), "valid_days": dr["valid"].tolist(),
           "group_rate_per_day_hz": np.round(np.stack([dr["rate"][:, 8 * g:8 * g + 8].mean(1) for g in range(4)], 1),
                                             3).tolist(),
           "spont": lag_changes(dr["rate"], dr["valid"], REAL["lags"]),
           "spont_without_last_days": lag_changes(dr["rate"], dr["valid"], REAL["lags"], last_days=REAL["last_days"]),
           "spont_lag_profile": {str(L): v for L, v in lag_changes(dr["rate"], dr["valid"], range(1, 11)).items()}}
    sws = sweeps(rec)
    res["sweeps"] = [{k: v for k, v in s.items() if k != "settings"}
                     | {"day": round((s["t_first"] - int(rec["t"][0])) / S / 86400, 2),
                        "n_settings": len(s["settings"])} for s in sws]
    res["evoked"] = evoked_changes(sws, int(rec["t"][0]))
    res["seconds"] = round(time.time() - t_start, 1)
    if not smoke:
        clv.write_json(path, res)
    return res


# --- Model ----------------------------------------------------------------------------------------------------------
def model_jobs(out: Path, sz: dict) -> list[tuple]:
    jobs = [("day0", 0.0, None, 0.0, out, sz), ("day0_rep", 0.0, None, 0.0, out, sz)]
    seeds = AGING_SEEDS[:1] if sz["smoke"] else AGING_SEEDS
    days = DAYS[:1] if sz["smoke"] else DAYS
    jobs += [(f"day{d}", float(d), a, 0.0, out, sz) for a in seeds for d in days]
    return jobs


def calibration_jobs(out: Path, sz: dict) -> list[tuple]:
    """Part 2: A2's aging (seed 11) at each lag, plus a shift of every neuron's excitability."""
    days = DAYS[:1] if sz["smoke"] else DAYS
    grid = SHIFTS[:2] if sz["smoke"] else SHIFTS
    return [(f"day{d}", float(d), CLS.AGING_SEED, x, out, sz) for d in days for x in grid]


def job_name(cond: str, aging_seed, sz: dict, shift: float = 0.0) -> str:
    name = f"{CFG}_{cond}" + ("" if aging_seed is None else f"_a{aging_seed}") + (f"_x{shift:+g}" if shift else "")
    return job_key(name, {k: sz[k] for k in MODEL_KEYS})


def model_job(job):
    import torch

    from biocompute.culture.aging import age
    from biocompute.runguard import Guard

    cond, days, aseed, shift, out, sz = job
    stem = job_name(cond, aseed, sz, shift)
    if clv.stop_requested():
        raise clv.Stopped(f"{stem} before start")
    g = Guard(stem.rsplit("_", 1)[0], out)
    O = clv.load_replay(sz["data"], 0)
    b0 = CLS.a1_blob(sz, CFG)
    row0 = b0["row"]
    ctrl = CLS.make_ctrl(CFG, "day0", O)
    dev, cu = ctrl.device, ctrl.device.culture
    g.check(cu.plasticity is None, "plasticity is on")
    dev.load_state(b0["warm"])
    alive0 = float(cu.alive.float().mean())
    if days:
        age(cu, days, seed=aseed)
    alive, exc_sd = float(cu.alive.float().mean()), float(cu.excitability.std())
    if days and not sz["smoke"]:
        g.check(abs(alive / alive0 - 0.98 ** days) < 0.01, f"alive share {alive / alive0:.4f} vs {0.98 ** days:.4f}")
        g.check(abs(exc_sd / (0.5 * days ** 0.5) - 1) < 0.1, f"excitability sd {exc_sd:.3f} vs {0.5 * days ** 0.5:.3f}")
    if shift:  # part 2: one shift for every neuron (the spread, checked above, is unchanged by it)
        cu.excitability += shift
    s = sum(map(ord, f"{CFG}_{cond}"))  # A2's noise seeding, by session name
    cu.torch_gen.manual_seed(1000 + s)
    dev.mea.gen.manual_seed(2000 + s)
    w_ref = cu.w.clone()
    dev.step(CLS.REST_MS)
    a, n = row0["split"], sz["recal_steps"]
    seg = O[a:a + n]
    F = np.zeros((n, ctrl.features.dim), np.float32)
    t0 = time.time()
    for i in range(n):
        if clv.stop_requested():
            raise clv.Stopped(f"{stem} at replay step {i}/{n} (redone on resume)")
        F[i] = ctrl.sense(seg[i])
        if g.due():
            g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
            g.culture(cu)
            g.beat(i + 1, n, phase="replay", ms_per_step=round((time.time() - t0) / (i + 1) * 1000, 1))
    C = F.shape[1] // 4
    rate = float((F[:, : 2 * C] ** 2).sum(1).mean() / C / 0.02)  # A2's replay rate (spikes per electrode per s)
    per_ch = ((F[:, :C] ** 2 + F[:, C:2 * C] ** 2).sum(0) / n / 0.02).astype(float)
    obs0 = clv.standing_obs()
    for _ in range(clv.WARM_STEPS):
        ctrl.sense(obs0)
    equals_a2 = None
    if (aseed == CLS.AGING_SEED or cond == "day0") and not shift and not sz["smoke"]:
        p = Path(sz["a2"]) / "sessions" / f"{CLS.session_stem(CFG, cond, sz)}.pkl"
        g.check(p.exists(), f"no stored A2 session {p}")
        old = pickle.loads(p.read_bytes())
        g.check(CLS.same(dev.state(), old["warm"]), f"{stem}: state after replay differs from A2's stored session")
        g.check(round(rate, 2) == old["row"]["rate_hz"], f"replay rate {rate:.2f} vs A2's {old['row']['rate_hz']}")
        equals_a2 = str(p)
    # New: spontaneous activity, no stimulation.
    n_settle, n_rec = int(round(sz["settle_s"] * 10)), int(round(sz["spont_s"] * 10))
    for i in range(n_settle):
        dev.step(100.0)
    counts = np.zeros(dev.n_channels)
    et, ec = [], []
    t_ms0 = float(cu.t_ms)
    for i in range(n_rec):
        if clv.stop_requested():
            raise clv.Stopped(f"{stem} at spontaneous read {i}/{n_rec} (redone on resume)")
        t_b = float(cu.t_ms)
        b = dev.step(100.0)
        counts += np.bincount(b.channels, minlength=dev.n_channels)
        et.append(((t_b - t_ms0) + b.times_ms) / 1000.0)
        ec.append(b.channels.astype(np.int16))
        if g.due():
            hz = counts.sum() / ((i + 1) * 0.1) / dev.n_channels
            g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
            g.culture(cu, rate_hz=max(hz, 1e-9))
            g.beat(i + 1, n_rec, phase="spontaneous", hz=round(hz, 2))
    g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
    alive_ch = np.asarray(dev.mea.alive_channels)
    spont = counts / sz["spont_s"]
    rec = out / "rec" / f"{stem}.npz"
    rec.parent.mkdir(parents=True, exist_ok=True)
    with open(rec.with_suffix(".tmp"), "wb") as f:
        np.savez(f, e_t_s=np.concatenate(et), e_ch=np.concatenate(ec), record_s=np.array(sz["spont_s"]),
                 alive_channels=alive_ch)
    rec.with_suffix(".tmp").replace(rec)
    row = {"name": stem, "cond": cond, "days": days, "aging_seed": aseed, "shift": shift,
           "alive_vs_day0": round(alive / alive0, 4),
           "excitability_sd": round(exc_sd, 3), "replay_rate_hz": round(rate, 3),
           "replay_channels": [int(x) for x in ctrl.features.channels], "replay_rate_per_channel": per_ch.tolist(),
           "spont_channels": alive_ch.tolist(), "spont_rate_per_channel": spont[alive_ch].tolist(),
           "spont_rate_hz": round(float(spont[alive_ch].mean()), 3), "equals_a2_session": equals_a2,
           "smoke": sz["smoke"]}
    clv.write_json(out / "jobs" / f"{stem}.json", row)
    g.beat(1, 1, force=True)
    print(json.dumps({k: row[k] for k in ("name", "alive_vs_day0", "excitability_sd", "replay_rate_hz",
                                          "spont_rate_hz", "equals_a2_session")}), flush=True)
    del ctrl, F
    torch.cuda.empty_cache()
    return row


def _model_safe(job):
    try:
        return model_job(job)
    except clv.Stopped as e:
        return {"stopped": str(e)}


def model_changes(rows: list[dict], p=REAL) -> dict:
    """Model day L against day 0: |log2| per electrode (pooled over aging seeds) and per culture (one per seed), for
    spontaneous and replay rates; the day0_rep noise floor."""
    by = {(r["cond"], r["aging_seed"]): r for r in rows}
    d0 = by.get(("day0", None))
    out = {}
    if d0 is None:
        return out

    def change(r, key):
        a, b = np.array(d0[key]), np.array(r[key])
        live = a >= p["live_hz"]
        el = np.log2(np.maximum(b[live], p["floor_hz"]) / a[live])
        cul = float(np.log2(max(b.mean(), p["floor_hz"]) / a.mean()))
        return el, cul

    for kind, key in (("spont", "spont_rate_per_channel"), ("evoked", "replay_rate_per_channel")):
        out[kind] = {}
        for d in DAYS:
            els, culs = [], []
            for a in AGING_SEEDS:
                r = by.get((f"day{d}", a))
                if r is None:
                    continue
                el, cul = change(r, key)
                els += el.tolist()
                culs.append(cul)
            out[kind][str(d)] = {"electrode": summary(np.abs(els)) | {"signed_median": _med(els)},
                                 "group": summary(np.abs(culs)) | {"signed_median": _med(culs), "per_seed": culs}}
        rep = by.get(("day0_rep", None))
        if rep is not None:
            el, cul = change(rep, key)
            out[kind]["noise_floor"] = {"electrode": summary(np.abs(el)), "group": round(abs(cul), 3)}
    return out


def compare(real: dict, model: dict) -> dict:
    """The pre-registered reading (module docstring)."""
    def at(real_s, model_s):
        if not real_s.get("n") or not model_s.get("n"):
            return None
        m = model_s["median"]
        return "too gentle" if m < real_s["p25"] else "too strong" if m > real_s["p75"] else "in range"

    def verdict(calls, need):
        calls = [c for c in calls if c is not None]
        for v in ("too gentle", "too strong"):
            if sum(c == v for c in calls) >= need:
                return v
        return "in range" if len(calls) >= need else "no data"

    rows, verdicts = [], {}
    for level in ("electrode", "group"):
        calls = []
        for L in map(str, REAL["lags"]):
            c = at(real["spont"][L][level], model["spont"][L][level])
            calls.append(c)
            rows.append({"kind": "spont", "level": level, "lag": L, "real": real["spont"][L][level],
                         "model": model["spont"][L][level], "call": c})
        verdicts[f"spont_{level}"] = verdict(calls, 2)
    for level in ("electrode", "group"):
        calls = []
        for cls, d in (("1d", "1"), ("7d", "7")):
            c = at(real["evoked"]["classes"][cls][level], model["evoked"][d][level])
            calls.append(c)
            rows.append({"kind": "evoked", "level": level, "lag": cls, "real": real["evoked"]["classes"][cls][level],
                         "model": model["evoked"][d][level], "call": c})
        verdicts[f"evoked_{level}"] = verdict(calls, 2) + ("" if level == "electrode" else " (reported, not used)")
    used = [verdicts[k] for k in ("spont_electrode", "spont_group", "evoked_electrode")]
    decision = ("part 2: fit a realistic day-to-day change, then rerun A2c under it"
                if any(v in ("too gentle", "too strong") for v in used) else
                "aging.py's drift is realistic in size: A2c stands, no rerun" if all(v == "in range" for v in used)
                else "undecided: missing data")
    return {"rows": rows, "verdicts": verdicts, "decision": decision}


def _evoked(r0: dict, r: dict, p=REAL) -> dict:
    """Change of ``r`` against day 0 ``r0``: evoked (replay) per electrode and culture, spontaneous per culture."""
    a, b = np.array(r0["replay_rate_per_channel"]), np.array(r["replay_rate_per_channel"])
    live = a >= p["live_hz"]
    el = np.log2(np.maximum(b[live], p["floor_hz"]) / a[live])
    sa, sb = np.array(r0["spont_rate_per_channel"]), np.array(r["spont_rate_per_channel"])
    return {"shift": r.get("shift", 0.0), "evoked_electrode_abs": round(float(np.median(np.abs(el))), 3),
            "evoked_electrode_signed": round(float(np.median(el)), 3),
            "evoked_culture": round(float(np.log2(b.mean() / a.mean())), 3),
            "spont_culture": round(float(np.log2(max(sb.mean(), p["floor_hz"]) / sa.mean())), 3)}


def dose_for(points: list[dict], sign: int, target: float) -> dict:
    """The smallest |shift| of ``sign`` (linear between calibration points, x = 0 included) at which the evoked
    electrode change reaches ``target``; the largest tried, flagged, if none does."""
    pts = sorted([q for q in points if q["shift"] == 0 or np.sign(q["shift"]) == sign], key=lambda q: abs(q["shift"]))
    if not pts:
        return {"shift": None, "reached": False}
    for q0, q1 in zip(pts, pts[1:]):
        y0, y1 = q0["evoked_electrode_abs"], q1["evoked_electrode_abs"]
        if y0 <= target <= y1 and y1 > y0:
            f = (target - y0) / (y1 - y0)
            lerp = {k: round(q0[k] + f * (q1[k] - q0[k]), 3) for k in q0 if k != "shift"}
            return {"shift": round(q0["shift"] + f * (q1["shift"] - q0["shift"]), 3), "reached": True, **lerp}
    if pts and pts[0]["evoked_electrode_abs"] >= target:
        return {**pts[0], "reached": True}
    return {**pts[-1], "reached": False}


def calibration(rows0: list[dict], cal: list[dict], real: dict) -> dict:
    """Part 2 (module docstring): the calibration curves, the targets and the doses."""
    d0 = next(r for r in rows0 if r["cond"] == "day0")
    ev = real["evoked"]["classes"]
    t1, t7 = ev["1d"]["electrode"]["median"], ev["7d"]["electrode"]["median"]
    targets = {1: t1, 3: round(t1 + (t7 - t1) * (3 - 1) / (7 - 1), 3), 7: t7}
    stress = ev["7d"]["electrode"]["p75"]
    curves, doses = {}, {}
    for L in DAYS:
        base = [r for r in rows0 if r["days"] == L and r["aging_seed"] == CLS.AGING_SEED and not r.get("shift")]
        pts = [_evoked(d0, r) for r in base + [c for c in cal if c["days"] == L]]
        if not pts:  # a lag without sessions (the smoke run)
            continue
        curves[str(L)] = sorted(pts, key=lambda q: q["shift"])
        for sign, label in ((-1, "down"), (1, "up")):
            doses[f"day{L}_{label}"] = {"days": L, "target": targets[L], **dose_for(pts, sign, targets[L])}
        if L == 7:
            doses["day7_stress"] = {"days": 7, "target": stress, **dose_for(pts, -1, stress)}
    real_spont = {L: real["spont"][str(L)]["group"]["signed_median"] for L in DAYS}
    return {"targets": {str(k): v for k, v in targets.items()}, "stress_target": stress, "curves": curves,
            "doses": doses, "real_spont_group_signed": {str(k): v for k, v in real_spont.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/day_drift")
    ap.add_argument("--a1", default="outputs/closed_loop_vestibular_affine")
    ap.add_argument("--a2", default="outputs/closed_loop_sessions")
    ap.add_argument("--data", default="outputs/probe/teacher_data_20k.pkl")
    ap.add_argument("--real-only", action="store_true", help="only the real data (CPU)")
    ap.add_argument("--calibrate", action="store_true", help="part 2: the excitability-shift calibration and doses")
    ap.add_argument("--smoke", action="store_true", help="whole pipeline at tiny scale into <out>/smoke: day0, "
                                                         "day0_rep, day1 seed 11; 100 replay steps, 2 s spontaneous")
    a = ap.parse_args()
    import faulthandler

    faulthandler.register(signal.SIGUSR1)
    out = Path(a.out)
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    sz = {"data": a.data, "a1": a.a1, "a2": a.a2, "recal_steps": 100 if a.smoke else CLS.RECAL_STEPS,
          "settle_s": 0.5 if a.smoke else 5.0, "spont_s": 2.0 if a.smoke else 120.0, "smoke": a.smoke}
    print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "workers": a.workers, **sz}), flush=True)
    real = {}
    for fs in FS:
        real[fs] = real_part(fs, out, a.smoke)
        r = real[fs]
        print(json.dumps({"real": fs, "days": r["n_days"], "valid_days": int(sum(r["valid_days"])),
                          "spont": {L: {k: v[k].get("median") for k in v} for L, v in r["spont"].items()},
                          "sweeps": r["sweeps"], "evoked_classes": r["evoked"]["classes"],
                          "seconds": r["seconds"]}), flush=True)
    if not a.smoke:
        sw = real[MAIN_FS]["sweeps"]
        ok = sum(s["delivered"] for s in sw)
        if ok < 3:
            print(f"GUARD FAIL [real]: {ok} delivered whole-array sweeps in {MAIN_FS}, expected 4", flush=True)
            raise SystemExit(3)
    if a.real_only:
        return
    jobs = model_jobs(out, sz)
    todo = [j for j in jobs if not (out / "jobs" / f"{job_name(j[0], j[2], sz, j[3])}.json").exists()]
    if todo:
        threads = max(1, min(4, (os.cpu_count() or 4) // max(a.workers, 1)))  # as closed_loop_sessions.py
        clv.pool_map(_model_safe, todo, min(a.workers, len(todo)), threads)
    rows = [json.loads((out / "jobs" / f"{job_name(j[0], j[2], sz, j[3])}.json").read_text()) for j in jobs]
    model = model_changes(rows)
    res = {"real": real, "model": model, "model_rows": [{k: v for k, v in r.items() if "per_channel" not in k
                                                          and k not in ("replay_channels", "spont_channels")}
                                                         for r in rows]}
    if not a.smoke:
        res["comparison"] = compare(real[MAIN_FS], model)
        print(json.dumps(res["comparison"]["verdicts"]), flush=True)
        print(json.dumps({"decision": res["comparison"]["decision"]}), flush=True)
    clv.write_json(out / "day_drift.json", res)
    if a.calibrate:
        cjobs = calibration_jobs(out, sz)
        todo = [j for j in cjobs if not (out / "jobs" / f"{job_name(j[0], j[2], sz, j[3])}.json").exists()]
        if todo:
            threads = max(1, min(4, (os.cpu_count() or 4) // max(a.workers, 1)))
            clv.pool_map(_model_safe, todo, min(a.workers, len(todo)), threads)
        cal = [json.loads((out / "jobs" / f"{job_name(j[0], j[2], sz, j[3])}.json").read_text()) for j in cjobs]
        cres = calibration(rows, cal, real[MAIN_FS])
        clv.write_json(out / "calibration.json", cres)
        print(json.dumps({"targets": cres["targets"], "stress": cres["stress_target"]}), flush=True)
        for k, v in cres["doses"].items():
            print(json.dumps({k: v}), flush=True)


if __name__ == "__main__":
    main()
