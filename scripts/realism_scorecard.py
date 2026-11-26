"""Step 3 (docs/route-to-computation.md, "Order" item 4.3): realism scorecard. Does the model culture's spontaneous
activity look like real cultures' on the same kind of array, and how often does a pulse set off a burst?

Pre-registered 25 Nov 2026, before the first recording.

Recording (GPU): the gpu_20k culture (the one in every drift and relay run), plasticity off, on the ``mcs60`` array
(8 x 8 minus corners = 60 electrodes at 200 um, like Wagenaar's 59-electrode MCS arrays), in five cultures (seeds
0-4): 10 s to settle, then 600 s of spontaneous activity. Kept: every electrode's spikes and every neuron's spikes
(for burst spread, where the model's truth is known). Then the replication's evoked-burst measure on the same culture
seed exactly as it ran there (cartpole_replication.build + characterise: grid256, the 32 most active electrodes, 25
pulses each at 4 Hz; a pulse sets off a network burst when the 0-200 ms count over the candidates exceeds median +
3 MAD). Negative control: culture 0 with its synapses removed (``dead``), recorded the same way.

Scorecard: ``spike_stats`` (scripts/public_data_targets.py, unchanged: B0's yardstick) on each culture's 600 s,
against outputs/public_data/targets.json (Wagenaar dense cultures, DIV 21-32, spread across 29 cultures). A measure
is in range when the median over the five cultures lies inside the real cultures' 5-95% range; how many single
cultures do is reported too. Measures, target median [5-95%]:

| Measure                                   | Real cultures      |
|-------------------------------------------|--------------------|
| electrode rate, median electrode (Hz)     | 2.8 [0.7-10.3]     |
| network bursts per minute                 | 4.2 [1.2-15.4]     |
| burst duration (ms)                       | 556 [79-1788]      |
| inter-burst interval CV                   | 0.87 [0.56-1.99]   |
| share of spikes inside bursts             | 0.63 [0.10-0.93]   |
| evoked: pulses that set off a burst       | 6.5% [0-16%] (Robbins organoids, median over stimulated units) |

Also reported: active electrodes, mean rate, inter-burst interval, recruitment. Nothing is decided on the scorecard
alone; step 4's gate (only if track B reopens) uses it.

Controls (GUARD FAIL otherwise): no weight moves; rates finite and plausible while recording; the evoked measure
on cultures 0-2 reproduces the replication's logged mean burst shares (0.20 / 0.21 / 0.21, run_task.log); the dead
culture shows fewer than 1.2 network bursts per minute (the real cultures' 5th percentile): the detector must not call
unconnected activity bursting.

Expected (reasoning, not measured): bursts too regular (the README's cpu_2k figure: every 2-3 s, CV ~0.3) and
evoked bursts too common (0.20-0.21 in the replication vs 0.065).

Run (guarded; a recording checkpoints every 5 min and on stop):

    uv run scripts/launch.py --unit bio-real --log outputs/realism_scorecard/run.log -- \\
        scripts/realism_scorecard.py --workers 3
    uv run scripts/watch.py --unit bio-real --log outputs/realism_scorecard/run.log

``--score-only`` recomputes the scorecard from the recordings (CPU). Outputs (outputs/realism_scorecard):
rec/<name>_<key>.npz (spikes; not committed), jobs/<name>_<key>.json (per-culture rows), scorecard.json.

Results (run 25 Nov 2026, 18:56-19:36). Controls held: no weight moved; the evoked measure reproduced the
replication's burst shares (0.199 / 0.213 / 0.209 against 0.20 / 0.21 / 0.21); the dead culture showed no network
bursts (0 per minute). Model median over five cultures (range), against the real cultures:

| Measure                                 | Model                | Real median [5-95%] | In range           |
|-----------------------------------------|----------------------|---------------------|--------------------|
| electrode rate, median electrode (Hz)   | 5.0 (3.9-7.2)        | 2.8 [0.7-10.3]      | yes, 5 of 5        |
| network bursts per minute               | 44.7 (43.7-47.3)     | 4.2 [1.2-15.4]      | **no**, 0 of 5     |
| burst duration (ms)                     | 103 (77-134)         | 556 [79-1788]       | yes, 4 of 5        |
| inter-burst interval CV                 | 0.32 (0.31-0.35)     | 0.87 [0.56-1.99]    | **no**, 0 of 5     |
| share of spikes inside bursts           | 0.85 (0.79-0.89)     | 0.63 [0.10-0.93]    | yes, 5 of 5        |
| evoked: pulses that set off a burst     | 20% (20-21%)         | 6.5% [0-16%]        | **no**, 0 of 5     |

Three of six in range. The model bursts ten times too often (every 1.4 s against every 12 s), like clockwork
(CV 0.32), and a single pulse sets off a burst three times too often. Its bursts are short (103 ms, near the real
5th percentile) and hold more of the spikes. Not gated: all 60 electrodes active (real 51-59 of 59). As expected:
too regular (CV ~0.3 expected) and evoked bursts too common (0.20-0.21 expected). Burst spread: burst_spread.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartpole_replication as CP  # noqa: E402
import public_data_targets as PDT  # noqa: E402

from biocompute.runguard import job_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CULTURES = (0, 1, 2, 3, 4)
WARM_S, RECORD_S = 10.0, 600.0
TARGETS = ROOT / "outputs/public_data/targets.json"
GATED = ("rate_hz_median", "burst_rate_per_min", "burst_duration_ms_median", "ibi_cv", "spikes_in_bursts")
REPORTED = ("n_active", "rate_hz_mean", "ibi_s_median", "recruitment_median")
LOGGED_BURST_MEAN = {0: 0.20, 1: 0.21, 2: 0.21}  # run_task.log, configure()'s burst_mean
DEAD_MAX_PER_MIN = 1.2


def job_name(kind: str, seed: int, sz: dict) -> str:
    """Named by the settings; ``culture`` (config overrides, weak_synapses.py) only when set, so the stored jobs of the
    unchanged culture keep their names."""
    return job_key(f"{kind}_s{seed}", {k: sz[k] for k in ("warm_s", "record_s", "evoked")} | (
        {"culture": sz["culture"]} if sz.get("culture") else {}))


def record_job(job):
    import torch

    from biocompute.runguard import Guard

    kind, seed, out, sz = job
    stem = job_name(kind, seed, sz)
    CP._start_job()
    g = Guard(f"record_{kind}_s{seed}{sz.get('tag', '')}", out)  # tag: a culture variant's name, for the log
    torch.set_num_threads(1)
    rec = out / "rec" / f"{stem}.npz"
    if not rec.exists():  # a finished recording isn't redone when only the evoked measure was stopped
        record(kind, seed, out, sz, stem, rec, g)
    with np.load(rec) as z:
        row = {"kind": kind, "seed": seed, "record_s": sz["record_s"], "rec": str(rec),
               "n_channels": int(len(z["electrode_xy"])),
               "neuron_rate_hz": round(float(len(z["n_id"]) / len(z["pos"]) / sz["record_s"]), 3)}
    if kind == "live" and sz["evoked"]:
        CP.STOP["phase"] = "setup"  # a stop here redoes the evoked measure (~5 min; ~30 min for s1000)
        dev2, _, _ = CP.build(seed, sz.get("culture"))
        cand, rate, first, burst, _, _ = CP.characterise(dev2, reps=sz["evoked_reps"],
                                                         beat=lambda i, n: g.beat(i, n, phase="evoked"))
        row |= {"evoked_burst_median": round(float(np.median(burst)), 3),
                "evoked_burst_mean": round(float(burst.mean()), 3), "evoked_burst_per_candidate": burst.tolist(),
                "evoked_candidates": [int(x) for x in cand],
                "config": CP.configure(cand, rate, first, burst)}  # the replication's encoders and decoders
        if seed in LOGGED_BURST_MEAN and not sz["smoke"] and not sz.get("culture"):
            g.check(round(float(burst.mean()), 2) == LOGGED_BURST_MEAN[seed],
                    f"evoked burst mean {burst.mean():.3f} != the replication's {LOGGED_BURST_MEAN[seed]}")
        del dev2
        torch.cuda.empty_cache()
        CP.STOP["phase"] = "idle"
    p = out / "jobs" / f"{stem}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(row))
    g.beat(1, 1, force=True)
    return row


def record(kind, seed, out, sz, stem, rec, g):
    """Settle, then record ``record_s`` of spontaneous activity: every electrode's and every neuron's spikes."""
    import torch

    from biocompute.culture.network import make_config
    from biocompute.device import SimulatedBiocomputer
    from biocompute.mea.electrodes import make_mea_config

    dev = SimulatedBiocomputer(make_config("gpu_20k", seed=seed, **(sz.get("culture") or {})),
                               make_mea_config("mcs60", seed=seed), keep_raster=True)
    c = dev.culture
    if kind == "dead":  # as biocompute.train.setup builds the dead culture
        c.w.zero_()
        c.w0.zero_()
    g.check(c.plasticity is None, "plasticity is on")
    w_ref = c.w.clone()
    ck = out / "ckpt" / f"{stem}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    per = int(round(100.0 / c.cfg.dt_ms))  # substeps per 100 ms read
    n = int(round(sz["record_s"] * 10))
    if ck.exists():
        saved = torch.load(ck, weights_only=False)
        dev.load_state(saved["dev"])
        S = saved["script"]
    else:
        for _ in range(int(round(sz["warm_s"] * 10))):
            dev.step(100.0)
        S = {"i": 0, "t0": float(c.t_ms), "et": [], "ec": [], "nt": [], "nn": []}
    CP.STOP["phase"] = "resumable"
    t_save, t_wall = time.time(), time.time()
    while S["i"] < n:
        if CP.STOP["requested"] or time.time() - t_save > 300:
            torch.save({"dev": dev.state(), "script": S}, ck.with_suffix(".tmp"))
            os.replace(ck.with_suffix(".tmp"), ck)
            t_save = time.time()
            if CP.STOP["requested"]:
                raise CP.Stopped(f"record {kind}_s{seed} at read {S['i']}/{n}")
        t0 = float(c.t_ms)
        b = dev.step(100.0)
        S["et"].append(((t0 - S["t0"]) + b.times_ms) / 1000.0)
        S["ec"].append(b.channels.astype(np.int16))
        nz = torch.nonzero(dev.last_raster).cpu().numpy()  # (substep, neuron), row-major: ordered by time
        S["nt"].append((nz[:, 0] + S["i"] * per).astype(np.int32))
        S["nn"].append(nz[:, 1].astype(np.int32))
        S["i"] += 1
        if g.due():
            k = min(S["i"], 50)
            hz = sum(len(x) for x in S["ec"][-k:]) / (k * 0.1) / dev.n_channels
            g.culture(c, rate_hz=max(hz, 1e-9) if kind == "live" else None)
            g.check(bool(torch.equal(c.w, w_ref)), "weights moved with plasticity off")
            g.beat(S["i"], n, hz=round(hz, 2), ms_per_read=round((time.time() - t_wall) / max(S["i"], 1) * 1000))
    CP.STOP["phase"] = "idle"
    g.check(bool(torch.equal(c.w, w_ref)), "weights moved with plasticity off")
    rec.parent.mkdir(parents=True, exist_ok=True)
    with open(rec.with_suffix(".tmp"), "wb") as f:
        np.savez(f, e_t_s=np.concatenate(S["et"]), e_ch=np.concatenate(S["ec"]), n_step=np.concatenate(S["nt"]),
                 n_id=np.concatenate(S["nn"]), dt_ms=np.array(c.cfg.dt_ms), record_s=np.array(sz["record_s"]),
                 pos=c.pos.cpu().numpy().astype(np.float32), excitatory=c.excitatory.cpu().numpy(),
                 electrode_xy=np.asarray(dev.mea.xy, np.float32), alive_channels=np.asarray(dev.mea.alive_channels))
    os.replace(rec.with_suffix(".tmp"), rec)
    ck.unlink(missing_ok=True)
    del dev, c
    torch.cuda.empty_cache()


def _record_safe(job):
    try:
        return record_job(job)
    except CP.Stopped as e:
        return {"stopped": str(e)}


def load_row(out: Path, kind: str, seed: int, sz: dict) -> dict | None:
    """A finished job's row with B0's statistics (``spike_stats``) of its recording; None while it hasn't finished."""
    p = out / "jobs" / f"{job_name(kind, seed, sz)}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    with np.load(r["rec"]) as z:
        r["stats"] = PDT.spike_stats(z["e_t_s"], z["e_ch"].astype(int), float(z["record_s"]), r["n_channels"])
    return r


def score(rows: list, targets: dict, smoke: bool = False) -> dict:
    tw = targets["wagenaar_2006_dense_div21_35"]["targets_across_cultures"]
    te = targets["robbins_2026_organoids"]["evoked_burst_percent_median_per_experiment"]
    live = [r for r in rows if r["kind"] == "live"]
    dead = [r for r in rows if r["kind"] == "dead"]
    card = {}
    for k in GATED + REPORTED + ("evoked_burst_median",):
        vals = [r["stats"].get(k) if k != "evoked_burst_median" else r.get(k) for r in live]
        vals = [v for v in vals if v is not None]
        lo, hi = (te if k == "evoked_burst_median" else tw[k])["p5_p95"]
        med = float(np.median(vals)) if vals else None
        card[k] = {"model_per_culture": vals, "model_median": None if med is None else round(med, 3),
                   "real_median": (te if k == "evoked_burst_median" else tw[k])["median"], "real_p5_p95": [lo, hi],
                   "cultures_in_range": int(sum(lo <= v <= hi for v in vals)), "n": len(vals),
                   "in_range": med is not None and lo <= med <= hi,
                   "gated": k in GATED or k == "evoked_burst_median"}
    dead_rate = dead[0]["stats"].get("burst_rate_per_min", 0.0) if dead else None
    return {"scorecard": card, "dead_burst_rate_per_min": dead_rate,
            "dead_control_ok": None if dead_rate is None else dead_rate < DEAD_MAX_PER_MIN,
            "in_range": [k for k, v in card.items() if v["gated"] and v["in_range"]],
            "out_of_range": [k for k, v in card.items() if v["gated"] and not v["in_range"]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cultures", type=int, nargs="+", default=list(CULTURES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/realism_scorecard")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="culture 0 and its dead control, 3 s each, 2 pulses per "
                                                         "candidate, into <out>/smoke")
    a = ap.parse_args()
    out = Path(a.out)
    sz = {"warm_s": WARM_S, "record_s": RECORD_S, "evoked": True, "evoked_reps": 25, "smoke": a.smoke}
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
        sz |= {"warm_s": 1.0, "record_s": 3.0, "evoked_reps": 2}
        a.cultures = [0]
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("live", s, out, sz) for s in a.cultures] + [("dead", a.cultures[0], out, sz)]
    print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "cultures": a.cultures, **sz,
                      "workers": a.workers}), flush=True)
    todo = [j for j in jobs if not (out / "jobs" / f"{job_name(j[0], j[1], sz)}.json").exists()]
    if todo and not a.score_only:
        CP.pool_map(_record_safe, todo, min(a.workers, len(todo)))
    rows = []
    for kind, seed, *_ in jobs:
        r = load_row(out, kind, seed, sz)
        if r is None:
            continue
        rows.append(r)
        print(json.dumps({"kind": kind, "seed": seed, "neuron_rate_hz": r["neuron_rate_hz"], **r["stats"],
                          **{k: r[k] for k in ("evoked_burst_median", "evoked_burst_mean") if k in r}}), flush=True)
    res = score(rows, json.loads(TARGETS.read_text()), a.smoke)
    if res["dead_control_ok"] is False and not a.smoke:
        print(f"GUARD FAIL [scorecard]: the dead culture shows {res['dead_burst_rate_per_min']} bursts per minute",
              flush=True)
    (out / "scorecard.json").write_text(json.dumps({"rows": rows} | res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "scorecard"}), flush=True)
    for k, v in res["scorecard"].items():
        print(json.dumps({k: {x: v[x] for x in ("model_median", "real_median", "real_p5_p95", "cultures_in_range",
                                                 "in_range")}}), flush=True)


if __name__ == "__main__":
    main()
