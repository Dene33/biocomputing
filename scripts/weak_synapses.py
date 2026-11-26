"""Step 4 (docs/route-to-computation.md, "Order" item 4.4): many weak synapses, the same total input. One time-boxed
round. A failure ends track B for good.

Pre-registered 25 Nov 2026, before any variant was recorded (only speed and memory had been measured). After it
was written, the smoke tests printed a glimpse of s1000, culture 0 (3-4 s: quiet, no burst); nothing was changed.

Why. The takeover rerun (takeover_rerun.py) showed that STDP takes its direction from activity spreading out from
the stronger input. The realism scorecard (realism_scorecard.py) showed the model bursting ten times too often, too
regularly, with a pulse setting off a burst three times too often. Muller et al. (Neuron 2026): with many weak
synapses, waves are sparse; small networks with strong synapses give dense waves (docs/literature-notes.md).

The change. Each neuron gets k times as many synapses, each 1/k as strong: the mean excitatory weight, the mean
inhibitory weight and the weight cap w_max are divided by k. STDP's step is a fraction of w_max, so it shrinks with
it; weight-dependent STDP, synaptic scaling, the 5 s total-input step and forgetting are all relative. So each
neuron's total input and total plastic change stay as they are; only their graininess changes (at equal mean, input
fluctuations fall about sqrt(k)-fold). Unchanged: neurons, positions, cell types, background drive, short-term
depression, conduction speed, stimulation, recording.

The constraint (measured before the run). Connection probability is p_max exp(-d / 150 um), and in gpu_20k p_max is
already 0.51, so a neuron can have at most ~194 synapses (p_max = 1). Beyond that the reach has to grow. The
variants, fixed now, each with the smallest reach that fits (p_max 0.95-0.97):

| Variant | Synapses per neuron | Weights and cap | Reach            | Delay buckets  |
|---------|---------------------|-----------------|------------------|----------------|
| s190    | 190 (x1.9)          | / 1.9           | 150 um, as now   | 8, as now      |
| s400    | 400 (x4)            | / 4             | 225 um           | 16             |
| s1000   | 1000 (x10)          | / 10            | 370 um           | 16             |

Delay buckets are set so that no more synapses hit the delay cap than now (0.3%). With 8 buckets (4 ms, 1050 um),
3.6% of s400's synapses and 15.6% of s1000's would be capped. Measured beforehand, culture 0:
- synapses per neuron: 99.2 (now), then 188.6 / 398.7 / 1000.3 (x1.90 / 4.02 / 10.08);
- wall seconds per simulated second, plasticity off / STDP plus scaling: 0.65 / 1.34 (now), then 0.67 / 1.30,
  1.05 / 1.67 and 2.22 / 3.12;
- GPU peak: 0.7 / 0.8 / 1.2 / 2.5 GB with STDP (s1000 with plasticity off: 1.3 GB).

Part 1, realism (plasticity off). Each variant in cultures 0-4, exactly as the scorecard runs it: 10 s to settle, then
600 s of spontaneous activity on mcs60, then the replication's evoked-burst measure on grid256 (characterise, 25
pulses per candidate). The evoked measure also yields the replication's encoders and decoders for that culture
(configure), which part 2 uses. A variant passes when all six of the scorecard's gated measures are in range, each
measured in all five cultures: electrode rate, bursts per minute, burst duration, inter-burst interval CV, spikes in
bursts, and evoked bursts (6.5% [0-16%]). "In range" means the median over the five cultures lies within the real
5-95% range.
Controls (GUARD FAIL otherwise):
- run through the same code, the unchanged culture (seeds 0-2) equals the stored scorecard rows bit for bit
  (statistics and evoked measure) and picks the stored encoders and decoders (DRIFT_CONFIGS);
- for each variant, culture 0's mean total excitatory and inhibitory input per neuron lies within 2% of the unchanged
  culture's, and at most 0.5% of synapses hit the delay cap;
- no weight moves (the scorecard's own check).

Part 2, stability. Runs only for variants that pass part 1, smallest k first. Per culture, on its own encoders and
decoders, the drift test as cartpole_replication.py --drift runs it (pulses for tilts held 1 s, 600 s, 20 s of rest
first). Two arms: no plasticity, and STDP x1 (additive) with the 5 s total-input step, which is gated.
Why this arm and not STDP with B1's scaling, which B2's wording ("combined with B1") suggests (reasoning, not
measured): B1's scaling pulls each neuron toward its resting rate within seconds; at eta 1e-5, a 5 Hz excess shrinks
the weights 20-fold in a minute. In a culture whose rate rises under stimulation, as a more realistic one's may
(stimulation adds bursts), the scaling would push weights down by itself, so a fail could come from the brake rather
than the takeover. The 5 s step keeps each neuron's total input and has no rate target. It is the arm in which the
takeover was first found (x1: 6 -> 43 Hz per electrode, one input driving both outputs).
Per culture, the gated arm passes when all three hold:
1. calm: its rate per electrode over the last 3 minutes is within 2/3-1.5x the no-plasticity arm's over the same
   minutes. That is B1's band, against a yardstick stimulation can't shift (same culture, same pulses). In the stored
   runs the no-plasticity arm sits at 0.97-1.14x rest, so there it equals B1's band against rest;
2. pathways kept: none of its four encoder -> decoder gains (own and cross) moved against the no-plasticity arm, by
   pathways_vs_off's test (last 3 minutes, beyond 2 SE from both arms' per-minute spread over minutes 2-10);
3. a pathway to keep: the no-plasticity arm is separate on its 10-minute mean gains (each decoder: own >= 2x |cross|,
   |cross| <= 1). This holds in all three stored cultures.
A culture for which configure finds no encoders and decoders fails. Gate: at least 3 of the 5 cultures pass.
Positive control (GUARD FAIL otherwise): the same rule, applied to the stored runs of the unchanged culture
(drift_scaling.json, cultures 0-2, no plasticity and STDP x1 with the 5 s step), fails all three.
Reported, not gated: rates per minute against rest and against the no-plasticity arm, gains per minute, and the E->E
weight against its start.

Step 4 passes when a variant passes both parts. If no variant passes part 1, step 4 fails and part 2 doesn't run,
since no decision depends on it.

Expected (reasoning, not measured):
- The mean drive is unchanged, so the cultures still burst.
- Smaller fluctuations should make bursts start less easily: fewer per minute, and fewer evoked.
- But bursts that wait for recovery from short-term depression may get more regular, not less. So the interval CV
  probably stays under 0.56, and part 1 fails.
- s400's and s1000's longer reach spreads each neuron's input wider and may speed bursts up.
- If part 2 runs: STDP's mean push is unchanged (k times the synapses, each step 1/k), so a variant can pass only
  through calmer activity.

Results (run 25-26 Nov 2026, 22:26-03:51, with one stop at 00:12; see below). The controls held. The unchanged
culture (seeds 0-2) repeated the stored scorecard rows bit for bit and picked the stored encoders and decoders. Every
variant kept each neuron's total input within 1% (excitatory 174.6 / 175.4 / 176.0, against 174.4), and at most
0.37% of synapses hit the delay cap. Median over five cultures (range):

| Measure                               | Now (stored) | s190             | s400             | s1000            | Real [5-95%]   |
|---------------------------------------|--------------|------------------|------------------|------------------|----------------|
| electrode rate (Hz)                   | 5.0          | 0.56 (0.55-0.58) | 0.55 (0.54-0.56) | 0.54 (0.53-0.56) | 2.8 [0.7-10.3] |
| network bursts per minute             | 44.7         | 0                | 0                | 0                | 4.2 [1.2-15.4] |
| burst ms / interval CV / in-burst     | 103/0.32/0.85| no bursts        | no bursts        | no bursts        | see scorecard  |
| evoked: pulses that set off a burst   | 20%          | 0% (0-4%)        | 4% (2-4%)        | 2% (0-4%)        | 6.5% [0-16%]   |
| neuron rate (Hz, not gated)           | 1.47         | 0.08             | 0.064            | 0.053            |                |

No variant passes part 1. Each has one of six measures in range, the evoked share. Step 4 fails and part 2 didn't
run. drift.json records the verdict and the positive control (calm 6.52 / 7.15 / 5.51; all three fail). Track B
closes for good.

Against the expectation: wrong on the main point. The cultures did not keep bursting: there was not one network burst
in 10 minutes in any of the 15 cultures. A check that isn't gated, on raw neuron spikes in 10 ms bins: the busiest bin
held at most 0.36% of the neurons. In the unchanged culture, the 99.9th percentile is 74-83%. Halving each synapse
(s190, same reach) is already enough, and the neurons then fire 18 times less (1.47 -> 0.08 Hz). "Fewer bursts" was
right in direction, but the loss was total.

The evoked share is in range. With no network bursts, though, a "burst" there is only a response above the culture's
own median + 3 MAD (reasoning, not checked). Also not gated: each culture still has clean local pairs. Each encoder
reaches its decoder within 10 ms on 76-100% of its pulses, and the other decoder on 0-4%.

Reading (reasoning, not tested): in this model, bursts start when strong single inputs coincide. At the same mean
input, smaller jolts never reach ignition. So the model has no setting in this direction that both bursts and has
fine-grained input. A model that bursts with many weak synapses would need other machinery, for example slower
synaptic currents or stronger drive, and that is outside the time-box.

The stop at 00:12: the watcher stopped the run after 15 minutes without a heartbeat. s1000's evoked measure takes
about 30 minutes, because three parallel workers saturate the GPU (6.8 s per simulated second each), and characterise
had no heartbeat. I added an optional per-electrode beat, which changes nothing measured. The rerun reused the
finished recordings and the control rows.

Run (guarded; every recording and drift arm checkpoints and resumes):

    uv run scripts/launch.py --unit bio-weak --log outputs/weak_synapses/run_realism.log -- \\
        scripts/weak_synapses.py --part realism --workers 3
    uv run scripts/watch.py --unit bio-weak --log outputs/weak_synapses/run_realism.log
    uv run scripts/launch.py --unit bio-weak2 --log outputs/weak_synapses/run_drift.log -- \\
        scripts/weak_synapses.py --part drift
    uv run scripts/watch.py --unit bio-weak2 --log outputs/weak_synapses/run_drift.log

Outputs (outputs/weak_synapses): realism/ (rec/*.npz not committed, jobs/), realism.json; drift/ (jobs/, ckpt/),
drift.json.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartpole_replication as CP  # noqa: E402
import realism_scorecard as RS  # noqa: E402

from biocompute.runguard import job_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Variant: (k, reach in um or None for unchanged, delay buckets or None for unchanged).
VARIANTS = {"s190": (1.9, None, None), "s400": (4.0, 225.0, 16), "s1000": (10.0, 370.0, 16)}
CULTURES = (0, 1, 2, 3, 4)
CONTROL_SEEDS = (0, 1, 2)
STORED_SCORECARD = ROOT / "outputs/realism_scorecard/scorecard.json"
STORED_DRIFT = ROOT / "outputs/drift_scaling/drift_scaling.json"
EVOKED_FIELDS = ("neuron_rate_hz", "evoked_burst_median", "evoked_burst_mean", "evoked_burst_per_candidate",
                 "evoked_candidates")
INPUT_TOL, CAP_MAX = 0.02, 0.005
DRIFT_S, REST_S = 600.0, 20.0
NEED = 3
DRIFT_WORKERS = {"s190": 3, "s400": 3, "s1000": 2}  # 8 GB card: s1000 with STDP takes ~2.5 GB plus the context


def overrides(variant: str) -> dict:
    """A variant's culture config overrides: k times the synapses, each 1/k as strong (mean excitatory and inhibitory
    weights, and the cap), plus the reach and delay buckets it needs. Only fields that change."""
    from biocompute.culture.network import make_config

    k, length, buckets = VARIANTS[variant]
    base = make_config("gpu_20k")
    o = {"synapses_per_neuron": base.synapses_per_neuron * k, "w_exc": base.w_exc / k, "w_inh": base.w_inh / k,
         "w_max": base.w_max / k}
    return o | ({"length_um": length} if length else {}) | ({"delay_buckets": buckets} if buckets else {})


def structure(culture: dict | None, seed: int = 0) -> dict:
    """Wiring of one culture, built on the CPU: synapses per neuron, mean total excitatory and inhibitory input per
    neuron (rows of the delivery matrix, an ordered sum), and the share of synapses whose delay hits the cap."""
    import torch

    from biocompute.culture.network import OrganoidCulture, make_config

    torch.set_num_threads(4)
    c = OrganoidCulture(make_config("gpu_20k", seed=seed, device="cpu", **(culture or {})))
    A = c._delivery_matrix()
    tot = (A @ torch.ones(A.shape[1], 1)).squeeze(1)
    exc, inh = tot[: c.n], tot[c.n:]
    d = (c.pos[c.pre] - c.pos[c.post]).norm(dim=1)
    raw = ((c.cfg.min_delay_ms + d / c.cfg.conduction_um_per_ms) / c.cfg.dt_ms).round()
    return {"synapses_per_neuron": round(c.n_synapses / c.n, 2), "exc_input_mean": round(float(exc.mean()), 3),
            "inh_input_mean": round(float(inh.mean()), 3), "exc_input_cv": round(float(exc.std() / exc.mean()), 3),
            "delay_capped": round(float((raw > c.cfg.delay_buckets).float().mean()), 4)}


def fail(what: str) -> None:
    print(f"GUARD FAIL [weak_synapses]: {what}", flush=True)
    raise SystemExit(1)


# --------------------------------------------------------------------------------------------------------- part 1
def realism(a, out: Path) -> None:
    sz0 = {"warm_s": RS.WARM_S, "record_s": RS.RECORD_S, "evoked": True, "evoked_reps": 25, "smoke": a.smoke}
    seeds, control = CULTURES, CONTROL_SEEDS
    variants = list(VARIANTS)
    if a.smoke:
        sz0 |= {"warm_s": 1.0, "record_s": 3.0, "evoked_reps": 2}
        seeds, control, variants = (0,), (0,), ["s1000"]
    groups = {"control": (sz0, control)} | {
        v: (sz0 | {"culture": overrides(v), "tag": f"_{v}"}, seeds) for v in variants}

    # Wiring first (CPU, ~1 min): the variants must keep each neuron's total input.
    st = {g: structure(sz.get("culture")) for g, (sz, _) in groups.items()}
    print(json.dumps({"structure": st}), flush=True)
    for v in variants:
        for k in ("exc_input_mean", "inh_input_mean"):
            rel = st[v][k] / st["control"][k] - 1
            if abs(rel) > INPUT_TOL:
                fail(f"{v}: {k} {st[v][k]} vs {st['control'][k]} ({rel:+.1%})")
        if st[v]["delay_capped"] > CAP_MAX:
            fail(f"{v}: {st[v]['delay_capped']:.2%} of synapses at the delay cap")

    def jobs_of(g):
        sz, ss = groups[g]
        return [("live", s, out, sz) for s in ss]

    def todo(js):
        return [j for j in js if not (out / "jobs" / f"{RS.job_name('live', j[1], j[3])}.json").exists()]

    def rows_of(g):
        sz, ss = groups[g]
        return {s: RS.load_row(out, "live", s, sz) for s in ss}

    # Positive control first: the unchanged culture must repeat the stored scorecard and the stored configurations.
    CP.pool_map(RS._record_safe, todo(jobs_of("control")), min(a.workers, len(control)))
    ctrl = rows_of("control")
    checks = {}
    if not a.smoke:
        stored = {r["seed"]: r for r in json.loads(STORED_SCORECARD.read_text())["rows"] if r["kind"] == "live"}
        for s, r in ctrl.items():
            same = r["stats"] == stored[s]["stats"] and all(r[k] == stored[s][k] for k in EVOKED_FIELDS)
            cfg_ok = r["config"] is not None and (r["config"]["enc"], r["config"]["dec"]) == tuple(
                map(list, CP.DRIFT_CONFIGS[s]))
            checks[str(s)] = {"equals_stored_scorecard": same, "picks_stored_electrodes": cfg_ok}
            if not (same and cfg_ok):
                fail(f"control culture {s}: equals stored scorecard {same}, stored electrodes {cfg_ok}")
    print(json.dumps({"controls": checks}), flush=True)

    # The variants, longest first (s1000's recordings take ~3x s190's).
    js = [j for v in sorted(variants, key=lambda v: -VARIANTS[v][0]) for j in jobs_of(v)]
    CP.pool_map(RS._record_safe, todo(js), a.workers)

    targets = json.loads(RS.TARGETS.read_text())
    res = {"pre_registered": "see scripts/weak_synapses.py", "structure": st, "controls": checks, "variants": {}}
    for v in variants:
        rows = rows_of(v)
        sc = RS.score(list(rows.values()), targets, a.smoke)
        gated = {k: c for k, c in sc["scorecard"].items() if c["gated"]}
        complete = all(c["n"] == len(seeds) for c in gated.values())
        res["variants"][v] = {
            "overrides": overrides(v), "scorecard": sc["scorecard"], "in_range": sc["in_range"],
            "out_of_range": sc["out_of_range"], "complete": complete,
            "pass": complete and not sc["out_of_range"],
            "configs": {str(s): r.get("config") for s, r in rows.items()},
            "rows": [{k: r[k] for k in ("seed", "neuron_rate_hz", "stats", "evoked_burst_median", "config")}
                     for r in rows.values()]}
        print(json.dumps({v: {"pass": res["variants"][v]["pass"], "out_of_range": sc["out_of_range"]}}), flush=True)
        for k, c in gated.items():
            print(json.dumps({v: {k: {x: c[x] for x in ("model_median", "model_per_culture", "real_p5_p95",
                                                         "cultures_in_range", "in_range")}}}), flush=True)
    passed = [v for v in variants if res["variants"][v]["pass"]]
    res["passed"] = passed
    res["verdict"] = ("part 2 runs on " + ", ".join(passed)) if passed else (
        "no variant passes part 1: step 4 fails, part 2 doesn't run, track B closes for good")
    (out.parent / ("realism_smoke.json" if a.smoke else "realism.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({"verdict": res["verdict"]}), flush=True)


# --------------------------------------------------------------------------------------------------------- part 2
def culture_verdict(off: dict | None, on: dict | None) -> dict:
    """The pre-registered per-culture rule for the gated arm ``on`` against its culture's no-plasticity arm."""
    if off is None or on is None:
        return {"pass": False, "why": "no encoders and decoders, or arm missing"}
    lo, hi = CP.B1_BAND
    tested = CP.long_enough(off) and CP.long_enough(on)
    r_off, r_on = float(np.mean(off["rate_ratio_60s"][-3:])), float(np.mean(on["rate_ratio_60s"][-3:]))
    calm_ratio = r_on / r_off if r_off > 0 else math.inf
    pw = CP.pathway_change(on, off)
    kept = bool(pw["tested"]) and pw["moved_own"] + pw["moved_cross"] == 0
    m = np.array(off["gain_60s"]).mean(0)  # [decoder, encoder]
    base = all(m[k, k] >= 2 * abs(m[k, 1 - k]) and abs(m[k, 1 - k]) <= 1.0 for k in (0, 1))
    return {"tested": tested, "calm_ratio_last3": round(calm_ratio, 2) if math.isfinite(calm_ratio) else None,
            "calm": lo <= calm_ratio <= hi, "pathways": pw, "pathways_kept": kept,
            "off_mean_gains": np.round(m, 2).tolist(), "off_separate": bool(base),
            "rate_vs_rest_last3": [round(r_off, 2), round(r_on, 2)],
            "pass": bool(tested and lo <= calm_ratio <= hi and kept and base)}


def drift_gate(rows: list, seeds=CULTURES) -> dict:
    """Per culture: the no-plasticity arm (scale 0) and the gated arm (STDP x1, the 5 s step: scale 1, no scaling,
    additive). At least NEED cultures must pass."""
    by = {(r.get("seed", 0), r["scale"]): r for r in rows
          if not r["eta"] and not r.get("homeo") and r["rule"] == "additive"}
    per = {str(s): culture_verdict(by.get((s, 0.0)), by.get((s, 1.0))) for s in seeds}
    n = sum(v["pass"] for v in per.values())
    return {"per_culture": per, "passed": n, "need": NEED, "of": len(seeds), "pass": n >= NEED}


def drift_part(a, out: Path) -> None:
    # Positive control: the rule must fail the stored runs of the unchanged culture, where the takeover was found.
    pos = drift_gate(json.loads(STORED_DRIFT.read_text())["rows"], seeds=(0, 1, 2))
    print(json.dumps({"positive_control": {s: {k: v[k] for k in ("calm_ratio_last3", "pathways_kept", "off_separate",
                                                                 "pass")} for s, v in pos["per_culture"].items()}}),
          flush=True)
    if any(v["pass"] for v in pos["per_culture"].values()):
        fail("the rule passes a stored run of the unchanged culture")
    if a.smoke:
        variants, seeds, secs, rest = ["s1000"], (0,), 4.0, 2.0
        configs = {"s1000": {"0": {"enc": CP.DRIFT_CONFIGS[0][0], "dec": CP.DRIFT_CONFIGS[0][1]}}}
    else:
        real = json.loads((out.parent / "realism.json").read_text())
        variants = sorted(real["passed"], key=lambda v: VARIANTS[v][0])
        seeds, secs, rest = CULTURES, DRIFT_S, REST_S
        configs = {v: real["variants"][v]["configs"] for v in variants}
    res = {"pre_registered": "see scripts/weak_synapses.py", "positive_control": pos, "variants": {}}
    if not variants:
        res["verdict"] = "no variant passed part 1: step 4 fails, track B closes for good"
        (out.parent / "drift.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"verdict": res["verdict"]}), flush=True)
        return
    for v in variants:
        arms = []
        for s in seeds:
            cfg = configs[v].get(str(s))
            if cfg is None:
                continue
            for x in (0.0, 1.0):
                sp = {"scale": x, "rule": "additive", "eta": 0.0, "w_max_inh": 30.0, "floor": 0.5, "seed": s,
                      "culture": overrides(v), "enc": cfg["enc"], "dec": cfg["dec"]}
                name = f"{v}_x{x:g}_s{s}"
                key = job_key(f"drift_{name}", {**sp, "seconds": secs, "rest_s": rest})
                arms.append((name, key, sp, secs, out, {"rest_s": rest}))
        if a.smoke:
            arms = arms[1:]  # the plastic arm only: the most machinery
        todo = [j for j in arms if not (out / "jobs" / f"{j[1]}.json").exists()]
        CP.pool_map(CP._drift_safe, todo, 1 if a.smoke else min(DRIFT_WORKERS[v], len(todo) or 1))
        rows = [json.loads((out / "jobs" / f"{j[1]}.json").read_text()) for j in arms]
        for s in seeds:  # both arms rest before any plasticity, so their rest must agree exactly
            pair = [r for r in rows if r["seed"] == s]
            if len(pair) == 2 and pair[0]["rest_hz_per_electrode"] != pair[1]["rest_hz_per_electrode"]:
                fail(f"{v} culture {s}: the arms' resting rates differ")
        gate = drift_gate(rows, seeds)
        res["variants"][v] = {"overrides": overrides(v), "gate": gate, "rows": rows}
        print(json.dumps({v: {"passed": gate["passed"], "need": NEED, "pass": gate["pass"],
                              "per_culture": {s: {k: c.get(k) for k in ("calm_ratio_last3", "pathways_kept",
                                                                        "off_separate", "pass")}
                                              for s, c in gate["per_culture"].items()}}}), flush=True)
    passed = [v for v in variants if res["variants"][v]["gate"]["pass"]]
    res["verdict"] = (f"step 4 passes with {', '.join(passed)}: track B reopens" if passed else
                      "no variant passes part 2: step 4 fails, track B closes for good")
    (out.parent / ("drift_smoke.json" if a.smoke else "drift.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({"verdict": res["verdict"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=["realism", "drift"], required=True)
    ap.add_argument("--workers", type=int, default=3, help="part 1; part 2 uses DRIFT_WORKERS per variant")
    ap.add_argument("--out", default="outputs/weak_synapses")
    ap.add_argument("--smoke", action="store_true", help="the whole part at tiny scale into <out>/smoke")
    a = ap.parse_args()
    root = Path(a.out) / ("smoke" if a.smoke else "")
    out = root / a.part
    if a.smoke:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "part": a.part, "smoke": a.smoke,
                      "workers": a.workers}), flush=True)
    (realism if a.part == "realism" else drift_part)(a, out)


if __name__ == "__main__":
    main()
