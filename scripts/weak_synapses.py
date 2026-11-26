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
DRIFT_WORKERS = {"s190": 3, "s400": 3, "s1000": 2}


def overrides(variant: str) -> dict:
    from biocompute.culture.network import make_config

    k, length, buckets = VARIANTS[variant]
    base = make_config("gpu_20k")
    o = {"synapses_per_neuron": base.synapses_per_neuron * k, "w_exc": base.w_exc / k, "w_inh": base.w_inh / k,
         "w_max": base.w_max / k}
    return o | ({"length_um": length} if length else {}) | ({"delay_buckets": buckets} if buckets else {})


def structure(culture: dict | None, seed: int = 0) -> dict:
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


def realism(a, out: Path) -> None:
    sz0 = {"warm_s": RS.WARM_S, "record_s": RS.RECORD_S, "evoked": True, "evoked_reps": 25, "smoke": a.smoke}
    seeds, control = CULTURES, CONTROL_SEEDS
    variants = list(VARIANTS)
    if a.smoke:
        sz0 |= {"warm_s": 1.0, "record_s": 3.0, "evoked_reps": 2}
        seeds, control, variants = (0,), (0,), ["s1000"]
    groups = {"control": (sz0, control)} | {
        v: (sz0 | {"culture": overrides(v), "tag": f"_{v}"}, seeds) for v in variants}

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


def culture_verdict(off: dict | None, on: dict | None) -> dict:
    if off is None or on is None:
        return {"pass": False, "why": "no encoders and decoders, or arm missing"}
    lo, hi = CP.B1_BAND
    tested = CP.long_enough(off) and CP.long_enough(on)
    r_off, r_on = float(np.mean(off["rate_ratio_60s"][-3:])), float(np.mean(on["rate_ratio_60s"][-3:]))
    calm_ratio = r_on / r_off if r_off > 0 else math.inf
    pw = CP.pathway_change(on, off)
    kept = bool(pw["tested"]) and pw["moved_own"] + pw["moved_cross"] == 0
    m = np.array(off["gain_60s"]).mean(0)
    base = all(m[k, k] >= 2 * abs(m[k, 1 - k]) and abs(m[k, 1 - k]) <= 1.0 for k in (0, 1))
    return {"tested": tested, "calm_ratio_last3": round(calm_ratio, 2) if math.isfinite(calm_ratio) else None,
            "calm": lo <= calm_ratio <= hi, "pathways": pw, "pathways_kept": kept,
            "off_mean_gains": np.round(m, 2).tolist(), "off_separate": bool(base),
            "rate_vs_rest_last3": [round(r_off, 2), round(r_on, 2)],
            "pass": bool(tested and lo <= calm_ratio <= hi and kept and base)}


def drift_gate(rows: list, seeds=CULTURES) -> dict:
    by = {(r.get("seed", 0), r["scale"]): r for r in rows
          if not r["eta"] and not r.get("homeo") and r["rule"] == "additive"}
    per = {str(s): culture_verdict(by.get((s, 0.0)), by.get((s, 1.0))) for s in seeds}
    n = sum(v["pass"] for v in per.values())
    return {"per_culture": per, "passed": n, "need": NEED, "of": len(seeds), "pass": n >= NEED}


def drift_part(a, out: Path) -> None:
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
            arms = arms[1:]
        todo = [j for j in arms if not (out / "jobs" / f"{j[1]}.json").exists()]
        CP.pool_map(CP._drift_safe, todo, 1 if a.smoke else min(DRIFT_WORKERS[v], len(todo) or 1))
        rows = [json.loads((out / "jobs" / f"{j[1]}.json").read_text()) for j in arms]
        for s in seeds:
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
    ap = argparse.ArgumentParser()
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
