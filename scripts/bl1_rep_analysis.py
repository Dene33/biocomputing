"""Cross-seed summary of outputs/bl1_drift/rep (bl1_drift.py --seed 42 43 44 45 46 --pulse-ms 20 0.5 --controls): rates
against the off arm, pathway gains in the last 3 minutes against the off arm with SE from minutes 2-10, and the seed-42
rerun against the first run. Plain python3 (numpy only):

    python3 scripts/bl1_rep_analysis.py
"""
import glob
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

root = str(Path(__file__).resolve().parent.parent / "outputs/bl1_drift/rep")
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/*/jobs/drift_*.json")) if "/smoke/" not in f]
_off = {(r["variant"], r["seed"], r["pulse_ms"]): r for r in rows if r["arm"] == "off"}
for r in rows:
    o = _off.get((r["variant"], r["seed"], r["pulse_ms"]))
    r["vs_off"] = r["hz_last60"] / o["hz_last60"] if o else float("nan")
pc = {}
for f in sorted(glob.glob(f"{root}/*/setup.json")):
    if "/smoke/" in f:
        continue
    d = json.load(open(f))
    m = re.search(r"/(shipped|fixed)(?:_seed(\d+))?(?:_pulse([\d.]+)ms)?/setup.json", f)
    seed, pulse = int(m.group(2) or 42), float(m.group(3) or 20)
    pc[(seed, pulse)] = (d["positive_control"]["rate_hz"], d["positive_control"]["bursts_per_min"],
                         d["config"]["c1_ab"], d["config"]["c1_cd"], d["config"]["c1_ad"], d["config"]["c1_cb"])
print("positive control (seed, pulse): rate Hz, bursts/min | C1 ab cd ad cb")
for k in sorted(pc):
    if k[1] == 20.0:
        print(" ", k[0], pc[k][:2], "|", pc[k][2:])

def changes(r):
    g = np.array(r["gain_lag_60s"])  # [window, decoder, encoder]
    own = [g[-1, k, k] - g[0, k, k] for k in (0, 1)]
    cross = [g[-1, k, 1 - k] - g[0, k, 1 - k] for k in (0, 1)]
    return own, cross, g

by = {(r["variant"], r["seed"], r["pulse_ms"], r["arm"]): r for r in rows}
seeds = sorted({r["seed"] for r in rows})
for pulse in (20.0, 0.5):
    print(f"\n=== pulse {pulse} ms  (median [min, max] over seeds {seeds}) ===")
    print(f"{'variant/arm':24s} {'vs_off':>18s} {'nonenc/rest':>18s} {'last/first':>18s} {'spec':>5s} {'pass':>5s} "
          f"{'max cross up vs off':>22s} {'own change vs off':>20s}")
    for variant, arm in [("shipped", "off"), ("shipped", "stdp"), ("shipped", "homeo"), ("shipped", "stdp_homeo"),
                         ("fixed", "stdp"), ("fixed", "homeo"), ("fixed", "stdp_homeo"), ("fixed", "stdp_ltp"),
                         ("fixed", "stdp_ltp_homeo")]:
        rs = [by.get((variant, s, pulse, arm)) for s in seeds]
        rs = [r for r in rs if r and (variant, r["seed"], pulse, "off") in by]
        if not rs:
            continue
        def st(key):
            v = np.array([r[key] for r in rs], float)
            return f"{np.median(v):5.2f} [{v.min():4.2f},{v.max():4.2f}]"
        xs, os_ = [], []
        for r in rs:
            off = by[(variant, r["seed"], pulse, "off")]
            o, c, _ = changes(r); o0, c0, _ = changes(off)
            dc = [c[k] - c0[k] for k in (0, 1)]
            k = int(np.argmax(dc))
            xs.append(dc[k]); os_.append(o[k] - o0[k])
        spec = sum(r["specific_end_lag"] for r in rs); ps = sum(r["pass"] for r in rs)
        print(f"{variant + '/' + arm:24s} {st('vs_off'):>18s} {st('rate_ratio_noenc'):>18s} {st('last_vs_first_min'):>18s} "
              f"{spec}/{len(rs):<3d} {ps}/{len(rs):<3d} {np.median(xs):6.2f} [{min(xs):5.2f},{max(xs):5.2f}] "
              f"{np.median(os_):6.2f} [{min(os_):5.2f},{max(os_):5.2f}]")
# off arm specificity with plasticity off, per seed (lag gains, first and last minute)
print("\noff arm lag gains [dec][enc], first -> last minute:")
for pulse in (20.0, 0.5):
    for s in seeds:
        r = by.get(("shipped", s, pulse, "off"))
        if r:
            g = np.array(r["gain_lag_60s"])
            print(f"  pulse {pulse} seed {s}: {np.round(g[0], 2).tolist()} -> {np.round(g[-1], 2).tolist()}")

# --- Pathways against the noise floor: last 3 minutes vs the same culture's off arm, SE from minutes 2-10 ---
print("\n=== pathway change vs off arm (last 3 min, lag gains), per cross / own pathway; sig = |diff| > 2 SE ===")
def last3(r):
    g = np.array(r["gain_lag_60s"])  # [w, dec, enc]
    return g[-3:].mean(0), g[1:].std(0, ddof=1)
for pulse in (20.0, 0.5):
    print(f"-- pulse {pulse} ms")
    for variant, arm in [("shipped", "stdp"), ("shipped", "homeo"), ("shipped", "stdp_homeo"), ("fixed", "stdp"),
                         ("fixed", "homeo"), ("fixed", "stdp_homeo"), ("fixed", "stdp_ltp"), ("fixed", "stdp_ltp_homeo")]:
        cu, cd, ou, od, cdiffs, odiffs = 0, 0, 0, 0, [], []
        for s in seeds:
            r, off = by.get((variant, s, pulse, arm)), by.get((variant, s, pulse, "off"))
            if not r or not off:
                continue
            m, sd = last3(r); m0, sd0 = last3(off)
            se = np.sqrt(sd ** 2 / 3 + sd0 ** 2 / 3)
            for k in (0, 1):
                dc, do = m[k, 1 - k] - m0[k, 1 - k], m[k, k] - m0[k, k]
                cdiffs.append(dc); odiffs.append(do)
                cu += dc > 2 * se[k, 1 - k]; cd += dc < -2 * se[k, 1 - k]
                ou += do > 2 * se[k, k]; od += do < -2 * se[k, k]
        n = len(cdiffs)
        if n:
            print(f"  {variant + '/' + arm:22s} cross: median {np.median(cdiffs):+5.2f}, up {cu}/{n}, down {cd}/{n} | "
                  f"own: median {np.median(odiffs):+5.2f}, up {ou}/{n}, down {od}/{n}")
# --- Repeatability: seed 42 rerun vs the original run (same code path, GPU atomics only) ---
print("\n=== seed 42 rerun vs original run ===")
for v in ("shipped", "fixed"):
    orig = {r["arm"]: r for r in json.load(open(Path(root).parent / v / f"drift_{v}.json"))}
    for arm in ("off", "stdp", "homeo", "stdp_homeo"):
        a, b = orig[arm], by[(v, 42, 20.0, arm)]
        print(f"  {v}/{arm:11s} rate_ratio {a['rate_ratio']:.2f} vs {b['rate_ratio']:.2f}; noenc {a['rate_ratio_noenc']:.2f} vs "
              f"{b['rate_ratio_noenc']:.2f}; w {a['w_exc_mean']:.4f} vs {b['w_exc_mean']:.4f}")
