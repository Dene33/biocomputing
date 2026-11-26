from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalspark_data as FD  # noqa: E402

P = dict(art_uv=300.0, post_ms=(14.0, 60.0), pre_ms=(-46.0, 0.0), live_hz=0.1, min_gain=0.1, resp_se=3.0,
         moved_se=2.0, min_pairs=20, hold_pts=10.0, drift_pts=25.0, session_gap_s=300.0, min_events=1000,
         series_gap_s=4 * 3600.0, block_gap_s=60.0, series_min_stims=200, series_min_blocks=3, series_min_isi_ms=100.0,
         delivered_ratio=2.0)


def find_mapping_session(ev: dict) -> np.ndarray | None:
    for a, b in FD.stim_sessions(ev["t"], P["session_gap_s"]):
        idx = np.arange(a, b + 1)
        if len(idx) < P["min_events"]:
            continue
        ok = True
        for i in idx:
            els = [x for x in range(32) if (int(ev["mask"][i]) >> x) & 1]
            if len(els) != 4 or sorted(x // 8 for x in els) != [0, 1, 2, 3] or len({x % 8 for x in els}) != 1:
                ok = False
                break
        if ok:
            return idx
    return None


def cycles_of(local: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(np.diff(local) < 0)]


def mean_se(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))) if n > 1 else (float("nan"), float("nan"))


def pair_tests(d: np.ndarray, local: np.ndarray, cyc: np.ndarray, pairs, groups, p=P) -> dict:
    rows = []
    for c1, c2 in pairs:
        for g in groups:
            for s in range(8):
                k1 = np.flatnonzero((cyc == c1) & (local == s))
                k2 = np.flatnonzero((cyc == c2) & (local == s))
                if len(k1) < 4 or len(k2) < 4:
                    continue
                for r in range(8):
                    if r == s:
                        continue
                    col = 8 * g + r
                    (g1, se1), (g2, se2) = mean_se(d[k1, col]), mean_se(d[k2, col])
                    m, se_m = (g1 + g2) / 2, np.hypot(se1, se2) / 2
                    if not (m >= p["min_gain"] and m > p["resp_se"] * se_m):
                        continue
                    se_d = np.hypot(se1, se2)
                    null = []
                    for k in (k1, k2):
                        (ga, sa), (gb, sb) = mean_se(d[k[0::2], col]), mean_se(d[k[1::2], col])
                        null.append(bool(abs(ga - gb) > p["moved_se"] * np.hypot(sa, sb)))
                    rows.append({"pair": [int(c1), int(c2)], "group": g, "s": s, "r": r, "g1": round(g1, 4),
                                 "g2": round(g2, 4), "se1": round(se1, 4), "se2": round(se2, 4),
                                 "gain": round(m, 4), "two_se": round(p["moved_se"] * se_d, 4),
                                 "moved": bool(abs(g2 - g1) > p["moved_se"] * se_d), "null_moved": null})
    n = len(rows)
    share = float(np.mean([x["moved"] for x in rows])) if n else float("nan")
    null = float(np.mean([y for x in rows for y in x["null_moved"]])) if n else float("nan")
    med_2se = float(np.median([x["two_se"] for x in rows])) if n else float("nan")
    med_gain = float(np.median([x["gain"] for x in rows])) if n else float("nan")
    return {"rows": rows, "n": n, "share_moved": share, "null_share": null, "median_two_se": med_2se,
            "median_gain": med_gain, "power": bool(n and med_2se <= med_gain)}


def reading(t: dict, p=P) -> str:
    if t["n"] < p["min_pairs"] or not t["power"]:
        return "inconclusive" + (" (too few responding pathway-pairs)" if t["n"] < p["min_pairs"] else " (no power)")
    excess = 100 * (t["share_moved"] - t["null_share"])
    if excess <= p["hold_pts"]:
        return "real pathways hold over ~5 minutes"
    if excess >= p["drift_pts"]:
        return "real pathways drift"
    return "in between"


def find_series(ev: dict, t: np.ndarray, p=P) -> list[dict]:
    single = [i for i in range(len(ev["t"])) if ev["n"][i] == 1 and ev["setting"][i] is not None]
    by = {}
    for i in single:
        by.setdefault((int(ev["mask"][i]).bit_length() - 1, ev["setting"][i]), []).append(i)
    out = []
    for (el, s), ii in sorted(by.items(), key=lambda kv: kv[1][0]):
        ii = np.array(ii)
        for a, b in FD.stim_sessions(ev["t"][ii], p["series_gap_s"]):
            jj = ii[a:b + 1]
            tt = ev["t"][jj]
            blk = np.r_[0, np.cumsum(np.diff(tt) > p["block_gap_s"] * FD.S)]
            within = np.diff(tt)[np.diff(blk) == 0]
            if (len(jj) < p["series_min_stims"] or blk.max() + 1 < p["series_min_blocks"] or len(within) == 0
                    or np.median(within) / FD.MS < p["series_min_isi_ms"]):
                continue
            out.append({"electrode": el, "setting": s, "idx": jj, "block": blk})
    return out


def delivery_ratio(t: np.ndarray, times: np.ndarray) -> float:
    after = sum(np.searchsorted(t, u + 10 * FD.MS) - np.searchsorted(t, u) for u in times) / 10.0
    before = sum(np.searchsorted(t, u) - np.searchsorted(t, u - 100 * FD.MS) for u in times) / 100.0
    return float(after / max(before, 1e-9))


def series_test(d: np.ndarray, pre: np.ndarray, blk: np.ndarray, el: int, p=P) -> dict:
    g = el // 8
    outs = [8 * g + r for r in range(8) if 8 * g + r != el]
    win_s = (p["pre_ms"][1] - p["pre_ms"][0]) / 1000
    first, last = blk == 0, blk == blk.max()
    base = {"first_block_hz": round(float(pre[first][:, 8 * g:8 * g + 8].mean() / win_s), 3),
            "last_block_hz": round(float(pre[last][:, 8 * g:8 * g + 8].mean() / win_s), 3)}
    rows = []
    for col in outs:
        m, se = mean_se(d[:, col])
        if not (m >= p["min_gain"] and m > p["resp_se"] * se):
            continue
        (g1, s1), (g2, s2) = mean_se(d[first, col]), mean_se(d[last, col])
        null = []
        for k in (np.flatnonzero(first), np.flatnonzero(last)):
            (ga, sa), (gb, sb) = mean_se(d[k[0::2], col]), mean_se(d[k[1::2], col])
            null.append(bool(abs(ga - gb) > p["moved_se"] * np.hypot(sa, sb)))
        rows.append({"output": int(col), "gain": round(m, 3), "first": round(g1, 3), "last": round(g2, 3),
                     "two_se": round(p["moved_se"] * float(np.hypot(s1, s2)), 3),
                     "moved": bool(abs(g2 - g1) > p["moved_se"] * np.hypot(s1, s2)), "null_moved": null,
                     "blocks": [round(float(d[blk == b, col].mean()), 3) for b in range(blk.max() + 1)]})
    n = len(rows)
    power = bool(n and np.median([x["two_se"] for x in rows]) <= 0.5 * np.median([x["gain"] for x in rows]))
    moved = [x for x in rows if x["moved"]]
    if n < 2 or not power:
        read = "inconclusive" + (" (fewer than 2 responding outputs)" if n < 2 else " (no power)")
    elif not moved:
        read = "the pathways hold under their own repeated stimulation"
    elif 2 * len(moved) >= n and len({np.sign(x["last"] - x["first"]) for x in moved}) == 1:
        read = "they drift " + ("up (potentiation)" if moved[0]["last"] > moved[0]["first"] else "down")
    else:
        read = "in between"
    return {"rows": rows, "responding": n, "moved": len(moved), "power": power,
            "null_share": round(float(np.mean([y for x in rows for y in x["null_moved"]])), 3) if n else None,
            "group_baseline": base, "reading": read}


def map_part(rec: dict, ev: dict) -> dict | None:
    idx = find_mapping_session(ev)
    if idx is None:
        return None
    times = ev["t"][idx]
    first_el = np.array([min(x for x in range(32) if (int(ev["mask"][i]) >> x) & 1) for i in idx])
    local = first_el % 8
    cyc = cycles_of(local)
    setting = [ev["setting"][i] for i in idx]
    cyc_setting = {}
    for c in np.unique(cyc):
        ss = {setting[i] for i in np.flatnonzero(cyc == c)}
        cyc_setting[int(c)] = ss.pop() if len(ss) == 1 else None
    by_setting = {}
    for c, s in cyc_setting.items():
        if s is not None:
            by_setting.setdefault(s, []).append(c)
    pairs = sorted((cs[i], cs[i + 1]) for cs in by_setting.values() for i in range(len(cs) - 1))
    pre = FD.window_counts(rec["t"], rec["e"], rec["amp"], times, *P["pre_ms"], P["art_uv"])
    post = FD.window_counts(rec["t"], rec["e"], rec["amp"], times, *P["post_ms"], P["art_uv"])
    d = post - pre
    base_hz = pre.sum(0) / len(times) / ((P["pre_ms"][1] - P["pre_ms"][0]) / 1000)
    group_hz = [float(base_hz[8 * g:8 * g + 8].mean()) for g in range(4)]
    live = [g for g in range(4) if group_hz[g] >= P["live_hz"]]
    silent = [g for g in range(4) if g not in live]
    gains, resp = {}, {}
    for g in range(4):
        G = np.zeros((8, 8))
        R = np.zeros((8, 8), bool)
        for s_ in range(8):
            k = local == s_
            for r in range(8):
                if r != s_:
                    m, se = mean_se(d[k, 8 * g + r])
                    G[s_, r], R[s_, r] = m, m >= P["min_gain"] and m > P["resp_se"] * se
        gains[g], resp[g] = G, R
    if any(resp[g].any() for g in silent):
        print(f"GUARD FAIL [real_pathways]: responding pathways in silent groups {silent}", flush=True)
        raise SystemExit(3)
    keys = ev["keys"]
    amp_of = {c: s[keys.index("a1")] for c, s in cyc_setting.items() if s is not None}
    dose = {}
    for amp_ in sorted(set(amp_of.values())):
        cs = [c for c, v in amp_of.items() if v == amp_]
        vals = [d[np.isin(cyc, cs) & (local == s_), 8 * g + r].mean() for g in live for s_ in range(8)
                for r in range(8) if resp[g][s_, r]]
        dose[str(amp_)] = round(float(np.median(vals)), 3) if vals else None
    res = {"session": {"first_day": round((times[0] - int(rec["t"][0])) / FD.S / 86400, 2),
                       "minutes": round((times[-1] - times[0]) / FD.S / 60, 1), "stimulations": int(len(idx)),
                       "cycles": int(cyc.max() + 1), "distinct_settings": len(by_setting)},
           "pairs": [list(x) for x in pairs], "group_baseline_hz": [round(x, 3) for x in group_hz], "live_groups": live,
           "map": {f"group{g}": {"responding": int(resp[g].sum()), "of": 56,
                                 "median_responding_gain": (round(float(np.median(gains[g][resp[g]])), 3)
                                                            if resp[g].any() else None),
                                 "gains": np.round(gains[g], 2).tolist(),
                                 "strongest_input_per_output": [int(np.argmax(np.where(np.eye(8, dtype=bool)[:, r],
                                                                                        -np.inf, gains[g][:, r])))
                                                                for r in range(8)]} for g in live},
           "median_responding_gain_by_amplitude_uA": dose}
    if pairs:
        test = pair_tests(d, local, cyc, pairs, live)
        res |= {"pair_test": {k: v for k, v in test.items() if k != "rows"}, "pair_reading": reading(test)}
    else:
        res["pair_reading"] = "not possible: no setting recurs in the mapping session"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", default="fs369")
    ap.add_argument("--out", default="outputs/real_pathways")
    a = ap.parse_args()
    t_start = time.time()
    rec = FD.load_record(FD.FS[a.fs], a.fs)
    ev = FD.stim_events(rec["st"])
    keys = ev["keys"]
    res = {"fs": a.fs, "params": P, "mapping_session": map_part(rec, ev), "series": []}
    for sr in find_series(ev, rec["t"]):
        times = ev["t"][sr["idx"]]
        ratio = delivery_ratio(rec["t"], times)
        pre = FD.window_counts(rec["t"], rec["e"], rec["amp"], times, *P["pre_ms"], P["art_uv"])
        post = FD.window_counts(rec["t"], rec["e"], rec["amp"], times, *P["post_ms"], P["art_uv"])
        g = sr["electrode"] // 8
        win_s = (P["pre_ms"][1] - P["pre_ms"][0]) / 1000
        group_hz = float(pre[:, 8 * g:8 * g + 8].mean() / win_s)
        blk = sr["block"]
        row = {"electrode": sr["electrode"], "group": g,
               "setting": {k: v for k, v in zip(keys, sr["setting"]) if k in ("a1", "a2", "d1", "d2", "nb_stim_pulse",
                                                                               "pulse_train_period", "stim_polarity",
                                                                               "stim_shape")},
               "stimulations": int(len(times)), "blocks": int(blk.max() + 1),
               "block_sizes": np.bincount(blk).tolist(),
               "block_start_h": [round(float((times[blk == b][0] - times[0]) / FD.S / 3600), 2)
                                 for b in range(blk.max() + 1)],
               "day": round((times[0] - int(rec["t"][0])) / FD.S / 86400, 2),
               "delivery_ratio": round(ratio, 2), "delivered": ratio >= P["delivered_ratio"],
               "group_baseline_hz": round(group_hz, 3), "live": group_hz >= P["live_hz"]}
        if row["delivered"] and row["live"]:
            row["test"] = series_test(post - pre, pre, blk, sr["electrode"])
        res["series"].append(row)
    res["seconds"] = round(time.time() - t_start, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"real_pathways_{a.fs}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("params",)}, indent=1))


if __name__ == "__main__":
    main()
