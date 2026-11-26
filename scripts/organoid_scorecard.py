from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_vestibular as clv  # noqa: E402
import finalspark_data as FD  # noqa: E402
import public_data_targets as PDT  # noqa: E402

from biocompute.runguard import job_key  # noqa: E402

ROOT = FD.ROOT
GATED = ("rate_hz_median", "burst_rate_per_min", "burst_duration_ms_median", "ibi_cv", "spikes_in_bursts")
REPORTED = ("n_active", "rate_hz_mean", "ibi_s_median", "recruitment_median")
P = dict(art_uv=300.0, window_s=600.0, pre_stim_s=60.0, min_windows=6, min_active=3, artifact_ms=14.0,
         burst_window_ms=200.0, chance_s=3600.0)
MODEL_REC = ROOT / "outputs/realism_scorecard/rec"
MODEL_STEM = "live_s{seed}_ed5a2d31"
CULTURES = (0, 1, 2, 3, 4)
PROTO = dict(warm_s=10.0, isi_ms=200.0, stims=38, cycles=5, amplitude=1.0)
BLOCK_XY = [(x, y) for y in (-100.0, 100.0) for x in (-300.0, -100.0, 100.0, 300.0)]


def real_windows(rec: dict, p=P) -> list[dict]:
    t, e, amp = rec["t"], rec["e"], rec["amp"]
    keep = np.abs(amp) <= p["art_uv"]
    t, e = t[keep], e[keep]
    ut = np.unique(rec["st"]["time_of_stim"])
    W = int(p["window_s"] * FD.S)
    t0, rows = int(t[0]), []
    for a in range(t0, int(t[-1]) - W + 1, W):
        k0, k1 = np.searchsorted(ut, a - int(p["pre_stim_s"] * FD.S)), np.searchsorted(ut, a + W)
        if k1 > k0:
            continue
        lo, hi = np.searchsorted(t, a), np.searchsorted(t, a + W)
        for g in range(4):
            m = (e[lo:hi] // 8) == g
            if not m.any():
                continue
            st = PDT.spike_stats((t[lo:hi][m] - a) / FD.S, e[lo:hi][m] - 8 * g, p["window_s"], 8)
            rows.append(st | {"group": g, "day": int((a - t0) // (86400 * FD.S))})
    return rows


def group_days(rows: list[dict], p=P) -> list[dict]:
    out = []
    for g, d in sorted({(r["group"], r["day"]) for r in rows}):
        R = [r for r in rows if r["group"] == g and r["day"] == d]
        if len(R) < p["min_windows"] or np.median([r["n_active"] for r in R]) < p["min_active"]:
            continue
        v = {"group": g, "day": d, "windows": len(R)}
        for k in GATED + REPORTED:
            x = [r[k] for r in R if r.get(k) is not None]
            v[k] = float(np.median(x)) if x else None
        out.append(v)
    return out


def targets(gd: list[dict]) -> dict:
    out = {}
    for k in GATED + REPORTED:
        x = [r[k] for r in gd if r[k] is not None]
        if x:
            lo, med, hi = np.percentile(x, [5, 50, 95])
            out[k] = {"median": round(float(med), 3), "p5_p95": [round(float(lo), 3), round(float(hi), 3)],
                      "n": len(x)}
    return out


def blocks_of(xy: np.ndarray, pitch: float = 200.0) -> list[list[int]]:
    ij = {(int(round(x / pitch * 2)), int(round(y / pitch * 2))): c for c, (x, y) in enumerate(xy)}
    xs, ys = sorted({a for a, _ in ij}), sorted({b for _, b in ij})
    out = []
    for w, h in ((4, 2), (2, 4)):
        for i in range(len(xs) - w + 1):
            for j in range(len(ys) - h + 1):
                cells = [(xs[i + a], ys[j + b]) for a in range(w) for b in range(h)]
                if all(c in ij for c in cells):
                    out.append(sorted(ij[c] for c in cells))
    return out


def model_blocks(seed: int) -> dict:
    with np.load(MODEL_REC / f"{MODEL_STEM.format(seed=seed)}.npz") as z:
        t, ch, xy, dur = z["e_t_s"], z["e_ch"].astype(int), z["electrode_xy"], float(z["record_s"])
    per = []
    for blk in blocks_of(xy):
        lut = -np.ones(len(xy), int)
        lut[blk] = np.arange(8)
        m = lut[ch] >= 0
        per.append(PDT.spike_stats(t[m], lut[ch[m]], dur, 8))
    med = {}
    for k in GATED + REPORTED:
        x = [s[k] for s in per if s.get(k) is not None]
        med[k] = float(np.median(x)) if x else None
    return {"seed": seed, "blocks": len(per), **med}


def score(model: list[dict], tg: dict) -> dict:
    card = {}
    for k in GATED:
        vals = [m[k] for m in model if m[k] is not None]
        med = float(np.median(vals)) if vals else None
        lo, hi = tg[k]["p5_p95"] if k in tg else (None, None)
        card[k] = {"model_per_culture": [round(v, 3) for v in vals], "model_median": None if med is None else round(med, 3),
                   "real_median": tg.get(k, {}).get("median"), "real_p5_p95": [lo, hi],
                   "in_range": med is not None and lo is not None and lo <= med <= hi,
                   "cultures_in_range": int(sum(lo <= v <= hi for v in vals)) if lo is not None else 0}
    return {"scorecard": card, "in_range": [k for k, v in card.items() if v["in_range"]],
            "out_of_range": [k for k, v in card.items() if not v["in_range"]]}


def real_map_bursts(rec: dict, p=P) -> dict:
    import real_pathways as RP

    ev = FD.stim_events(rec["st"])
    idx = RP.find_mapping_session(ev)
    times = ev["t"][idx]
    t, e, amp = rec["t"], rec["e"], rec["amp"]
    a0 = int(times[0] - p["chance_s"] * FD.S)
    lo, hi = np.searchsorted(t, a0), np.searchsorted(t, int(times[-1]) + 2 * FD.S)
    t, e, amp = t[lo:hi], e[lo:hi], amp[lo:hi]
    k = np.searchsorted(times, t, side="right") - 1
    art = (k >= 0) & (t - times[np.maximum(k, 0)] < p["artifact_ms"] * FD.MS)
    keep = (np.abs(amp) <= p["art_uv"]) & ~art
    t, e = t[keep], e[keep]
    out = {}
    for g in range(4):
        m = (e // 8) == g
        if m.sum() < 100:
            continue
        dur = (int(times[-1]) + 2 * FD.S - a0) / FD.S
        _, _, bursts, _ = PDT.network_bursts((t[m] - a0) / FD.S, e[m] - 8 * g, dur, 8)
        on = np.array(sorted(b[0] for b in bursts)) * FD.S + a0 if bursts else np.array([], np.int64)
        before = on[on < times[0]]
        rate = len(before) / p["chance_s"]
        chance = 1 - np.exp(-rate * p["burst_window_ms"] / 1000)
        hit = np.array([((on >= u) & (on < u + p["burst_window_ms"] * FD.MS)).any() for u in times])
        out[f"group{g}"] = {"bursts_per_min_before": round(rate * 60, 3), "chance_pct": round(100 * chance, 2),
                            "after_stim_pct": round(100 * float(hit.mean()), 2),
                            "excess_pct": round(100 * (float(hit.mean()) - chance), 2)}
    return out


def map_job(job):
    import torch

    from biocompute.culture.network import make_config
    from biocompute.device import SimulatedBiocomputer
    from biocompute.mea.electrodes import make_mea_config
    from biocompute.runguard import Guard

    seed, out, pz = job
    stem = job_key(f"map_s{seed}", pz)
    if clv.stop_requested():
        raise clv.Stopped(f"{stem} before start")
    g = Guard(f"map_s{seed}", out)
    dev = SimulatedBiocomputer(make_config("gpu_20k", seed=seed), make_mea_config("mcs60", seed=seed))
    c = dev.culture
    g.check(c.plasticity is None, "plasticity is on")
    w_ref = c.w.clone()
    xy = np.asarray(dev.mea.xy, float)
    block = [int(np.argmin(np.hypot(*(xy - np.array(p)).T))) for p in BLOCK_XY]
    g.check(all(np.hypot(*(xy[b] - np.array(p))) < 1 for b, p in zip(block, BLOCK_XY)), "block electrodes not found")
    for _ in range(int(round(pz["warm_s"] * 10))):
        dev.step(100.0)
    n = pz["cycles"] * 8 * pz["stims"]
    st_t, st_in, et, ec = np.zeros(n), np.zeros(n, int), [], []
    i = 0
    for cyc in range(pz["cycles"]):
        for s_ in range(8):
            for _ in range(pz["stims"]):
                if clv.stop_requested():
                    raise clv.Stopped(f"{stem} at stimulation {i}/{n} (redone on resume)")
                t0 = float(c.t_ms)
                dev.stimulate(np.array([block[s_]]), np.array([0.0]), np.array([pz["amplitude"]]))
                b = dev.step(pz["isi_ms"])
                st_t[i], st_in[i] = t0, s_
                et.append(t0 + b.times_ms)
                ec.append(b.channels.astype(np.int16))
                i += 1
                if g.due():
                    hz = sum(len(x) for x in ec[-50:]) / (min(len(ec), 50) * pz["isi_ms"] / 1000) / dev.n_channels
                    g.check(bool(torch.equal(c.w, w_ref)), "weights moved with plasticity off")
                    g.culture(c, rate_hz=max(hz, 1e-9))
                    g.beat(i, n, hz=round(hz, 2))
    g.check(bool(torch.equal(c.w, w_ref)), "weights moved with plasticity off")
    p = out / "rec" / f"{stem}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".tmp"), "wb") as f:
        np.savez(f, e_t_ms=np.concatenate(et), e_ch=np.concatenate(ec), stim_t_ms=st_t, stim_input=st_in,
                 block=np.array(block), seed=np.array(seed))
    p.with_suffix(".tmp").replace(p)
    g.beat(n, n, force=True)
    del dev, c
    torch.cuda.empty_cache()
    return {"seed": seed, "rec": str(p)}


def _map_safe(job):
    try:
        return map_job(job)
    except clv.Stopped as e:
        return {"stopped": str(e)}


def map_measures(e_t_ms, e_ch, stim_t_ms, stim_input, block, spont: tuple | None, p=P) -> dict:
    import real_pathways as RP

    lut = -np.ones(int(max(e_ch.max(initial=0), max(block))) + 1, int)
    lut[block] = np.arange(8)
    m = (e_ch < len(lut)) & (lut[np.minimum(e_ch, len(lut) - 1)] >= 0)
    t = np.round(e_t_ms[m] * FD.MS).astype(np.int64)
    e = lut[e_ch[m]].astype(np.int64)
    o = np.argsort(t, kind="stable")
    t, e = t[o], e[o]
    amp = np.zeros(len(t))
    times = np.round(stim_t_ms * FD.MS).astype(np.int64)
    d = (FD.window_counts(t, e, amp, times, *RP.P["post_ms"], p["art_uv"])
         - FD.window_counts(t, e, amp, times, *RP.P["pre_ms"], p["art_uv"]))[:, :8]
    resp, gains = np.zeros((8, 8), bool), np.zeros((8, 8))
    for s_ in range(8):
        k = stim_input == s_
        for r in range(8):
            if r != s_:
                mu, se = RP.mean_se(d[k, r])
                gains[s_, r], resp[s_, r] = mu, mu >= RP.P["min_gain"] and mu > RP.P["resp_se"] * se
    k = np.searchsorted(times, t, side="right") - 1
    art = (k >= 0) & (t - times[np.maximum(k, 0)] < p["artifact_ms"] * FD.MS)
    tt, ee = t[~art], e[~art]
    dur = (tt.max() - times[0]) / FD.S + 1 if len(tt) else 1.0
    _, _, bursts, _ = PDT.network_bursts((tt - times[0]) / FD.S, ee, dur, 8)
    on = np.array(sorted(b[0] for b in bursts)) * FD.S + times[0] if bursts else np.array([], np.int64)
    hit = np.array([((on >= u) & (on < u + p["burst_window_ms"] * FD.MS)).any() for u in times])
    chance = None
    if spont is not None:
        st_, sc_, sdur = spont
        lut2 = -np.ones(int(max(sc_.max(initial=0), max(block))) + 1, int)
        lut2[block] = np.arange(8)
        m2 = lut2[np.minimum(sc_, len(lut2) - 1)] >= 0
        _, _, b2, _ = PDT.network_bursts(st_[m2], lut2[sc_[m2]], sdur, 8)
        rate = len(b2) / sdur
        chance = 1 - np.exp(-rate * p["burst_window_ms"] / 1000)
    return {"responding": int(resp.sum()), "responding_share": round(float(resp.sum() / 56), 3),
            "median_responding_gain": round(float(np.median(gains[resp])), 3) if resp.any() else None,
            "after_stim_pct": round(100 * float(hit.mean()), 2),
            "chance_pct": None if chance is None else round(100 * chance, 2),
            "excess_pct": None if chance is None else round(100 * (float(hit.mean()) - chance), 2),
            "strongest_input_per_output": [int(np.argmax(np.where(np.eye(8, dtype=bool)[:, r], -np.inf, gains[:, r])))
                                           for r in range(8)]}


def like_real(model: list[dict], real: dict) -> dict:
    rs, rg, rx = real["responding_share"], real["median_responding_gain"], real["excess_pct"]
    def med(k):
        v = [m[k] for m in model if m[k] is not None]
        return float(np.median(v)) if v else None

    calls = {"responding_share": [med("responding_share"), [min(rs) / 2, 2 * max(rs)]],
             "median_responding_gain": [med("median_responding_gain"), [min(rg) / 2, 2 * max(rg)]],
             "excess_pct": [med("excess_pct"), [None, max(2 * max(rx), max(rx) + 2)]]}
    return {k: {"model_median": None if v is None else round(v, 3), "band": b,
                "like_real": v is not None and (b[0] is None or v >= b[0]) and v <= b[1]}
            for k, (v, b) in calls.items()}


def part_b(out: Path, workers: int, smoke: bool) -> dict:
    import real_pathways as RP  # noqa: F401  (the real map's code)

    pz = dict(PROTO)
    seeds = CULTURES
    if smoke:
        pz |= {"warm_s": 1.0, "stims": 3, "cycles": 1}
        seeds = CULTURES[:1]
    jobs = [(s_, out, pz) for s_ in seeds]
    todo = [j for j in jobs if not (out / "rec" / f"{job_key(f'map_s{j[0]}', pz)}.npz").exists()]
    if todo:
        threads = max(1, min(4, (os.cpu_count() or 4) // max(workers, 1)))
        clv.pool_map(_map_safe, todo, min(workers, len(todo)), threads)
    model = []
    for s_ in seeds:
        with np.load(out / "rec" / f"{job_key(f'map_s{s_}', pz)}.npz") as z:
            rec = {k: z[k] for k in z.files}
        with np.load(MODEL_REC / f"{MODEL_STEM.format(seed=s_)}.npz") as z:
            spont = (z["e_t_s"], z["e_ch"].astype(int), float(z["record_s"]))
        model.append({"seed": s_} | map_measures(rec["e_t_ms"], rec["e_ch"].astype(int), rec["stim_t_ms"],
                                                 rec["stim_input"], list(rec["block"]), spont))
    rp = json.loads((ROOT / "outputs/real_pathways/real_pathways_fs369.json").read_text())["mapping_session"]["map"]
    sp = json.loads((out.parent / "spontaneous.json" if smoke else out / "spontaneous.json").read_text())
    mb = sp["real"]["fs369"]["map_bursts"]
    live = sorted(rp)
    real = {"groups": live, "responding_share": [rp[g]["responding"] / 56 for g in live],
            "median_responding_gain": [rp[g]["median_responding_gain"] for g in live],
            "excess_pct": [mb[g]["excess_pct"] for g in live]}
    return {"protocol": pz, "real": real, "model": model, "reading": like_real(model, real)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/organoid_scorecard")
    ap.add_argument("--evoked", action="store_true", help="part B (GPU)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="part B at tiny scale into <out>/smoke")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.evoked:
        import faulthandler

        faulthandler.register(signal.SIGUSR1)
        if a.smoke:
            out = out / "smoke"
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
        print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "part": "B", "smoke": a.smoke}), flush=True)
        res = part_b(out, a.workers, a.smoke)
        (out / "evoked.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1), flush=True)
        return
    t_start = time.time()
    real, gd_all = {}, []
    for fs in FD.FS:
        rec = FD.load_record(FD.FS[fs], fs)
        gd = group_days(real_windows(rec))
        gd_all += [x | {"fs": fs} for x in gd]
        real[fs] = {"group_days": len(gd), "targets": targets(gd)}
        if fs == "fs369":
            real[fs]["map_bursts"] = real_map_bursts(rec)
        del rec
    tg = targets(gd_all)
    model = [model_blocks(s) for s in CULTURES]
    res = {"params": P, "real": real, "targets_pooled": tg, "group_days": gd_all, "model": model,
           **score(model, tg), "seconds": round(time.time() - t_start, 1)}
    (out / "spontaneous.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("group_days", "params")}, indent=1))


if __name__ == "__main__":
    main()
