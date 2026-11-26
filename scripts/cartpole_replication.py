import argparse
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np

from biocompute.runguard import job_key

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("coach_probe", HERE / "coach_probe.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

STEP_S = 0.3
PROFICIENT_S = 20.85
LIMIT = math.radians(16.0)


class CartPole:
    def __init__(self, rng):
        self.s = rng.uniform(-0.05, 0.05, 4)

    @classmethod
    def resume(cls, s):
        env = cls.__new__(cls)
        env.s = np.array(s)
        return env

    def step(self, a):
        x, xd, th, thd = self.s
        g, mc, mp_, l, tau = 9.8, 1.0, 0.1, 0.5, 0.02
        f = 10.0 * float(np.clip(a, -1, 1))
        s, c = math.sin(th), math.cos(th)
        temp = (f + mp_ * l * thd ** 2 * s) / (mc + mp_)
        tha = (g * s - c * temp) / (l * (4 / 3 - mp_ * c * c / (mc + mp_)))
        xa = temp - mp_ * l * tha * c / (mc + mp_)
        self.s = np.array([x + tau * xd, xd + tau * xa, th + tau * thd, thd + tau * tha])
        return abs(self.s[2]) > LIMIT


def build(seed, culture=None):
    import torch

    from biocompute.culture.network import make_config
    from biocompute.culture.plasticity import STDP
    from biocompute.device import SimulatedBiocomputer
    from biocompute.mea.electrodes import make_mea_config

    torch.set_num_threads(1)
    dev = SimulatedBiocomputer(make_config("gpu_20k", seed=seed, **(culture or {})),
                               make_mea_config("grid256", seed=seed))
    dev.step(2000.0)
    c = dev.culture
    w_in0 = c.excitatory_input()
    return dev, STDP(), w_in0


def characterise(dev, reps=25, isi_ms=250.0, beat=None):
    bins = np.zeros((200, dev.n_channels))
    neuron = np.zeros(dev.culture.n)
    for i in range(200):
        b = dev.step(100.0)
        bins[i] = np.bincount(b.channels, minlength=dev.n_channels)
        neuron += dev.oracle_counts
    counts = bins.sum(0)
    c200 = bins.reshape(100, 2, -1).sum(1)
    rest_stats = np.stack([c200.mean(0), c200.std(0)], 1)
    alive = dev.mea.alive_channels
    cand = alive[np.argsort(counts[alive])[::-1][:32]]
    k = len(cand)
    first = np.zeros((k, k))
    totals = np.zeros((k, reps))
    for i, ch in enumerate(cand):
        for r in range(reps):
            dev.stimulate(np.array([ch]), np.array([0.0]), np.array([1.0]))
            b = dev.step(isi_ms)
            on = np.isin(b.channels, cand)
            idx = np.searchsorted(np.sort(cand), b.channels[on])
            order = np.argsort(cand)
            j = order[idx]
            t = b.times_ms[on]
            hit = np.zeros(k, bool)
            hit[j[t < 10.0]] = True
            first[i] += hit
            totals[i, r] = float((t < 200.0).sum())
        if beat is not None:
            beat(i + 1, k)
    first /= reps
    np.fill_diagonal(first, 0.0)
    med = np.median(totals)
    thr = med + 3 * np.median(np.abs(totals - med))
    burst = (totals > thr).mean(1)
    return cand, counts[cand], first, burst, rest_stats, neuron / 20.0


def configure(cand, rate, first, burst):
    k = len(cand)
    pairs = sorted(((first[a, b], a, b) for a in range(k) for b in range(k) if a != b and burst[a] <= 0.5),
                   reverse=True)[:120]
    best, bs = None, -1e9
    for (_, a, b), (_, c, d) in itertools.combinations(pairs, 2):
        if len({a, b, c, d}) < 4:
            continue
        s = first[a, b] + first[c, d] - first[a, d] - first[c, b] - 0.5 * (first[a, c] + first[c, a])
        if s > bs:
            best, bs = (a, b, c, d), s
    if best is None:
        return None
    a, b, c, d = best
    rest = [i for i in np.argsort(rate)[::-1] if i not in best][:8]
    return {"enc": [int(cand[a]), int(cand[c])], "dec": [int(cand[b]), int(cand[d])],
            "train": [int(cand[i]) for i in rest], "score": round(float(bs), 3),
            "c1_ab": round(float(first[a, b]), 3), "c1_cd": round(float(first[c, d]), 3),
            "c1_ad": round(float(first[a, d]), 3), "c1_cb": round(float(first[c, b]), 3),
            "burst_enc": [round(float(burst[a]), 2), round(float(burst[c]), 2)],
            "c1_mean": round(float(first.mean()), 3), "burst_mean": round(float(burst.mean()), 2)}


def schedule(dev, enc, th, last, log=None):
    hz = [(7 * (0.15 - math.sin(th))) ** 2, (7 * (0.15 + math.sin(th))) ** 2]
    ch, tt, n = [], [], [0, 0]
    now = dev.culture.t_ms
    for i, (e, f) in enumerate(zip(enc, hz)):
        nxt = max(last[e] + 1000.0 / f, now)
        while nxt < now + 200.0:
            ch.append(e)
            tt.append(nxt - now)
            last[e] = nxt
            nxt += 1000.0 / f
            n[i] += 1
    if ch:
        dev.stimulate(np.array(ch), np.array(tt), np.ones(len(ch)))
        if log is not None:
            log.append((now + np.array(tt, np.float64), np.array(ch, np.int16)))
    return n


def calibrate(dev, cfg, rng, seconds=120.0):
    c = dev.culture
    c.plasticity = None
    enc, dec = cfg["enc"], cfg["dec"]
    last = {e: -1e9 for e in enc}
    X, Y, S = [], [], []
    for i in range(int(seconds * 5)):
        if i % 5 == 0:
            th = math.radians(rng.uniform(-12.0, 12.0))
        n = schedule(dev, enc, th, last)
        b = dev.step(200.0)
        X.append(n)
        Y.append([(b.channels == dec[0]).sum(), (b.channels == dec[1]).sum()])
        S.append(math.sin(th))
    X, Y, S = np.array(X, float), np.array(Y, float), np.array(S)
    A = np.column_stack([np.ones(len(X)), X])
    gain = np.linalg.lstsq(A, Y, rcond=None)[0][1:].T
    diff = Y[:, 0] - Y[:, 1]
    rho = float(np.corrcoef(diff, -S)[0, 1])
    return {"dec_task": [[round(float(m), 3), round(float(v), 3)] for m, v in zip(Y.mean(0), Y.std(0))],
            "gain": np.round(gain, 2).tolist(), "reflex_r": round(rho, 3), "reflex_r2": round(rho * rho, 3),
            "pulses_per_read": round(float(X.sum(1).mean()), 2)}


STOP = {"requested": False, "phase": "idle"}


class Stopped(Exception):
    pass


def _on_term(signum, frame):
    STOP["requested"] = True
    if STOP["phase"] == "setup":
        raise Stopped("setup")


def _start_job():
    if STOP["requested"]:
        raise Stopped("not started")
    STOP["phase"] = "setup"


def _worker_init():
    signal.signal(signal.SIGTERM, _on_term)


class Checkpoint:
    def __init__(self, path, every_s=300.0):
        self.path, self.every_s, self.t = Path(path), every_s, time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        import torch

        return torch.load(self.path, weights_only=False) if self.path.exists() else None

    def save(self, dev, stdp, script):
        import torch

        tmp = self.path.with_suffix(".tmp")
        torch.save({"dev": dev.state(), "stdp": stdp.state(), "script": script}, tmp)
        os.replace(tmp, self.path)
        self.t = time.time()

    def tick(self, dev, stdp, script_fn):
        if STOP["requested"] or time.time() - self.t > self.every_s:
            self.save(dev, stdp, script_fn())
            if STOP["requested"]:
                raise Stopped(str(self.path))


def make_stdp(spec):
    from biocompute.culture.plasticity import STDP

    return STDP(a_plus=0.01 * spec["scale"], a_minus=0.012 * spec["scale"],
                multiplicative=spec["rule"] == "multiplicative", istdp_eta=spec["eta"],
                w_max_inh=spec["w_max_inh"] if spec["eta"] else None)


def spec_tag(spec):
    t = ("" if spec["scale"] == 1.0 else f"_stdp{spec['scale']:g}") + (
        "" if spec["rule"] == "additive" else f"_{spec['rule']}")
    return t + (f"_istdp{spec['eta']:g}" if spec["eta"] else "") + (
        f"_cap{spec['w_max_inh']:g}" if spec["eta"] and spec["w_max_inh"] != 30.0 else "") + (
        f"_homeo{spec['homeo']:g}" if spec.get("homeo") else "")


DRIFT_CONFIGS = {0: ([136, 155], [137, 139]), 1: ([129, 34], [113, 35]), 2: ([217, 100], [233, 116])}
B1_BAND = (2 / 3, 1.5)
GATE_MINUTES = 10


def long_enough(r):
    return r is not None and r["seconds"] >= 60 * GATE_MINUTES


def resting_rates(dev, seconds=20.0):
    c = dev.culture
    keep, c.plasticity = c.plasticity, None
    counts, det = np.zeros(c.n), 0
    for _ in range(int(seconds * 10)):
        det += len(dev.step(100.0).channels)
        counts += dev.oracle_counts
    c.plasticity = keep
    return counts / seconds, det / seconds / dev.n_channels


def gains(X, Y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, Y, rcond=None)[0][1:].T


def per10(x):
    k = max(len(x) // 50, 1) * 50 if len(x) >= 50 else len(x)
    return x[:k].reshape(-1, min(50, k), *x.shape[1:]).mean(1)


def gate_b1(rows, eta=1e-5):
    lo, hi = B1_BAND
    by = {(r.get("seed", 0), r["scale"], r.get("homeo", 0.0)): r for r in rows if not r["eta"]}
    seeds = sorted({k[0] for k in by})
    inb = lambda r: r is not None and lo <= r["rate_ratio"] <= hi  # noqa: E731
    out = {"band": [round(lo, 3), hi], "eta": eta, "per_culture": {}}
    for sd in seeds:
        row = {name: (by.get((sd, x, h)) or {}).get("rate_ratio") for name, x, h in
               (("off", 0.0, 0.0), ("scaling", 0.0, eta), ("stdp_scaling", 1.0, eta), ("stdp_old_step", 1.0, 0.0))}
        gated = [by.get((sd, 0.0, eta)), by.get((sd, 1.0, eta))]
        row["pass"] = all(inb(r) for r in gated)
        row["controls_ok"] = inb(by.get((sd, 0.0, 0.0))) and (by.get((sd, 1.0, 0.0)) or {}).get("rate_ratio", 0) > hi
        row["tested"] = all(long_enough(r) for r in gated)
        out["per_culture"][str(sd)] = row
    out["tested"] = len(seeds) >= 3 and all(v["tested"] for v in out["per_culture"].values())
    out["pass"] = out["tested"] and all(v["pass"] for v in out["per_culture"].values())
    return out


def pathway_change(r, off):
    ok = long_enough(r) and long_enough(off)
    g, g0 = np.array(r["gain_60s"]), np.array(off["gain_60s"])
    se = np.sqrt(g[1:].std(0, ddof=1) ** 2 / 3 + g0[1:].std(0, ddof=1) ** 2 / 3) if ok else np.nan
    diff = g[-3:].mean(0) - g0[-3:].mean(0)
    moved = np.abs(diff) > 2 * se
    return {"tested": ok, "own_diff": [round(float(diff[k, k]), 2) for k in (0, 1)],
            "cross_diff": [round(float(diff[k, 1 - k]), 2) for k in (0, 1)],
            "moved_own": int(moved[0, 0] + moved[1, 1]) if ok else None,
            "moved_cross": int(moved[0, 1] + moved[1, 0]) if ok else None}


def pathways_vs_off(rows, eta=1e-5, max_moved=1):
    by = {(r.get("seed", 0), r["scale"], r.get("homeo", 0.0)): r for r in rows if not r["eta"]}
    out = {"per_culture": {}}
    for sd in sorted({k[0] for k in by}):
        off = by.get((sd, 0.0, 0.0))
        if off is None:
            continue
        out["per_culture"][str(sd)] = {f"x{x:g}_homeo{h:g}": pathway_change(r, off)
                                       for (s2, x, h), r in sorted(by.items()) if s2 == sd and (x or h)}
    key = f"x1_homeo{eta:g}"
    arms = [c.get(key) for c in out["per_culture"].values()]
    rates = [by.get((int(sd), 1.0, eta), {}).get("rate_ratio") for sd in out["per_culture"]]
    lo, hi = B1_BAND
    tested = len(arms) >= 3 and all(a and a["tested"] for a in arms)
    out["moved_at_eta"] = sum(a["moved_own"] + a["moved_cross"] for a in arms) if tested else None
    out["reading"] = (tested and all(r is not None and lo <= r <= hi for r in rates)
                      and out["moved_at_eta"] <= max_moved)
    return out


def drift_summary(rows, rule):
    return {"stdp_rule": rule} | ({"gate_B1": gate_b1(rows)} if rule == "additive" else {}) | {
        "pathways": pathways_vs_off(rows)}


def drift(job):
    import torch

    from biocompute.runguard import Guard

    from biocompute.culture.plasticity import Chain, SynapticScaling

    arm, key, spec, seconds, out, opts = job
    seed = spec.get("seed", 0)
    dump, ref = opts.get("dump", False), opts.get("ref")
    _start_job()
    g = Guard(f"drift_{arm}", out)
    enc, dec = (spec["enc"], spec["dec"]) if "enc" in spec else DRIFT_CONFIGS[seed]
    ck = Checkpoint(out / "ckpt" / f"{key}.pt")
    rng = np.random.default_rng(1)
    dev, _, w_in0 = build(seed, spec.get("culture"))
    stdp = make_stdp(spec)
    c = dev.culture
    stdp_on = spec["scale"] > 0 or spec["eta"] > 0
    homeo = SynapticScaling(eta=spec["homeo"]) if spec.get("homeo") else None
    rule = stdp if homeo is None else Chain(stdp, homeo) if stdp_on else homeo
    plastic = stdp_on or homeo is not None
    ee = c.syn_exc & c.excitatory[c.post]
    saved = ck.load()
    if saved:
        dev.load_state(saved["dev"])
        rule.load_state(c, saved["stdp"])
        S = saved["script"]
        rng.bit_generator.state = S.pop("rng")
    else:
        rates, rest_hz = resting_rates(dev, seconds=opts["rest_s"])
        target = np.maximum(rates, spec["floor"])
        if spec["eta"]:
            stdp.set_targets(c, target)
        if homeo is not None:
            homeo.set_targets(c, target)
        S = {"i": 0, "X": [], "Y": [], "N": [], "last": {e: -1e9 for e in enc}, "th": 0.0, "target_hz": target,
             "neuron_rest_hz": round(float(rates.mean()), 3), "rest_hz_per_electrode": round(float(rest_hz), 2),
             "w_ee_start": float(c.w[ee].mean())}
        if dump:
            S |= {"spk": [], "pulses": [], "t_start_ms": float(c.t_ms)}
            p = out / "dump" / f"{key}_start.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p.with_suffix(".tmp.npy"), c.w.cpu().numpy())
            os.replace(p.with_suffix(".tmp.npy"), p)
        if ref:
            for k in ("neuron_rest_hz", "rest_hz_per_electrode"):
                g.check(S[k] == ref[k], f"{k} {S[k]} != stored {ref[k]}")
    c.plasticity = rule if plastic else None
    n_reads = int(seconds * 5)
    STOP["phase"] = "resumable"
    while S["i"] < n_reads:
        ck.tick(dev, rule, lambda: S | {"rng": rng.bit_generator.state})
        i = S["i"]
        if i % 5 == 0:
            S["th"] = math.radians(rng.uniform(-12.0, 12.0))
        S["X"].append(schedule(dev, enc, S["th"], S["last"], S["pulses"] if dump else None))
        t0 = float(c.t_ms)
        b = dev.step(200.0)
        if dump:
            S["spk"].append((t0 + b.times_ms.astype(np.float64), b.channels.astype(np.int16)))
        S["Y"].append([int((b.channels == dec[0]).sum()), int((b.channels == dec[1]).sum())])
        S["N"].append(len(b.channels))
        if i >= n_reads - 300:
            S["neuron_end"] = S.get("neuron_end", 0) + dev.oracle_counts
        if (i + 1) % 25 == 0 and stdp_on and homeo is None:
            C.scale(c, w_in0)
        S["i"] += 1
        if ref and S["i"] % 300 == 0:
            m = S["i"] // 300 - 1
            gm = np.round(gains(np.array(S["X"][-300:], float), np.array(S["Y"][-300:], float)), 2).tolist()
            g.check(m < len(ref["gain_60s"]) and gm == ref["gain_60s"][m],
                    f"minute {m + 1} gains {gm} != stored {ref['gain_60s'][m] if m < len(ref['gain_60s']) else None}")
        if g.due():
            hz = float(np.mean(S["N"][-50:])) / 0.2 / dev.n_channels
            g.culture(c, stdp if stdp_on else None, rate_hz=hz)
            g.beat(S["i"], n_reads, hz=round(hz, 2), w_inh=round(float(c.w[~c.syn_exc].mean()), 3))
    STOP["phase"] = "idle"
    guard = g
    X, Y, N = np.array(S["X"], float), np.array(S["Y"], float), np.array(S["N"], float)
    g = [np.round(gains(X[k:k + 300], Y[k:k + 300]), 2).tolist() for k in range(0, len(X) - 299, 300)] or [
        np.round(gains(X, Y), 2).tolist()]
    hz = N / 0.2 / dev.n_channels
    ratio = float(hz[-300:].mean() / max(S["rest_hz_per_electrode"], 1e-9))
    end = np.array(g[-1])
    specific = all(end[k, k] >= 2 * abs(end[k, 1 - k]) and abs(end[k, 1 - k]) <= 1.0 for k in (0, 1))
    inh = ~c.syn_exc
    w_inh = c.w[inh]
    end_hz = S["neuron_end"] / (min(300, n_reads) * 0.2)
    diag = {"inh_at_cap": round(float((w_inh >= (spec["w_max_inh"] if spec["eta"] else c.cfg.w_max) - 1e-3)
                                      .float().mean()), 3),
            "inh_at_zero": round(float((w_inh <= 1e-3).float().mean()), 3), "w_inh_max": round(float(w_inh.max()), 2),
            "neuron_end_hz": round(float(end_hz.mean()), 3),
            "neurons_over_1p5x_target": round(float((end_hz > 1.5 * S["target_hz"]).mean()), 3)}
    row = {"arm": arm, **spec, "seconds": seconds, **diag,
           "hz_per_electrode_10s": np.round(per10(hz), 1).tolist(),
           "dec_hz_10s": np.round(per10(Y) / 0.2, 1).tolist(),
           "gain_60s": g, "rate_ratio": round(ratio, 2), "specific_end": bool(specific),
           "seed": seed, "rate_ratio_60s": [round(float(m / max(S["rest_hz_per_electrode"], 1e-9)), 2)
                                            for m in hz[: len(hz) // 300 * 300].reshape(-1, 300).mean(1)],
           "w_ee_mean_vs_start": round(float(c.w[ee].mean()) / S["w_ee_start"], 3),
           "pass": bool(ratio <= 1.5 and specific), "neuron_rest_hz": S["neuron_rest_hz"],
           "rest_hz_per_electrode": S["rest_hz_per_electrode"],
           "w_exc_mean": round(float(c.w[c.syn_exc].mean()), 3), "w_inh_mean": round(float(c.w[inh].mean()), 3),
           "w_inh_mean_start": round(float(c.w0[inh].mean()), 3)}
    if ref:
        same = json.loads(json.dumps(row))
        diff = [k for k, v in ref.items() if same.get(k) != v]
        guard.check(not diff, f"rerun differs from the stored row in {diff}")
        row["equals_stored_row"] = True
    if dump:
        p = out / "dump" / f"{key}.npz"
        spk, pul = S["spk"], S["pulses"]
        cpu = lambda t: t.cpu().numpy()  # noqa: E731
        arrays = dict(
            w_start=np.load(out / "dump" / f"{key}_start.npy"), w_end=cpu(c.w), w0=cpu(c.w0),
            pre=cpu(c.pre).astype(np.int32), post=cpu(c.post).astype(np.int32), syn_exc=cpu(c.syn_exc),
            excitatory=cpu(c.excitatory), pos=cpu(c.pos).astype(np.float32), alive=cpu(c.alive),
            electrode_xy=np.asarray(dev.mea.xy, np.float32), enc=np.array(enc), dec=np.array(dec),
            spike_t_ms=np.concatenate([s[0] for s in spk]) if spk else np.zeros(0),
            spike_ch=np.concatenate([s[1] for s in spk]) if spk else np.zeros(0, np.int16),
            pulse_t_ms=np.concatenate([s[0] for s in pul]) if pul else np.zeros(0),
            pulse_ch=np.concatenate([s[1] for s in pul]) if pul else np.zeros(0, np.int16),
            t_start_ms=np.array(S["t_start_ms"]), t_end_ms=np.array(float(c.t_ms)), X=X, Y=Y, N=N,
            target_hz=np.asarray(S["target_hz"], np.float32), pulses_delivered_dropped=np.array(
                [dev.pulses_delivered, dev.pulses_dropped]))
        with open(p.with_suffix(".tmp"), "wb") as f:
            np.savez(f, **arrays)
        os.replace(p.with_suffix(".tmp"), p)
        row["dump"] = str(p)
    (out / "jobs").mkdir(exist_ok=True)
    (out / "jobs" / f"{key}.json").write_text(json.dumps(row))
    ck.path.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in row.items() if k not in ("dec_hz_10s",)}), flush=True)
    del dev
    torch.cuda.empty_cache()
    return row


def new_cycle(mode, minutes):
    return {"mode": mode, "t_ms": 0.0, "limit": minutes * 60000.0, "last": None, "steps_all": [], "n_train": 0,
            "dec_counts": [], "ep_hz": [], "r": np.zeros(2), "env": None, "steps": 0}


def cycle(dev, stdp, w_in0, cfg, cs, rng, agent, decoder, tick):
    c = dev.culture
    mode = cs["mode"]
    c.plasticity = None if mode == "frozen" else stdp
    enc, dec, pool = cfg["enc"], cfg["dec"], cfg["train"]
    if cs["last"] is None:
        cs["last"] = {e: -1e9 for e in enc}
    mu, sd = np.array(cfg[f"dec_{decoder}"]).T if decoder != "raw" else (np.zeros(2), np.ones(2))
    sd = np.maximum(sd, 1.0)
    while cs["t_ms"] < cs["limit"]:
        if cs["env"] is None:
            cs["env"], cs["steps"] = CartPole(rng).s, 0
        env = CartPole.resume(cs["env"])
        while True:
            tick()
            schedule(dev, enc, env.s[2], cs["last"])
            b = dev.step(200.0)
            cnt = np.array([(b.channels == dec[0]).sum(), (b.channels == dec[1]).sum()], float)
            cs["dec_counts"].append(cnt)
            cs["r"] = 0.2 * cs["r"] + 0.8 * (cnt - mu) / sd
            cs["t_ms"] += 200.0
            cs["steps"] += 1
            done = env.step(-(cs["r"][0] - cs["r"][1]))
            cs["env"] = env.s
            if done or cs["t_ms"] >= cs["limit"]:
                break
        steps = cs["steps"]
        cs["steps_all"].append(steps)
        cs["ep_hz"].append(np.mean(cs["dec_counts"][-steps:], 0) / 0.2)
        cs["env"] = None
        C.scale(c, w_in0)
        change = np.mean(cs["steps_all"][-5:]) - np.mean(cs["steps_all"][-20:])
        if mode in ("adaptive", "frozen") and agent["last"] is not None:
            agent["traces"] *= 0.3
            agent["traces"][agent["last"]] += 1.0
            agent["values"] += 0.3 * (change - agent["values"]) * agent["traces"]
            np.maximum(agent["values"], 0.1, out=agent["values"])
        if cs["t_ms"] >= cs["limit"]:
            break
        pat = None
        if not change > 0:
            if mode == "random":
                pat = list(rng.choice(pool, 5, replace=False))
            elif mode in ("adaptive", "frozen"):
                p = agent["values"] / agent["values"].sum()
                agent["last"] = int(rng.choice(len(p), p=p))
                pat = list(agent["pairs"][agent["last"]])
        if pat is not None:
            cs["n_train"] += 1
            chs = [int(x) for _ in range(4) for x in pat]
            tts = [k * 100.0 + j * 10.0 for k in range(4) for j in range(len(pat))]
            dev.stimulate(np.array(chs), np.array(tts), np.ones(len(chs)))
        dev.step(400.0)
        for _ in range(6):
            dev.step(500.0)
        cs["t_ms"] += 3400.0
    t = np.array(cs["steps_all"]) * STEP_S
    p90 = float(np.percentile(t, 90))
    ep_hz = cs["ep_hz"]
    return {"mode": mode, "episodes": len(t), "p90_s": round(p90, 2), "mean_s": round(float(t.mean()), 2),
            "proficient": bool(p90 > PROFICIENT_S), "trainings": cs["n_train"],
            "decoder_hz": [round(float(x) / 0.2, 2) for x in np.mean(cs["dec_counts"], 0)],
            "dec_hz_first10": np.round(np.mean(ep_hz[:10], 0), 1).tolist(),
            "dec_hz_last10": np.round(np.mean(ep_hz[-10:], 0), 1).tolist(),
            "w_exc_mean": round(float(c.w[c.syn_exc].mean()), 3),
            "w_inh_mean": round(float(c.w[~c.syn_exc].mean()), 3),
            "steps": [int(s) for s in cs["steps_all"]], "dec_hz_ep": np.round(ep_hz, 1).tolist()}


def run(job):
    import torch

    from biocompute.culture.plasticity import rest
    from biocompute.runguard import Guard

    seed, key, opt, out = job
    _start_job()
    tag = opt["tag"]
    ck = Checkpoint(out / "ckpt" / f"{key}.pt")
    rng = np.random.default_rng(seed + 900)
    dev, _, w_in0 = build(seed)
    stdp = make_stdp(opt["spec"])
    c = dev.culture
    saved = ck.load()
    if saved:
        dev.load_state(saved["dev"])
        stdp.load_state(c, saved["stdp"])
        S = saved["script"]
        rng.bit_generator.state = S.pop("rng")
        print(json.dumps({"seed": seed, "resumed_at": S["k"], "cycle_t_s": None if S["cs"] is None else
                          round(S["cs"]["t_ms"] / 1000)}), flush=True)
    else:
        t0 = time.time()
        cand, rate, first, burst, rest_stats, neuron_hz = characterise(dev, reps=opt["char_reps"])
        cfg = configure(cand, rate, first, burst)
        cfg["dec_rest"] = [[round(float(x), 3) for x in rest_stats[ch]] for ch in cfg["dec"]]
        cfg |= calibrate(dev, cfg, rng, seconds=opt["cal_s"])
        if opt["spec"]["eta"]:
            stdp.set_targets(c, np.maximum(neuron_hz, opt["spec"]["floor"]))
        print(json.dumps({"seed": seed, "config": cfg, "wall_s": round(time.time() - t0)}), flush=True)
        pairs = [p for p in itertools.permutations(cfg["train"], 2)]
        S = {"cfg": cfg, "cycles": [], "k": 0, "cs": None,
             "agent": {"pairs": pairs, "values": np.ones(len(pairs)), "traces": np.zeros(len(pairs)), "last": None},
             "frozen_agent": {"pairs": pairs, "values": np.ones(len(pairs)), "traces": np.zeros(len(pairs)),
                              "last": None}}
        ck.save(dev, stdp, S | {"rng": rng.bit_generator.state})
    plan = [(rep, mode) for rep in range(opt["repeats"]) for mode in opt["order"]]
    snapshot = lambda: S | {"rng": rng.bit_generator.state}  # noqa: E731
    g = Guard(f"{tag}_seed{seed}", out)

    def tick():
        ck.tick(dev, stdp, snapshot)
        if g.due():
            cs = S["cs"]
            g.culture(c, None if cs["mode"] == "frozen" else stdp)
            g.beat(round(S["k"] + cs["t_ms"] / cs["limit"], 2), len(plan), mode=cs["mode"],
                   episodes=len(cs["steps_all"]))

    STOP["phase"] = "resumable"
    while S["k"] < len(plan):
        rep, mode = plan[S["k"]]
        if S["cs"] is None:
            S["cs"] = new_cycle(mode, opt["minutes"])
        agent = S["frozen_agent"] if mode == "frozen" else S["agent"]
        res = cycle(dev, stdp, w_in0, S["cfg"], S["cs"], rng, agent, opt["decoder"], tick)
        res |= {"seed": seed, "repeat": rep, "decoder": opt["decoder"], **opt["spec"]}
        S["cycles"].append(res)
        print(json.dumps({k: v for k, v in res.items() if k not in ("steps", "dec_hz_ep")}), flush=True)
        rest(c, 2700.0, stdp)
        S["k"] += 1
        S["cs"] = None
        ck.save(dev, stdp, snapshot())
    STOP["phase"] = "idle"
    result = {"seed": seed, "config": S["cfg"], "cycles": S["cycles"]}
    (out / "jobs").mkdir(exist_ok=True)
    (out / "jobs" / f"{key}.json").write_text(json.dumps(result))
    ck.path.unlink(missing_ok=True)
    del dev
    torch.cuda.empty_cache()
    return result


def _guarded(fn):
    def call(job):
        try:
            return fn(job)
        except Stopped as e:
            return {"stopped": str(e)}
    return call


def _run_safe(job):
    return _guarded(run)(job)


def _drift_safe(job):
    return _guarded(drift)(job)


def pool_map(fn, jobs, workers):
    if not jobs:  # an idle pool's shutdown can hang on workers that catch SIGTERM
        return []
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with mp.get_context("spawn").Pool(workers, initializer=_worker_init) as pool:
        res = list(pool.imap_unordered(fn, jobs))
    stopped = [r["stopped"] for r in res if "stopped" in r]
    if stopped:
        print(json.dumps({"stopped": stopped, "resume": "run the same command again"}), flush=True)
        raise SystemExit(0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--order", nargs="+", default=["null", "random", "adaptive", "frozen"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cycle-min", type=float, default=15.0)
    ap.add_argument("--decoder", choices=["task", "rest", "raw"], default="task")
    ap.add_argument("--stdp-scale", type=float, default=1.0)
    ap.add_argument("--stdp-rule", choices=["additive", "multiplicative"], default="additive")
    ap.add_argument("--istdp-eta", type=float, default=0.0, help="inhibitory plasticity rate (0: off)")
    ap.add_argument("--w-max-inh", type=float, default=30.0)
    ap.add_argument("--target-floor", type=float, default=0.5, help="lowest per-neuron target rate (Hz)")
    ap.add_argument("--drift", action="store_true", help="STDP drift test instead of the replication")
    ap.add_argument("--drift-scales", type=float, nargs="+", default=[0.1, 0.03, 0.3, 1.0, 0.0])
    ap.add_argument("--drift-etas", type=float, nargs="+", default=[0.0])
    ap.add_argument("--drift-pairs", nargs="+", default=None, metavar="ETA:CAP",
                    help="inhibitory plasticity rate and cap per arm, instead of --drift-etas at --w-max-inh")
    ap.add_argument("--drift-seeds", type=int, nargs="+", default=[0], choices=sorted(DRIFT_CONFIGS),
                    help="cultures for the drift test (their encoders and decoders from the replication)")
    ap.add_argument("--drift-homeo", type=float, nargs="+", default=[0.0],
                    help="per-neuron synaptic scaling rate eta per ms (0: off, the 5 s total-input step when STDP is "
                         "on); B1 uses 1e-5")
    ap.add_argument("--drift-seconds", type=float, default=None,
                    help="per arm (default: 120 s at x0 and x1, 300 s at x0.3, 600 s below)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/cartpole_replication")
    ap.add_argument("--smoke", action="store_true",
                    help="whole pipeline at tiny scale into <out>/smoke: one arm 4 s (--drift), or one culture with "
                         "a short setup and one 6 s null cycle")
    a = ap.parse_args()
    out = Path(a.out)
    sizes = {"rest_s": 20.0, "char_reps": 25, "cal_s": 120.0}
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
        sizes = {"rest_s": 2.0, "char_reps": 2, "cal_s": 4.0}
        a.drift_seconds, a.workers = 4.0, 1
        a.seeds, a.order, a.repeats, a.cycle_min = a.seeds[:1], a.order[:1], 1, 0.1
    out.mkdir(parents=True, exist_ok=True)

    def spec(scale, eta, cap=a.w_max_inh, homeo=0.0, seed=0):
        sp = {"scale": scale, "rule": a.stdp_rule if scale else "additive", "eta": eta, "w_max_inh": cap,
              "floor": a.target_floor}
        return sp | ({"homeo": homeo} if homeo else {}) | ({"seed": seed} if seed else {})

    if a.drift:
        arms = []
        pairs = [tuple(map(float, p.split(":"))) for p in a.drift_pairs] if a.drift_pairs else [
            (eta, a.w_max_inh) for eta in a.drift_etas]
        for h, seed, x, (eta, cap) in itertools.product(a.drift_homeo, a.drift_seeds, a.drift_scales, pairs):
            sp = spec(x, eta, cap, h, seed)
            secs = a.drift_seconds or (120.0 if x in (0.0, 1.0) else 300.0 if x >= 0.3 else 600.0)
            name = f"x{x:g}" + spec_tag({**sp, "scale": 1.0}) + (f"_s{seed}" if seed else "")
            key = job_key(f"drift_{name}", {**sp, "seconds": secs, "rest_s": sizes["rest_s"]})
            arms.append((name, key, sp, secs, out, sizes))
        if a.smoke:
            arms = [max(arms, key=lambda j: (j[2]["scale"] > 0) + bool(j[2].get("homeo")))]
        todo = [j for j in arms if not (out / "jobs" / f"{j[1]}.json").exists()]
        pool_map(_drift_safe, todo, a.workers)
        res = [json.loads((out / "jobs" / f"{j[1]}.json").read_text()) for j in arms]
        name = "drift" + ("" if a.stdp_rule == "additive" else f"_{a.stdp_rule}") + (
            "_istdp" if any(eta for eta, _ in pairs) else "") + ("_scaling" if any(a.drift_homeo) else "")
        summary = drift_summary(res, a.stdp_rule) if any(a.drift_homeo) else {}
        if summary:
            print(json.dumps(summary), flush=True)
        (out / f"{name}.json").write_text(json.dumps({"rows": res} | summary if summary else res, indent=1))
        return
    sp = spec(a.stdp_scale, a.istdp_eta)
    tag = a.decoder + spec_tag(sp)
    opt = {"tag": tag, "spec": sp, "order": a.order, "repeats": a.repeats, "minutes": a.cycle_min,
           "decoder": a.decoder} | sizes
    jobs = [(s, job_key(f"{tag}_seed{s}", opt | {"seed": s}), opt, out) for s in a.seeds]
    todo = [j for j in jobs if not (out / "jobs" / f"{j[1]}.json").exists()]
    pool_map(_run_safe, todo, a.workers)
    res = [json.loads((out / "jobs" / f"{j[1]}.json").read_text()) for j in jobs]
    (out / f"results_{tag}.json").write_text(json.dumps(res))
    summ = {}
    for m in a.order:
        cs = [c for r in res for c in r["cycles"] if c["mode"] == m]
        summ[m] = {"cycles": len(cs), "proficient_share": round(float(np.mean([c["proficient"] for c in cs])), 3),
                   "p90_median_s": round(float(np.median([c["p90_s"] for c in cs])), 2),
                   "mean_s": round(float(np.mean([c["mean_s"] for c in cs])), 2)}
    print(json.dumps(summ), flush=True)
    (out / f"summary_{tag}.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
