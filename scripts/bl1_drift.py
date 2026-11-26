import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # several arms share the 8 GB GPU

import argparse
import itertools
import json
import math
import multiprocessing as mp
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

N, SEED, DT = 5000, 42, 0.5
READ_MS = 200.0
STEPS = int(READ_MS / DT)
CHUNK = 25
PULSE_STEPS = int(20.0 / DT)
NMDA_RATIO, BG_MEAN, BG_STD = 0.37, 1.0, 3.0
G_EXC, PONG_G_EXC = 0.12, 0.05
TARGET_FLOOR = 0.5
ARMS = {"off": (False, False, False), "stdp": (True, False, False), "homeo": (False, True, False),
        "stdp_homeo": (True, True, False), "stdp_ltp": (True, False, True), "stdp_ltp_homeo": (True, True, True)}
DEFAULT_ARMS = ["off", "stdp", "homeo", "stdp_homeo"]


def set_pulse(ms):
    global PULSE_STEPS
    PULSE_STEPS = max(1, int(round(ms / DT)))


class Culture:
    def __init__(self, variant, seed=SEED, ltp=False, closedloop=False):
        import jax
        import jax.numpy as jnp

        from bl1.core.izhikevich import create_population
        from bl1.core.synapses import create_synapse_state
        from bl1.loop.encoding import STIM_AMPLITUDE_MV
        from bl1.mea.electrode import MEA, build_neuron_electrode_map
        from bl1.network.topology import build_connectivity, place_neurons
        from bl1.plasticity.homeostatic import HomeostaticParams, init_homeostatic_state
        from bl1.plasticity.stdp import STDPParams, init_stdp_state
        from bl1.plasticity.stp import STPParams, init_stp_state

        self.variant, self.seed, self.closedloop = variant, seed, closedloop
        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        positions = place_neurons(k1, N, (3000.0, 3000.0))
        self.params, state, self.is_exc = create_population(k2, N)
        W_exc, self.W_inh, _ = build_connectivity(k3, positions, self.is_exc, lambda_um=200.0, p_max=0.21,
                                                  g_exc=0.12, g_inh=0.36)
        self.idx, self.w0 = W_exc.indices, W_exc.data
        U = jnp.where(self.is_exc, 0.30, 0.04)
        tau_rec = jnp.where(self.is_exc, 800.0, 100.0)
        tau_fac = jnp.where(self.is_exc, 0.001, 1000.0)
        self.stp_params = STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)
        self.noise_key = k4
        mea = MEA("cl1_64ch")
        self.emap = build_neuron_electrode_map(positions, mea.positions, mea.detection_radius_um)
        self.n_el = int(self.emap.shape[0])
        self.emap_f = self.emap.astype(jnp.float32)
        self.stim_map = self.emap_f * STIM_AMPLITUDE_MV
        self.stdp_params = STDPParams()
        self.homeo_params = HomeostaticParams()
        if variant == "fixed":
            k, d = G_EXC / PONG_G_EXC, STDPParams()
            self.stdp_params = d._replace(A_plus=d.A_plus * k, A_minus=d.A_minus * k, w_max=d.w_max * k)
            self.homeo_params = HomeostaticParams(w_max=self.stdp_params.w_max)
        if ltp:
            p_ = self.stdp_params
            self.stdp_params = p_._replace(tau_plus=p_.tau_minus, tau_minus=p_.tau_plus)
        self.w_bounds = (min(self.stdp_params.w_min, self.homeo_params.w_min),
                         max(self.stdp_params.w_max, self.homeo_params.w_max, float(self.w0.max())))
        self.carry = (state, create_synapse_state(N), init_stp_state(N, self.stp_params), init_stdp_state(N),
                      init_homeostatic_state(N), self.w0)
        self.t = 0
        self._fns = {}

    def set_targets(self, rates, reset_estimate=True):
        import jax.numpy as jnp

        from bl1.plasticity.homeostatic import HomeostaticState

        tgt = jnp.asarray(np.maximum(rates, TARGET_FLOOR), dtype=jnp.float32)
        self.homeo_params = self.homeo_params._replace(r_target=tgt)
        if reset_estimate:
            ns, ss, stp_st, sd, _, w = self.carry
            self.carry = (ns, ss, stp_st, sd, HomeostaticState(rate_estimate=tgt), w)
        self._fns = {k: f for k, f in self._fns.items() if k in ("fine", (False, False))}

    def _step(self, stdp_on, homeo_on, fine):
        import jax
        import jax.numpy as jnp
        from jax.experimental.sparse import BCOO

        from bl1.core.izhikevich import izhikevich_step
        from bl1.core.synapses import (SynapseState, ampa_step, compute_synaptic_current, gaba_a_step, gaba_b_step,
                                       nmda_step)
        from bl1.plasticity.homeostatic import update_rate_estimate
        from bl1.plasticity.stdp import stdp_update
        from bl1.plasticity.stp import stp_step

        params, stp_params, W_inh, idx = self.params, self.stp_params, self.W_inh, self.idx

        def step(carry, x):
            ns, ss, stp_st, sd, hs, w = carry
            t, flags = x
            I_t = (BG_MEAN + BG_STD * jax.random.normal(jax.random.fold_in(self.noise_key, t), (N,))
                   + flags.astype(jnp.float32) @ self.stim_map)
            W_exc = BCOO((w, idx), shape=(N, N))
            I_syn = compute_synaptic_current(ss, ns.v)
            ns = izhikevich_step(ns, params, I_syn + I_t, DT)
            stp_st, scale = stp_step(stp_st, stp_params, ns.spikes, DT)
            W_ampa, W_nmda = (W_exc, W_exc) if self.closedloop else (W_exc * (1.0 - NMDA_RATIO), W_exc * NMDA_RATIO)
            new_ampa = ampa_step(ss.g_ampa, scale, W_ampa, DT)
            new_gaba = gaba_a_step(ss.g_gaba_a, scale, W_inh, DT)
            new_nmda_rise, new_nmda_decay, _ = nmda_step(ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT)
            gb_rise, gb_decay = ss.g_gaba_b_rise, ss.g_gaba_b_decay
            if self.closedloop:
                gb_rise, gb_decay, _ = gaba_b_step(gb_rise, gb_decay, scale, W_inh, DT)
            ss = SynapseState(g_ampa=new_ampa, g_gaba_a=new_gaba, g_nmda_rise=new_nmda_rise,
                              g_nmda_decay=new_nmda_decay, g_gaba_b_rise=gb_rise, g_gaba_b_decay=gb_decay)
            if stdp_on:
                sd, W_new = stdp_update(sd, self.stdp_params, ns.spikes, W_exc, self.is_exc, DT)
                w = W_new.data
            if homeo_on:
                hs = update_rate_estimate(hs, ns.spikes, DT)
            s = ns.spikes.astype(jnp.float32)
            out = (self.emap_f @ s, s.sum()) if fine else ns.spikes
            return (ns, ss, stp_st, sd, hs, w), out

        return step

    def _reads_fn(self, stdp_on, homeo_on):
        import jax
        import jax.numpy as jnp
        from jax.experimental.sparse import BCOO

        from bl1.plasticity.homeostatic import homeostatic_scaling

        step = self._step(stdp_on, homeo_on, fine=False)

        def read(carry, x):
            t0, flags = x
            carry, spikes = jax.lax.scan(step, carry, (t0 + jnp.arange(STEPS), flags))
            if homeo_on:
                ns, ss, stp_st, sd, hs, w = carry
                hs, W_new = homeostatic_scaling(hs, self.homeo_params, BCOO((w, self.idx), shape=(N, N)),
                                                self.is_exc, dt_ms=READ_MS)
                carry = (ns, ss, stp_st, sd, hs, W_new.data)
            return carry, spikes.sum(0).astype(jnp.int16)

        def run(carry, t0, flags):
            k = flags.shape[0]
            return jax.lax.scan(read, carry, (t0 + STEPS * jnp.arange(k), flags))

        return jax.jit(run)

    def _fine_fn(self):
        import jax
        import jax.numpy as jnp

        step = self._step(False, False, fine=True)
        return jax.jit(lambda carry, t0, flags: jax.lax.scan(step, carry, (t0 + jnp.arange(flags.shape[0]), flags)))

    def reads(self, flags, stdp_on=False, homeo_on=False):
        key = (stdp_on, homeo_on)
        if key not in self._fns:
            self._fns[key] = self._reads_fn(stdp_on, homeo_on)
        self.carry, counts = self._fns[key](self.carry, self.t, flags)
        self.t += flags.shape[0] * STEPS
        return np.asarray(counts, dtype=np.int64)

    def fine(self, flags):
        if "fine" not in self._fns:
            self._fns["fine"] = self._fine_fn()
        self.carry, (el, pop) = self._fns["fine"](self.carry, self.t, flags)
        self.t += flags.shape[0]
        return np.asarray(el), np.asarray(pop)

    def quiet(self, n_reads):
        return np.zeros((n_reads, STEPS, self.n_el), bool)

    def electrode_counts(self, counts):
        return counts @ np.asarray(self.emap, dtype=np.int64).T

    def w(self):
        return np.asarray(self.carry[5])

    def state(self):
        import jax

        return {"carry": jax.device_get(self.carry), "t": self.t}

    def load_state(self, s):
        import jax
        import jax.numpy as jnp

        self.carry = jax.tree.map(jnp.asarray, s["carry"])
        self.t = s["t"]


STOP = {"requested": False, "phase": "idle"}


class Stopped(Exception):
    pass


EVENT = {"stop": None}


def stop_requested():
    flag = EVENT["stop"]
    return STOP["requested"] or (flag is not None and flag.value == 1)


def _on_term(signum, frame):
    STOP["requested"] = True
    if EVENT["stop"] is not None:
        EVENT["stop"].value = 1
    if STOP["phase"] == "setup" and not STOP.get("raised"):
        STOP["raised"] = True  # once: a second SIGTERM mid-unwind must not escape _guarded
        raise Stopped("setup")


def _worker_init(ev=None):
    import faulthandler

    EVENT["stop"] = ev
    signal.signal(signal.SIGTERM, _on_term)
    faulthandler.register(signal.SIGUSR1)


class Checkpoint:
    def __init__(self, path, every_s=300.0):
        self.path, self.every_s, self.t = Path(path), every_s, time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        return pickle.loads(self.path.read_bytes()) if self.path.exists() else None

    def save(self, culture, script):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({"culture": culture.state(), "script": script}))
        os.replace(tmp, self.path)
        self.t = time.time()

    def tick(self, culture, script_fn):
        if stop_requested() or time.time() - self.t > self.every_s:
            self.save(culture, script_fn())
            if stop_requested():
                raise Stopped(str(self.path))


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
    a, b, c, d = best
    rest = [i for i in np.argsort(rate)[::-1] if i not in best][:8]
    return {"enc": [int(cand[a]), int(cand[c])], "dec": [int(cand[b]), int(cand[d])],
            "train": [int(cand[i]) for i in rest], "score": round(float(bs), 3),
            "c1_ab": round(float(first[a, b]), 3), "c1_cd": round(float(first[c, d]), 3),
            "c1_ad": round(float(first[a, d]), 3), "c1_cb": round(float(first[c, b]), 3),
            "burst_enc": [round(float(burst[a]), 2), round(float(burst[c]), 2)],
            "c1_mean": round(float(first.mean()), 3), "burst_mean": round(float(burst.mean()), 2)}


def schedule(enc, th, last, now_ms, flags_read, spill):
    hz = [(7 * (0.15 - math.sin(th))) ** 2, (7 * (0.15 + math.sin(th))) ** 2]
    n = [0, 0]
    for i, (e, f) in enumerate(zip(enc, hz)):
        nxt = max(last[e] + 1000.0 / f, now_ms)
        while nxt < now_ms + READ_MS:
            s0 = int(round((nxt - now_ms) / DT))
            s1 = s0 + PULSE_STEPS
            flags_read[s0:min(s1, STEPS), e] = True
            if s1 > STEPS:
                spill[: s1 - STEPS, e] = True
            last[e] = nxt
            nxt += 1000.0 / f
            n[i] += 1
    return n


def gains(X, Y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, Y, rcond=None)[0][1:].T


def gains_lag(X, Y):
    A = np.column_stack([np.ones(len(X) - 1), X[1:], X[:-1]])
    c = np.linalg.lstsq(A, Y[1:], rcond=None)[0]
    return (c[1:3] + c[3:5]).T


def trend(series):
    y = np.asarray(series, float)
    if len(y) < 3:
        return [None, None]
    x = np.arange(len(y), dtype=float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    se = math.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return [round(float(b), 3), round(se, 3)]


def per10(x):
    k = max(len(x) // 50, 1) * 50 if len(x) >= 50 else len(x)
    return x[:k].reshape(-1, min(50, k), *x.shape[1:]).mean(1)


def setup(job):
    from bl1.analysis.bursts import burst_statistics, detect_bursts

    from biocompute.runguard import Guard

    variant, seed, out, sz = job
    set_pulse(sz["pulse_ms"])
    STOP["phase"] = "setup"
    if stop_requested():
        raise Stopped("not started")
    g = Guard("setup", out)
    t0 = time.time()
    cu = Culture(variant, seed)
    cu.reads(cu.quiet(10))
    T = int(sz["spont_s"] * 1000 / DT)
    el, pop = cu.fine(np.zeros((T, cu.n_el), bool))
    fr = float(pop.sum()) / N / sz["spont_s"]
    bursts = detect_bursts(pop[:, None], dt_ms=DT, threshold_std=1.5, min_duration_ms=50.0)
    bs = burst_statistics(bursts) if bursts else {}
    pc ={"rate_hz": round(fr, 3), "bursts_per_min": round(len(bursts) / (sz["spont_s"] / 60), 2),
          "burst_ms": round(float(bs["duration_mean"]), 1) if bursts else None,
          "ibi_ms": round(float(bs["ibi_mean"]), 1) if len(bursts) > 1 else None,
          "ibi_cv": round(float(bs["ibi_cv"]), 2) if len(bursts) > 2 else None,
          "bl1_reported": {"rate_hz": 1.6, "bursts_per_min": "6-11", "ibi_ms": "4500-10300", "burst_ms": "95-116"}}
    print(json.dumps({"positive_control": pc, "wall_s": round(time.time() - t0)}), flush=True)
    g.check(0.1 <= fr <= 5.0, f"spontaneous rate {fr:.3g} Hz outside Wagenaar's 0.1-5 Hz: not bl1's culture")
    if not sz["smoke"]:
        g.check(0.2 <= pc["bursts_per_min"] <= 20, f"{pc['bursts_per_min']} bursts/min outside Wagenaar's 0.2-20")
    counts = el.sum(0)
    c200 = el[: (T // STEPS) * STEPS].reshape(-1, STEPS, cu.n_el).sum(1)
    rest_stats = np.stack([c200.mean(0), c200.std(0)], 1)
    cand = np.argsort(counts)[::-1][: sz["n_cand"]]
    k, reps, isi = len(cand), sz["reps"], int(250.0 / DT)
    first, totals = np.zeros((k, k)), np.zeros((k, reps))
    for i, ch in enumerate(cand):
        flags = np.zeros((reps * isi, cu.n_el), bool)
        for r in range(reps):
            flags[r * isi: r * isi + PULSE_STEPS, ch] = True
        el, _ = cu.fine(flags)
        el = el.reshape(reps, isi, cu.n_el)[:, :, cand]
        first[i] += (el[:, : int(10.0 / DT)].sum(1) > 0).sum(0)
        totals[i] = el[:, : int(200.0 / DT)].sum((1, 2))
        g.beat(i + 1, k, force=i + 1 == k, rate_hz=round(fr, 3))
    first /= reps
    np.fill_diagonal(first, 0.0)
    med = np.median(totals)
    thr = med + 3 * np.median(np.abs(totals - med))
    burst = (totals > thr).mean(1)
    cfg = configure(cand, counts[cand], first, burst)
    cfg["dec_rest"] = [[round(float(x), 3) for x in rest_stats[ch]] for ch in cfg["dec"]]
    res = {"positive_control": pc, "config": cfg, "candidates": cand.tolist(), "wall_s": round(time.time() - t0)}
    tmp = out / "setup.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, out / "setup.json")
    print(json.dumps({"config": cfg}), flush=True)
    STOP["phase"] = "idle"
    return res


def drift(job):
    from biocompute.runguard import Guard

    arm, variant, seed, seconds, out, sz, cfg = job
    set_pulse(sz["pulse_ms"])
    STOP["phase"] = "setup"
    if stop_requested():
        raise Stopped("not started")
    stdp_on, homeo_on, ltp = ARMS[arm]
    g = Guard(f"drift_{arm}", out)
    enc, dec = cfg["enc"], cfg["dec"]
    ck = Checkpoint(out / "ckpt" / f"drift_{arm}.pkl")
    rng = np.random.default_rng(1)
    cu = Culture(variant, seed, ltp=ltp)
    w_lo, w_hi = cu.w_bounds
    saved = ck.load()
    if saved:
        cu.load_state(saved["culture"])
        S = saved["script"]
        rng.bit_generator.state = S.pop("rng")
        print(json.dumps({"arm": arm, "resumed_at_read": S["i"]}), flush=True)
    else:
        cu.reads(cu.quiet(10))
        rest = cu.reads(cu.quiet(int(sz["rest_s"] * 5)))
        rates = rest.sum(0) / sz["rest_s"]
        rest_el = cu.electrode_counts(rest)
        S = {"i": 0, "X": [], "Y": [], "N": [], "N_noenc": [], "last": {e: -1e9 for e in enc}, "th": 0.0,
             "spill": np.zeros((PULSE_STEPS, cu.n_el), bool), "w_exc_60s": [], "neuron_rest_hz": rates,
             "rest_hz_per_electrode": float(rest_el.sum() / sz["rest_s"] / cu.n_el),
             "rest_hz_per_electrode_noenc": float(np.delete(rest_el, enc, 1).sum() / sz["rest_s"] / (cu.n_el - 2))}
        if variant == "fixed":
            cu.set_targets(rates)
    if saved and variant == "fixed":
        cu.set_targets(S["neuron_rest_hz"], reset_estimate=False)
    n_reads = int(seconds * 5)
    STOP["phase"] = "resumable"
    while S["i"] < n_reads:
        ck.tick(cu, lambda: S | {"rng": rng.bit_generator.state})
        k = min(CHUNK, n_reads - S["i"])
        flags = np.zeros((k, STEPS, cu.n_el), bool)
        X = []
        for r in range(k):
            i = S["i"] + r
            if i % 5 == 0:
                S["th"] = math.radians(rng.uniform(-12.0, 12.0))
            flags[r, :PULSE_STEPS] |= S["spill"]
            S["spill"] = np.zeros_like(S["spill"])
            X.append(schedule(enc, S["th"], S["last"], i * READ_MS, flags[r], S["spill"]))
        counts = cu.reads(flags, stdp_on, homeo_on)
        el = cu.electrode_counts(counts)
        S["X"] += X
        S["Y"] += el[:, dec].tolist()
        S["N"] += el.sum(1).tolist()
        S["N_noenc"] += np.delete(el, enc, 1).sum(1).tolist()
        tail = min(k, max(0, S["i"] + k - (n_reads - 300)))
        if tail:
            S["neuron_end"] = S.get("neuron_end", 0) + counts[k - tail:].sum(0)
        S["i"] += k
        if S["i"] % 300 < k or S["i"] == n_reads:
            S["w_exc_60s"].append(round(float(cu.w().mean()), 5))
        if g.due() or S["i"] == n_reads:
            w = cu.w()
            v, u = np.asarray(cu.carry[0].v), np.asarray(cu.carry[0].u)
            g.check(bool(np.isfinite(v).all() and np.isfinite(u).all() and np.isfinite(w).all()), "NaN/Inf in state")
            g.check(w.min() >= w_lo - 1e-6 and w.max() <= w_hi + 1e-6,
                    f"weights [{w.min():.4g}, {w.max():.4g}] outside bl1's bounds [{w_lo}, {w_hi}]")
            hz = float(np.mean(S["N"][-50:])) / 0.2 / cu.n_el
            g.check(0.0 < hz < 3000.0, f"rate {hz:.4g} Hz per electrode")
            g.beat(S["i"], n_reads, hz=round(hz, 2), w_exc=round(float(w.mean()), 5))
    STOP["phase"] = "idle"
    X, Y = np.array(S["X"], float), np.array(S["Y"], float)
    Nel, Nne = np.array(S["N"], float), np.array(S["N_noenc"], float)
    gw = [np.round(gains(X[k:k + 300], Y[k:k + 300]), 2).tolist() for k in range(0, len(X) - 299, 300)] or [
        np.round(gains(X, Y), 2).tolist()]
    gl = [np.round(gains_lag(X[k:k + 300], Y[k:k + 300]), 2) for k in range(0, len(X) - 299, 300)] or [
        np.round(gains_lag(X, Y), 2)]
    gl_trend = [[trend([g[k, i] for g in gl]) for i in (0, 1)] for k in (0, 1)]
    end_l = gl[-1]
    specific_lag = all(end_l[k, k] >= 2 * abs(end_l[k, 1 - k]) and abs(end_l[k, 1 - k]) <= 1.0 for k in (0, 1))
    hz, hz_ne = Nel / 0.2 / cu.n_el, Nne / 0.2 / (cu.n_el - 2)
    last = min(300, len(hz))
    ratio = float(hz[-last:].mean() / max(S["rest_hz_per_electrode"], 1e-9))
    end = np.array(gw[-1])
    specific = all(end[k, k] >= 2 * abs(end[k, 1 - k]) and abs(end[k, 1 - k]) <= 1.0 for k in (0, 1))
    w = cu.w()
    end_hz = S["neuron_end"] / (last * 0.2)
    rest_n = S["neuron_rest_hz"]
    row = {"arm": arm, "variant": variant, "seed": seed, "stdp": stdp_on, "homeo": homeo_on, "ltp": ltp,
           "seconds": seconds,
           "rate_ratio": round(ratio, 2), "specific_end": bool(specific), "pass": bool(ratio <= 1.5 and specific),
           "rate_ratio_noenc": round(float(hz_ne[-last:].mean() / max(S["rest_hz_per_electrode_noenc"], 1e-9)), 2),
           "last_vs_first_min": round(float(hz[-last:].mean() / max(hz[:last].mean(), 1e-9)), 2),
           "hz_last60": round(float(hz[-last:].mean()), 2),
           "rest_hz_per_electrode": round(S["rest_hz_per_electrode"], 2),
           "neuron_rest_hz": round(float(rest_n.mean()), 3), "neuron_end_hz": round(float(end_hz.mean()), 3),
           "neurons_over_1p5x_rest": round(float((end_hz > 1.5 * np.maximum(rest_n, 0.05)).mean()), 3),
           "hz_per_electrode_10s": np.round(per10(hz), 1).tolist(),
           "hz_noenc_10s": np.round(per10(hz_ne), 1).tolist(),
           "dec_hz_10s": np.round(per10(Y) / 0.2, 1).tolist(), "gain_60s": gw,
           "gain_lag_60s": [g.tolist() for g in gl], "gain_lag_trend_per_min": gl_trend,
           "specific_end_lag": bool(specific_lag), "pulse_ms": sz["pulse_ms"],
           "w_exc_mean_start": round(float(np.asarray(cu.w0).mean()), 5), "w_exc_mean_60s": S["w_exc_60s"],
           "w_exc_mean": round(float(w.mean()), 5), "w_at_max": round(float((w >= w.max() - 1e-6).mean()), 3),
           "w_at_min": round(float((w <= w_lo + 1e-6).mean()), 3), "w_bounds": [w_lo, w_hi],
           "pulses_per_read": round(float(X.sum(1).mean()), 2)}
    (out / "jobs").mkdir(exist_ok=True)
    np.savez_compressed(out / "jobs" / f"drift_{arm}_series.npz", X=X, Y=Y, N=Nel, N_noenc=Nne,
                        neuron_rest_hz=rest_n, neuron_end_hz=end_hz, w_end=w, w0=np.asarray(cu.w0))
    (out / "jobs" / f"drift_{arm}.json").write_text(json.dumps(row))
    ck.path.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in row.items() if not k.endswith("_10s")}), flush=True)
    return row


def _guarded(fn):
    def call(job):
        try:
            return fn(job)
        except Stopped as e:
            return {"stopped": str(e)}
        finally:
            STOP["phase"] = "idle"  # a later SIGTERM must not raise Stopped inside the pool's own code
    return call


def _setup_safe(job):
    return _guarded(setup)(job)


def _drift_safe(job):
    return _guarded(drift)(job)


_MAIN_EVENT = {}


def pool_map(fn, jobs, workers):
    ctx = mp.get_context("spawn")
    if "ev" not in _MAIN_EVENT:
        _MAIN_EVENT["ev"] = ctx.RawValue("b", 0)

        def on_term(signum, frame):
            _MAIN_EVENT["ev"].value = 1

        signal.signal(signal.SIGTERM, on_term)
    ev = _MAIN_EVENT["ev"]
    if ev.value == 1:
        print(json.dumps({"stopped": "before start", "resume": "run the same command again"}), flush=True)
        raise SystemExit(0)
    # close() + join(), not the context manager: its terminate() hangs on workers that catch SIGTERM.
    pool = ctx.Pool(workers, initializer=_worker_init, initargs=(ev,), maxtasksperchild=1)
    try:
        res = list(pool.imap_unordered(fn, jobs))
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        raise
    stopped = [r["stopped"] for r in res if "stopped" in r]
    if stopped:
        print(json.dumps({"stopped": stopped, "resume": "run the same command again"}), flush=True)
        raise SystemExit(0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", choices=["shipped", "fixed"], default=["shipped"])
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--controls", action="store_true",
                    help="also the potentiation-dominated positive controls (stdp_ltp, stdp_ltp_homeo) in --variant fixed")
    ap.add_argument("--seed", nargs="+", type=int, default=[SEED], help="culture seeds (bl1's calibration used 42-49)")
    ap.add_argument("--pulse-ms", nargs="+", type=float, default=[20.0],
                    help="pulse length; 20 = ClosedLoop's (16-20 spikes per stimulated neuron), 0.5 = one step")
    ap.add_argument("--seconds", type=float, default=600.0, help="stimulation per arm")
    ap.add_argument("--workers", type=int, default=2, help="each grows to ~2.2 GB GPU memory")
    ap.add_argument("--root", default="outputs/bl1_drift")
    ap.add_argument("--out", default=None, help="one configuration only: its folder (default <root>/<variant>"
                                                "[_seed<n>][_pulse<ms>ms])")
    ap.add_argument("--smoke", action="store_true",
                    help="whole pipeline at tiny scale into <folder>/smoke for every variant x pulse (first seed): "
                         "2 s spontaneous, 4 candidates x 2, every arm 4 s")
    a = ap.parse_args()
    import faulthandler

    faulthandler.register(signal.SIGUSR1)
    seeds = a.seed[:1] if a.smoke else a.seed
    configs = list(itertools.product(a.variant, seeds, a.pulse_ms))
    if a.out and len(configs) > 1:
        sys.exit("--out takes one configuration")

    def folder(variant, seed, pulse):
        if a.out:
            return Path(a.out)
        return Path(a.root) / (variant + ("" if seed == SEED else f"_seed{seed}")
                               + ("" if pulse == 20.0 else f"_pulse{pulse:g}ms"))

    def sizes(pulse):
        if a.smoke:
            return {"spont_s": 2.0, "n_cand": 4, "reps": 2, "rest_s": 2.0, "smoke": True, "pulse_ms": pulse}
        return {"spont_s": 60.0, "n_cand": 32, "reps": 25, "rest_s": 20.0, "smoke": False, "pulse_ms": pulse}

    if a.smoke:
        a.seconds = 4.0
        a.controls = True
    outs = {}
    for c in configs:
        o = folder(*c) / "smoke" if a.smoke else folder(*c)
        if a.smoke:
            shutil.rmtree(o, ignore_errors=True)
        o.mkdir(parents=True, exist_ok=True)
        outs[c] = o
    first = {}
    for v, sd, pl in configs:
        first.setdefault((sd, pl), (v, sd, pl))
    todo = [(c[0], c[1], outs[c], sizes(c[2])) for c in first.values() if not (outs[c] / "setup.json").exists()]
    if todo:
        pool_map(_setup_safe, todo, min(a.workers, len(todo)))
    for c in configs:
        if not (outs[c] / "setup.json").exists():
            shutil.copy(outs[first[(c[1], c[2])]] / "setup.json", outs[c] / "setup.json")

    def arms_of(c):
        return a.arms + (["stdp_ltp", "stdp_ltp_homeo"] if a.controls and c[0] == "fixed" else [])

    jobs = []
    for c in configs:
        cfg = json.loads((outs[c] / "setup.json").read_text())["config"]
        jobs += [(arm, c[0], c[1], a.seconds, outs[c], sizes(c[2]), cfg) for arm in arms_of(c)]
    todo = [j for j in jobs if not (j[4] / "jobs" / f"drift_{j[0]}.json").exists()]
    if todo:
        pool_map(_drift_safe, todo, min(a.workers, len(todo)))
    summary = []
    for c in configs:
        rows = [json.loads((outs[c] / "jobs" / f"drift_{arm}.json").read_text()) for arm in arms_of(c)]
        off = next((r for r in rows if r["arm"] == "off"), None)
        for r in rows:
            r["vs_off"] = round(r["hz_last60"] / off["hz_last60"], 2) if off and off["hz_last60"] > 0 else None
        (outs[c] / f"drift_{c[0]}.json").write_text(json.dumps(rows, indent=1))
        summary += [{"variant": c[0], "seed": c[1], "pulse_ms": c[2],
                     **{k: r.get(k) for k in ("arm", "rate_ratio", "vs_off", "rate_ratio_noenc", "last_vs_first_min",
                                              "specific_end", "specific_end_lag", "pass", "w_exc_mean_start",
                                              "w_exc_mean", "gain_lag_trend_per_min")}} for r in rows]
    for row in summary:
        print(json.dumps(row), flush=True)
    if not a.smoke and len(configs) > 1:
        tag = "_".join([*a.variant, *[f"s{x}" for x in seeds], *[f"p{x:g}" for x in a.pulse_ms]])
        (Path(a.root) / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
