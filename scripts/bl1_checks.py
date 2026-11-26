import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("bl1_drift", HERE / "bl1_drift.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

RESULTS = []


def report(name, ok, **info):
    RESULTS.append(ok)
    print(json.dumps({"check": name, "result": "PASS" if ok else "FAIL", **info}), flush=True)


def pulses(cu, n_reads, rng, enc):
    flags = np.zeros((n_reads, D.STEPS, cu.n_el), bool)
    spill = np.zeros((D.PULSE_STEPS, cu.n_el), bool)
    last, X, th = {e: -1e9 for e in enc}, [], 0.0
    for i in range(n_reads):
        if i % 5 == 0:
            th = math.radians(rng.uniform(-12.0, 12.0))
        flags[i, :D.PULSE_STEPS] |= spill
        spill = np.zeros_like(spill)
        X.append(D.schedule(enc, th, last, i * D.READ_MS, flags[i], spill))
    return flags, np.array(X)


def noise(cu, t0, T):
    import jax
    import jax.numpy as jnp

    return jax.vmap(lambda t: D.BG_MEAN + D.BG_STD * jax.random.normal(jax.random.fold_in(cu.noise_key, t), (D.N,)))(
        t0 + jnp.arange(T))


def spont_port():
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    from bl1.core.izhikevich import izhikevich_step
    from bl1.core.synapses import (SynapseState, ampa_step, compute_synaptic_current, create_synapse_state,
                                   gaba_a_step, nmda_step)
    from bl1.plasticity.stp import init_stp_state, stp_step

    T = 40000
    cu = D.Culture("shipped")
    _, pop_port = cu.fine(np.zeros((T, cu.n_el), bool))
    ref = D.Culture("shipped")
    params, W_inh, stp_params = ref.params, ref.W_inh, ref.stp_params
    W_exc = BCOO((ref.w0, ref.idx), shape=(D.N, D.N))
    W_ampa, W_nmda = W_exc * (1.0 - 0.37), W_exc * 0.37
    DT = 0.5

    def step_fn(carry, I_t):
        ns, ss, stp_st = carry
        I_syn = compute_synaptic_current(ss, ns.v)
        I_total = I_syn + I_t
        ns = izhikevich_step(ns, params, I_total, DT)
        stp_st, scale = stp_step(stp_st, stp_params, ns.spikes, DT)
        new_ampa = ampa_step(ss.g_ampa, scale, W_ampa, DT)
        new_gaba = gaba_a_step(ss.g_gaba_a, scale, W_inh, DT)
        new_nmda_rise, new_nmda_decay, _ = nmda_step(ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT)
        ss = SynapseState(g_ampa=new_ampa, g_gaba_a=new_gaba, g_nmda_rise=new_nmda_rise, g_nmda_decay=new_nmda_decay,
                          g_gaba_b_rise=ss.g_gaba_b_rise, g_gaba_b_decay=ss.g_gaba_b_decay)
        return (ns, ss, stp_st), ns.spikes

    _, spikes = jax.lax.scan(step_fn, (ref.carry[0], create_synapse_state(D.N), init_stp_state(D.N, stp_params)),
                             noise(ref, 0, T))
    pop_ref = np.asarray(spikes).sum(1)
    differ = int((pop_ref != pop_port).sum())
    report("spont_port", differ == 0, steps=T, spikes_port=int(pop_port.sum()), spikes_bl1=int(pop_ref.sum()),
           steps_differing=differ)


def dishbrain_weights(cu):
    cu.w0 = cu.w0 * (0.05 / 0.12)
    cu.W_inh = cu.W_inh * (0.20 / 0.36)
    cu.carry = cu.carry[:5] + (cu.w0,)
    return cu


def plastic_port():
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    from bl1.core.synapses import create_synapse_state
    from bl1.loop.controller import _make_scan_step
    from bl1.plasticity.stdp import init_stdp_state
    from bl1.plasticity.stp import init_stp_state

    n_reads = 50
    cu = dishbrain_weights(D.Culture("shipped", closedloop=True))
    flags, X = pulses(cu, n_reads, np.random.default_rng(1), [3, 34])
    counts = []
    for k in range(0, n_reads, D.CHUNK):
        counts.append(cu.reads(flags[k:k + D.CHUNK], True, False))
    counts = np.concatenate(counts)
    ref = dishbrain_weights(D.Culture("shipped"))
    step = _make_scan_step(ref.params, ref.W_inh, 0.5, True, ref.stdp_params, ref.is_exc, stp_params=ref.stp_params)
    T = n_reads * D.STEPS
    I_ext = noise(ref, 0, T) + jnp.asarray(flags.reshape(T, -1), jnp.float32) @ ref.stim_map
    carry = (ref.carry[0], create_synapse_state(D.N), init_stdp_state(D.N), BCOO((ref.w0, ref.idx), shape=(D.N, D.N)),
             init_stp_state(D.N, ref.stp_params))
    carry, spikes = jax.lax.scan(step, carry, I_ext)
    ref_counts = np.asarray(spikes).reshape(n_reads, D.STEPS, D.N).sum(1).astype(np.int64)
    w_ref, w_port = np.asarray(carry[3].data), cu.w()
    reads_differing = int((ref_counts != counts).any(1).sum())
    report("plastic_port", reads_differing == 0 and np.array_equal(w_ref, w_port), reads=n_reads,
           pulses=int(X.sum()), spikes_port=int(counts.sum()), spikes_bl1=int(ref_counts.sum()),
           reads_differing=reads_differing, max_weight_diff=float(np.abs(w_ref - w_port).max()),
           weights_changed=float((w_port != np.asarray(cu.w0)).mean()))


def homeo_port():
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    from bl1.plasticity.homeostatic import HomeostaticState, homeostatic_scaling, update_rate_estimate

    n_reads = 50
    cu = D.Culture("fixed")
    rates = np.random.default_rng(0).uniform(0.5, 3.0, D.N)
    cu.set_targets(rates)
    flags, _ = pulses(cu, n_reads, np.random.default_rng(1), [3, 34])
    ref = D.Culture("fixed")
    ref.set_targets(rates)
    hp = ref.homeo_params
    counts, ref_counts = [], []
    for k in range(0, n_reads, D.CHUNK):
        counts.append(cu.reads(flags[k:k + D.CHUNK], False, True))
    step = jax.jit(lambda c, x: jax.lax.scan(ref._step(False, False, fine=False), c, x))
    ema = jax.jit(lambda hs, sp: jax.lax.scan(lambda h, s: (update_rate_estimate(h, s, 0.5), None), hs, sp)[0])
    carry, hs = ref.carry, HomeostaticState(rate_estimate=jnp.asarray(np.maximum(rates, D.TARGET_FLOOR), jnp.float32))
    for r in range(n_reads):
        carry, spikes = step(carry, (r * D.STEPS + jnp.arange(D.STEPS), flags[r]))
        ref_counts.append(np.asarray(spikes).sum(0))
        hs = ema(hs, spikes)
        _, W = homeostatic_scaling(hs, hp, BCOO((carry[5], ref.idx), shape=(D.N, D.N)), ref.is_exc, dt_ms=200.0)
        carry = carry[:5] + (W.data,)
    counts, ref_counts = np.concatenate(counts), np.array(ref_counts, np.int64)
    w_ref, w_port = np.asarray(carry[5]), cu.w()
    rel = float(np.abs(w_ref - w_port).max() / w_ref.max())
    reads_differing = int((ref_counts != counts).any(1).sum())
    report("homeo_port", reads_differing == 0 and rel < 1e-5, reads=n_reads, reads_differing=reads_differing,
           max_rel_weight_diff=rel, weights_changed=float((np.abs(w_port - np.asarray(cu.w0)) > 1e-7).mean()),
           rate_estimate_diff=float(np.abs(np.asarray(hs.rate_estimate) - np.asarray(cu.carry[4].rate_estimate)).max()))


def reads_vs_fine():
    n_reads = 25
    a, b = D.Culture("shipped"), D.Culture("shipped")
    flags, _ = pulses(a, n_reads, np.random.default_rng(1), [3, 34])
    counts = a.reads(flags)
    el, _ = b.fine(flags.reshape(n_reads * D.STEPS, -1))
    el_reads = el.reshape(n_reads, D.STEPS, -1).sum(1)
    ok = np.array_equal(a.electrode_counts(counts), np.rint(el_reads).astype(np.int64))
    report("reads_vs_fine", ok, reads=n_reads, spikes=int(counts.sum()))


def determinism():
    n_reads = 150
    runs = []
    for _ in range(2):
        cu = D.Culture("fixed")
        cu.set_targets(np.full(D.N, 1.5))
        flags, _ = pulses(cu, n_reads, np.random.default_rng(1), [3, 34])
        c = np.concatenate([cu.reads(flags[k:k + D.CHUNK], True, True) for k in range(0, n_reads, D.CHUNK)])
        runs.append((c, cu.w()))
    (c1, w1), (c2, w2) = runs
    first = np.nonzero((c1 != c2).any(1))[0]
    report("determinism", bool(np.array_equal(c1, c2) and np.array_equal(w1, w2)), reads=n_reads,
           first_differing_read=int(first[0]) if len(first) else None,
           spikes=[int(c1.sum()), int(c2.sum())], max_weight_diff=float(np.abs(w1 - w2).max()))


def stimulus():
    cu = D.Culture("shipped")
    enc = [3, 34]
    out = {}
    ok = True
    for deg in (-12.0, 0.0, 6.0, 12.0):
        th = math.radians(deg)
        last, n_reads = {e: -1e9 for e in enc}, 50
        flags = np.zeros((n_reads, D.STEPS, cu.n_el), bool)
        spill = np.zeros((D.PULSE_STEPS, cu.n_el), bool)
        X = []
        for i in range(n_reads):
            flags[i, :D.PULSE_STEPS] |= spill
            spill = np.zeros_like(spill)
            X.append(D.schedule(enc, th, last, i * D.READ_MS, flags[i], spill))
        X = np.array(X).sum(0)
        want = [(7 * (0.15 - math.sin(th))) ** 2 * 10, (7 * (0.15 + math.sin(th))) ** 2 * 10]
        on = flags.reshape(-1, cu.n_el)[:, enc].sum(0)
        full = all(abs(on[k] - D.PULSE_STEPS * X[k]) <= D.PULSE_STEPS for k in (0, 1))
        good = all(abs(X[k] - want[k]) <= 1.0 for k in (0, 1)) and full
        ok &= good
        out[f"{deg:g}deg"] = {"pulses": X.tolist(), "expected": [round(w, 1) for w in want], "on_steps": on.tolist()}
    sm = np.asarray(cu.stim_map)
    stim_vals = sorted(set(np.round(sm[enc].ravel(), 3).tolist()))
    ok &= stim_vals == [0.0, 150.0]
    cu.reads(cu.quiet(10))
    flags = np.zeros((D.STEPS, cu.n_el), bool)
    flags[:D.PULSE_STEPS, enc[0]] = True
    import jax.numpy as jnp
    step = cu._step(False, False, fine=False)
    import jax
    carry, spikes = jax.lax.scan(step, cu.carry, (cu.t + jnp.arange(D.STEPS), flags))
    spikes = np.asarray(spikes)
    m = np.asarray(cu.emap[enc[0]])
    per_neuron = spikes[:D.PULSE_STEPS, m].sum(0)
    report("stimulus", bool(ok), schedule=out, stim_values=stim_vals, neurons_per_electrode=int(m.sum()),
           spikes_per_stimulated_neuron_per_pulse=[int(per_neuron.min()), round(float(per_neuron.mean()), 1),
                                                   int(per_neuron.max())],
           others_spiking_in_200ms=int((spikes[:, ~m].sum(0) > 0).sum()))


def homeo_tracks():
    cu = D.Culture("fixed")
    cu.reads(cu.quiet(10))
    rest = cu.reads(cu.quiet(100))
    rates = rest.sum(0) / 20.0
    cu.set_targets(rates)
    flags, _ = pulses(cu, 150, np.random.default_rng(1), [3, 34])
    counts = [cu.reads(flags[k:k + D.CHUNK], False, True) for k in range(0, 150, D.CHUNK)]
    w_before = cu.w()
    counts.append(cu.reads(flags[:1], False, True))
    w_after = cu.w()
    counts = np.concatenate(counts)
    measured = counts[-50:].sum(0) / 10.0
    est = np.asarray(cu.carry[4].rate_estimate)
    r = float(np.corrcoef(est, measured)[0, 1])
    tgt = np.asarray(cu.homeo_params.r_target)
    rows = np.asarray(cu.idx[:, 0]); cols = np.asarray(cu.idx[:, 1])
    is_exc = np.asarray(cu.is_exc)
    ee = is_exc[rows] & is_exc[cols]
    err = (tgt - est)[rows]
    dw = w_after - w_before
    lo, hi = cu.homeo_params.w_min, cu.homeo_params.w_max
    sel = ee & (np.abs(err) > 0.05) & (w_before > lo) & (w_before < hi)
    agree = float((np.sign(dw[sel]) == np.sign(err[sel])).mean())
    non_ee_changed = float((dw[~ee] != 0).mean())
    report("homeo_tracks", r > 0.8 and agree > 0.95, estimate_vs_measured_r=round(r, 3),
           estimate_mean=round(float(est.mean()), 3), measured_mean=round(float(measured.mean()), 3),
           target_mean=round(float(tgt.mean()), 3), sign_agreement_ee=round(agree, 3),
           non_ee_weights_changed=non_ee_changed)


CHECKS = [spont_port, plastic_port, homeo_port, reads_vs_fine, determinism, stimulus, homeo_tracks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=[c.__name__ for c in CHECKS])
    a = ap.parse_args()
    for c in CHECKS:
        if a.only and c.__name__ not in a.only:
            continue
        try:
            c()
        except Exception as e:
            import traceback

            traceback.print_exc()
            report(c.__name__, False, error=repr(e)[:300])
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
