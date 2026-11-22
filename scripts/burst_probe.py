"""Route finding: why can't the model culture be coached? The burst hypothesis.

scripts/coach_probe.py found that pairing moves the culture's response a lot, but to both contexts alike: every
probe sets off the same network burst. Spiking-network models of learning met the same wall. Sinapayen, Masumori &
Ikegami (PLoS ONE 2017, learning by stimulation avoidance, Izhikevich neurons + STDP): in a bursting network "the
firing rates of Output Zone A and Output Zone B increase in equivalent proportions"; burst suppression (sparser
connectivity, more noise, or short-term depression) was "essential to obtain learning". In real cultures,
distributed background stimulation suppresses bursts (Wagenaar, Madhavan, Pine & Potter 2005), and the Potter lab's
goal-directed learning ran training on top of random background stimulation (Chao, Bakkum & Potter 2008).

Per condition, 2 cultures, the 60-electrode MEA and electrode groups of coach_probe.py:
1. Burstiness: 20 s of activity. Population count in 10 ms bins: share of spikes in the busiest 5% of bins
   (~10% for independent firing, most for a bursting culture), and the bins' coefficient of variation.
2. Site specificity: 40 probes on each context group (alternating). Response = spike counts per electrode
   5-60 ms after the probe, on electrodes not stimulated. Which group was probed, decoded leave-one-out (nearest
   centroid, z-scored); and d' of the readout score (A - B) between the two contexts.
3. Pathway-specific plasticity: 400 pairings of context 0 with output B (``causal``: probe, then B 8 ms later; or
   ``reverse`` order), synaptic scaling every 100 pairings. Shift of each context's A - B score, plasticity off while
   measuring. Specific share = shift(context 0) - shift(context 1); the target is negative.

Conditions:
- ``base``     the culture as used so far
- ``bg50``, ``bg200``  Poisson background pulses on the 44 unused electrodes, 50 / 200 Hz in total; paused during
               each response window (the MEA blanks every channel for 2 ms after any pulse)
- ``noise``    noise current 3 -> 8 (LSA's desynchronizing fix)
- ``sparse``   100 -> 40 synapses per neuron (LSA: learning best at 20-30 per neuron)
- ``inhib``    inhibitory weights -6 -> -12

    uv run scripts/burst_probe.py
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("coach_probe", HERE / "coach_probe.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

P, G = C.P, C.G
USED = set(P[0] + P[1] + G[0] + G[1])
BG = [c for c in range(60) if c not in USED]
STIM = set(P[0] + P[1])
OTHER = [c for c in range(60) if c not in STIM]

CONDITIONS = {
    "base": ({}, 0.0),
    "bg50": ({}, 50.0),
    "bg200": ({}, 200.0),
    "noise": ({"noise_std": 8.0}, 0.0),
    "sparse": ({"synapses_per_neuron": 40}, 0.0),
    "inhib": ({"w_inh": -12.0}, 0.0),
}


def build(seed, overrides):
    import torch

    from biocompute.culture.network import make_config
    from biocompute.culture.plasticity import STDP
    from biocompute.device import SimulatedBiocomputer
    from biocompute.mea.electrodes import make_mea_config

    torch.set_num_threads(1)
    dev = SimulatedBiocomputer(make_config("gpu_20k", seed=seed, **overrides),
                               make_mea_config("mcs60", seed=seed, stim_length_um=120.0))
    dev.step(2000.0)
    c = dev.culture
    w_in0 = torch.zeros(c.n, device=c.device).index_add_(0, c.post[c.syn_exc], c.w[c.syn_exc])
    return dev, STDP(), w_in0


def advance(dev, ms, bg_hz, rng, ch=(), t=()):
    """Run ``ms`` with the given pulses plus Poisson background pulses on the unused electrodes."""
    ch, t = list(ch), list(t)
    n = rng.poisson(bg_hz * ms / 1000.0) if bg_hz else 0
    if n:
        ch += list(rng.choice(BG, n))
        t += list(rng.uniform(0.0, ms, n))
    if ch:
        dev.stimulate(np.array(ch), np.array(t), np.ones(len(ch)))
    return dev.step(ms)


def burstiness(dev, bg_hz, rng, seconds=20.0):
    counts = []
    for _ in range(int(seconds * 10)):
        b = advance(dev, 100.0, bg_hz, rng)
        counts.append(np.histogram(b.times_ms, bins=10, range=(0.0, 100.0))[0])
    x = np.concatenate(counts).astype(float)
    top = np.sort(x)[::-1][: max(1, len(x) // 20)]
    return {"rate_hz_per_electrode": round(float(x.sum() / seconds / 60), 2),
            "top5pct_share": round(float(top.sum() / max(x.sum(), 1)), 3),
            "bin_cv": round(float(x.std() / max(x.mean(), 1e-9)), 2)}


def respond(dev, k, bg_hz, rng):
    """One probe on context group k (two pulses 10 ms apart, as coach_probe); counts per electrode 5-60 ms."""
    n = len(P[k])
    b = advance(dev, 60.0, 0.0, rng, P[k] * 2, [0.0] * n + [10.0] * n)  # background paused in the window
    m = (b.times_ms >= 5.0) & (b.times_ms < 60.0)
    v = np.bincount(b.channels[m], minlength=60).astype(float)
    advance(dev, 150.0, bg_hz, rng)
    return v


def measure(dev, stdp, bg_hz, rng, reps=40):
    c = dev.culture
    c.plasticity = None
    V, K = [], []
    for _ in range(reps):
        for k in (0, 1):
            V.append(respond(dev, k, bg_hz, rng))
            K.append(k)
    c.plasticity = stdp
    V, K = np.array(V), np.array(K)
    score = V[:, G[0]].sum(1) - V[:, G[1]].sum(1)
    return V, K, score


def site_decoding(V, K):
    X = V[:, OTHER]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ok = 0
    for i in range(len(X)):
        m = np.ones(len(X), bool)
        m[i] = False
        c0 = X[m & (K == 0)].mean(0)
        c1 = X[m & (K == 1)].mean(0)
        guess = int(np.linalg.norm(X[i] - c1) < np.linalg.norm(X[i] - c0))
        ok += guess == K[i]
    return round(ok / len(X), 3)


def dprime(score, K):
    a, b = score[K == 0], score[K == 1]
    return round(float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9)), 3)


def run(job):
    import torch

    cond, seed, mode, pairings = job
    overrides, bg_hz = CONDITIONS[cond]
    rng = np.random.default_rng(seed + 500)
    t0 = time.time()
    dev, stdp, w_in0 = build(seed, overrides)
    c = dev.culture
    row = {"condition": cond, "seed": seed, "mode": mode, "pairings": pairings}
    if mode == "causal":  # the culture before pairing is the same in both modes; measure it once
        row["burst"] = burstiness(dev, bg_hz, rng)
    V, K, s0 = measure(dev, stdp, bg_hz, rng)
    row["site_decoding"] = site_decoding(V, K)
    row["dprime_before"] = dprime(s0, K)
    row["evoked_per_probe"] = round(float(V[:, OTHER].sum(1).mean()), 1)
    row["evoked_cv"] = round(float(V[:, OTHER].sum(1).std() / max(V[:, OTHER].sum(1).mean(), 1e-9)), 2)
    c.plasticity = stdp
    first, second = (P[0], G[1]) if mode == "causal" else (G[1], P[0])
    for i in range(pairings):
        advance(dev, 30.0, 0.0, rng, first + second, [0.0] * 4 + [8.0] * 4)
        advance(dev, 70.0, bg_hz, rng)
        if (i + 1) % 100 == 0:
            C.scale(c, w_in0)
    V2, K2, s1 = measure(dev, stdp, bg_hz, rng)
    d0 = s1[K2 == 0].mean() - s0[K == 0].mean()
    d1 = s1[K2 == 1].mean() - s0[K == 1].mean()
    se = np.sqrt(sum(x[k == j].var() / (k == j).sum() for x, k in ((s0, K), (s1, K2)) for j in (0, 1)))
    row |= {"shift_ctx0": round(float(d0), 2), "shift_ctx1": round(float(d1), 2),
            "specific": round(float(d0 - d1), 2), "specific_se": round(float(se), 2),
            "common": round(float((d0 + d1) / 2), 2), "score_sd": round(float(s0.std()), 2),
            "dprime_after": dprime(s1, K2), "w_exc_mean_end": round(float(c.w[c.syn_exc].mean()), 3),
            "wall_s": round(time.time() - t0)}
    print(json.dumps(row), flush=True)
    del dev
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--pairings", type=int, default=400)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/burst_probe")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s, m, a.pairings) for c in a.conditions for s in a.seeds for m in ("causal", "reverse")]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
