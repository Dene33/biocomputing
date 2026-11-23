"""Route finding: does our model culture reproduce a published learning effect?

Robbins et al. 2026, "Goal-directed learning in cortical organoids" (Cell Reports 45:116984; code on Zenodo,
10.5281/zenodo.17684862): mouse cortical organoids on a MaxOne HD-MEA balance a cartpole. Proficient cycles:
null 2.3%, random training 4.5%, adaptive training 23.1% (continuous adaptive 46.4%). Our coaching probe
(coach_probe.py) used neither their electrode selection nor their training rule, so here is their protocol, step by
step, on the simulated culture (20,000 neurons, 256-electrode grid at 100 um, focal stimulation):

1. Record: 20 s of spontaneous activity; candidates = the 32 most active electrodes (MaxOne stimulates <= 32).
2. Stimulate: each candidate 25x at 4 Hz (paper: 50x at 2 Hz). First-order causal connectivity C1(i, j) =
   P(spike on j within 10 ms); a stimulation evokes a network burst when the total count over all candidates
   (0-200 ms) exceeds median + 3 MAD.
3. Configuration: encoders a, c and decoders b, d maximising C1(a,b) + C1(c,d) - C1(a,d) - C1(c,b); encoders with
   burst probability > 0.5 excluded; 8 training units from the remaining candidates (most active first).
4. Train: gym cartpole (0.02 s per step, force +-10 N continuous, fail at |theta| > 16 deg); one game step per
   200 ms read. Encoders pulsed at regular intervals at f = (7 (0.15 -/+ sin theta))^2 Hz (1.1 Hz upright,
   0.8 / 8.9 Hz at 16 deg). Force = 10 N x clip(-(r_b - r_d), -1, 1), r = 0.2 r + 0.8 count (200 ms counts).
   ``--decoder raw`` uses the counts as they are. That rule assumes near-silent decoders (organoid units fire a few
   Hz); ours fire 40-90 Hz, so the difference sits far past the clip and an unbalanced pair pushes one way forever
   (first run: two of three cultures fell in 13 steps every episode in every mode). ``--decoder rest`` (default)
   puts each decoder on the paper's scale: (count - resting mean) / resting SD, from the spontaneous recording.
   That still pinned two of three cultures: under stimulation one decoder fires 85-100 Hz and the other ~48 Hz
   (both ~12 Hz at rest). ``--decoder task`` normalises with the mean and SD measured in a calibration block.
   Calibration (every run, plasticity off): 120 s of encoder pulses at the encoding rates for tilts held 1 s each,
   drawn uniformly within +-12 deg, game off. Per read, each decoder's count is regressed on both encoders' pulse
   counts: the 2 x 2 gain matrix says how specific the 200 ms decoder windows are; ``reflex_r2`` is the share of
   the variance of the decoder difference that the tilt explains.
   Calibration found specific windows in all three cultures (own encoder 1.6-3.3 spikes per pulse, other 0.2-0.5;
   reflex R^2 0.11-0.23, decoders ~15 Hz) but in play, with STDP on, decoders ran at 30-100 Hz within 10 episodes.
   ``--drift``: is that the plasticity? The calibration stimulation (seed 0's configuration) with STDP off or scaled
   by ``--drift-scales``, synaptic scaling every 5 s as in play; rates per 10 s and encoder->decoder gains per 60 s.
   ``--stdp-scale`` scales STDP's amplitudes (a_plus, a_minus) in the replication itself.
   Result: it is the plasticity, at every strength. STDP off: 6 Hz per electrode throughout, decoders specific. On:
   the encoder with the slightly stronger response is potentiated until its pulses fire the whole network
   (x1: 6 -> 45 Hz in 20 s; x0.3: 39 Hz; x0.1: 28 Hz; x0.03: 6.5 -> 13 Hz in 10 min) and then drives both decoders
   (end gains on the other decoder 24.6 / 14.7 / 19.2 / 7.5 spikes per pulse vs 0.3-2.9 from its own encoder).
   Scaling total input doesn't stop it; weaker STDP only slows it. The rule is additive with hard bounds, the
   textbook unstable form (Song, Miller & Abbott 2000; van Rossum, Bi & Turrigiano 2000 for the stable,
   weight-dependent form).

   ``--stdp-rule multiplicative``: the weight-dependent form, in the drift test and the replication.

Logs: run_raw_decoder.log (round 1), run_rest.log (round 1), run_task.log (calibration + round 1 null); the runs
were stopped there because the decoders pinned. drift.json: the drift test.
   After each episode, if the mean of the last 5 episodes is not above that of the last 20: a 400 ms training
   signal, then 3 s wait (null: the same time without stimulation).
   - ``random``: 5 training units in random order, 10 ms apart, repeated at 10 Hz
   - ``adaptive``: an ordered pair of training units, 10 ms apart, repeated at 10 Hz, sampled in proportion to its
     value; after every episode the last pattern's value moves towards the change (5-episode mean - 20-episode
     mean) through eligibility traces (learning rate 0.3, trace decay 0.3, floor 0.1, start 1)
   - ``frozen``: adaptive with plasticity off (our control: the agent alone, steering a culture that can't learn)
   Cycles of ``--cycle-min`` minutes, 45 min rest between (fast-forwarded: weights decay towards baseline).
   STDP on, with synaptic scaling after each episode (the model's weights run away without it).
5. Score (theirs): 90th percentile of time balanced per cycle, time = steps x 0.3 s (their 300 ms base);
   proficient above 20.85 s, mean + 3 SD of the best random controller with this physics (8.73 +- 4.04 s; paper
   8.64 +- 4.00 s -> 20.64 s).

    uv run scripts/cartpole_replication.py
"""

import argparse
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("coach_probe", HERE / "coach_probe.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

STEP_S = 0.3  # their time base per game step
PROFICIENT_S = 20.85
LIMIT = math.radians(16.0)


class CartPole:
    """gym CartPole physics (Euler, tau 0.02 s), continuous force, no cart-position limit, fail at 16 degrees."""

    def __init__(self, rng):
        self.s = rng.uniform(-0.05, 0.05, 4)

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


def build(seed):
    import torch

    from biocompute.culture.network import make_config
    from biocompute.culture.plasticity import STDP
    from biocompute.device import SimulatedBiocomputer
    from biocompute.mea.electrodes import make_mea_config

    torch.set_num_threads(1)
    dev = SimulatedBiocomputer(make_config("gpu_20k", seed=seed), make_mea_config("grid256", seed=seed))
    dev.step(2000.0)
    c = dev.culture
    w_in0 = torch.zeros(c.n, device=c.device).index_add_(0, c.post[c.syn_exc], c.w[c.syn_exc])
    return dev, STDP(), w_in0


def characterise(dev, reps=25, isi_ms=250.0):
    bins = np.zeros((200, dev.n_channels))
    for i in range(200):
        b = dev.step(100.0)
        bins[i] = np.bincount(b.channels, minlength=dev.n_channels)
    counts = bins.sum(0)
    c200 = bins.reshape(100, 2, -1).sum(1)  # counts per 200 ms read
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
    first /= reps
    np.fill_diagonal(first, 0.0)
    med = np.median(totals)
    thr = med + 3 * np.median(np.abs(totals - med))
    burst = (totals > thr).mean(1)
    return cand, counts[cand], first, burst, rest_stats


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


def schedule(dev, enc, th, last):
    """Queue the encoder pulses of the next 200 ms read for pole angle ``th``; returns pulses per encoder."""
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
    gain = np.linalg.lstsq(A, Y, rcond=None)[0][1:].T  # gain[k, i]: spikes on decoder k per pulse on encoder i
    diff = Y[:, 0] - Y[:, 1]
    rho = float(np.corrcoef(diff, -S)[0, 1])
    return {"dec_task": [[round(float(m), 3), round(float(v), 3)] for m, v in zip(Y.mean(0), Y.std(0))],
            "gain": np.round(gain, 2).tolist(), "reflex_r": round(rho, 3), "reflex_r2": round(rho * rho, 3),
            "pulses_per_read": round(float(X.sum(1).mean()), 2)}


def drift(job):
    import torch

    scale, seconds, rule = job
    enc, dec = [136, 155], [137, 139]  # seed 0's configuration (characterise/configure, see run_task.log)
    dev, stdp, w_in0 = build(0)
    c = dev.culture
    stdp.multiplicative = rule == "multiplicative"
    stdp.a_plus *= scale
    stdp.a_minus *= scale
    c.plasticity = stdp if scale > 0 else None
    rng = np.random.default_rng(1)
    last = {e: -1e9 for e in enc}
    X, Y, N = [], [], []
    for i in range(int(seconds * 5)):
        if i % 5 == 0:
            th = math.radians(rng.uniform(-12.0, 12.0))
        X.append(schedule(dev, enc, th, last))
        b = dev.step(200.0)
        Y.append([(b.channels == dec[0]).sum(), (b.channels == dec[1]).sum()])
        N.append(len(b.channels))
        if (i + 1) % 25 == 0 and scale > 0:
            C.scale(c, w_in0)
    X, Y, N = np.array(X, float), np.array(Y, float), np.array(N, float)
    gains = []
    for k in range(0, len(X) - 299, 300):
        A = np.column_stack([np.ones(300), X[k:k + 300]])
        gains.append(np.round(np.linalg.lstsq(A, Y[k:k + 300], rcond=None)[0][1:].T, 2).tolist())
    row = {"stdp_scale": scale, "stdp_rule": rule, "seconds": seconds,
           "hz_per_electrode_10s": np.round(N.reshape(-1, 50).mean(1) / 0.2 / dev.n_channels, 1).tolist(),
           "dec_hz_10s": np.round(Y.reshape(-1, 50, 2).mean(1) / 0.2, 1).tolist(),
           "gain_60s": gains, "w_exc_mean": round(float(c.w[c.syn_exc].mean()), 3)}
    print(json.dumps(row), flush=True)
    del dev
    torch.cuda.empty_cache()
    return row


def cycle(dev, stdp, w_in0, cfg, mode, minutes, rng, agent, decoder="rest"):
    c = dev.culture
    c.plasticity = None if mode == "frozen" else stdp
    enc, dec, pool = cfg["enc"], cfg["dec"], cfg["train"]
    t_ms, limit = 0.0, minutes * 60000.0
    last = {e: -1e9 for e in enc}
    steps_all, n_train, dec_counts, ep_hz = [], 0, [], []
    r = np.zeros(2)
    mu, sd = np.array(cfg[f"dec_{decoder}"]).T if decoder != "raw" else (np.zeros(2), np.ones(2))
    sd = np.maximum(sd, 1.0)
    while t_ms < limit:
        env, steps = CartPole(rng), 0
        while True:
            schedule(dev, enc, env.s[2], last)
            b = dev.step(200.0)
            cnt = np.array([(b.channels == dec[0]).sum(), (b.channels == dec[1]).sum()], float)
            dec_counts.append(cnt)
            r = 0.2 * r + 0.8 * (cnt - mu) / sd
            t_ms += 200.0
            steps += 1
            if env.step(-(r[0] - r[1])) or t_ms >= limit:
                break
        steps_all.append(steps)
        ep_hz.append(np.mean(dec_counts[-steps:], 0) / 0.2)
        C.scale(c, w_in0)
        change = np.mean(steps_all[-5:]) - np.mean(steps_all[-20:])
        if mode in ("adaptive", "frozen") and agent["last"] is not None:
            agent["traces"] *= 0.3
            agent["traces"][agent["last"]] += 1.0
            agent["values"] += 0.3 * (change - agent["values"]) * agent["traces"]
            np.maximum(agent["values"], 0.1, out=agent["values"])
        if t_ms >= limit:
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
            n_train += 1
            chs = [int(x) for _ in range(4) for x in pat]
            tts = [k * 100.0 + j * 10.0 for k in range(4) for j in range(len(pat))]
            dev.stimulate(np.array(chs), np.array(tts), np.ones(len(chs)))
        dev.step(400.0)
        for _ in range(6):
            dev.step(500.0)
        t_ms += 3400.0
    t = np.array(steps_all) * STEP_S
    p90 = float(np.percentile(t, 90))
    return {"mode": mode, "episodes": len(t), "p90_s": round(p90, 2), "mean_s": round(float(t.mean()), 2),
            "proficient": bool(p90 > PROFICIENT_S), "trainings": n_train,
            "decoder_hz": [round(float(x) / 0.2, 2) for x in np.mean(dec_counts, 0)],
            "dec_hz_first10": np.round(np.mean(ep_hz[:10], 0), 1).tolist(),
            "dec_hz_last10": np.round(np.mean(ep_hz[-10:], 0), 1).tolist(),
            "w_exc_mean": round(float(c.w[c.syn_exc].mean()), 3), "steps": [int(s) for s in steps_all],
            "dec_hz_ep": np.round(ep_hz, 1).tolist()}


def run(job):
    import torch

    from biocompute.culture.plasticity import rest

    seed, order, repeats, minutes, decoder, stdp_scale, rule = job
    t0 = time.time()
    rng = np.random.default_rng(seed + 900)
    dev, stdp, w_in0 = build(seed)
    stdp.multiplicative = rule == "multiplicative"
    stdp.a_plus *= stdp_scale
    stdp.a_minus *= stdp_scale
    cand, rate, first, burst, rest_stats = characterise(dev)
    cfg = configure(cand, rate, first, burst)
    cfg["dec_rest"] = [[round(float(x), 3) for x in rest_stats[ch]] for ch in cfg["dec"]]
    cfg |= calibrate(dev, cfg, rng)
    print(json.dumps({"seed": seed, "config": cfg, "wall_s": round(time.time() - t0)}), flush=True)
    pairs = [p for p in itertools.permutations(cfg["train"], 2)]
    agent = {"pairs": pairs, "values": np.ones(len(pairs)), "traces": np.zeros(len(pairs)), "last": None}
    frozen_agent = {"pairs": pairs, "values": np.ones(len(pairs)), "traces": np.zeros(len(pairs)), "last": None}
    cycles = []
    for rep in range(repeats):
        for mode in order:
            res = cycle(dev, stdp, w_in0, cfg, mode, minutes, rng, frozen_agent if mode == "frozen" else agent,
                        decoder)
            res |= {"seed": seed, "repeat": rep, "decoder": decoder, "stdp_scale": stdp_scale, "stdp_rule": rule,
                    "wall_s": round(time.time() - t0)}
            cycles.append(res)
            print(json.dumps({k: v for k, v in res.items() if k not in ("steps", "dec_hz_ep")}), flush=True)
            rest(dev.culture, 2700.0, stdp)
    del dev
    torch.cuda.empty_cache()
    return {"seed": seed, "config": cfg, "cycles": cycles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--order", nargs="+", default=["null", "random", "adaptive", "frozen"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cycle-min", type=float, default=15.0)
    ap.add_argument("--decoder", choices=["task", "rest", "raw"], default="task")
    ap.add_argument("--stdp-scale", type=float, default=1.0)
    ap.add_argument("--stdp-rule", choices=["additive", "multiplicative"], default="additive")
    ap.add_argument("--drift", action="store_true", help="STDP drift test instead of the replication")
    ap.add_argument("--drift-scales", type=float, nargs="+", default=[0.1, 0.03, 0.3, 1.0, 0.0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/cartpole_replication")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.drift:
        if a.stdp_rule == "additive":
            jobs = [(x, 120.0 if x in (0.0, 1.0) else 300.0 if x >= 0.3 else 600.0, "additive") for x in a.drift_scales]
        else:
            jobs = [(x, 600.0, a.stdp_rule) for x in a.drift_scales]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(drift, jobs, chunksize=1)
        name = "drift.json" if a.stdp_rule == "additive" else f"drift_{a.stdp_rule}.json"
        (out / name).write_text(json.dumps(res, indent=1))
        return
    jobs = [(s, a.order, a.repeats, a.cycle_min, a.decoder, a.stdp_scale, a.stdp_rule) for s in a.seeds]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    tag = a.decoder + ("" if a.stdp_scale == 1.0 else f"_stdp{a.stdp_scale:g}") + (
        "" if a.stdp_rule == "additive" else f"_{a.stdp_rule}")
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
