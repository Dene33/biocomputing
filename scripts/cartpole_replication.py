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
   (x1: 6 -> 12 Hz by 20 s, 41 Hz by 40 s, 43 Hz over the second minute; last-minute means x0.3: 38 Hz,
   x0.1: 30 Hz; x0.03: 6.5 -> 13 Hz in 10 min) and then drives both decoders
   (end gains on the other decoder 24.6 / 14.7 / 19.2 / 7.5 spikes per pulse vs 0.3-2.9 from its own encoder).
   Scaling total input doesn't stop it; weaker STDP only slows it. The rule is additive with hard bounds, the
   textbook unstable form (Song, Miller & Abbott 2000; van Rossum, Bi & Turrigiano 2000 for the stable,
   weight-dependent form).

   ``--stdp-rule multiplicative``: the weight-dependent form. It plateaus lower (~22 Hz) but the same encoder
   takes over both decoders (drift_multiplicative.json).
   ``--istdp-eta`` adds inhibitory plasticity (Vogels et al. 2011, see plasticity.py), each neuron's target its
   resting rate (floor ``--target-floor``), inhibitory weights up to ``--w-max-inh``. Drift gate, fixed before the
   sweep: rate per electrode in the last 60 s <= 1.5x the resting rate (20 s before any stimulation; a 60 s trial
   showed the runaway can happen inside the first minute, so "last vs first minute" would miss it), and in the
   last 60 s each decoder's own gain >= 2x its cross gain with the cross gain <= 1 spike per pulse.
   First sweep (drift_istdp_clamped10.json, STDP x1, 10 min): no rate passes. The rate falls as eta rises but
   levels off (last 60 s vs rest: eta 0.001 4.1x, 0.003 3.8x, 0.01 3.4x, 0.03 2.8x, 0.1 2.2x), and the mean
   inhibitory weight ends at 9.7-9.9 in every arm: the synaptic scaling step (coach_probe.scale) clamped *all*
   weights to w_max = 10, so inhibition was capped at 10, not ``--w-max-inh``, and saturated. Fixed (scaling now
   bounds excitatory weights only; without inhibitory plasticity nothing changes, inhibitory weights start <= 10).
   ``--drift-pairs ETA:CAP`` sweeps rate and cap together; the drift row now reports the share of inhibitory
   synapses at the cap and at zero, and the share of neurons above 1.5x their target in the last 60 s.
   Refinement round, the last of the time box (drift_istdp.json; eta:cap 0.01:30 0.03:30 0.1:30 0.03:100 0.1:100):
   no arm passes. Rate vs rest, last 60 s: 1.64 / 1.67 / 1.65 / 1.56 / 1.51 (gate 1.5); 59-72% of neurons above
   1.5x their own target. Inhibition is not at its cap (2-4% of synapses at 30, <1% at 100) and still climbing (mean
   5.7 -> 23-25 at cap 30, 38-51 at cap 100), yet rates don't come down: more inhibition stops buying lower rates.
   Untested guess: the rule also strengthens inhibitory -> inhibitory synapses (Vogels et al. used I -> E only), so
   it partly silences the inhibition itself. Evoked responses fade: own gains 2-4 -> 0.5-1.5 spikes per pulse, and
   in three arms decoder 1 follows the wrong encoder (own / cross 0.57 / 2.0 at 0.01:30). Only 0.1:100 is specific
   in the last window (1.06 / 0.45, 0.92 / 0.29), not a minute earlier (0.57 / 0.59), with gains near zero (0.09 /
   0.01 at 9 min). Inhibitory plasticity cuts the runaway (43 Hz -> 9-10 Hz per electrode) but doesn't make the model
   hold rates and pathways; no replication run.

   Track B, step B1 (docs/route-to-computation.md, "Order of work without a dish"): a rate controller.
   ``--drift-homeo ETA`` replaces the 5 s total-input step with per-neuron synaptic scaling (plasticity.py,
   SynapticScaling: bl1's rule, each neuron's target its own resting rate with floor ``--target-floor``, rate
   estimate a 10 s moving average, E->E weights x exp(eta x 200 ms x (target - rate)) every 200 ms; the runs of
   24 Nov 2026 used bl1's 1 + eta x 200 ms x (target - rate), which can go negative, see plasticity.py).
   ``--drift-seeds`` runs the test on the replication's cultures 0, 1, 2 (their encoders and decoders from
   run_task.log). Arms per culture: no plasticity; scaling alone; STDP x1 with the old 5 s step (the runaway, as
   above); STDP x1 with scaling instead. bl1's eta 1e-5 is the gate's; 1e-4 runs after it, exploratory.
   Gate B1, fixed before the run: at eta 1e-5, in each of the 3 cultures, the rate per electrode in the last 60 s stays
   within a factor 1.5 of rest (2/3x to 1.5x) both with scaling alone and with STDP x1 plus scaling. The lower bound
   is ours: bl1's arms "passed" the old gate by running down, and calm by muting doesn't count. Controls: no
   plasticity stays in that band, and the old 5 s step runs away (> 1.5x), so the test can see what B1 must stop.
   Reported, not gated: the rate per minute (``rate_ratio_60s``), encoder -> decoder gains per minute (B2's half of
   the drift gate), the mean E->E weight against its start. Output: <out>/drift_scaling.json (rows and gate_B1).

        uv run scripts/launch.py --unit bio-b1 --log outputs/drift_scaling/run.log -- \
            scripts/cartpole_replication.py --drift --drift-seeds 0 1 2 --drift-scales 0 1 \
            --drift-homeo 0 1e-5 1e-4 --drift-seconds 600 --workers 4 --out outputs/drift_scaling
   Result (24 Nov 2026): gate B1 not passed; the controls behave. Last-minute rate per electrode against rest,
   cultures 0 / 1 / 2: no plasticity 1.10 / 1.14 / 0.97; scaling alone 1.04 / 1.05 / 1.01 (0.99-1.07 in every
   minute; neurons above 1.5x their own target 19-25% -> 0.5-1%); STDP x1 with the old 5 s step 6.97 / 9.10 / 4.54;
   STDP x1 with scaling at eta 1e-5 7.40 / 9.59 / 5.96, no better than the old step: 5-9x rest by minute 2 and about
   flat after, mean E->E weight 1.47-1.75x its start (the old step holds it at 0.99x). Exploratory eta 1e-4: 1.92 /
   1.86 / 1.92, about 1.9x from minute 5 on, E->E weight 0.52-0.68x. Scaling alone doesn't mute the inputs (own gains
   1-3 spikes per pulse, as with no plasticity). With STDP and eta 1e-4 the pathways mix: cross gains climb from ~1 to
   2-5 spikes per pulse over the 10 minutes (no plasticity: 0-2), and in culture 2 one decoder ends up following the
   other encoder (10.1 against 1.5). Ten times the eta cut the excess over rest from 6.7x to 0.9x: the brake works,
   but against a push that doesn't let up. Likely reason, not tested: our STDP is additive (a fixed
   step per pairing) and the scaling multiplicative, so they balance only at a rate error that grows as eta falls
   and as the weights shrink. A rate brake also does nothing for the pathways.
   Check before closing track B (24 Nov 2026; B2's first candidate, run early and labelled as such): does
   weight-dependent STDP close the gap? ``--stdp-rule multiplicative`` (depression scales with the weight, van Rossum
   et al. 2000) with the same scaling, cultures and folder. Arms without STDP don't involve the rule, so they are
   named without it and B1's no-plasticity and scaling-alone arms are reused (their rows say "additive"). New arms
   per culture: weight-dependent STDP x1 with the old 5 s step (control; culture 0 ran to ~22 Hz before), with
   scaling at eta 1e-5 (gated) and 1e-4 (exploratory). Reading, fixed before the run: it supports opening B2 only if,
   at eta 1e-5, the rate gate above holds in all 3 cultures and at most 1 of the 12 pathways (own and cross, 4 per
   culture) has moved against the no-plasticity arm: last 3 minutes, beyond 2 SE from both arms' per-minute spread
   over minutes 2-10 (``pathways_vs_off``, bl1_rep_analysis.py's test). On B1's rows that test finds 2 of 4 moved in
   every culture at every setting with additive STDP, and 1 of 24 with scaling alone (the noise floor). Otherwise
   the track closes, as its rule says. Output: <out>/drift_multiplicative_scaling.json.

        uv run scripts/launch.py --unit bio-b1m --log outputs/drift_scaling/run_multiplicative.log -- \
            scripts/cartpole_replication.py --drift --stdp-rule multiplicative --drift-seeds 0 1 2 \
            --drift-scales 0 1 --drift-homeo 0 1e-5 1e-4 --drift-seconds 600 --workers 4 --out outputs/drift_scaling
   Result (24 Nov 2026): the reading fails, so track B closes. Last-minute rate against rest, cultures 0 / 1 / 2:
   weight-dependent STDP with the old step 3.98 / 3.85 / 2.59 (control); with scaling at eta 1e-5 4.19 / 5.36 / 3.63
   (additive: 7.40 / 9.59 / 5.96), above the old step again, E->E weight 1.11-1.31x; 6 of 12 pathways moved (2 of
   4 in every culture: a cross gain up by 14-18 spikes per pulse). Exploratory eta 1e-4: 1.40 / 1.45 / 1.32, within
   the band in 29 of 30 minutes (1.05-1.50; culture 1 minute 8 at 1.56), E->E weight 0.67-0.76x, inputs still
   answered (own gains 2.5-5.8): the first setting that holds rates. But the pathways still merge: 2 of 4 moved in
   every culture, and in each culture one encoder drives both decoders by the end (the other decoder: 6.2 / 5.8 /
   5.1 spikes per pulse from the wrong encoder against 1.3 / 0.9 / 1.1 from its own); separate in 0 of 30 minutes.
   So the likely reason above holds for the rates (weight dependence plus faster scaling brings them in), and the
   pathways are a separate problem: under repeated stimulation our STDP hands the network to whichever input drives
   it, whatever holds the rates.
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

Logs: run_raw_decoder.log (round 1), run_rest.log (round 1), run_task.log (calibration + round 1 null); the runs
were stopped there because the decoders pinned. drift.json, drift_multiplicative.json, drift_istdp_clamped10.json,
drift_istdp.json: the drift tests; outputs/drift_scaling/drift_scaling.json and run.log: B1;
drift_multiplicative_scaling.json and run_multiplicative.log there: the weight-dependent check.

Stop and resume: every job checkpoints (device, plasticity, script state) to <out>/ckpt every 5 minutes of wall
time, and on SIGTERM at the next 200 ms read, then exits. Run the same command again to resume; finished jobs are
kept in <out>/jobs and skipped. Result and checkpoint files are named by every setting the job runs with (a digest
after the arm name, ``job_key``), so a rerun with any setting changed runs anew; code changes are not part of the key,
so rerun into a fresh --out after changing the model. A stop during a job's setup (characterisation, calibration, ~15 min) restarts that
job's setup.

    uv run scripts/cartpole_replication.py
    uv run scripts/cartpole_replication.py --drift --drift-scales 1 --drift-pairs 0.01:30 0.1:100 --drift-seconds 600
    systemctl --user stop <unit>   # graceful: saves, then exits; relaunch the same command to resume

Long runs go through the launcher (smoke test, detached unit, start-up check) and the watchdog:

    uv run scripts/launch.py --unit bio-cp --log outputs/cartpole_replication/run.log -- \
        scripts/cartpole_replication.py --drift --drift-scales 1 --drift-pairs 0.01:30 --drift-seconds 600
    uv run scripts/watch.py --unit bio-cp --log outputs/cartpole_replication/run.log

Every job runs ``Guard`` checks every 30 s (finite state, weights within their caps and not piled up below them,
plausible rate) and writes a heartbeat to <out>/heartbeat. After a GUARD FAIL caused by a bug, delete the job's
checkpoint in <out>/ckpt before rerunning: it holds the state the bug produced.
"""

import argparse
import hashlib
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
    neuron = np.zeros(dev.culture.n)
    for i in range(200):
        b = dev.step(100.0)
        bins[i] = np.bincount(b.channels, minlength=dev.n_channels)
        neuron += dev.oracle_counts
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


# --- Stop and resume ------------------------------------------------------------------------------------------
STOP = {"requested": False, "phase": "idle"}


class Stopped(Exception):
    pass


def _on_term(signum, frame):
    STOP["requested"] = True  # in a running job: saved and stopped at the next read
    if STOP["phase"] == "setup":
        raise Stopped("setup")  # nothing worth saving yet; a rerun starts this job's setup over


def _start_job():
    if STOP["requested"]:
        raise Stopped("not started")
    STOP["phase"] = "setup"


def _worker_init():
    signal.signal(signal.SIGTERM, _on_term)


class Checkpoint:
    """Atomic snapshots of one job: device, plasticity and the script's state; every ``every_s`` of wall time and
    when a stop is requested (then raises ``Stopped``)."""

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


def job_key(name, settings):
    """File stem of a job's result and checkpoint: its readable name plus a digest of every setting it runs with.
    A rerun into the same --out with any setting changed (target floor, length, setup sizes, ...) then runs anew
    instead of reusing a result, or resuming a checkpoint, made with other settings."""
    return f"{name}_{hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]}"


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


# Encoders and decoders per culture seed, from characterise/configure in the replication (run_task.log).
DRIFT_CONFIGS = {0: ([136, 155], [137, 139]), 1: ([129, 34], [113, 35]), 2: ([217, 100], [233, 116])}
B1_BAND = (2 / 3, 1.5)  # gate B1: last-minute rate per electrode within a factor 1.5 of rest, either way


def resting_rates(dev, seconds=20.0):
    """Per-neuron rates (Hz) and the mean rate per electrode over ``seconds`` of spontaneous activity, plasticity
    off."""
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
    return np.linalg.lstsq(A, Y, rcond=None)[0][1:].T  # [decoder, encoder]: spikes per pulse


def per10(x):
    """Means over 10 s windows (50 reads); a run shorter than 10 s (smoke tests) gives one window."""
    k = max(len(x) // 50, 1) * 50 if len(x) >= 50 else len(x)
    return x[:k].reshape(-1, min(50, k), *x.shape[1:]).mean(1)


def gate_b1(rows, eta=1e-5):
    """Gate B1 (fixed before the run): in every culture, the last-minute rate per electrode stays within B1_BAND of
    rest (i) with scaling alone and (ii) with STDP x1 plus scaling, at bl1's eta. Other rates are reported only.
    Controls: the no-plasticity arm stays in the band (stimulation alone doesn't lift the rate), and STDP with the old
    5 s step runs away (> 1.5x: the test can catch what B1 must stop)."""
    lo, hi = B1_BAND
    by = {(r.get("seed", 0), r["scale"], r.get("homeo", 0.0)): r for r in rows if not r["eta"]}
    seeds = sorted({k[0] for k in by})
    inb = lambda r: r is not None and lo <= r["rate_ratio"] <= hi  # noqa: E731
    out = {"band": [round(lo, 3), hi], "eta": eta, "per_culture": {}}
    for sd in seeds:
        row = {name: (by.get((sd, x, h)) or {}).get("rate_ratio") for name, x, h in
               (("off", 0.0, 0.0), ("scaling", 0.0, eta), ("stdp_scaling", 1.0, eta), ("stdp_old_step", 1.0, 0.0))}
        row["pass"] = inb(by.get((sd, 0.0, eta))) and inb(by.get((sd, 1.0, eta)))
        row["controls_ok"] = inb(by.get((sd, 0.0, 0.0))) and (by.get((sd, 1.0, 0.0)) or {}).get("rate_ratio", 0) > hi
        out["per_culture"][str(sd)] = row
    out["tested"] = len(seeds) >= 3 and all(v["stdp_scaling"] is not None for v in out["per_culture"].values())
    out["pass"] = out["tested"] and all(v["pass"] for v in out["per_culture"].values())
    return out


def pathways_vs_off(rows, eta=1e-5, max_moved=1):
    """Pathways against the noise floor (bl1_rep_analysis.py's test): for each plastic arm, each encoder -> decoder
    gain (own and cross) over the last 3 minutes against the same culture's no-plasticity arm; it has moved when the
    difference exceeds 2 SE, SE from both arms' per-minute spread over minutes 2-10. Reported per culture and arm;
    ``reading`` (fixed before the weight-dependent check) holds when, at ``eta``, STDP plus scaling is in B1_BAND in
    every culture and at most ``max_moved`` of all its pathways moved."""
    by = {(r.get("seed", 0), r["scale"], r.get("homeo", 0.0)): r for r in rows if not r["eta"]}
    out = {"per_culture": {}}
    for sd in sorted({k[0] for k in by}):
        off = by.get((sd, 0.0, 0.0))
        if off is None:
            continue
        g0 = np.array(off["gain_60s"])  # [minute, decoder, encoder]
        cult = {}
        for (s2, x, h), r in sorted(by.items()):
            if s2 != sd or (x == 0 and not h):
                continue
            g = np.array(r["gain_60s"])
            se = np.sqrt(g[1:].std(0, ddof=1) ** 2 / 3 + g0[1:].std(0, ddof=1) ** 2 / 3)
            diff = g[-3:].mean(0) - g0[-3:].mean(0)
            moved = np.abs(diff) > 2 * se
            cult[f"x{x:g}_homeo{h:g}"] = {
                "own_diff": [round(float(diff[k, k]), 2) for k in (0, 1)],
                "cross_diff": [round(float(diff[k, 1 - k]), 2) for k in (0, 1)],
                "moved_own": int(moved[0, 0] + moved[1, 1]), "moved_cross": int(moved[0, 1] + moved[1, 0])}
        out["per_culture"][str(sd)] = cult
    key = f"x1_homeo{eta:g}"
    arms = [c.get(key) for c in out["per_culture"].values()]
    rates = [by.get((int(sd), 1.0, eta), {}).get("rate_ratio") for sd in out["per_culture"]]
    lo, hi = B1_BAND
    out["moved_at_eta"] = sum(a["moved_own"] + a["moved_cross"] for a in arms if a) if all(arms) else None
    out["reading"] = (len(arms) >= 3 and all(arms) and all(r is not None and lo <= r <= hi for r in rates)
                      and out["moved_at_eta"] <= max_moved)
    return out


def drift(job):
    import torch

    from biocompute.runguard import Guard

    from biocompute.culture.plasticity import Chain, SynapticScaling

    arm, key, spec, seconds, out, opts = job
    seed = spec.get("seed", 0)
    _start_job()
    g = Guard(f"drift_{arm}", out)
    enc, dec = DRIFT_CONFIGS[seed]
    ck = Checkpoint(out / "ckpt" / f"{key}.pt")
    rng = np.random.default_rng(1)
    dev, _, w_in0 = build(seed)
    stdp = make_stdp(spec)
    c = dev.culture
    stdp_on = spec["scale"] > 0 or spec["eta"] > 0
    # Per-neuron synaptic scaling (B1) replaces the 5 s total-input step (coach_probe.scale) when it is on.
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
        if spec["eta"]:
            stdp.set_targets(c, np.maximum(rates, spec["floor"]))
        if homeo is not None:
            homeo.set_targets(c, np.maximum(rates, spec["floor"]))
        S = {"i": 0, "X": [], "Y": [], "N": [], "last": {e: -1e9 for e in enc}, "th": 0.0,
             "neuron_rest_hz": round(float(rates.mean()), 3), "rest_hz_per_electrode": round(float(rest_hz), 2),
             "w_ee_start": float(c.w[ee].mean())}
    c.plasticity = rule if plastic else None
    n_reads = int(seconds * 5)
    STOP["phase"] = "resumable"
    while S["i"] < n_reads:
        ck.tick(dev, rule, lambda: S | {"rng": rng.bit_generator.state})
        i = S["i"]
        if i % 5 == 0:
            S["th"] = math.radians(rng.uniform(-12.0, 12.0))
        S["X"].append(schedule(dev, enc, S["th"], S["last"]))
        b = dev.step(200.0)
        S["Y"].append([int((b.channels == dec[0]).sum()), int((b.channels == dec[1]).sum())])
        S["N"].append(len(b.channels))
        if i >= n_reads - 300:
            S["neuron_end"] = S.get("neuron_end", 0) + dev.oracle_counts
        if (i + 1) % 25 == 0 and stdp_on and homeo is None:
            C.scale(c, w_in0)
        S["i"] += 1
        if g.due():
            hz = float(np.mean(S["N"][-50:])) / 0.2 / dev.n_channels
            g.culture(c, stdp if stdp_on else None, rate_hz=hz)
            g.beat(S["i"], n_reads, hz=round(hz, 2), w_inh=round(float(c.w[~c.syn_exc].mean()), 3))
    STOP["phase"] = "idle"
    X, Y, N = np.array(S["X"], float), np.array(S["Y"], float), np.array(S["N"], float)
    g = [np.round(gains(X[k:k + 300], Y[k:k + 300]), 2).tolist() for k in range(0, len(X) - 299, 300)] or [
        np.round(gains(X, Y), 2).tolist()]  # runs under 60 s (smoke tests): the whole run
    hz = N / 0.2 / dev.n_channels
    ratio = float(hz[-300:].mean() / max(S["rest_hz_per_electrode"], 1e-9))
    end = np.array(g[-1])
    specific = all(end[k, k] >= 2 * abs(end[k, 1 - k]) and abs(end[k, 1 - k]) <= 1.0 for k in (0, 1))
    inh = ~c.syn_exc
    w_inh = c.w[inh]
    end_hz = S["neuron_end"] / (min(300, n_reads) * 0.2)
    target = (stdp._targets if spec["eta"] else homeo._targets if homeo is not None else None)
    target = target.cpu().numpy() if target is not None else np.full(c.n, S["neuron_rest_hz"])
    diag = {"inh_at_cap": round(float((w_inh >= (spec["w_max_inh"] if spec["eta"] else c.cfg.w_max) - 1e-3)
                                      .float().mean()), 3),
            "inh_at_zero": round(float((w_inh <= 1e-3).float().mean()), 3), "w_inh_max": round(float(w_inh.max()), 2),
            "neuron_end_hz": round(float(end_hz.mean()), 3),
            "neurons_over_1p5x_target": round(float((end_hz > 1.5 * target).mean()), 3)}
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
    """Run, or resume, the cycle held in ``cs``; ``tick()`` checkpoints between reads."""
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
        # Episode end: everything below runs between checkpoints.
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
    """Run jobs in worker processes that save and stop on SIGTERM; this process ignores SIGTERM and waits."""
    if not jobs:  # nothing to do (all saved): an idle pool's shutdown can hang on workers that catch SIGTERM
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
        # With STDP off (scale 0) the rule doesn't act: such arms say "additive" and are shared across rules.
        sp = {"scale": scale, "rule": a.stdp_rule if scale else "additive", "eta": eta, "w_max_inh": cap,
              "floor": a.target_floor}
        return sp | ({"homeo": homeo} if homeo else {}) | ({"seed": seed} if seed else {})

    if a.drift:
        arms = []
        pairs = [tuple(map(float, p.split(":"))) for p in a.drift_pairs] if a.drift_pairs else [
            (eta, a.w_max_inh) for eta in a.drift_etas]
        # Rates in the order given (B1's 1e-5 before the exploratory ones), each over every culture.
        for h, seed, x, (eta, cap) in itertools.product(a.drift_homeo, a.drift_seeds, a.drift_scales, pairs):
            sp = spec(x, eta, cap, h, seed)
            secs = a.drift_seconds or (120.0 if x in (0.0, 1.0) else 300.0 if x >= 0.3 else 600.0)
            name = f"x{x:g}" + spec_tag({**sp, "scale": 1.0}) + (f"_s{seed}" if seed else "")
            key = job_key(f"drift_{name}", {**sp, "seconds": secs, "rest_s": sizes["rest_s"]})
            arms.append((name, key, sp, secs, out, sizes))
        if a.smoke:  # the arm with the most machinery: STDP and scaling together if asked for
            arms = [max(arms, key=lambda j: (j[2]["scale"] > 0) + bool(j[2].get("homeo")))]
        todo = [j for j in arms if not (out / "jobs" / f"{j[1]}.json").exists()]
        pool_map(_drift_safe, todo, a.workers)
        res = [json.loads((out / "jobs" / f"{j[1]}.json").read_text()) for j in arms]
        name = "drift" + ("" if a.stdp_rule == "additive" else f"_{a.stdp_rule}") + (
            "_istdp" if any(eta for eta, _ in pairs) else "") + ("_scaling" if any(a.drift_homeo) else "")
        summary = {"gate_B1": gate_b1(res), "pathways": pathways_vs_off(res)} if any(a.drift_homeo) else {}
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
