"""Drift test on bl1 (github.com/m9h/bl1, MIT): does its culture hold rates and pathways under repeated stimulation
with its plasticity on? Our model failed this test (additive STDP runs away; the iSTDP fix closed without a pass in
9355cbf). bl1 lists homeostatic scaling, the piece our model lacks, so its culture gets the same test.

bl1 code, used as is (clone at /mnt/datadisk/repos/bl1, commit 91f7c01, run in its own venv with jax[cuda12]):
- Culture: configs/wagenaar_calibrated.yaml, built the way scripts/tune_bursting.py builds it (the calibration code
  behind that yaml): place_neurons (3000 x 3000 um), create_population, build_connectivity (lambda 200 um, p_max 0.21,
  g_exc 0.12, g_inh 0.36), Tsodyks-Markram STP (excitatory U 0.30, tau_rec 800 ms; inhibitory U 0.04, tau_rec 100 ms,
  tau_fac 1000 ms), excitatory weights split AMPA 63% / NMDA 37%, background current 1.0 + 3.0 N(0, 1) per neuron per
  step, dt 0.5 ms, 5,000 neurons, seed 42. One step = tune_bursting's step_fn (izhikevich_step, stp_step, ampa_step,
  gaba_a_step, nmda_step; GABA_B left out, as there).
- MEA: MEA("cl1_64ch") (8 x 8, 200 um pitch); spikes per electrode = spikes of the neurons in
  build_neuron_electrode_map at its 100 um detection radius (ClosedLoop's decoding: no noise, no sorting).
- Stimulation pulse: what ClosedLoop delivers for one sensory pulse: STIM_AMPLITUDE_MV (150) added to the input of
  every neuron in the electrode's map for one game step (20 ms).
- STDP: stdp_update with STDPParams() defaults on the excitatory weights every step, as ClosedLoop applies it.
- Homeostatic scaling: bl1 ships it (plasticity/homeostatic.py) but no loop calls it, so the wiring is ours:
  update_rate_estimate every step (tau 10 s, init_homeostatic_state at its 5 Hz), homeostatic_scaling with
  HomeostaticParams() defaults after every 200 ms read with dt_ms = 200, i.e. the documented continuous rule
  dW/dt = eta (r_target - r) W with eta per ms.
Ours (glue only): time chunked into 200 ms reads inside one jitted scan per 5 s, background noise drawn per step from
fold_in(key, step) (tune_bursting pre-draws one (T, N) array; same distribution), and the protocol below.

Setup job (plasticity off), once:
1. Positive control: 60 s spontaneous. bl1 reports for this config 1.6 Hz mean rate, 6-11 bursts/min, IBI 4.5-10 s
   (bl1.analysis.bursts.detect_bursts on the population count, threshold 1.5 SD, 50 ms minimum, as the yaml). GUARD
   FAIL if the rate or burst rate leaves Wagenaar's ranges (0.1-5 Hz, 0.2-20 /min): the port would not be bl1's culture.
2. Electrode selection as in cartpole_replication.py (Robbins et al. 2026): the 32 most active electrodes, each
   stimulated 25x at 4 Hz, C1(i, j) = P(spike on j within 10 ms), burst probability per encoder, configure(): encoder
   a -> decoder b and c -> d maximising C1(a,b) + C1(c,d) - C1(a,d) - C1(c,b).

Drift protocol, per arm (cartpole_replication.py --drift, unchanged): fresh culture, 2 s warm-up, 20 s rest
(plasticity off) for the resting rates, then ``--seconds`` (600) of the calibration stimulation: a tilt drawn
uniformly in +-12 deg every 1 s, both encoders pulsed at regular intervals at f = (7 (0.15 -/+ sin theta))^2 Hz.
Rates per 10 s, encoder -> decoder gains per 60 s (decoder counts per 200 ms read regressed on pulses per read).
Gate (fixed before our iSTDP sweep): rate per electrode in the last 60 s <= 1.5x the resting rate, and in the last 60 s
each decoder's own gain >= 2x its cross gain with the cross gain <= 1 spike per pulse. bl1's pulses are much stronger
than ours, so stimulation alone may lift the rate; each row also reports the rate against the no-plasticity arm
(``vs_off``), the rate without the two encoder electrodes, and the last vs the first minute of stimulation.

Arms: off, stdp, homeo, stdp_homeo. ``--variant shipped``: bl1's defaults. ``--variant fixed``: two bl1 bugs fixed
(below): STDP scaled to this culture's weights (A+-, w_max x 0.12/0.05 = 2.4, so w_max 0.24, also the homeostatic
cap), and each neuron's homeostatic target its own resting rate (floor 0.5 Hz), the rate estimate starting there.
``--controls`` adds positive controls to the fixed variant: stdp_ltp (bl1's STDP with tau+ and tau- swapped, so
potentiation-dominated) and stdp_ltp_homeo. ``--pulse-ms 0.5``: a one-step pulse (~1 spike per stimulated neuron, like
a real biphasic MEA pulse) instead of ClosedLoop's 20 ms. ``--seed`` takes several cultures; one invocation runs every
variant x seed x pulse, setups shared per (seed, pulse). Rows also carry lag-1 gains (a 20 ms pulse spills into the next
read) and their per-minute trends; per-read series go to jobs/drift_<arm>_series.npz.

Checks (scripts/bl1_checks.py, all pass): with the same noise our step is spike-for-spike bl1's tune_bursting step
(20 s, 158,894 spikes, 0 of 40,000 steps differ); with STDP and pulses it is bl1's own ClosedLoop _make_scan_step (50
reads, 46,403 spikes, identical weights, at bl1's DishBrain weights); homeostasis matches an independent reference built
from bl1's calls; reads and fine paths agree; a 30 s plastic run repeats bit for bit; schedule and pulse delivery as
specified; the rate estimate tracks measured rates (r 0.986) and scaling moves every unclipped E->E weight the right
way. bl1's own tests for the modules used: 91/91 pass on JAX 0.12.1. Stop/resume: bit-identical to an uninterrupted
run (CPU). Seed-42 rerun vs first run (GPU): every rate within 0.01.
Positive control per culture (60 s, our noise stream) vs bl1's own tune_bursting code (30 s, its stream), seeds 42-46:
1.51-1.67 vs 1.56-1.71 Hz; bursts/min 1/14/5/10/20 vs 4/16/6/18/16. The yaml's "6-11/min" holds for some seeds only.

Bugs found in bl1 (commit 91f7c01):
1. STDPParams() caps excitatory weights at 0.1, below the calibrated culture's (0.108-0.132): the first STDP step
   clips them, an 18% cut in excitation that isn't plasticity.
2. HomeostaticParams() targets 5 Hz in every neuron; the calibrated culture fires 1.6 Hz, so scaling pushes all E->E
   weights up to its 0.2 cap instead of holding a set point.
3. Homeostatic scaling (and structural plasticity) is called by no simulation loop; the README's "4 timescales"
   run as STDP + STP only.
4. ClosedLoop's step feeds the full excitatory weight to both AMPA and NMDA (the calibration splits it 63/37): the
   calibrated culture in ClosedLoop's step fires 183 Hz per neuron with no stimulation (1.51 Hz in the calibration
   step), a seizure. (ClosedLoop's own config uses g_exc 0.05; the validated culture can't be dropped into it.)
5. STIM_AMPLITUDE_MV = 150 is added as current for a whole 20 ms game step: each of the ~27 neurons under the electrode
   fires 16-20 spikes per pulse (~900 Hz), and 27% of the culture spikes within 200 ms of one pulse.
Not exercised: simulate() re-initialises STP state on every call without returning it, so chunked runs reset depression.

Results, 600 s per arm, 5 cultures (seeds 42-46), outputs/bl1_drift/rep (summary_*.json); median [range] over
cultures; rate per electrode, last 60 s; vs off = against the same culture's no-plasticity arm:

    arm                    20 ms pulse (ClosedLoop's)                   0.5 ms pulse
                           vs off  non-enc vs rest  last/first min      vs off  non-enc vs rest  last/first min
    off                    1.00    1.23 [1.17,1.29]  0.98               1.00    1.05 [1.01,1.06]  1.00
    shipped stdp           0.66    0.57 [0.53,0.58]  0.91               0.54    0.55              0.86
    shipped homeo          1.43    2.11 [1.90,2.29]  1.07               1.91    2.05              1.13
    shipped stdp_homeo     0.75    0.75              0.98               0.65    0.67              1.00
    fixed stdp             1.01    1.29 [1.09,1.51]  1.02               1.25    1.35 [1.22,1.52]  1.40 [1.24,1.44]
    fixed homeo            0.88    1.01 [1.00,1.01]  0.93               0.95    1.00 [1.00,1.01]  0.98
    fixed stdp_homeo       0.99    1.25              1.01               1.20    1.29              1.25
    fixed stdp_ltp (+)     1.38    2.06              0.98               2.11    2.23              1.01
    fixed stdp_ltp_homeo   1.25    1.74              0.96               1.55    1.65              0.90

- Gate: passed only by arms that run down (shipped stdp 5/5 at 20 ms and 2/5 at 0.5 ms, shipped stdp_homeo 1/5 each:
  the bug-1 clip plus depression); no fixed arm passes in any culture. The pairs
  are mostly not specific even with plasticity off (lag gains: 2/5 cultures at 20 ms, 0/5 at 0.5 ms), so the
  specificity half can't be met for reasons that have nothing to do with plasticity.
- Rates: fixed homeostasis alone holds the non-encoder electrodes at rest in every culture with both pulses (1.00-1.01x),
  so per-neuron synaptic scaling works as a rate controller. It does not hold them with STDP on: with the realistic
  pulse fixed STDP lifts rates 1.25x the off arm and still climbing after 10 min (last/first 1.24-1.44), and
  homeostasis only trims that (1.20, 1.25). The positive control shows the test catches potentiation (1.38x / 2.11x);
  homeostasis contains part of it (1.25x / 1.55x). Nothing runs away the way our model does (7x within minutes).
- Pathways, last 3 min vs the off arm with SE from minutes 2-10 (10 pathways per arm, |diff| > 2 SE): STDP moves each
  decoder's own gain by ~2 spikes per pulse (fixed stdp: 20 ms median -2.2, 5/10 down; 0.5 ms median +2.4, 5/10 up;
  stdp_homeo the same), and cross gains scatter both ways (fixed stdp 20 ms: 3 up, 2 down, median -0.08). The first
  run's "stronger encoder takes over the other decoder" (seed 42) is one culture's pathway, not a general effect.
Conclusion: bl1's culture is not a model in which learning questions can be asked either: pathways move by a few
spikes per pulse within 10 minutes of the same stimulation, homeostasis or not, and with realistic pulses rates drift
up too. Per-neuron synaptic scaling (bug 2 fixed) is the piece worth porting: it holds rates on its own.

Stop and resume: every arm checkpoints (pickled state) to <out>/ckpt every 5 minutes of wall time and on SIGTERM at the
next 5 s chunk, then exits; queued arms see a shared stop flag and return at once (stop takes ~1 s; tested with more
arms than workers). Run the same command again to resume. Finished arms are kept in <out>/jobs. A stop during the
setup job restarts that job. Guards every 30 s: finite state, weights within bl1's bounds, plausible rate.
``kill -USR1 <pid>`` dumps every thread's stack to the log.

    uv run scripts/launch.py --unit bio-bl1-rep --log outputs/bl1_drift/rep/run.log -- \\
        --project /mnt/datadisk/repos/bl1 --no-sync python scripts/bl1_drift.py --variant shipped fixed \\
        --seed 42 43 44 45 46 --pulse-ms 20 0.5 --controls --workers 3 --root outputs/bl1_drift/rep
    uv run scripts/watch.py --unit bio-bl1-rep --log outputs/bl1_drift/rep/run.log
    systemctl --user stop bio-bl1-rep   # graceful; relaunch the same command to resume
"""

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
sys.path.insert(0, str(HERE.parent / "src"))  # biocompute.runguard (no torch at import)

N, SEED, DT = 5000, 42, 0.5
READ_MS = 200.0
STEPS = int(READ_MS / DT)  # per read
CHUNK = 25  # reads per jitted call (5 s)
PULSE_STEPS = int(20.0 / DT)  # ClosedLoop game step; --pulse-ms changes it (set_pulse, in each worker)
NMDA_RATIO, BG_MEAN, BG_STD = 0.37, 1.0, 3.0
G_EXC, PONG_G_EXC = 0.12, 0.05  # wagenaar_calibrated.yaml; dishbrain_pong.yaml, the culture STDPParams() pairs with
TARGET_FLOOR = 0.5  # Hz, lowest homeostatic target in --variant fixed (as our iSTDP's --target-floor)
# arm: (STDP, homeostasis, potentiation-dominated STDP). The *_ltp arms are positive controls (opt-in via --arms):
# bl1's STDP with tau_plus and tau_minus swapped (A+ tau+ / A- tau- = 2.4), which should run away.
ARMS = {"off": (False, False, False), "stdp": (True, False, False), "homeo": (False, True, False),
        "stdp_homeo": (True, True, False), "stdp_ltp": (True, False, True), "stdp_ltp_homeo": (True, True, True)}
DEFAULT_ARMS = ["off", "stdp", "homeo", "stdp_homeo"]


def set_pulse(ms):
    global PULSE_STEPS
    PULSE_STEPS = max(1, int(round(ms / DT)))


# --- bl1 culture ------------------------------------------------------------------------------------------------
class Culture:
    """bl1's Wagenaar-calibrated culture (tune_bursting.py) with ClosedLoop's pulses and bl1's plasticity."""

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
        if variant == "fixed":  # bl1's STDP in this culture's weight units; one E->E cap for both rules
            k, d = G_EXC / PONG_G_EXC, STDPParams()
            self.stdp_params = d._replace(A_plus=d.A_plus * k, A_minus=d.A_minus * k, w_max=d.w_max * k)
            self.homeo_params = HomeostaticParams(w_max=self.stdp_params.w_max)  # r_target: set_targets()
        if ltp:
            p_ = self.stdp_params
            self.stdp_params = p_._replace(tau_plus=p_.tau_minus, tau_minus=p_.tau_plus)
        self.w_bounds = (min(self.stdp_params.w_min, self.homeo_params.w_min),
                         max(self.stdp_params.w_max, self.homeo_params.w_max, float(self.w0.max())))
        self.carry = (state, create_synapse_state(N), init_stp_state(N, self.stp_params), init_stdp_state(N),
                      init_homeostatic_state(N), self.w0)
        self.t = 0  # global step: the noise of step t is fold_in(noise_key, t)
        self._fns = {}

    def set_targets(self, rates, reset_estimate=True):
        """--variant fixed: each neuron's homeostatic target is its own resting rate (floor TARGET_FLOOR), and the
        rate estimate starts there instead of at 5 Hz. Call before the first plastic read (the jitted reads capture
        the parameters)."""
        import jax.numpy as jnp

        from bl1.plasticity.homeostatic import HomeostaticState

        tgt = jnp.asarray(np.maximum(rates, TARGET_FLOOR), dtype=jnp.float32)
        self.homeo_params = self.homeo_params._replace(r_target=tgt)
        if reset_estimate:
            ns, ss, stp_st, sd, _, w = self.carry
            self.carry = (ns, ss, stp_st, sd, HomeostaticState(rate_estimate=tgt), w)
        self._fns = {k: f for k, f in self._fns.items() if k in ("fine", (False, False))}

    # One step = tune_bursting.step_fn, plus ClosedLoop's STDP call and the homeostatic rate estimate.
    # closedloop=True (bl1_checks.py only): ClosedLoop's own step instead, i.e. the full excitatory weight into both
    # AMPA and NMDA and GABA_B updated, to compare against bl1's loop.controller._make_scan_step.
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
        """Jitted: K reads of 200 ms; returns spike counts per neuron per read. Homeostatic scaling after each read."""
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

        def run(carry, t0, flags):  # flags (K, STEPS, E)
            k = flags.shape[0]
            return jax.lax.scan(read, carry, (t0 + STEPS * jnp.arange(k), flags))

        return jax.jit(run)

    def _fine_fn(self):
        """Jitted, plasticity off: per-step spike counts per electrode and in the whole culture."""
        import jax
        import jax.numpy as jnp

        step = self._step(False, False, fine=True)
        return jax.jit(lambda carry, t0, flags: jax.lax.scan(step, carry, (t0 + jnp.arange(flags.shape[0]), flags)))

    def reads(self, flags, stdp_on=False, homeo_on=False):
        """Advance len(flags) reads; flags (K, STEPS, E) bool. Returns counts per neuron per read (K, N)."""
        key = (stdp_on, homeo_on)
        if key not in self._fns:
            self._fns[key] = self._reads_fn(stdp_on, homeo_on)
        self.carry, counts = self._fns[key](self.carry, self.t, flags)
        self.t += flags.shape[0] * STEPS
        return np.asarray(counts, dtype=np.int64)

    def fine(self, flags):
        """Advance len(flags) steps, plasticity off; flags (T, E). Returns (T, E) and (T,) spike counts."""
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


# --- Stop and resume (as cartpole_replication.py) ---------------------------------------------------------------
STOP = {"requested": False, "phase": "idle"}


class Stopped(Exception):
    pass


EVENT = {"stop": None}  # in workers: the shared stop flag (a RawValue byte the main process sets on SIGTERM)


def stop_requested():
    """SIGTERM reached this process, or the main process saw one: workers spawned after the signal (one per arm,
    maxtasksperchild=1) never receive it themselves, so they read the shared flag. It is a lock-free RawValue: a
    multiprocessing Event takes a lock in set(), and SIGTERM arrives twice (systemd and uv's forward), so the
    re-entered handler deadlocked on it and every worker blocked in is_set()."""
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
    faulthandler.register(signal.SIGUSR1)  # `kill -USR1 <pid>` dumps every thread's stack to the log


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


# --- Protocol (ported from cartpole_replication.py) -------------------------------------------------------------
def configure(cand, rate, first, burst):
    """cartpole_replication.configure, unchanged."""
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
    """cartpole_replication.schedule on bl1 pulses: queue the encoder pulses of the read starting at ``now_ms`` (pole
    angle ``th``) as 20 ms on-flags; a pulse's tail past the read goes to ``spill`` (next read). Pulses per encoder."""
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
    return np.linalg.lstsq(A, Y, rcond=None)[0][1:].T  # [decoder, encoder]: spikes per pulse


def gains_lag(X, Y):
    """Gains with a one-read lag: a 20 ms pulse late in a read spills into the next one, and so can its response.
    Y_t ~ X_t + X_{t-1}; returns the summed coefficients [decoder, encoder] (spikes per pulse over 400 ms)."""
    A = np.column_stack([np.ones(len(X) - 1), X[1:], X[:-1]])
    c = np.linalg.lstsq(A, Y[1:], rcond=None)[0]
    return (c[1:3] + c[3:5]).T


def trend(series):
    """OLS slope per window and its SE over a list of per-window values."""
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
    """Positive control (spontaneous statistics vs bl1's reported ones) and electrode selection."""
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
    cu.reads(cu.quiet(10))  # 2 s warm-up
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
        el = el.reshape(reps, isi, cu.n_el)[:, :, cand]  # [rep, step, candidate]
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
        cu.reads(cu.quiet(10))  # 2 s warm-up
        rest = cu.reads(cu.quiet(int(sz["rest_s"] * 5)))  # plasticity off
        rates = rest.sum(0) / sz["rest_s"]
        rest_el = cu.electrode_counts(rest)
        S = {"i": 0, "X": [], "Y": [], "N": [], "N_noenc": [], "last": {e: -1e9 for e in enc}, "th": 0.0,
             "spill": np.zeros((PULSE_STEPS, cu.n_el), bool), "w_exc_60s": [], "neuron_rest_hz": rates,
             "rest_hz_per_electrode": float(rest_el.sum() / sz["rest_s"] / cu.n_el),
             "rest_hz_per_electrode_noenc": float(np.delete(rest_el, enc, 1).sum() / sz["rest_s"] / (cu.n_el - 2))}
        if variant == "fixed":
            cu.set_targets(rates)
    if saved and variant == "fixed":
        cu.set_targets(S["neuron_rest_hz"], reset_estimate=False)  # the estimate comes back with the carry
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
    gl_trend = [[trend([g[k, i] for g in gl]) for i in (0, 1)] for k in (0, 1)]  # [decoder][encoder]: slope/min, SE
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
    """Run jobs in worker processes (a fresh one per job) that save and stop on SIGTERM. This process doesn't die on
    SIGTERM: it sets the shared stop event, so queued jobs start, see it and return at once, and waits."""
    ctx = mp.get_context("spawn")
    if "ev" not in _MAIN_EVENT:  # one flag for every pool of this run, so a stop between pools still counts
        _MAIN_EVENT["ev"] = ctx.RawValue("b", 0)

        def on_term(signum, frame):
            _MAIN_EVENT["ev"].value = 1  # no locks in a signal handler (see stop_requested)

        signal.signal(signal.SIGTERM, on_term)
    ev = _MAIN_EVENT["ev"]
    if ev.value == 1:
        print(json.dumps({"stopped": "before start", "resume": "run the same command again"}), flush=True)
        raise SystemExit(0)
    # close() + join(), not the context manager: its terminate() SIGTERMs the workers and joins them, and a worker
    # that catches SIGTERM to save its state keeps running, so the join hung whenever a replacement worker had
    # finished starting (every stop; normal ends only by timing luck). close() lets them take the end-of-queue
    # sentinel and exit by themselves.
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
    # Setup is plasticity-off, so one per (seed, pulse), shared by the variants.
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
