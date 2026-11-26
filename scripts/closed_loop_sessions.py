"""Track A, step A2 (docs/route-to-computation.md, "Order of work without a dish"): sessions and drift. Does the
simulated culture still relay the IMU after it ages, or after a new dish replaces it, and does one short open-loop
calibration plus one weight edit restore it? This tests route 3's claim that a session needs only one weight edit.

Everything not listed here is A1b exactly (scripts/closed_loop_vestibular.py, ``--edit affine``): the dish setup, the
ridge decoder, the affine edit of the teacher's first layer, the controller, stand_push for 20 s with 1-step action
latency, and plasticity off.

Conditions ("sessions"), each starting from A1b's warm day-0 state of the same configuration:

- ``day0``     no change (the control for the short calibration);
- ``day1``, ``day3``, ``day7``: ``biocompute.culture.aging.age(days, seed=11)``. Excitability drifts by
  0.5 x sqrt(days) current units per neuron, and 2% of the neurons die per day. All three come from the same day-0
  state with the same aging seed, so they form one dose series: the same drift direction, larger with days, and nested
  sets of dead neurons;
- ``newdish``  a new culture (culture seed 1: new positions and wiring) on the same MEA, electrodes and encoder.

Every session then runs the same open-loop history: 1 s with no input, a 60 s replay of teacher stand_push episodes
(replay steps 15587-18586, the start of A1b's held-out part), then 1 s of the standing IMU. Every episode of that
session starts from this warm state.

Arms, 40 fresh seeds (9200-9239), per configuration and condition:

- ``stale``   A1b's decoder and A1b's affine edit, unchanged;
- ``recal``   a decoder refitted on the session's 60 s replay (penalty by contiguous 5-fold CV) and an affine edit
  refitted on its out-of-fold predictions around the standing posture (as A1b, which fitted it on held-out replay).

Plus the ideal-IMU and zero-action baselines on the same seeds. The edit is the affine one, not the plan's 1/gain edit:
A1 showed the 1/gain edit fails (0/40) because of the decoder's offset. The affine edit is still a single edit of the
teacher's first layer.

Main configuration: 1 well x 32 electrodes (``s32``), the one that passed A1's bar. 1 x 12 failed A1 (23/40 at day 0),
so a drift test on it can't reach the bar; it can still run as a comparison (``--configs s12 s32``).

Gate A2 (fixed before any closed-loop seed): on s32, ``recal`` reaches >= 34/40 full runs in each of day1, day3, day7
and newdish. The gate tests the claim only where ``stale`` drops below 34/40; if it drops nowhere, these aging rates
don't break the relay and the claim is untested, which is reported as such. If ``recal`` fails at day0, 60 s of
calibration is too short, whatever aging does.

Positive controls (GUARD FAIL otherwise): at day0, the stale decoder on the session's replay reproduces A1b's held-out
gains around standing (within 0.15), and the replay rate matches A1b's (within 15%). Aging is checked against its
nominal dose: the share of neurons alive is (0.98)^days, and the spread of excitability is 0.5 x sqrt(days)
(within 10%). In every step, as in A1, no weight moves, the rate stays in range and the decoded IMU stays finite.

Run (guarded, resumable):

    uv run scripts/launch.py --unit bio-a2 --log outputs/closed_loop_sessions/run.log -- \\
        scripts/closed_loop_sessions.py --workers 3
    uv run scripts/watch.py --unit bio-a2 --log outputs/closed_loop_sessions/run.log
    systemctl --user stop bio-a2     # graceful; relaunch the same command to resume

Outputs (outputs/closed_loop_sessions): sessions/<cfg>_<cond>_<key>.json (+ .pkl: decoder, edit, warm state),
jobs/<cfg>/<cond>/<arm>/<seed>_<key>.json (one per episode), results.json (the table and the gate). <key> is a
digest of the settings the file depends on (``job_key``: data, A1's folder, recalibration steps; for an episode also
its length), so a rerun into the same --out with a setting changed runs anew. A session's 60 s
replay takes about 2 minutes and is redone if stopped (no checkpoint); a stopped episode is redone; finished
episodes are kept.

Results (24 Nov 2026, s32, seeds 9200-9239; results.json). Full 20 s runs of 40:

| Session  | Neurons alive | Excitability SD | Replay rate (Hz) | Stale         | Recalibrated (60 s) |
|----------|---------------|-----------------|------------------|---------------|---------------------|
| day 0    | 100%          | 0               | 66.0             | 39/40         | 36/40               |
| day 1    | 98.0%         | 0.50            | 64.6             | 36/40         | 36/40               |
| day 3    | 94.1%         | 0.86            | 62.8             | 39/40         | **33/40**           |
| day 7    | 86.8%         | 1.32            | 58.5             | 39/40         | 37/40               |
| new dish | 100%          | 0               | 66.8             | **0/40**, 3.3 s | 36/40             |

Ideal IMU 40/40, zero action 0/40 (5.0 s).

Gate A2: not passed, by one episode. Recalibration falls short at day 3 (33/40); every other condition clears 34/40.
The gate was tested only on the new dish: there the stale relay fails completely (closed-loop biases of 0.8-2.6
spreads on four channels, falls in 2.7-3.9 s), and one 60 s calibration plus one edit brings it back to 36/40. Aging up to 7 days at
aging.py's rates doesn't break the stale relay (36-39/40). Its open-loop gains stay within 0.1 of day 0; day 7 adds a
gyro_x bias of -0.4 spreads.

Pooled, the stale relay reaches 153/160 on the old dish, and the recalibrated one 178/200 across all five sessions.
The gap is the short calibration, not aging: the 60 s decoder reads grav_x with closed-loop noise 0.84-0.89 spreads,
against 0.53-0.55 for the decoder fitted on 312 s, and it scores 36/40 even at day 0. Its failures fall on different
seeds in each session (22 failures on 16 seeds; 9204 fails in three sessions), as expected of a marginal channel.

Positive controls held: day-0 stale gains reproduce A1b's within 0.04, the replay rate is the same (66.0 vs 66.0 Hz),
aging matches its nominal dose, and no weight moved.

Stop handling, tested on the run itself with more jobs than workers. During the sessions (5 jobs on 3 workers), the
stop took 2.1 s: the three running replays were redone, and the two queued jobs returned without building a culture.
During the episodes (48 jobs on 3 workers), it took 23.7 s: the 27 finished episodes were kept and the three running
ones were redone. Each queued job still starts a worker process before it sees the flag.

A2c: keep the old decoder, refit only the edit (pre-registered 25 Nov 2026, before its first seed; route doc
"Order" item 4.1, docs/literature-notes.md "Track A2")
--------------------------------------------------------------------------------------------------------------------

Why, from A2's stored sessions (sessions/*.json; open loop, in-range steps around standing, teacher spreads, channels
gyro x/y/z, grav x/y). On the new dish the stale decoder's output is shifted: offsets -1.01/-1.03/+2.49/-0.41/-0.14.
Once an affine edit corrects scale and offset, its error (noise / gain) is 0.68/0.73/0.70/0.98/0.59, against
0.62/0.75/0.75/1.70/0.71 for the decoder refitted on 60 s, and 0.53/0.52/0.62/0.69/0.49 for the stale decoder at
day 0. Closed loop on the new dish, grav_x noise / gain: stale 1.18, recal 1.32. Against it: the stale decoder's
noise is slower (correlated over 1.2-1.7 steps, against 0.8-1.05).

Sessions: rebuilt exactly as A2's (same conditions, aging, replay, warm state). Arms on s32, every condition:

- ``refit60`` (gated), ``refit30``, ``refit10``: A1b's decoder unchanged, applied to the session's replay. Only the
  affine edit (gain and intercept per IMU channel: a line through decoded vs true on the in-range steps around the
  measured standing posture, as A1b) is refitted, on the first 3000 / 1500 / 500 replay steps (60 / 30 / 10 s);
- ``recal``: A2's arm (decoder and edit refitted on the 60 s), on the same seeds, for a paired comparison;
- ``ideal_imu``, ``zero_action``: the baselines.

Fresh seeds 9300-9339 (A1 used 9000-39, A1b 9100-39, A2 9200-39), the same 40 for every arm. Fresh --out, since
job_key covers settings, not code:

    uv run scripts/launch.py --unit bio-a2c --log outputs/closed_loop_sessions_refit/run.log -- \\
        scripts/closed_loop_sessions.py --arms refit60 refit30 refit10 recal ideal_imu zero_action \\
        --seed-base 9300 --out outputs/closed_loop_sessions_refit --workers 3

Gate A2c: ``refit60`` reaches >= 34/40 full 20 s runs in every condition, day 0 included. Pass: route 3's session
protocol becomes "refit the edit" (two numbers per channel), and A2 is done. Fail: keep refitting the decoder, from
a longer calibration.

Positive controls (GUARD FAIL otherwise): each session equals A2's stored one exactly (decoder, edits, warm state,
open-loop statistics; ``--a2``); at day 0 the stale gains are within 0.04 of A1b's held-out ones and the replay rate
within 1% of A1b's 66.0 Hz; aging on its nominal dose; no weight moves; decoded IMU finite. Reported alongside: each
short edit's error on the rest of the 60 s replay (``rest_bias``, ``rest_rmse``: what the edited teacher sees minus
the truth), in every episode the same in closed loop (``edited_channel``; ``closed_loop_channel`` is the decoder's
raw output, as in A2), and per condition the seeds where refit60 and recal disagree.

Expected (reasoning, not measured): new dish >= 36/40. At day 0-7 the refitted edit stays close to A1b's (stale gains
within 0.1 of day 0 in A2), so refit60 should score near stale's 36-39/40 there. A 10 s fit should pin the offset to
~0.07 spread; whether 10 s pins the gain is uncertain.

Results A2c (run 25 Nov 2026, 15:26-17:57, 880 episodes; ``outputs/closed_loop_sessions_refit``). Full 20 s runs of
40, same seeds in every arm; the last column counts seeds full with only one of refit60 and recal:

| Session  | refit60 (gated) | refit30 | refit10 | recal | refit60 only / recal only |
|----------|-----------------|---------|---------|-------|---------------------------|
| day 0    | 37/40           | 40/40   | 38/40   | 36/40 | 4 / 3                     |
| day 1    | 39/40           | 39/40   | 40/40   | 36/40 | 4 / 1                     |
| day 3    | 40/40           | 38/40   | 39/40   | 34/40 | 6 / 0                     |
| day 7    | 37/40           | 36/40   | 40/40   | 35/40 | 4 / 2                     |
| new dish | 37/40           | 39/40   | 36/40   | 30/40 | 10 / 3                    |

Ideal IMU 39/40, zero action 0/40 (4.8 s).

Gate A2c: passed. refit60 reaches 37-40/40 in every condition (bar 34). Pooled over the five sessions: refit60
190/200, refit30 192/200, refit10 193/200, recal 171/200. Paired, 28 seeds stand only with refit60 against 9 only
with recal (sign test p = 0.003, treating the 200 pairs as independent though the 40 seeds repeat across sessions;
alone, only day 3 is significant, 6 / 0, p = 0.03). On these seeds the 60 s decoder refit fails the bar on the new
dish (30/40; A2's seeds gave 36/40): A2's verdict that 60 s is too short for a new decoder holds.

Against the prediction: new dish 37/40 (expected >= 36); days 0-7 37-40/40 (expected near stale's 36-39). The 10 s
edit leaves an offset of median 0.03, max 0.09 spread on the rest of the 60 s replay (21 of 25 channels <= 0.07; 30 s:
max 0.07). Its gain is within 7% of the 60 s fit on the aged dishes, but on the new dish the grav_x and grav_y gains
come out 22% and 29% low, so in closed loop the edited teacher sees those channels 1.41 and 1.43 times too large and the
robot stands 36/40, still over the bar. 30 s pins every gain to within 5%.

Why the edit beats the refit (closed loop, the edited channel, medians): the kept decoder reads grav_x with noise
0.67-0.69 spreads on the old dish and 0.98 on the new one, against 1.03-1.06 and 1.61 for the decoder refitted on 60 s.
The old decoder, fitted on 312 s, is the less noisy one, and its offsets (raw -1.0/-1.0/+2.5 spreads on the new dish,
-0.42 on gyro_x at day 7) are removed by the edit (edited offsets <= 0.09 spread with the 60 s edit, <= 0.13 with
any edit or with recal).

Positive controls held: all five sessions equal A2's stored ones exactly; day-0 stale gains within 0.04 of A1b's,
replay rate 65.99 vs 65.96 Hz; no GUARD FAIL.

Decision: route 3's session protocol is "refit the edit" (two numbers per IMU channel, from 30-60 s of replay; 10 s
suffices on an aged dish but leaves the new dish's gains loose). A2 is done. Limit: one new dish (one culture seed),
so the transfer of a decoder to a new culture is shown once, not across cultures.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_vestibular as clv  # noqa: E402

from biocompute.runguard import job_key  # noqa: E402

ROOT = clv.ROOT
SEEDS = list(range(9200, 9240))
CONDITIONS = {"day0": dict(days=0.0), "day1": dict(days=1.0), "day3": dict(days=3.0), "day7": dict(days=7.0),
              "newdish": dict(days=0.0, culture_seed=1)}
AGED = ["day1", "day3", "day7", "newdish"]
REFIT = {"refit60": 3000, "refit30": 1500, "refit10": 500}  # A2c: replay steps (20 ms) the edit is refitted on
ARMS = {"stale": dict(dish=True), "recal": dict(dish=True), **{k: dict(dish=True) for k in REFIT},
        "ideal_imu": dict(dish=False), "zero_action": dict(dish=False)}
DISH_ARMS = [k for k, v in ARMS.items() if v["dish"]]
A2_ARMS = ["stale", "recal", "ideal_imu", "zero_action"]  # the default run (A2); A2c names its arms
AGING_SEED = 11
REST_MS = 1000.0
RECAL_STEPS = 3000  # 60 s
FOLDS = 5
SESSION_KEYS = ("data", "a1", "recal_steps")  # the settings a session depends on; an episode also on its length


def _session_safe(job):
    try:
        return session_job(job)
    except clv.Stopped as e:
        return {"stopped": str(e)}


def _episode_safe(job):
    try:
        return episode_job(job)
    except clv.Stopped as e:
        return {"stopped": str(e)}


def session_stem(cfg: str, cond: str, sz: dict) -> str:
    """File stem of a session (sessions/): ``<cfg>_<cond>`` plus a digest of the settings it is built with."""
    return job_key(f"{cfg}_{cond}", {k: sz[k] for k in SESSION_KEYS})


def episode_file(out: Path, cfg: str, cond: str, arm: str, seed: int, sz: dict) -> Path:
    """One episode's result: jobs/<cfg>/<cond>/<arm>/<seed>_<digest of its settings>.json."""
    key = job_key(str(seed), {k: sz[k] for k in SESSION_KEYS + ("seconds",)})
    return out / "jobs" / cfg / cond / arm / f"{key}.json"


def a1_setup(sz: dict, cfg: str) -> Path:
    """A1's day-0 setup for ``cfg``, under the name A1 gave it (built on the whole replay of this run's data)."""
    return Path(sz["a1"]) / "setups" / f"{clv.setup_stem(cfg, 'live', {'data': sz['data'], 'replay_steps': 0})}.pkl"


def a1_blob(sz, cfg):
    return pickle.loads(a1_setup(sz, cfg).read_bytes())


def same(a, b) -> bool:
    """Exact equality of nested results: dicts, lists, tuples, numpy arrays, torch tensors, numbers (NaN == NaN)."""
    import torch

    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(map(same, a, b)))
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and bool(torch.equal(a.cpu(), b.cpu())))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
    return a == b or (a != a and b != b)


def refit_edit(T: np.ndarray, P: np.ndarray, stand: np.ndarray, k: int) -> dict:
    """A2c's edit: the affine channel of the kept decoder (decoded ``P`` vs true ``T``) fitted on the first ``k``
    steps, around the standing posture, as A1b fitted its edit. With steps left over, what the edited teacher would
    see on them, (P - intercept) / gain, minus the truth, in teacher spreads (in-range steps only)."""
    from biocompute.eval.dish_channel import IMU_SPREAD

    st = clv.channel_stats(T[:k], P[:k], center=stand)
    g, a = st.get("gain"), st.get("intercept")
    ok = g is not None and all(x is not None and 0.05 < x < 1.5 for x in g)
    e = {"steps": int(k), "in_range": st["in_range"], "gain": g if ok else None, "intercept": a if ok else None,
         "fit_gain": g, "noise_over_gain": np.round(np.array(st["noise"]) / np.array(g), 3).tolist() if ok else None,
         "tau": st.get("tau")}
    if ok and k < len(T):
        sp = IMU_SPREAD[:5]
        Tr, Pr = T[k:, :5], P[k:, :5]
        m = (np.abs(Tr - np.asarray(stand)[:5]) <= 3 * sp).all(1)
        err = (Pr[m] - np.array(a)) / np.array(g) - Tr[m]
        e |= {"rest_in_range": int(m.sum()), "rest_bias": np.round(err.mean(0) / sp, 3).tolist(),
              "rest_rmse": np.round(np.sqrt((err**2).mean(0)) / sp, 3).tolist()}
    return e


def make_ctrl(cfg: str, cond: str, O: np.ndarray):
    """A1's live dish for ``cfg``; for ``newdish`` the culture is swapped for one with another seed (same MEA,
    electrodes, encoder and feature layout)."""
    from biocompute.train.setup import build

    setup = clv.make_setup(clv.CONFIGS[cfg], "live")
    ctrl = build(setup, O)
    cs = CONDITIONS[cond].get("culture_seed")
    if cs is not None:
        import torch

        from biocompute.culture.network import make_config
        from biocompute.device import SimulatedBiocomputer
        from biocompute.mea.electrodes import make_mea_config

        old = ctrl.device.mea.alive_channels.copy()
        ctrl.device = None
        torch.cuda.empty_cache()
        ctrl.device = SimulatedBiocomputer(
            make_config(setup.culture, seed=cs, wells=setup.wells, **setup.culture_overrides),
            make_mea_config(setup.mea, seed=setup.seed, wells=setup.wells, **setup.mea_overrides))
        if not np.array_equal(old, ctrl.device.mea.alive_channels):
            raise RuntimeError("new dish changed the recording channels; the stale decoder would not apply")
    return ctrl


def session_job(job):
    import torch

    from biocompute.culture.aging import age
    from biocompute.interface.decoders import LinearReadout
    from biocompute.runguard import Guard

    cfg, cond, out, sz = job
    name = f"{cfg}_{cond}"
    if clv.stop_requested():
        raise clv.Stopped(f"session {name} before start")
    g = Guard(f"session_{name}", out)
    O = clv.load_replay(sz["data"], 0)  # the encoder is fitted on the whole replay, as in A1
    b0 = a1_blob(sz, cfg)
    row0 = b0["row"]
    ctrl = make_ctrl(cfg, cond, O)
    dev, cu = ctrl.device, ctrl.device.culture
    g.check(cu.plasticity is None, "plasticity is on")
    days = CONDITIONS[cond]["days"]
    if "culture_seed" not in CONDITIONS[cond]:
        dev.load_state(b0["warm"])
    alive0 = float(cu.alive.float().mean())
    if days:
        age(cu, days, seed=AGING_SEED)
    alive = float(cu.alive.float().mean())
    exc_sd = float(cu.excitability.std())
    if days and not sz["smoke"]:
        g.check(abs(alive / alive0 - 0.98 ** days) < 0.01, f"alive share {alive / alive0:.4f} vs {0.98 ** days:.4f}")
        g.check(abs(exc_sd / (0.5 * days ** 0.5) - 1) < 0.1, f"excitability sd {exc_sd:.3f} vs {0.5 * days ** 0.5:.3f}")
    s = sum(map(ord, name))
    cu.torch_gen.manual_seed(1000 + s)
    dev.mea.gen.manual_seed(2000 + s)
    w_ref = cu.w.clone()
    dev.step(REST_MS)
    a, n = row0["split"], sz["recal_steps"]
    seg = O[a:a + n]
    F = np.zeros((n, ctrl.features.dim), np.float32)
    t0 = time.time()
    for i in range(n):
        if clv.stop_requested():
            raise clv.Stopped(f"session {name} at replay step {i}/{n} (redone on resume)")
        F[i] = ctrl.sense(seg[i])
        if g.due():
            win = F[i - 49:i + 1, :2 * (F.shape[1] // 4)] if i >= 49 else None
            g.culture(cu, rate_hz=None if win is None else float((win ** 2).sum(1).mean() / (F.shape[1] // 4) / 0.02))
            g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
            g.beat(i + 1, n, ms_per_step=round((time.time() - t0) / (i + 1) * 1000, 1))
    C = F.shape[1] // 4
    cols = 3 * C
    g.check(cols == b0["cols"], f"feature layout {cols} != A1's {b0['cols']}")
    zero_rows = float((F[:, : 2 * C].sum(1) == 0).mean())
    g.check(zero_rows < 0.01, f"{zero_rows:.1%} of replay steps have no spikes at all")
    rate = float((F[:, : 2 * C] ** 2).sum(1).mean() / C / 0.02)
    T = seg[:, :6]
    stand = np.array(row0["stand_imu"])
    # Stale: A1b's decoder on this session's replay.
    ro0 = LinearReadout(b0["cols"], 6)
    for k, v in b0["readout"].items():
        setattr(ro0, k, v)
    P_stale = clv.rebuild_grav_z(ro0.predict(F[:, :cols]))
    stale = clv.channel_stats(T, P_stale, center=stand)
    # Recalibrated: decoder on all 60 s, affine edit on out-of-fold predictions.
    ro = LinearReadout(cols, 6)
    cv = ro.fit_cv(F[:, :cols], T, alphas=np.logspace(-1, 5, 13), standardize_y=True)
    edges = np.linspace(0, n, FOLDS + 1).astype(int)
    P_oof = np.zeros_like(T)
    for k in range(FOLDS):
        te = np.arange(edges[k], edges[k + 1])
        tr = np.setdiff1d(np.arange(n), te)
        rk = LinearReadout(cols, 6)
        rk.fit(F[tr, :cols], T[tr], alpha=cv["alpha"], standardize_y=True)
        P_oof[te] = rk.predict(F[te, :cols])
    P_oof = clv.rebuild_grav_z(P_oof)
    g.check(bool(np.isfinite(P_oof).all() and np.isfinite(P_stale).all()), "decoder predictions not finite")
    recal = clv.channel_stats(T, P_oof, center=stand)
    aff_g, aff_a = recal.get("gain"), recal.get("intercept")
    if not (aff_g is not None and all(x is not None and 0.05 < x < 1.5 for x in aff_g)):
        aff_g = aff_a = None
        if not sz["smoke"]:
            g.fail(f"recalibrated affine fit unusable: gain {recal.get('gain')}")
    # A2c: A1b's decoder kept, only its edit refitted on the first 10 / 30 / 60 s of this replay.
    g.check(sz["smoke"] or n >= max(REFIT.values()), f"{n} replay steps, A2c's edits need {max(REFIT.values())}")
    refit = {arm: refit_edit(T, P_stale, stand, min(k, n)) for arm, k in REFIT.items()}
    for arm, e in refit.items():
        if e["gain"] is None and not sz["smoke"]:
            g.fail(f"{arm}: affine fit on {e['steps']} steps unusable: gain {e['fit_gain']}")
    if cond == "day0" and not sz["smoke"]:
        ref = np.array(row0["held_out_at_stand"]["gain"])
        g.check(stale.get("gain") is not None and bool((np.abs(np.array(stale["gain"]) - ref) < 0.04).all()),
                f"day0 stale gains {stale.get('gain')} don't reproduce A1b's {ref.tolist()} within 0.04")
        g.check(abs(rate / row0["mean_rate_hz"] - 1) < 0.01, f"day0 rate {rate:.2f} vs A1b {row0['mean_rate_hz']}")
    obs0 = clv.standing_obs()
    for _ in range(clv.WARM_STEPS):
        ctrl.sense(obs0)
    g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
    row = {"cfg": cfg, "cond": cond, "days": days, "culture_seed": CONDITIONS[cond].get("culture_seed", 0),
           "alive_share": round(alive, 4), "alive_vs_day0": round(alive / alive0, 4),
           "excitability_sd": round(exc_sd, 3), "replay_steps": [int(a), int(a + n)], "rate_hz": round(rate, 2),
           "a1_rate_hz": row0["mean_rate_hz"], "stale_open_loop": stale, "recal_alpha": cv["alpha"],
           "recal_cv_r2": round(cv["val_r2"], 3), "recal_open_loop_oof": recal, "recal_affine_gain": aff_g,
           "recal_affine_icpt": aff_a, "stale_affine_gain": row0["affine_gain"],
           "stale_affine_icpt": row0["affine_icpt"], "smoke": sz["smoke"]}
    blob = {"row": row, "readout": {k: getattr(ro, k) for k in ("W", "mean", "std", "y_mean", "y_std")},
            "warm": dev.state(), "obs0": obs0, "cols": cols}
    p = out / "sessions" / f"{session_stem(cfg, cond, sz)}.pkl"
    a2 = Path(sz["a2"]) / "sessions" / p.name if sz.get("a2") else None
    if a2 is not None and not sz["smoke"] and a2.resolve() != p.resolve():
        # Positive control: plasticity off, the GPU sums ordered: the session must repeat A2's bit for bit.
        g.check(a2.exists(), f"no stored A2 session {a2} to compare with (--a2 '' skips the check)")
        old = pickle.loads(a2.read_bytes())
        diff = [k for k in ("readout", "warm", "obs0", "cols") if not same(blob[k], old[k])]
        diff += [f"row.{k}" for k, v in old["row"].items() if not same(row.get(k), v)]
        g.check(not diff, f"session {name} differs from A2's stored one in {diff}")
        row["equals_a2_session"] = str(a2)
    row["refit_edit"] = refit
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(blob))
    os.replace(tmp, p)
    clv.write_json(p.with_suffix(".json"), row)
    g.beat(n, n, force=True)
    print(json.dumps({"session": name, "alive_vs_day0": row["alive_vs_day0"], "exc_sd": row["excitability_sd"],
                      "rate_hz": row["rate_hz"], "stale_gain": stale.get("gain"),
                      "stale_bias": stale.get("bias_over_spread"), "recal_gain": aff_g,
                      "recal_cv_r2": row["recal_cv_r2"], "equals_a2": "equals_a2_session" in row}
                     | {f"{k}_gain": e["gain"] for k, e in refit.items()}
                     | {f"{k}_rest_bias": e["rest_bias"] for k, e in refit.items() if "rest_bias" in e}), flush=True)
    del ctrl, F
    torch.cuda.empty_cache()
    return row


def episode_job(job):
    from biocompute.interface.decoders import LinearReadout
    from biocompute.loop import run_episode
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.runguard import Guard
    from biocompute.train.tasks import make_task

    cfg, cond, arm, seeds, out, sz = job
    todo = [s for s in seeds if not episode_file(out, cfg, cond, arm, s, sz).exists()]
    if not todo or clv.stop_requested():
        return {"stopped": f"{cfg}/{cond}/{arm} before start"} if todo else {"done": 0}
    g = Guard(f"{cfg}_{cond}_{arm}_{seeds[0]}", out)
    env = AsimovEnv()
    dish = readout = cu = w_ref = center = None
    cols, obs0, W, stim = 0, clv.standing_obs(), None, False
    if arm == "ideal_imu":
        W = clv.teacher_weights(None)
    elif ARMS[arm]["dish"]:
        import torch

        sb = pickle.loads((out / "sessions" / f"{session_stem(cfg, cond, sz)}.pkl").read_bytes())
        src = a1_blob(sz, cfg) if arm == "stale" or arm in REFIT else sb  # A1b's decoder, or the refitted one
        O = clv.load_replay(sz["data"], 0)
        dish = make_ctrl(cfg, cond, O)
        cu = dish.device.culture
        g.check(cu.plasticity is None, "plasticity is on")
        readout = LinearReadout(src["cols"], 6)
        for k, v in src["readout"].items():
            setattr(readout, k, v)
        cols, obs0, warm = sb["cols"], sb["obs0"], sb["warm"]
        r = sb["row"]
        if arm == "stale":
            gain, icpt = r["stale_affine_gain"], r["stale_affine_icpt"]
        elif arm == "recal":
            gain, icpt = r["recal_affine_gain"], r["recal_affine_icpt"]
        else:
            gain, icpt = r["refit_edit"][arm]["gain"], r["refit_edit"][arm]["intercept"]
        if gain is None:  # smoke only: the fit was unusable at tiny scale
            gain, icpt = r["stale_affine_gain"], r["stale_affine_icpt"]
        W = clv.teacher_weights(gain, icpt)
        center, stim = a1_blob(sz, cfg)["row"]["stand_imu"], True
        w_ref = torch.as_tensor(warm["culture"]["w"]).to(cu.device)
    ctl = clv.VestibularController(dish, readout, cols, W, stim, obs0)
    steps_total = int(round(sz["seconds"] / env.dt))
    n_done = 0
    for si, seed in enumerate(todo):
        if clv.stop_requested():
            raise clv.Stopped(f"{cfg}/{cond}/{arm} before seed {seed}")
        if dish is not None:
            dish.device.load_state(warm)
            cu.torch_gen.manual_seed(7 * seed + 1)
            dish.device.mea.gen.manual_seed(7 * seed + 2)
        task = make_task("stand_push", sz["seconds"], seed=seed)

        def on_step(step, env_, c_):
            if clv.stop_requested():
                raise clv.Stopped(f"{cfg}/{cond}/{arm} seed {seed} at step {step} (redone on resume)")
            if g.due():
                if cu is not None:
                    import torch

                    sim_s = (step + 1) * 0.02
                    g.culture(cu, rate_hz=max(c_.spikes / max(len(dish.features.channels), 1) / sim_s, 1e-6))
                    g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
                if c_.decoded:
                    g.check(bool(np.isfinite(c_.decoded[-1]).all()), "decoded IMU not finite")
                g.beat(si + step / steps_total, len(todo), seed=seed)

        ep = run_episode(env, ctl, sz["seconds"], task.command_fn, push_fn=task.push_fn, clip=1e6,
                         on_step=on_step)
        survived = float(ep.fell_at) if ep.fell_at >= 0 else sz["seconds"]
        row = {"cfg": cfg, "cond": cond, "arm": arm, "seed": seed, "survived_s": round(survived, 2),
               "full": bool(ep.fell_at < 0), "steps": ep.steps, "seconds": sz["seconds"]}
        if dish is not None:
            row["rate_hz_per_electrode"] = round(ctl.spikes / len(dish.features.channels) / (ep.steps * 0.02), 2)
            if ctl.decoded:
                T_, P_ = np.array(ctl.true), np.array(ctl.decoded)
                row["closed_loop_channel"] = clv.channel_stats(T_, P_, center=center)  # the decoder's output
                Q_ = P_.copy()  # what the edited teacher sees: (decoded - intercept) / gain (logging only)
                Q_[:, :5] = (P_[:, :5] - np.asarray(icpt, float)[:5]) / np.asarray(gain, float)[:5]
                row["edited_channel"] = clv.channel_stats(T_, Q_, center=center)
        clv.write_json(episode_file(out, cfg, cond, arm, seed, sz), row)
        n_done += 1
        g.beat(si + 1, len(todo), force=True, seed=seed, survived=row["survived_s"])
    return {"done": n_done}


def summarize(out: Path, configs, conds, seeds, sz) -> dict:
    table, fulls = [], {}
    for cfg, cond, arm in [(c, d, a) for c in configs for d in conds for a in DISH_ARMS] + \
            [("none", "none", a) for a in ("ideal_imu", "zero_action")]:
        rows = [json.loads(p.read_text()) for s in seeds if (p := episode_file(out, cfg, cond, arm, s, sz)).exists()]
        if not rows:
            continue
        fulls[(cfg, cond, arm)] = {r["seed"]: r["full"] for r in rows}
        surv = np.array([r["survived_s"] for r in rows])
        e = {"cfg": cfg, "cond": cond, "arm": arm, "n": len(rows), "full": int(sum(r["full"] for r in rows)),
             "mean_s": round(float(surv.mean()), 2), "survived_s": surv.tolist()}
        for src, pre in (("closed_loop_channel", "closed_loop"), ("edited_channel", "edited")):
            cl = [r[src] for r in rows if "gain" in r.get(src, {})]
            if cl:
                for k in ("gain", "bias_over_spread", "noise"):
                    e[f"{pre}_{k}_median"] = np.round(np.nanmedian(
                        [[np.nan if x is None else x for x in c[k]] for c in cl], 0), 3).tolist()
        rates = [r["rate_hz_per_electrode"] for r in rows if "rate_hz_per_electrode" in r]
        if rates:
            e["rate_hz_median"] = round(float(np.median(rates)), 2)
        table.append(e)
    by = {(e["cfg"], e["cond"], e["arm"]): e for e in table}
    bar = int(np.ceil(0.85 * len(seeds)))
    gate = None
    if "s32" in configs and any(k[2] == "stale" for k in by):
        per = {}
        for cond in [c for c in AGED if c in conds]:
            st, rc = by.get(("s32", cond, "stale")), by.get(("s32", cond, "recal"))
            if st and rc:
                per[cond] = {"stale": f"{st['full']}/{st['n']}", "recal": f"{rc['full']}/{rc['n']}",
                             "stale_drops": st["full"] < bar, "recal_holds": rc["full"] >= bar}
        complete = all(by.get(("s32", c, a), {}).get("n") == len(seeds) for c in AGED for a in ("stale", "recal"))
        d0 = by.get(("s32", "day0", "recal"))
        gate = {"bar": f">= {bar}/{len(seeds)}", "per_condition": per, "complete": complete,
                "day0_recal": None if d0 is None else f"{d0['full']}/{d0['n']}",
                "tested": any(v["stale_drops"] for v in per.values()),
                "pass": bool(complete and per and all(v["recal_holds"] for v in per.values()))}
    sessions = {f"{c}_{d}": json.loads(p.read_text()) for c in configs for d in sorted(conds)
                if (p := out / "sessions" / f"{session_stem(c, d, sz)}.json").exists()}
    res = {"seeds": [seeds[0], seeds[-1]], "seconds": sz["seconds"], "table": table}
    if gate is not None or not any(k[2] in REFIT for k in by):
        res["gate_A2"] = gate
    if any(k[2] in REFIT for k in by):
        res["gate_A2c"] = gate_a2c(by, fulls, len(seeds))
    res["sessions"] = sessions
    clv.write_json(out / "results.json", res)
    for e in table:
        print(json.dumps({k: v for k, v in e.items() if k != "survived_s"}), flush=True)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("gate")}), flush=True)
    return res


def gate_a2c(by: dict, fulls: dict, n_seeds: int) -> dict:
    """Gate A2c: refit60 >= 85% full runs on s32 in every condition (day 0 included), all seeds done. Per condition
    also the paired comparison with recal on the same seeds: seeds where only one of the two runs the full 20 s."""
    bar = int(np.ceil(0.85 * n_seeds))
    per = {}
    for cond in CONDITIONS:
        e = by.get(("s32", cond, "refit60"))
        if e is None:
            continue
        per[cond] = {"refit60": f"{e['full']}/{e['n']}", "holds": e["full"] >= bar}
        for arm in ("refit30", "refit10", "recal"):
            if (x := by.get(("s32", cond, arm))) is not None:
                per[cond][arm] = f"{x['full']}/{x['n']}"
        a, b = fulls.get(("s32", cond, "refit60"), {}), fulls.get(("s32", cond, "recal"), {})
        both = sorted(set(a) & set(b))
        if both:
            per[cond]["only_refit60_full"] = [s for s in both if a[s] and not b[s]]
            per[cond]["only_recal_full"] = [s for s in both if b[s] and not a[s]]
    complete = all(by.get(("s32", c, "refit60"), {}).get("n") == n_seeds for c in CONDITIONS)
    return {"bar": f">= {bar}/{n_seeds}", "arm": "refit60", "per_condition": per, "complete": complete,
            "pass": bool(complete and all(v["holds"] for v in per.values()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", default=["s32"], choices=list(clv.CONFIGS))
    ap.add_argument("--conds", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--arms", nargs="+", default=A2_ARMS, choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--seed-base", type=int, default=SEEDS[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--recal-steps", type=int, default=RECAL_STEPS)
    ap.add_argument("--a1", default="outputs/closed_loop_vestibular_affine", help="A1b's output (day-0 setups)")
    ap.add_argument("--a2", default="outputs/closed_loop_sessions",
                    help="A2's output: each new session must equal its stored one bit for bit ('' skips the check)")
    ap.add_argument("--data", default="outputs/probe/teacher_data_20k.pkl")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/closed_loop_sessions")
    ap.add_argument("--smoke", action="store_true",
                    help="whole pipeline at tiny scale into <out>/smoke: 300 replay steps, 2 seeds, 1 s episodes")
    a = ap.parse_args()
    import faulthandler

    faulthandler.register(signal.SIGUSR1)
    out = Path(a.out)
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
        a.recal_steps, a.seeds, a.seconds, a.chunk = 300, 2, 1.0, 1
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed_base, a.seed_base + a.seeds))
    threads = max(1, min(4, (os.cpu_count() or 4) // max(a.workers, 1)))
    sz = {"data": a.data, "a1": a.a1, "a2": a.a2, "recal_steps": a.recal_steps, "seconds": a.seconds,
          "smoke": a.smoke}
    for cfg in a.configs:
        if not a1_setup(sz, cfg).exists():
            sys.exit(f"no A1 setup for {cfg} in {a.a1}")
        for d in a.conds:
            p = out / "sessions" / f"{session_stem(cfg, d, sz)}.json"
            if set(a.arms) & set(REFIT) and p.exists() and "refit_edit" not in json.loads(p.read_text()):
                sys.exit(f"{p} predates A2c (no refit edits): use a fresh --out")
    print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": a.configs, "conds": a.conds,
                      "arms": a.arms, "seeds": [seeds[0], seeds[-1]], "seconds": a.seconds,
                      "workers": a.workers, "smoke": a.smoke}), flush=True)
    dish_arms = [x for x in a.arms if ARMS[x]["dish"]]
    todo = [(c, d, out, sz) for c in a.configs for d in a.conds
            if dish_arms and not (out / "sessions" / f"{session_stem(c, d, sz)}.pkl").exists()]
    if todo:
        clv.pool_map(_session_safe, todo, min(a.workers, len(todo)), threads)
    combos = [(c, d, x) for c in a.configs for d in a.conds for x in dish_arms]
    combos += [("none", "none", x) for x in a.arms if not ARMS[x]["dish"]]
    jobs = []
    for c, d, x in combos:
        left = [s for s in seeds if not episode_file(out, c, d, x, s, sz).exists()]
        jobs += [(c, d, x, left[i:i + a.chunk], out, sz) for i in range(0, len(left), a.chunk)]
    jobs.sort(key=lambda j: not ARMS[j[2]]["dish"])  # long jobs first
    if jobs:
        clv.pool_map(_episode_safe, jobs, min(a.workers, len(jobs)), threads)
    summarize(out, a.configs, a.conds, seeds, sz)


if __name__ == "__main__":
    main()
