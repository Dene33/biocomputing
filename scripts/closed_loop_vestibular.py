"""Track A, step A1 (docs/route-to-computation.md, "Order of work without a dish"; Phase 3.3 of docs/next-steps.md):
the simulated culture in the robot's loop, as its vestibular organ.

Only the 6 IMU values (obs[0:6]: gyro x0.25, projected gravity) go through the dish. Joint positions, joint
velocities and the last action go straight to the controller, the teacher's own network in numpy
(wsp/controller/track_a.py). Everything is fixed before the first closed-loop seed:

- Dish: Phase 3.1's setup exactly (scripts/vestibular_probe.py): gpu_20k culture, 1 well, grid256 MEA, stimulation
  reach 120 um, ``axis`` encoding of obs[0:6] on 12 (CL1-sized, main) or 32 electrodes, amplitude coding at gain 0.5,
  4 pulse slots per 20 ms step. Plasticity is off: a relay doesn't need it, and the model's plasticity is its known
  flaw (an in-run guard checks that no weight moves).
- Decoder: ridge from the spike features (two 10 ms bins + the 40 ms trace of every recording electrode) to the IMU,
  fitted on an open-loop replay of teacher stand_push episodes (outputs/probe/teacher_data_20k.pkl, 19k steps with
  DART action noise, as in Phase 3.1): the first ~80% (cut at an episode boundary), penalty by contiguous 5-fold CV.
  grav_z is rebuilt from x, y.
- Weight edit (WSP track a): the teacher's first-layer IMU columns x 1/gain, the gain being the slope of decoded on
  true IMU in the held-out ~20% of the replay (in-range steps: every IMU value within 3 teacher spreads of standing).
- Each episode starts from the same warmed-up culture state (after the replay, 1 s of the standing IMU as input) with
  its own noise stream (seeded by the push seed), and the decoder's traces are pre-rolled for 100 ms with the standing
  IMU. So every (configuration, arm, seed) is independent of run order, and a stop loses at most one episode.

Arms, stand_push, 20 s, 1-step action latency (as Phase 3.1's eval_controller.py), 40 fresh seeds (9000-9039; no
earlier script used them):

- ``culture_edit``   culture -> decoder -> teacher with the 1/gain edit (the main arm)
- ``culture_plain``  culture -> decoder -> teacher unchanged
- ``silenced``       the culture gets no stimulation (its decoder reads spontaneous activity) -> edited teacher
- ``dead_edit``      dead culture (recurrent synapses removed) with its own decoder and its own edit
- ``ideal_imu``      true IMU -> teacher unchanged (no culture; the ceiling)
- ``zero_action``    the default pose under PD (the floor)

The first four run per configuration (``s12`` = 1 well x 12, main; ``s32`` = 1 well x 32, comparison).

Gate A1: ``culture_edit`` on s12 reaches >= 34/40 full 20 s runs and ``silenced`` 0/40. The dead culture will
probably match the live one; if so, the dish relays and computes nothing, and we say so.

Positive control before the closed loop: the held-out open-loop channel must reproduce Phase 3.1's gains for the same
configuration (outputs/vestibular_probe/channel_models.json, within 0.2) and the numpy teacher must match the ONNX
teacher; otherwise GUARD FAIL. Every episode also records the closed-loop channel (decoded vs true IMU), to compare
with the open-loop one the edit was fitted on.

Run (guarded, resumable):

    uv run scripts/launch.py --unit bio-a1 --log outputs/closed_loop_vestibular/run.log -- \\
        scripts/closed_loop_vestibular.py --workers 3
    uv run scripts/watch.py --unit bio-a1 --log outputs/closed_loop_vestibular/run.log
    systemctl --user stop bio-a1     # graceful: replays checkpoint, the running episode is redone on resume

Outputs (<out> = outputs/closed_loop_vestibular): setups/<cfg>_<dish>.json (+ .pkl: decoder, edit, warm state),
jobs/<cfg>/<arm>/<seed>.json (one per episode), results.json (the table and the gate). Stop: workers share a
lock-free stop flag, checked every control step; a replay saves its checkpoint (ckpt/), an episode is dropped and
redone. Relaunch the same command to resume. GPU runs are not bit-reproducible (atomic adds).

Run 2 (A1b), after run 1 failed: ``--edit affine --seed-base 9100 --out outputs/closed_loop_vestibular_affine``.

Results (24 Nov 2026; results.json in each output folder). Full 20 s runs of 40, mean survival:

| Arm                         | Run 1: 1/gain edit, 9000-9039 | Run 2: affine edit, 9100-9139 |
|-----------------------------|-------------------------------|-------------------------------|
| 1 x 12 culture, edited      | **0/40**, 5.6 s               | **23/40**, 15.2 s             |
| 1 x 12 culture, unchanged   | 0/40, 3.7 s                   | 0/40, 3.5 s                   |
| 1 x 12 silenced             | 0/40, 1.6 s                   | 0/40, 1.0 s                   |
| 1 x 12 dead, own edit       | 23/40, 16.5 s                 | 32/40, 18.8 s                 |
| 1 x 32 culture, edited      | 36/40, 19.0 s                 | **40/40**, 20.0 s             |
| 1 x 32 culture, unchanged   | 38/40, 19.4 s                 | 38/40, 19.3 s                 |
| 1 x 32 silenced             | 0/40, 0.7 s                   | 0/40, 0.8 s                   |
| 1 x 32 dead, own edit       | 38/40, 19.3 s                 | 36/40, 19.2 s                 |
| ideal IMU                   | 40/40                         | 40/40                         |
| zero action                 | 0/40, 5.0 s                   | 0/40, 4.6 s                   |

Gate A1 (1 x 12): not passed in either run. The 1 x 32 well clears the same bar (40/40, silenced 0/40) with the
affine edit. The dead culture does as well as the live one or better in every configuration: the dish relays the
IMU and computes nothing, and the live culture's own activity costs fidelity.

Positive controls held: open-loop held-out gains reproduce Phase 3.1 (1 x 12: 0.53-0.63 vs 0.61-0.70; 1 x 32:
0.78-0.93 vs 0.84-0.94), replay rate 24.79 Hz as in Phase 3.1, numpy teacher = ONNX teacher to 7e-6, no weight moved.

Why run 1 failed (diagnosed by decoding in the shadow of an ideal-IMU episode, seed 9000): the robot stands at
grav_x = -0.063, not 0 (the teacher's posture, 3.4 of grav_x's teacher spreads from 0). The ridge decoder shrinks
toward the replay's mean, so the channel is p = a + g t with a = -0.022 on grav_x; at quiet standing it decodes
-0.060 for a true -0.063 (bias 0.17 spreads). The 1/gain edit rescales around 0, so the edited teacher saw
-0.060 / 0.56 = -0.107: a false forward lean of 2.35 spreads. It leaned back to grav_x = +0.078, beyond the replay's
range (-0.088 to -0.007), where the grav_x electrodes saturate (closed-loop slope 0.02), and fell. Phase 3.1's channel
model (gain around (0, 0, 0, 0, 0, -1), no offset) could not show this, and its "in range" window (3 spreads around
0) leaves out quiet standing. Run 2's edit inverts the affine channel (1/g on the IMU columns plus a first-layer bias
shift, a and g fitted on held-out replay around the measured standing posture) and was evaluated on fresh seeds
because it was designed after seeing run 1. Closed-loop bias fell to <= 0.5 spreads. The remaining 1 x 12 falls
are not explained by the stationary channel statistics (closed-loop gain 0.41-0.58, noise 0.46-0.63 spreads, close to
the open-loop model that gave 58/60 as a statistical channel). They follow pushes from the front (scripts/
closed_loop_falls.py): 12 of 41 such pushes lead to a fall (dead culture 2/52; side pushes 5/96 vs 6/119; pushes
from behind 0). In three traced falls, the relay reads the pitch rate 2.4-2.5 spreads too high in the second after
the push. The teacher under-corrects the forward lean of its recovery, the lean passes the encoder's input limit
(4 replay SDs, 7.9 spreads) about 1 s after the push, the teacher's view stops near 9 spreads while the lean reaches
33-42, and the robot falls about 1 s later.

Stop handling, tested on run 1 itself with more jobs than workers (4 replays on 3 workers, then 40 episode jobs):
stop during the replays took 5.9 s and resumed from the checkpoints (step 1778 of 19334); stop during the episodes
took 14.8 s, the three running episodes were redone, the 137 finished ones kept.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(9000, 9040))
CONFIGS = {"s12": 12, "s32": 32}
PROBE_NAME = {"s12": "axis_s12_w1", "s32": "axis_s32_w1"}
ARMS = {
    "culture_edit": dict(dish="live", teacher="edit", stim=True),
    "culture_plain": dict(dish="live", teacher="plain", stim=True),
    "silenced": dict(dish="live", teacher="edit", stim=False),
    "dead_edit": dict(dish="dead", teacher="edit", stim=True),
    "ideal_imu": dict(dish=None, teacher="plain", stim=False),
    "zero_action": dict(dish=None, teacher=None, stim=False),
}
CULTURE_ARMS = [k for k, v in ARMS.items() if v["dish"]]
PREROLL_STEPS = 5
WARM_STEPS = 50


# --- Stop and resume (as bl1_drift.py) -----------------------------------------------------------------------------
STOP = {"requested": False}
EVENT = {"stop": None}  # in workers: the shared stop flag, a RawValue byte (no locks in signal handlers)


class Stopped(Exception):
    pass


def stop_requested():
    flag = EVENT["stop"]
    return STOP["requested"] or (flag is not None and flag.value == 1)


def _on_term(signum, frame):
    STOP["requested"] = True
    if EVENT["stop"] is not None:
        EVENT["stop"].value = 1


def _worker_init(ev=None, threads=4):
    import faulthandler

    import torch

    torch.set_num_threads(threads)  # 16 threads per worker oversubscribed the CPU while cultures were built
    EVENT["stop"] = ev
    signal.signal(signal.SIGTERM, _on_term)
    faulthandler.register(signal.SIGUSR1)  # `kill -USR1 <pid>` dumps every thread's stack to the log


def _guarded(fn):
    def call(job):
        try:
            return fn(job)
        except Stopped as e:
            return {"stopped": str(e)}
    return call


def _setup_safe(job):
    return _guarded(setup_job)(job)


def _episode_safe(job):
    return _guarded(episode_job)(job)


_MAIN_EVENT = {}


def pool_map(fn, jobs, workers, threads=4):
    """Jobs in fresh worker processes (one per job). On SIGTERM this process sets the shared flag and waits: running
    jobs save and return, queued ones start, see the flag and return at once."""
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
    # close() + join(), not the context manager: its terminate() SIGTERMs workers that catch SIGTERM, then hangs.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # read when a worker imports numpy
        os.environ[k] = str(threads)
    pool = ctx.Pool(workers, initializer=_worker_init, initargs=(ev, threads), maxtasksperchild=1)
    try:
        res = list(pool.imap_unordered(fn, jobs))
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        raise
    stopped = [r["stopped"] for r in res if isinstance(r, dict) and "stopped" in r]
    if stopped or ev.value == 1:
        print(json.dumps({"stopped": stopped or "flag", "resume": "run the same command again"}), flush=True)
        raise SystemExit(0)
    return res


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# --- Pieces reused from the project -----------------------------------------------------------------------------------
def make_setup(n_stim: int, dish: str):
    """Phase 3.1's setup (vestibular_probe.py), with the dead culture as ``source="dead"``."""
    from biocompute.train.setup import Setup

    return Setup(culture="gpu_20k", mea="grid256", mea_overrides={"stim_length_um": 120.0}, n_stim=n_stim,
                 n_feedback=0, encoding="amplitude", encoder_gain=0.5, projection="axis",
                 select_dims=tuple(range(6)), select_spread=1.0, slots_ms=(0.0, 5.0, 10.0, 15.0), bins=2,
                 taus_ms=(40.0, 160.0), wells=1, diverse_wells=False,
                 source={"live": "mea", "dead": "dead"}[dish], seed=0)


_TRACK_A = {}


def track_a():
    if "mod" not in _TRACK_A:
        spec = importlib.util.spec_from_file_location("track_a", ROOT / "wsp" / "controller" / "track_a.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TRACK_A["mod"] = mod
    return _TRACK_A["mod"]


def teacher_weights(gain=None, icpt=None) -> dict:
    """The teacher's weights with its first-layer IMU columns x 1/gain (track_a.py's edit); ``None``: unchanged.
    With ``icpt`` the edit inverts the affine channel p = icpt + gain * t instead: the scaled columns also move the
    first-layer bias by -column / gain * icpt, so the teacher sees (p - icpt) / gain (track_a.py does the same for
    grav_z's operating point). grav_z keeps gain 1: it is rebuilt from x, y."""
    ta = track_a()
    g = np.ones(5) if gain is None else np.asarray(gain, float)
    old = ta.GYRO_GAIN, ta.GRAV_GAIN
    ta.GYRO_GAIN, ta.GRAV_GAIN = 1.0 / g[0:3], np.r_[1.0 / g[3:5], 1.0]
    try:
        W = ta.write_weights()
    finally:
        ta.GYRO_GAIN, ta.GRAV_GAIN = old
    if icpt is not None:
        W["l0_b"] -= W["l0_w"][:, 0:5] @ np.asarray(icpt, float)[:5]
    return W


def rebuild_grav_z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).copy()
    gxy = np.clip(v[..., 3:5], -0.99, 0.99)
    v[..., 5] = -np.sqrt(np.maximum(1.0 - (gxy**2).sum(-1), 1e-4))
    return v


def channel_stats(T: np.ndarray, P: np.ndarray, min_steps: int = 30, center=None) -> dict:
    """Decoded ``P`` vs true ``T`` IMU, in teacher spreads, on in-range steps (every IMU value within 3 spreads of
    ``center``; default Phase 3.1's (0, 0, 0, 0, 0, -1), but the robot stands at grav_x ~ -0.065, 3.5 spreads away,
    so later runs centre on the measured standing posture): per-dimension gain (slope), intercept, bias (mean
    decoded - true), noise around the fitted line, its correlation time (steps), RMSE; R² over all steps."""
    from biocompute.eval.dish_channel import IMU_SPREAD, STANDING_IMU

    sp = IMU_SPREAD
    st = STANDING_IMU if center is None else np.asarray(center, float)
    m = (np.abs(T[:, :5] - st[:5]) <= 3 * sp[:5]).all(1)
    out = {"steps": int(len(T)), "in_range": int(m.sum())}
    var = T.var(0)
    out["r2"] = np.round(1 - ((T - P) ** 2).mean(0)[:5] / np.maximum(var[:5], 1e-12), 3).tolist()
    if m.sum() < min_steps:
        return out
    gain, icpts, bias, noise, tau, rmse = [], [], [], [], [], []
    for d in range(5):
        t, p = T[m, d], P[m, d]
        if t.std() < 1e-9:
            for x in (gain, icpts, bias, noise, tau, rmse):
                x.append(None)
            continue
        slope, icpt = np.polyfit(t, p, 1)
        r = p - (icpt + slope * t)
        rho = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else 0.0
        gain.append(round(float(slope), 3))
        icpts.append(round(float(icpt), 5))
        bias.append(round(float((p - t).mean() / sp[d]), 3))
        noise.append(round(float(r.std() / sp[d]), 3))
        tau.append(round(float(-1.0 / np.log(rho)), 2) if 0 < rho < 1 else 0.0)
        rmse.append(round(float(np.sqrt(((p - t) ** 2).mean()) / sp[d]), 3))
    return out | {"gain": gain, "intercept": icpts, "bias_over_spread": bias, "noise": noise, "tau": tau,
                  "rmse_over_spread": rmse}


class VestibularController:
    """The controller ``run_episode`` drives: obs[0:6] through the dish (or not), the rest straight to the teacher."""

    def __init__(self, dish, readout, cols, W, stim, obs0):
        self.dish, self.readout, self.cols, self.W, self.stim, self.obs0 = dish, readout, cols, W, stim, obs0
        self.true, self.decoded, self.spikes = [], [], 0

    def reset_readout_state(self) -> None:
        self.true, self.decoded, self.spikes = [], [], 0
        if self.dish is not None:
            self.dish.reset_readout_state()
            for _ in range(PREROLL_STEPS):  # the dish runs before the controller engages, as on hardware
                self.dish.sense(self.obs0, stimulate=self.stim)
        if self.W is not None:
            track_a().reset(self.W)

    def vest(self, obs: np.ndarray) -> np.ndarray:
        if self.dish is None:
            return np.asarray(obs[0:6], float)
        f = self.dish.sense(obs, stimulate=self.stim)
        self.spikes += len(self.dish.last_block.channels)
        return rebuild_grav_z(self.readout(f[: self.cols]))

    def act(self, obs: np.ndarray):
        if self.W is None:
            return np.zeros(23, np.float32), np.zeros(0, np.float32)
        v = self.vest(obs)
        self.true.append(np.asarray(obs[0:6], float))
        self.decoded.append(v)
        a = track_a().act(self.W, obs[9:32], obs[32:55], obs[55:78], v)
        return np.asarray(a, np.float32), np.zeros(0, np.float32)


# --- Setup: replay, decoder, edit, warm state -------------------------------------------------------------------------
def load_replay(path: str, steps: int) -> np.ndarray:
    d = pickle.load(open(ROOT / path if not Path(path).is_absolute() else path, "rb"))
    return np.asarray(d["obs"][: steps or None], np.float64)


def standing_imu(seconds: float = 3.0) -> np.ndarray:
    """The robot's own standing posture: the unchanged teacher with the true IMU, no pushes, mean IMU over the last
    half (deterministic). grav_x settles near -0.065, not 0."""
    from biocompute.robot.asimov import AsimovEnv

    env, W = AsimovEnv(), teacher_weights(None)
    env.reset()
    track_a().reset(W)
    a, imu = np.zeros(23, np.float32), []
    for _ in range(int(round(seconds / env.dt))):
        o = env.observe()
        imu.append(o[0:6].copy())
        nxt = np.asarray(track_a().act(W, o[9:32], o[32:55], o[55:78], o[0:6]), np.float32)
        env.step(a)  # 1-step latency, as in the episodes
        a = nxt
    return np.mean(imu[len(imu) // 2:], 0)


def standing_obs() -> np.ndarray:
    from biocompute.robot.asimov import AsimovEnv

    env = AsimovEnv()
    env.reset()
    return env.observe()


def setup_job(job):
    import torch

    from biocompute.interface.decoders import LinearReadout
    from biocompute.runguard import Guard
    from biocompute.train.setup import build

    cfg, dish, out, sz = job
    name = f"{cfg}_{dish}"
    if stop_requested():  # a queued job started after the stop: don't build a culture just to save it
        raise Stopped(f"replay {name} before start")
    g = Guard(f"setup_{name}", out)
    O = load_replay(sz["data"], sz["replay_steps"])
    setup = make_setup(CONFIGS[cfg], dish)
    ctrl = build(setup, O)
    dev, cu = ctrl.device, ctrl.device.culture
    g.check(cu.plasticity is None, "plasticity is on")
    ck = out / "ckpt" / f"replay_{name}.pkl"
    ck.parent.mkdir(parents=True, exist_ok=True)
    n = len(O)
    F = None
    if ck.exists():
        s = pickle.loads(ck.read_bytes())
        dev.load_state(s["device"])
        ctrl.features.traces = s["traces"]
        F, i0 = s["F"], s["i"]
        print(json.dumps({"setup": name, "resumed_at": i0, "of": n}), flush=True)
    else:
        dev.step(1000.0)  # as vestibular_probe.py
        i0 = 0
    if F is None:
        F = np.zeros((n, ctrl.features.dim), np.float32)
    w_ref = cu.w.clone()
    t_save = time.time()
    t0 = time.time()
    for i in range(i0, n):
        if stop_requested() or time.time() - t_save > sz["ckpt_every_s"]:
            tmp = ck.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps({"device": dev.state(), "traces": ctrl.features.traces.copy(),
                                          "F": F, "i": i}))
            os.replace(tmp, ck)
            t_save = time.time()
            if stop_requested():
                raise Stopped(f"replay {name} at step {i}/{n}")
        F[i] = ctrl.sense(O[i])
        if g.due():
            # Rate over the last 1 s only once a full second has run: the first stimulated step after the quiet
            # warm-up answers with an onset burst (331 Hz per electrode in one 20 ms step, s32 smoke).
            win = F[i - 49:i + 1, :2 * (F.shape[1] // 4)] if i - i0 >= 49 else None
            g.culture(cu, rate_hz=None if win is None else float((win ** 2).sum(1).mean() / (F.shape[1] // 4) / 0.02))
            g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
            g.beat(i + 1, n, ms_per_step=round((time.time() - t0) / max(i + 1 - i0, 1) * 1000, 1))
    C = F.shape[1] // 4  # layout: 2 bins x channels, then the 40 and 160 ms traces
    cols = 3 * C
    # A resume that skipped steps would leave rows at zero (stimulated cultures fire ~25-70 Hz per electrode).
    zero_rows = float((F[:, : 2 * C].sum(1) == 0).mean())
    g.check(zero_rows < 0.01, f"{zero_rows:.1%} of replay steps have no spikes at all")
    T = O[:, :6]
    ep_start = [i for i in range(1, n) if np.abs(O[i, 9:55] - O[i - 1, 9:55]).max() > 0.5]
    split = min(ep_start, key=lambda i: abs(i - 0.8 * n)) if ep_start else int(0.8 * n)
    if not 0.6 * n <= split <= 0.9 * n:
        split = int(0.8 * n)
    ro = LinearReadout(cols, 6)
    cv = ro.fit_cv(F[:split, :cols], T[:split], alphas=np.logspace(-1, 5, 13), standardize_y=True)
    P = rebuild_grav_z(ro.predict(F[split:, :cols]))
    held = channel_stats(T[split:], P)
    stand = standing_imu()
    held_stand = channel_stats(T[split:], P, center=stand)  # the window the affine edit is fitted on
    rate = float((F[:, : 2 * C] ** 2).sum(1).mean() / C / 0.02)
    g.check(bool(np.isfinite(P).all()), "decoder predictions not finite")
    gain = held.get("gain")
    edit_gain = None
    if gain is not None and all(x is not None and x > 0.05 for x in gain):
        edit_gain = gain
    elif not sz["smoke"]:
        g.fail(f"held-out gain unusable for the edit: {gain}")
    aff_g, aff_a = held_stand.get("gain"), held_stand.get("intercept")
    if not (aff_g is not None and all(x is not None and 0.05 < x < 1.5 for x in aff_g)):
        aff_g = aff_a = None
        if not sz["smoke"]:
            g.fail(f"held-out affine fit unusable: gain {held_stand.get('gain')}")
    probe = json.loads((ROOT / "outputs/vestibular_probe/channel_models.json").read_text()).get(PROBE_NAME[cfg])
    if dish == "live" and not sz["smoke"]:
        g.check(probe is not None, f"no Phase 3.1 channel model for {cfg}")
        diff = np.abs(np.array(gain) - np.array(probe["gain"][:5]))
        g.check(bool((diff < 0.2).all()), f"held-out gains {gain} don't reproduce Phase 3.1's {probe['gain'][:5]}")
    # Teacher parity: the numpy teacher (gain 1) against the ONNX policy on replay observations.
    from biocompute.robot.teacher import Teacher

    teacher, W = Teacher(), teacher_weights(None)
    ta = track_a()
    idx = np.linspace(0, n - 1, min(n, 200)).astype(int)
    diffs = [np.abs(teacher(O[k]) - ta.act(W, O[k, 9:32], O[k, 32:55], O[k, 55:78], O[k, 0:6])).max() for k in idx]
    g.check(max(diffs) < 1e-3, f"numpy teacher differs from ONNX by {max(diffs):.3g}")
    # Warm state: 1 s of the standing IMU as input after the replay; every episode starts here.
    obs0 = standing_obs()
    for _ in range(WARM_STEPS):
        ctrl.sense(obs0)
    g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
    row = {"cfg": cfg, "dish": dish, "n_stim": CONFIGS[cfg], "replay_steps": n, "split": int(split),
           "channels_recorded": int(C), "decoder_features": int(cols), "alpha": cv["alpha"],
           "cv_r2_train": round(cv["val_r2"], 3), "held_out": held, "edit_gain": edit_gain,
           "edit_input_scale": None if edit_gain is None else np.round(1 / np.array(edit_gain), 3).tolist(),
           "phase31_gain": None if probe is None else probe["gain"][:5], "mean_rate_hz": round(rate, 2),
           "stand_imu": np.round(stand, 5).tolist(), "held_out_at_stand": held_stand,
           "affine_gain": aff_g, "affine_icpt": aff_a,
           "teacher_parity_max_diff": float(max(diffs)), "setup": setup.to_dict(), "smoke": sz["smoke"]}
    blob = {"row": row, "readout": {k: getattr(ro, k) for k in ("W", "mean", "std", "y_mean", "y_std")},
            "warm": dev.state(), "obs0": obs0, "cols": cols,
            "hold_T": T[split:].astype(np.float32), "hold_P": P.astype(np.float32)}
    p = out / "setups" / f"{name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(blob))
    os.replace(tmp, p)
    write_json(out / "setups" / f"{name}.json", row)
    ck.unlink(missing_ok=True)
    g.beat(n, n, force=True, gain=str(gain), r2=str(held["r2"]))
    print(json.dumps({"setup": name} | {k: row[k] for k in ("alpha", "edit_gain", "phase31_gain", "affine_gain",
                                                             "affine_icpt", "stand_imu", "mean_rate_hz",
                                                             "teacher_parity_max_diff")}
                     | {"held_out_rmse": held.get("rmse_over_spread"), "held_out_r2": held["r2"]}), flush=True)
    del ctrl, F
    torch.cuda.empty_cache()
    return row


# --- Closed-loop episodes -----------------------------------------------------------------------------------------------
def episode_job(job):
    from biocompute.interface.decoders import LinearReadout
    from biocompute.loop import run_episode
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.runguard import Guard
    from biocompute.train.tasks import make_task

    cfg, arm, seeds, out, sz = job
    spec = ARMS[arm]
    todo = [s for s in seeds if not (out / "jobs" / cfg / arm / f"{s}.json").exists()]
    if not todo or stop_requested():
        return {"stopped": f"{cfg}/{arm} before start"} if todo else {"done": 0}
    g = Guard(f"{cfg}_{arm}_{seeds[0]}", out)
    env = AsimovEnv()
    dish = readout = cu = w_ref = None
    cols, obs0, gain, icpt, warm, center = 0, standing_obs(), None, None, None, None
    if spec["dish"]:
        import torch

        from biocompute.train.setup import build

        blob = pickle.loads((out / "setups" / f"{cfg}_{spec['dish']}.pkl").read_bytes())
        O = load_replay(sz["data"], sz["replay_steps"])
        dish = build(make_setup(CONFIGS[cfg], spec["dish"]), O)  # same encoder fit as the setup's
        cu = dish.device.culture
        g.check(cu.plasticity is None, "plasticity is on")
        readout = LinearReadout(blob["cols"], 6)
        for k, v in blob["readout"].items():
            setattr(readout, k, v)
        cols, obs0, warm = blob["cols"], blob["obs0"], blob["warm"]
        if sz["edit"] == "affine":
            gain, icpt = blob["row"]["affine_gain"], blob["row"]["affine_icpt"]
        else:
            gain = blob["row"]["edit_gain"]
        center = blob["row"].get("stand_imu")
        w_ref = torch.as_tensor(warm["culture"]["w"]).to(cu.device)
    W = None if spec["teacher"] is None else (teacher_weights(gain, icpt) if spec["teacher"] == "edit"
                                              else teacher_weights(None))
    ctl = VestibularController(dish, readout, cols, W, spec["stim"], obs0)
    steps_total = int(round(sz["seconds"] / env.dt))
    n_done = 0
    for si, seed in enumerate(todo):
        if stop_requested():
            raise Stopped(f"{cfg}/{arm} before seed {seed}")
        if dish is not None:
            dish.device.load_state(warm)
            cu.torch_gen.manual_seed(7 * seed + 1)
            dish.device.mea.gen.manual_seed(7 * seed + 2)
        task = make_task("stand_push", sz["seconds"], seed=seed)

        def on_step(step, env_, c_):
            if stop_requested():
                raise Stopped(f"{cfg}/{arm} seed {seed} at step {step} (redone on resume)")
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
        row = {"cfg": cfg, "arm": arm, "edit": sz["edit"], "seed": seed, "survived_s": round(survived, 2),
               "full": bool(ep.fell_at < 0), "steps": ep.steps, "seconds": sz["seconds"]}
        if dish is not None:
            row["rate_hz_per_electrode"] = round(ctl.spikes / len(dish.features.channels)
                                                 / (ep.steps * 0.02), 2)
        if ctl.decoded and spec["dish"]:
            row["closed_loop_channel"] = channel_stats(np.array(ctl.true), np.array(ctl.decoded), center=center)
        write_json(out / "jobs" / cfg / arm / f"{seed}.json", row)
        n_done += 1
        g.beat(si + 1, len(todo), force=True, seed=seed, survived=row["survived_s"])
    return {"done": n_done}


# --- Summary ------------------------------------------------------------------------------------------------------------
def summarize(out: Path, combos, seeds, seconds, edit="gain") -> dict:
    table = []
    for cfg, arm in combos:
        rows = [json.loads(p.read_text()) for s in seeds if (p := out / "jobs" / cfg / arm / f"{s}.json").exists()]
        surv = np.array([r["survived_s"] for r in rows])
        full = int(sum(r["full"] for r in rows))
        e = {"cfg": cfg, "arm": arm, "n": len(rows), "full": full,
             "mean_s": round(float(surv.mean()), 2) if len(rows) else None,
             "se_s": round(float(surv.std(ddof=1) / np.sqrt(len(rows))), 2) if len(rows) > 1 else None,
             "survived_s": [r["survived_s"] for r in rows]}
        cl = [r["closed_loop_channel"] for r in rows if "closed_loop_channel" in r and "gain" in
              r["closed_loop_channel"]]
        if cl:
            e["closed_loop_gain_median"] = np.round(np.median([[np.nan if x is None else x for x in c["gain"]] for c in cl], 0),
                                                    3).tolist()
            e["closed_loop_noise_median"] = np.round(np.median([[np.nan if x is None else x for x in c["noise"]] for c in cl], 0),
                                                     3).tolist()
        rates = [r["rate_hz_per_electrode"] for r in rows if "rate_hz_per_electrode" in r]
        if rates:
            e["rate_hz_per_electrode_median"] = round(float(np.median(rates)), 2)
        table.append(e)
    by = {(e["cfg"], e["arm"]): e for e in table}
    gate = None
    if ("s12", "culture_edit") in by and ("s12", "silenced") in by:
        ce, si = by[("s12", "culture_edit")], by[("s12", "silenced")]
        complete = ce["n"] == len(seeds) and si["n"] == len(seeds)
        bar = int(np.ceil(0.85 * len(seeds)))  # 34/40
        gate = {"culture_edit_full": f"{ce['full']}/{ce['n']}", "silenced_full": f"{si['full']}/{si['n']}",
                "bar": f">= {bar}/{len(seeds)} and 0/{len(seeds)}", "complete": complete,
                "pass": bool(complete and ce["full"] >= bar and si["full"] == 0)}
    dead_vs_live = {}
    for cfg in CONFIGS:
        if (cfg, "culture_edit") in by and (cfg, "dead_edit") in by:
            a, b = by[(cfg, "culture_edit")], by[(cfg, "dead_edit")]
            if a["n"] and b["n"]:
                dead_vs_live[cfg] = {"live_full": f"{a['full']}/{a['n']}", "dead_full": f"{b['full']}/{b['n']}",
                                     "live_minus_dead_mean_s": round(a["mean_s"] - b["mean_s"], 2)}
    setups = {p.stem: json.loads(p.read_text()) for p in sorted((out / "setups").glob("*.json"))}
    res = {"edit": edit, "seeds": [seeds[0], seeds[-1]], "seconds": seconds, "table": table, "gate_A1": gate,
           "dead_vs_live": dead_vs_live,
           "setups": {k: {kk: v.get(kk) for kk in ("edit_gain", "phase31_gain", "held_out", "affine_gain", "affine_icpt",
                                                   "stand_imu", "held_out_at_stand", "alpha", "mean_rate_hz",
                                                   "replay_steps", "split")} for k, v in setups.items()}}
    write_json(out / "results.json", res)
    for e in table:
        print(json.dumps({k: v for k, v in e.items() if k != "survived_s"}), flush=True)
    print(json.dumps({"gate_A1": gate, "dead_vs_live": dead_vs_live}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="how many seeds from --seed-base")
    ap.add_argument("--seed-base", type=int, default=SEEDS[0],
                    help="9000 (run 1); 9100 for the affine-edit run, designed after run 1 used 9000-9039")
    ap.add_argument("--edit", choices=["gain", "affine"], default="gain",
                    help="gain: IMU columns x 1/gain (run 1, pre-registered); affine: also undo the decoder's offset")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--replay-steps", type=int, default=0, help="first N replay steps (0: all ~19k)")
    ap.add_argument("--data", default="outputs/probe/teacher_data_20k.pkl")
    ap.add_argument("--chunk", type=int, default=10, help="seeds per job")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ckpt-every-s", type=float, default=120.0)
    ap.add_argument("--out", default="outputs/closed_loop_vestibular")
    ap.add_argument("--smoke", action="store_true",
                    help="whole pipeline at tiny scale into <out>/smoke: 300 replay steps, 2 seeds, 1 s episodes")
    a = ap.parse_args()
    import faulthandler

    faulthandler.register(signal.SIGUSR1)
    out = Path(a.out)
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
        a.replay_steps, a.seeds, a.seconds, a.chunk, a.ckpt_every_s = 300, 2, 1.0, 1, 5.0
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed_base, a.seed_base + a.seeds))
    threads = max(1, min(4, (os.cpu_count() or 4) // max(a.workers, 1)))
    sz = {"data": a.data, "replay_steps": a.replay_steps, "seconds": a.seconds, "smoke": a.smoke,
          "ckpt_every_s": a.ckpt_every_s, "edit": a.edit}
    combos = [(c, arm) for c in a.configs for arm in a.arms if arm in CULTURE_ARMS]
    combos += [("none", arm) for arm in a.arms if arm not in CULTURE_ARMS]
    print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "combos": len(combos), "seeds": len(seeds),
                      "seconds": a.seconds, "workers": a.workers, "smoke": a.smoke, "edit": a.edit,
                      "seed_base": a.seed_base}), flush=True)
    dishes = sorted({(c, ARMS[arm]["dish"]) for c, arm in combos if ARMS[arm]["dish"]})
    todo = [(c, d, out, sz) for c, d in dishes if not (out / "setups" / f"{c}_{d}.pkl").exists()]
    if todo:
        pool_map(_setup_safe, todo, min(a.workers, len(todo)), threads)
    jobs = []
    for c, arm in combos:
        left = [s for s in seeds if not (out / "jobs" / c / arm / f"{s}.json").exists()]
        jobs += [(c, arm, left[i:i + a.chunk], out, sz) for i in range(0, len(left), a.chunk)]
    # Long jobs first, so the short ones fill in at the end.
    jobs.sort(key=lambda j: (ARMS[j[1]]["dish"] is None, j[1] == "silenced"))
    if jobs:
        pool_map(_episode_safe, jobs, min(a.workers, len(jobs)), threads)
    summarize(out, combos, seeds, a.seconds, a.edit)


if __name__ == "__main__":
    main()
