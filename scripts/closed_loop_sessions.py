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

Outputs (outputs/closed_loop_sessions): sessions/<cfg>_<cond>.json (+ .pkl: decoder, edit, warm state),
jobs/<cfg>/<cond>/<arm>/<seed>.json (one per episode), results.json (the table and the gate). A session's 60 s
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

ROOT = clv.ROOT
SEEDS = list(range(9200, 9240))
CONDITIONS = {"day0": dict(days=0.0), "day1": dict(days=1.0), "day3": dict(days=3.0), "day7": dict(days=7.0),
              "newdish": dict(days=0.0, culture_seed=1)}
AGED = ["day1", "day3", "day7", "newdish"]
ARMS = {"stale": dict(dish=True), "recal": dict(dish=True), "ideal_imu": dict(dish=False),
        "zero_action": dict(dish=False)}
AGING_SEED = 11
REST_MS = 1000.0
RECAL_STEPS = 3000  # 60 s
FOLDS = 5


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


def a1_blob(sz, cfg):
    return pickle.loads((Path(sz["a1"]) / "setups" / f"{cfg}_live.pkl").read_bytes())


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
    if cond == "day0" and not sz["smoke"]:
        ref = np.array(row0["held_out_at_stand"]["gain"])
        g.check(stale.get("gain") is not None and bool((np.abs(np.array(stale["gain"]) - ref) < 0.15).all()),
                f"day0 stale gains {stale.get('gain')} don't reproduce A1b's {ref.tolist()}")
        g.check(abs(rate / row0["mean_rate_hz"] - 1) < 0.15, f"day0 rate {rate:.1f} vs A1b {row0['mean_rate_hz']}")
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
    p = out / "sessions" / f"{name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(blob))
    os.replace(tmp, p)
    clv.write_json(out / "sessions" / f"{name}.json", row)
    g.beat(n, n, force=True)
    print(json.dumps({"session": name, "alive_vs_day0": row["alive_vs_day0"], "exc_sd": row["excitability_sd"],
                      "rate_hz": row["rate_hz"], "stale_gain": stale.get("gain"),
                      "stale_bias": stale.get("bias_over_spread"), "recal_gain": aff_g,
                      "recal_cv_r2": row["recal_cv_r2"]}), flush=True)
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
    path = out / "jobs" / cfg / cond / arm
    todo = [s for s in seeds if not (path / f"{s}.json").exists()]
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

        sb = pickle.loads((out / "sessions" / f"{cfg}_{cond}.pkl").read_bytes())
        src = a1_blob(sz, cfg) if arm == "stale" else sb
        O = clv.load_replay(sz["data"], 0)
        dish = make_ctrl(cfg, cond, O)
        cu = dish.device.culture
        g.check(cu.plasticity is None, "plasticity is on")
        readout = LinearReadout(src["cols"], 6)
        for k, v in src["readout"].items():
            setattr(readout, k, v)
        cols, obs0, warm = sb["cols"], sb["obs0"], sb["warm"]
        r = sb["row"]
        gain, icpt = ((r["stale_affine_gain"], r["stale_affine_icpt"]) if arm == "stale"
                      else (r["recal_affine_gain"], r["recal_affine_icpt"]))
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
                row["closed_loop_channel"] = clv.channel_stats(np.array(ctl.true), np.array(ctl.decoded),
                                                               center=center)
        clv.write_json(path / f"{seed}.json", row)
        n_done += 1
        g.beat(si + 1, len(todo), force=True, seed=seed, survived=row["survived_s"])
    return {"done": n_done}


def summarize(out: Path, configs, conds, seeds, seconds) -> dict:
    table = []
    for cfg, cond, arm in [(c, d, a) for c in configs for d in conds for a in ("stale", "recal")] + \
            [("none", "none", a) for a in ("ideal_imu", "zero_action")]:
        rows = [json.loads(p.read_text()) for s in seeds
                if (p := out / "jobs" / cfg / cond / arm / f"{s}.json").exists()]
        if not rows:
            continue
        surv = np.array([r["survived_s"] for r in rows])
        e = {"cfg": cfg, "cond": cond, "arm": arm, "n": len(rows), "full": int(sum(r["full"] for r in rows)),
             "mean_s": round(float(surv.mean()), 2), "survived_s": surv.tolist()}
        cl = [r["closed_loop_channel"] for r in rows if "gain" in r.get("closed_loop_channel", {})]
        if cl:
            for k in ("gain", "bias_over_spread", "noise"):
                e[f"closed_loop_{k}_median"] = np.round(np.nanmedian(
                    [[np.nan if x is None else x for x in c[k]] for c in cl], 0), 3).tolist()
        rates = [r["rate_hz_per_electrode"] for r in rows if "rate_hz_per_electrode" in r]
        if rates:
            e["rate_hz_median"] = round(float(np.median(rates)), 2)
        table.append(e)
    by = {(e["cfg"], e["cond"], e["arm"]): e for e in table}
    bar = int(np.ceil(0.85 * len(seeds)))
    gate = None
    if "s32" in configs:
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
    sessions = {p.stem: json.loads(p.read_text()) for p in sorted((out / "sessions").glob("*.json"))}
    res = {"seeds": [seeds[0], seeds[-1]], "seconds": seconds, "table": table, "gate_A2": gate,
           "sessions": sessions}
    clv.write_json(out / "results.json", res)
    for e in table:
        print(json.dumps({k: v for k, v in e.items() if k != "survived_s"}), flush=True)
    print(json.dumps({"gate_A2": gate}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", default=["s32"], choices=list(clv.CONFIGS))
    ap.add_argument("--conds", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--seed-base", type=int, default=SEEDS[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--recal-steps", type=int, default=RECAL_STEPS)
    ap.add_argument("--a1", default="outputs/closed_loop_vestibular_affine", help="A1b's output (day-0 setups)")
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
    sz = {"data": a.data, "a1": a.a1, "recal_steps": a.recal_steps, "seconds": a.seconds, "smoke": a.smoke}
    for cfg in a.configs:
        if not (Path(a.a1) / "setups" / f"{cfg}_live.pkl").exists():
            sys.exit(f"no A1 setup for {cfg} in {a.a1}")
    print(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": a.configs, "conds": a.conds,
                      "arms": a.arms, "seeds": [seeds[0], seeds[-1]], "seconds": a.seconds,
                      "workers": a.workers, "smoke": a.smoke}), flush=True)
    dish_arms = [x for x in a.arms if ARMS[x]["dish"]]
    todo = [(c, d, out, sz) for c in a.configs for d in a.conds
            if dish_arms and not (out / "sessions" / f"{c}_{d}.pkl").exists()]
    if todo:
        clv.pool_map(_session_safe, todo, min(a.workers, len(todo)), threads)
    combos = [(c, d, x) for c in a.configs for d in a.conds for x in dish_arms]
    combos += [("none", "none", x) for x in a.arms if not ARMS[x]["dish"]]
    jobs = []
    for c, d, x in combos:
        left = [s for s in seeds if not (out / "jobs" / c / d / x / f"{s}.json").exists()]
        jobs += [(c, d, x, left[i:i + a.chunk], out, sz) for i in range(0, len(left), a.chunk)]
    jobs.sort(key=lambda j: not ARMS[j[2]]["dish"])  # long jobs first
    if jobs:
        clv.pool_map(_episode_safe, jobs, min(a.workers, len(jobs)), threads)
    summarize(out, a.configs, a.conds, seeds, a.seconds)


if __name__ == "__main__":
    main()
