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
A2_CONDS = list(CONDITIONS)
A2D_DOSES = {"day1_down": (1.0, -0.597), "day1_up": (1.0, 0.734), "day3_down": (3.0, -1.003), "day3_up": (3.0, 1.626),
             "day7_down": (7.0, -1.583), "day7_up": (7.0, 2.0), "day7_stress": (7.0, -3.0)}
A2D_TARGETS = {"day1_down": 0.174, "day1_up": 0.174, "day3_down": 0.336, "day3_up": 0.336, "day7_down": 0.659,
               "day7_up": 0.659, "day7_stress": 1.99}
CONDITIONS |= {k: dict(days=d, shift=x) for k, (d, x) in A2D_DOSES.items()}
A2D_CONDS = list(A2D_DOSES)
REFIT = {"refit60": 3000, "refit30": 1500, "refit10": 500}
ARMS = {"stale": dict(dish=True), "recal": dict(dish=True), **{k: dict(dish=True) for k in REFIT},
        "ideal_imu": dict(dish=False), "zero_action": dict(dish=False)}
DISH_ARMS = [k for k, v in ARMS.items() if v["dish"]]
A2_ARMS = ["stale", "recal", "ideal_imu", "zero_action"]
AGING_SEED = 11
REST_MS = 1000.0
RECAL_STEPS = 3000
FOLDS = 5
SESSION_KEYS = ("data", "a1", "recal_steps")


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


def cond_settings(cond: str) -> dict:
    return {"shift": CONDITIONS[cond]["shift"]} if "shift" in CONDITIONS.get(cond, {}) else {}


def session_stem(cfg: str, cond: str, sz: dict) -> str:
    return job_key(f"{cfg}_{cond}", {k: sz[k] for k in SESSION_KEYS} | cond_settings(cond))


def episode_file(out: Path, cfg: str, cond: str, arm: str, seed: int, sz: dict) -> Path:
    key = job_key(str(seed), {k: sz[k] for k in SESSION_KEYS + ("seconds",)} | cond_settings(cond))
    return out / "jobs" / cfg / cond / arm / f"{key}.json"


def a1_setup(sz: dict, cfg: str) -> Path:
    return Path(sz["a1"]) / "setups" / f"{clv.setup_stem(cfg, 'live', {'data': sz['data'], 'replay_steps': 0})}.pkl"


def a1_blob(sz, cfg):
    return pickle.loads(a1_setup(sz, cfg).read_bytes())


def same(a, b) -> bool:
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
    O = clv.load_replay(sz["data"], 0)
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
    shift = CONDITIONS[cond].get("shift", 0.0)
    if shift:
        m0 = float(cu.excitability.mean())
        cu.excitability += shift
        g.check(abs(float(cu.excitability.mean()) - m0 - shift) < 1e-4, f"excitability shift {shift} not applied")
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
    rate_ch = ((F[:, :C] ** 2 + F[:, C:2 * C] ** 2).sum(0) / n / 0.02).round(4).tolist()
    T = seg[:, :6]
    stand = np.array(row0["stand_imu"])
    ro0 = LinearReadout(b0["cols"], 6)
    for k, v in b0["readout"].items():
        setattr(ro0, k, v)
    P_stale = clv.rebuild_grav_z(ro0.predict(F[:, :cols]))
    stale = clv.channel_stats(T, P_stale, center=stand)
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
    if a2 is not None and cond in A2_CONDS and not sz["smoke"] and a2.resolve() != p.resolve():
        g.check(a2.exists(), f"no stored A2 session {a2} to compare with (--a2 '' skips the check)")
        old = pickle.loads(a2.read_bytes())
        diff = [k for k in ("readout", "warm", "obs0", "cols") if not same(blob[k], old[k])]
        diff += [f"row.{k}" for k, v in old["row"].items() if not same(row.get(k), v)]
        g.check(not diff, f"session {name} differs from A2's stored one in {diff}")
        row["equals_a2_session"] = str(a2)
    row["refit_edit"] = refit
    row["shift"] = shift
    row["replay_rate_per_channel"] = rate_ch
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
        src = a1_blob(sz, cfg) if arm == "stale" or arm in REFIT else sb
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
        if gain is None:
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
                row["closed_loop_channel"] = clv.channel_stats(T_, P_, center=center)
                Q_ = P_.copy()
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
    if "s32" in configs and any(k[2] == "stale" for k in by) and any(c in conds for c in AGED):
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
    if any(k[1] in A2D_CONDS for k in by):
        res["gate_A2d"] = gate_a2d(by, len(seeds), sessions)
    elif any(k[2] in REFIT for k in by):
        res["gate_A2c"] = gate_a2c(by, fulls, len(seeds))
    res["sessions"] = sessions
    clv.write_json(out / "results.json", res)
    for e in table:
        print(json.dumps({k: v for k, v in e.items() if k != "survived_s"}), flush=True)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("gate")}), flush=True)
    return res


def gate_a2c(by: dict, fulls: dict, n_seeds: int) -> dict:
    bar = int(np.ceil(0.85 * n_seeds))
    per = {}
    for cond in A2_CONDS:
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
    complete = all(by.get(("s32", c, "refit60"), {}).get("n") == n_seeds for c in A2_CONDS)
    return {"bar": f">= {bar}/{n_seeds}", "arm": "refit60", "per_condition": per, "complete": complete,
            "pass": bool(complete and all(v["holds"] for v in per.values()))}


def gate_a2d(by: dict, n_seeds: int, sessions: dict) -> dict:
    bar = int(np.ceil(0.85 * n_seeds))
    d0 = sessions.get("s32_day0", {}).get("replay_rate_per_channel")
    per = {}
    for cond in ["day0"] + A2D_CONDS:
        e, st = by.get(("s32", cond, "refit60")), by.get(("s32", cond, "stale"))
        if e is None:
            continue
        per[cond] = {"refit60": f"{e['full']}/{e['n']}", "holds": e["full"] >= bar,
                     "stale": None if st is None else f"{st['full']}/{st['n']}",
                     "stale_drops": None if st is None else st["full"] < bar, "shift": CONDITIONS[cond].get("shift", 0.0)}
        r = sessions.get(f"s32_{cond}", {}).get("replay_rate_per_channel")
        if d0 is not None and r is not None and cond != "day0":
            a, b = np.array(d0), np.array(r)
            live = a >= 0.1
            per[cond]["evoked_change"] = round(float(np.median(np.abs(np.log2(np.maximum(b[live], 0.01) / a[live])))), 3)
            per[cond]["target"] = A2D_TARGETS[cond]
    complete = all(by.get(("s32", c, "refit60"), {}).get("n") == n_seeds for c in ["day0"] + A2D_CONDS)
    return {"bar": f">= {bar}/{n_seeds}", "arm": "refit60", "per_condition": per, "complete": complete,
            "tested": any(v["stale_drops"] for v in per.values() if v["stale_drops"] is not None),
            "pass": bool(complete and all(v["holds"] for v in per.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["s32"], choices=list(clv.CONFIGS))
    ap.add_argument("--conds", nargs="+", default=A2_CONDS, choices=list(CONDITIONS))
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
    jobs.sort(key=lambda j: not ARMS[j[2]]["dish"])
    if jobs:
        clv.pool_map(_episode_safe, jobs, min(a.workers, len(jobs)), threads)
    summarize(out, a.configs, a.conds, seeds, sz)


if __name__ == "__main__":
    main()
