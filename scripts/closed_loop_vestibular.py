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

from biocompute.runguard import job_key

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
SETUP_KEYS = ("data", "replay_steps")
EPISODE_KEYS = SETUP_KEYS + ("seconds", "edit")
PREROLL_STEPS = 5
WARM_STEPS = 50


STOP = {"requested": False}
EVENT = {"stop": None}


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

    torch.set_num_threads(threads)
    EVENT["stop"] = ev
    signal.signal(signal.SIGTERM, _on_term)
    faulthandler.register(signal.SIGUSR1)


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
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    # close() + join(), not the context manager: its terminate() hangs on workers that catch SIGTERM.
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


def make_setup(n_stim: int, dish: str):
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
    def __init__(self, dish, readout, cols, W, stim, obs0):
        self.dish, self.readout, self.cols, self.W, self.stim, self.obs0 = dish, readout, cols, W, stim, obs0
        self.true, self.decoded, self.spikes = [], [], 0

    def reset_readout_state(self) -> None:
        self.true, self.decoded, self.spikes = [], [], 0
        if self.dish is not None:
            self.dish.reset_readout_state()
            for _ in range(PREROLL_STEPS):
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


def setup_stem(cfg: str, dish: str, sz: dict) -> str:
    return job_key(f"{cfg}_{dish}", {k: sz[k] for k in SETUP_KEYS})


def episode_file(out: Path, cfg: str, arm: str, seed: int, sz: dict) -> Path:
    return out / "jobs" / cfg / arm / f"{job_key(str(seed), {k: sz[k] for k in EPISODE_KEYS})}.json"


def load_replay(path: str, steps: int) -> np.ndarray:
    d = pickle.load(open(ROOT / path if not Path(path).is_absolute() else path, "rb"))
    return np.asarray(d["obs"][: steps or None], np.float64)


def standing_imu(seconds: float = 3.0) -> np.ndarray:
    from biocompute.robot.asimov import AsimovEnv

    env, W = AsimovEnv(), teacher_weights(None)
    env.reset()
    track_a().reset(W)
    a, imu = np.zeros(23, np.float32), []
    for _ in range(int(round(seconds / env.dt))):
        o = env.observe()
        imu.append(o[0:6].copy())
        nxt = np.asarray(track_a().act(W, o[9:32], o[32:55], o[55:78], o[0:6]), np.float32)
        env.step(a)
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
    if stop_requested():
        raise Stopped(f"replay {name} before start")
    g = Guard(f"setup_{name}", out)
    O = load_replay(sz["data"], sz["replay_steps"])
    setup = make_setup(CONFIGS[cfg], dish)
    ctrl = build(setup, O)
    dev, cu = ctrl.device, ctrl.device.culture
    g.check(cu.plasticity is None, "plasticity is on")
    stem = setup_stem(cfg, dish, sz)
    ck = out / "ckpt" / f"replay_{stem}.pkl"
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
        dev.step(1000.0)
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
            win = F[i - 49:i + 1, :2 * (F.shape[1] // 4)] if i - i0 >= 49 else None
            g.culture(cu, rate_hz=None if win is None else float((win ** 2).sum(1).mean() / (F.shape[1] // 4) / 0.02))
            g.check(bool(torch.equal(cu.w, w_ref)), "weights moved with plasticity off")
            g.beat(i + 1, n, ms_per_step=round((time.time() - t0) / max(i + 1 - i0, 1) * 1000, 1))
    C = F.shape[1] // 4
    cols = 3 * C
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
    held_stand = channel_stats(T[split:], P, center=stand)
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
    from biocompute.robot.teacher import Teacher

    teacher, W = Teacher(), teacher_weights(None)
    ta = track_a()
    idx = np.linspace(0, n - 1, min(n, 200)).astype(int)
    diffs = [np.abs(teacher(O[k]) - ta.act(W, O[k, 9:32], O[k, 32:55], O[k, 55:78], O[k, 0:6])).max() for k in idx]
    g.check(max(diffs) < 1e-3, f"numpy teacher differs from ONNX by {max(diffs):.3g}")
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
    p = out / "setups" / f"{stem}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(blob))
    os.replace(tmp, p)
    write_json(p.with_suffix(".json"), row)
    ck.unlink(missing_ok=True)
    g.beat(n, n, force=True, gain=str(gain), r2=str(held["r2"]))
    print(json.dumps({"setup": name} | {k: row[k] for k in ("alpha", "edit_gain", "phase31_gain", "affine_gain",
                                                             "affine_icpt", "stand_imu", "mean_rate_hz",
                                                             "teacher_parity_max_diff")}
                     | {"held_out_rmse": held.get("rmse_over_spread"), "held_out_r2": held["r2"]}), flush=True)
    del ctrl, F
    torch.cuda.empty_cache()
    return row


def episode_job(job):
    from biocompute.interface.decoders import LinearReadout
    from biocompute.loop import run_episode
    from biocompute.robot.asimov import AsimovEnv
    from biocompute.runguard import Guard
    from biocompute.train.tasks import make_task

    cfg, arm, seeds, out, sz = job
    spec = ARMS[arm]
    todo = [s for s in seeds if not episode_file(out, cfg, arm, s, sz).exists()]
    if not todo or stop_requested():
        return {"stopped": f"{cfg}/{arm} before start"} if todo else {"done": 0}
    g = Guard(f"{cfg}_{arm}_{seeds[0]}", out)
    env = AsimovEnv()
    dish = readout = cu = w_ref = None
    cols, obs0, gain, icpt, warm, center = 0, standing_obs(), None, None, None, None
    if spec["dish"]:
        import torch

        from biocompute.train.setup import build

        blob = pickle.loads((out / "setups" / f"{setup_stem(cfg, spec['dish'], sz)}.pkl").read_bytes())
        O = load_replay(sz["data"], sz["replay_steps"])
        dish = build(make_setup(CONFIGS[cfg], spec["dish"]), O)
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
        write_json(episode_file(out, cfg, arm, seed, sz), row)
        n_done += 1
        g.beat(si + 1, len(todo), force=True, seed=seed, survived=row["survived_s"])
    return {"done": n_done}


def summarize(out: Path, combos, seeds, sz) -> dict:
    table = []
    for cfg, arm in combos:
        rows = [json.loads(p.read_text()) for s in seeds if (p := episode_file(out, cfg, arm, s, sz)).exists()]
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
        bar = int(np.ceil(0.85 * len(seeds)))
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
    setups = {f"{c}_{d}": json.loads(p.read_text()) for c, d in sorted({(c, ARMS[a]["dish"]) for c, a in combos
                                                                         if ARMS[a]["dish"]})
              if (p := out / "setups" / f"{setup_stem(c, d, sz)}.json").exists()}
    res = {"edit": sz["edit"], "seeds": [seeds[0], seeds[-1]], "seconds": sz["seconds"], "table": table,
           "gate_A1": gate, "dead_vs_live": dead_vs_live,
           "setups": {k: {kk: v.get(kk) for kk in ("edit_gain", "phase31_gain", "held_out", "affine_gain", "affine_icpt",
                                                   "stand_imu", "held_out_at_stand", "alpha", "mean_rate_hz",
                                                   "replay_steps", "split")} for k, v in setups.items()}}
    write_json(out / "results.json", res)
    for e in table:
        print(json.dumps({k: v for k, v in e.items() if k != "survived_s"}), flush=True)
    print(json.dumps({"gate_A1": gate, "dead_vs_live": dead_vs_live}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
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
    todo = [(c, d, out, sz) for c, d in dishes if not (out / "setups" / f"{setup_stem(c, d, sz)}.pkl").exists()]
    if todo:
        pool_map(_setup_safe, todo, min(a.workers, len(todo)), threads)
    jobs = []
    for c, arm in combos:
        left = [s for s in seeds if not episode_file(out, c, arm, s, sz).exists()]
        jobs += [(c, arm, left[i:i + a.chunk], out, sz) for i in range(0, len(left), a.chunk)]
    jobs.sort(key=lambda j: (ARMS[j[1]]["dish"] is None, j[1] == "silenced"))
    if jobs:
        pool_map(_episode_safe, jobs, min(a.workers, len(jobs)), threads)
    summarize(out, combos, seeds, sz)


if __name__ == "__main__":
    main()
