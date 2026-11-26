import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from biocompute.train import parallel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="stand_push")
    p.add_argument("--source", default="mea", choices=["mea", "oracle", "none", "dead"])
    p.add_argument("--culture", default="gpu_20k")
    p.add_argument("--mea", default="hd1024")
    p.add_argument("--n-stim", type=int, default=32)
    p.add_argument("--slots", type=float, nargs="+", default=[0.0])
    p.add_argument("--projection", default="supervised", choices=["pca", "random", "supervised", "identity"])
    p.add_argument("--culture-set", nargs="*", default=[], help="Culture overrides, key=value")
    p.add_argument("--mea-set", nargs="*", default=[], help="MEA overrides, key=value")
    p.add_argument("--bins", type=int, default=2)
    p.add_argument("--wells", type=int, default=1)
    p.add_argument("--diverse-wells", action="store_true")
    p.add_argument("--encoder-gain", type=float, default=1.5)
    p.add_argument("--readout", default="ridge", choices=["ridge", "mlp"])
    p.add_argument("--efference", action="store_true", help="Decoder also gets its own last command")
    p.add_argument("--silent-inputs", action="store_true", help="Control: no sensory stimulation")
    p.add_argument("--recovery-weight", type=float, default=4.0)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--beta0", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--latency", type=int, default=1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-seeds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="outputs/dagger")
    a = p.parse_args()

    from biocompute.robot.asimov import AsimovEnv
    from biocompute.robot.teacher import Teacher
    from biocompute.train import dagger
    from biocompute.train.setup import Setup, build
    from biocompute.train.tasks import make_task

    name = f"{a.task}_{a.source}_{a.culture}_{a.mea}_w{a.wells}_s{a.seed}{a.tag}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    env, teacher = AsimovEnv(), Teacher()
    obs = dagger.teacher_observations(env, teacher, [make_task(a.task, a.seconds, seed=a.seed)], seconds=a.seconds,
                                      episodes=6, noise=0.3, seed=a.seed)
    kv = lambda items: {k: float(v) for k, v in (i.split("=") for i in items)}
    setup = Setup(culture=a.culture, mea=a.mea, n_stim=a.n_stim, source=a.source, seed=a.seed,
                  slots_ms=tuple(a.slots), projection=a.projection, bins=a.bins, wells=a.wells,
                  diverse_wells=a.diverse_wells, encoder_gain=a.encoder_gain, efference=a.efference,
                  silent_inputs=a.silent_inputs,
                  culture_overrides=kv(a.culture_set), mea_overrides=kv(a.mea_set))
    run = env.run
    lo = (run.action_clip[:, 0] - run.default_joint_pos) / run.action_scale
    hi = (run.action_clip[:, 1] - run.default_joint_pos) / run.action_scale
    acts = np.clip(np.array([teacher(o) for o in obs]), lo, hi)
    if a.source == "none":
        dim = a.n_stim
    elif a.source == "oracle":
        from biocompute.culture.network import make_config

        dim = make_config(a.culture).n_neurons * 3
    else:
        from biocompute.mea.electrodes import make_mea_config

        cfg = make_mea_config(a.mea, seed=a.seed)
        n_ch = (cfg.rows * cfg.cols - (4 if cfg.skip_corners else 0)) * a.wells
        n_ch -= int(round(cfg.dead_fraction * n_ch))
        dim = n_ch * (setup.bins + len(setup.taus_ms))
    if a.efference:
        dim += 23
    pool = parallel.Pool(setup.to_dict(), obs, a.workers, threads=a.threads, action_samples=acts)
    print(f"{name}: {dim} features, {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    res = parallel.train(pool, dim, a.task, rounds=a.rounds, episodes_per_round=a.episodes, beta0=a.beta0,
                         seconds=a.seconds, latency=a.latency, eval_seeds=a.eval_seeds, seed=a.seed,
                         recovery_weight=a.recovery_weight, readout_kind=a.readout, skip_dim=23 if a.efference else 0,
                         ckpt_dir=out,
                         log=lambda r: (rows.append(r), print(json.dumps(r), flush=True)))
    from biocompute.interface.decoders import EnsembleReadout

    candidates = {"best_round": res["best_readout"], "ensemble_last4": EnsembleReadout(res["readouts"][-4:])}
    scores = {}
    for name_c, cand in candidates.items():
        ev_c = pool.map([dict(readout=cand, task=a.task, seconds=a.seconds, seed=500 + s, beta=0.0,
                              latency=a.latency, label=False) for s in range(8)])
        scores[name_c] = float(np.mean([e["survived_s"] for e in ev_c]))
    print(json.dumps({"selection": scores}), flush=True)
    ro = candidates[max(scores, key=scores.get)]
    with open(out / "readouts_all.pkl", "wb") as f:
        pickle.dump(res["readouts"], f)
    ev = pool.map([dict(readout=ro, task=a.task, seconds=a.seconds, seed=1000 + s, beta=0.0, latency=a.latency,
                        label=False) for s in range(10)])
    pool.close()
    surv = [e["survived_s"] for e in ev]
    summary = {"name": name, "setup": setup.to_dict(), "args": vars(a), "train_seconds": time.time() - t0,
               "selection": scores, "selected": max(scores, key=scores.get),
               "history": rows, "final_eval": ev, "final_survived_s_mean": float(np.mean(surv)),
               "final_full": f"{sum(s >= a.seconds for s in surv)}/{len(surv)}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "readout.pkl", "wb") as f:
        pickle.dump(ro, f)
    print(json.dumps({k: summary[k] for k in ("final_survived_s_mean", "final_full")}), flush=True)


if __name__ == "__main__":
    main()
