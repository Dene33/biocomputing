import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

OUTCOME = {0: {0: 19 / 20, 1: 0 / 20}, 1: {0: 0 / 20, 1: 19 / 20}}
GRID = [(r, c) for r in range(8) for c in range(8) if (r, c) not in {(0, 0), (0, 7), (7, 0), (7, 7)}]
CH = {rc: i for i, rc in enumerate(GRID)}
P = [[CH[x] for x in [(2, 2), (4, 2), (3, 5), (5, 5)]], [CH[x] for x in [(3, 2), (5, 2), (2, 5), (4, 5)]]]
G = [[CH[x] for x in [(2, 3), (4, 3), (3, 4), (5, 4)]], [CH[x] for x in [(3, 3), (5, 3), (2, 4), (4, 4)]]]


CULTURE = {"name": "gpu_20k"}


def build(seed, homeostasis):
    import torch

    from biocompute.culture.network import make_config
    from biocompute.culture.plasticity import STDP
    from biocompute.device import SimulatedBiocomputer
    from biocompute.mea.electrodes import make_mea_config

    torch.set_num_threads(1)
    dev = SimulatedBiocomputer(make_config(CULTURE["name"], seed=seed), make_mea_config("mcs60", seed=seed, stim_length_um=120.0))
    dev.step(2000.0)
    c = dev.culture
    w_in0 = c.excitatory_input()
    return dev, STDP(), w_in0


def scale(culture, w_in0):
    import torch

    exc = culture.syn_exc
    w_in = culture.excitatory_input()
    f = torch.where(w_in > 0, w_in0 / w_in.clamp(min=1e-6), torch.ones_like(w_in))
    culture.w[exc] *= f[culture.post[exc]]
    culture.w[exc] = culture.w[exc].clamp(0.0, culture.cfg.w_max)


def probe(dev, k, repeats=3, both=False):
    a = b_ = 0.0
    for _ in range(repeats):
        ch_p = P[0] + P[1] if both else P[k]
        n = len(ch_p)
        dev.stimulate(np.array(ch_p * 2), np.array([0.0] * n + [10.0] * n), np.ones(2 * n))
        b = dev.step(60.0)
        m = (b.times_ms >= 5.0) & (b.times_ms < 60.0)
        ch = b.channels[m]
        a += float(np.isin(ch, G[0]).sum())
        b_ += float(np.isin(ch, G[1]).sum())
        dev.step(150.0)
    return a, b_


def pair(dev, k, g, reps=20, both=False):
    ch_p = P[0] + P[1] if both else P[k]
    for _ in range(reps):
        dev.stimulate(np.array(ch_p + G[g]), np.array([0.0] * len(ch_p) + [8.0] * 4), np.ones(len(ch_p) + 4))
        dev.step(100.0)


def run(job):
    from biocompute.culture.plasticity import rest

    mode, seed, trials, homeostasis, context, block = job
    both = context == "none"
    rng = np.random.default_rng(seed + 100)
    dev, stdp, w_in0 = build(seed, homeostasis)
    c = dev.culture
    bias = []
    for k in (0, 1):
        s = [np.subtract(*probe(dev, k, both=both)) for _ in range(15)]
        bias.append(float(np.mean(s)))
    if both:
        bias = [float(np.mean(bias))] * 2
    c.plasticity = stdp
    log = []
    w0_mean = float(c.w[c.syn_exc].mean())
    t0 = time.time()
    for t in range(trials):
        k = (t // block) % 2 if both else int(rng.integers(2))
        a, b = probe(dev, k, both=both)
        score = (a - b) - bias[k]
        choice = 0 if score > 0 else 1 if score < 0 else int(rng.integers(2))
        ok = bool(rng.random() < OUTCOME[k][choice])
        if not ok:
            if mode == "adaptive":
                pair(dev, k, 1 - choice, both=both)
            elif mode == "random":
                pair(dev, k, int(rng.integers(2)), both=both)
            else:
                dev.step(2000.0)
        dev.step(200.0)
        rest(c, 5.0, stdp)
        if homeostasis:
            scale(c, w_in0)
        log.append({"t": t, "context": k, "choice": choice, "correct": int(choice == k), "survived": int(ok),
                    "score": round(score, 2)})
    corr = np.array([x["correct"] for x in log])
    return {"mode": mode, "seed": seed, "homeostasis": homeostasis, "context": context, "block": block,
            "trials": trials, "bias": bias,
            "correct_first30": float(corr[:30].mean()), "correct_last50": float(corr[-50:].mean()),
            "survived_last50": float(np.mean([x["survived"] for x in log[-50:]])),
            "w_exc_mean_start": round(w0_mean, 3), "w_exc_mean_end": round(float(c.w[c.syn_exc].mean()), 3),
            "wall_s": round(time.time() - t0), "log": log}


def dose(job):
    import torch

    mode, seed, doses = job
    dev, stdp, w_in0 = build(seed, True)
    c = dev.culture

    def measure():
        c.plasticity = None
        out = {}
        for k in (0, 1):
            s = np.array([np.subtract(*probe(dev, k, repeats=1)) for _ in range(45)])
            out[k] = (round(float(s.mean()), 2), round(float(s.std() / np.sqrt(len(s))), 2))
        c.plasticity = stdp
        return out

    rows, done = [], 0
    rows.append({"pairings": 0, **{f"ctx{k}": v for k, v in measure().items()},
                 "w_exc_mean": round(float(c.w[c.syn_exc].mean()), 4)})
    for d in doses:
        n = d - done
        c.plasticity = stdp
        for _ in range(n):
            if mode == "causal":
                dev.stimulate(np.array(P[0] + G[1]), np.array([0.0] * 4 + [8.0] * 4), np.ones(8))
            elif mode == "reverse":
                dev.stimulate(np.array(G[1] + P[0]), np.array([0.0] * 4 + [8.0] * 4), np.ones(8))
            dev.step(100.0)
        done = d
        rows.append({"pairings": d, **{f"ctx{k}": v for k, v in measure().items()},
                     "w_exc_mean": round(float(c.w[c.syn_exc].mean()), 4)})
        print(mode, seed, rows[-1], flush=True)
    del dev
    torch.cuda.empty_cache()
    return {"mode": mode, "seed": seed, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", action="store_true", help="dose-response of pairing instead of coaching")
    ap.add_argument("--modes", nargs="+", default=["adaptive", "random", "none"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--no-homeostasis", action="store_true")
    ap.add_argument("--context", choices=["given", "none"], default="given")
    ap.add_argument("--block", type=int, default=30, help="--context none: trials between unannounced switches")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/coach_probe")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dose:
        jobs = [(m, s, [100, 400, 1000]) for m in ("causal", "reverse", "none") for s in a.seeds]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(dose, jobs)
        (out / "dose.json").write_text(json.dumps(res, indent=1))
        for r in res:
            print(r["mode"], r["seed"], [(x["pairings"], x["ctx0"], x["ctx1"]) for x in r["rows"]], flush=True)
        return
    jobs = [(m, s, a.trials, not a.no_homeostasis, a.context, a.block) for m in a.modes for s in a.seeds]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(run, jobs)
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != "log"}), flush=True)
    tag = ("" if not a.no_homeostasis else "_nohomeo") + ("" if a.context == "given" else "_nocontext")
    with open(out / f"results{tag}.json", "w") as f:
        json.dump(res, f)
    summ = {}
    for m in a.modes:
        rs = [r for r in res if r["mode"] == m]
        summ[m] = {"correct_first30": round(float(np.mean([r["correct_first30"] for r in rs])), 3),
                   "correct_last50": round(float(np.mean([r["correct_last50"] for r in rs])), 3),
                   "survived_last50": round(float(np.mean([r["survived_last50"] for r in rs])), 3)}
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
