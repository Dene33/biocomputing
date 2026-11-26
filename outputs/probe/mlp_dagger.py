import sys

import numpy as np
import torch
from collections import deque

from biocompute.robot.asimov import AsimovEnv
from biocompute.robot.teacher import Teacher
from biocompute.train.tasks import make_task

torch.set_num_threads(4)
env, teacher = AsimovEnv(), Teacher()
lo = (env.run.action_clip[:, 0] - env.run.default_joint_pos) / env.run.action_scale
hi = (env.run.action_clip[:, 1] - env.run.default_joint_pos) / env.run.action_scale
rng = np.random.default_rng(0)
hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 256
mode = sys.argv[2] if len(sys.argv) > 2 else "mlp"
from biocompute.interface.decoders import LinearReadout
import os
GAIN = float(os.environ.get("GAIN", 1.5)); WEIGHT = float(os.environ.get("WEIGHT", 0)); MEM = int(os.environ.get("MEM", 0))
TAG = os.environ.get("TAG", "")
DRIVES = int(os.environ.get("DRIVES", 0))
ENC = None
if DRIVES:
    from biocompute.interface.encoders import Encoder, ObsNormalizer
    from biocompute.train import dagger as _dg
    from biocompute.train.tasks import make_task as _mt
    NS = int(os.environ.get("NSTIM", 32))
    if os.environ.get("ENC_FIT", "teacher") == "dart":
        import pickle as _pk
        _d = _pk.load(open("outputs/probe/teacher_data_20k.pkl", "rb")); _O, _A = _d["obs"], _d["y"]
    else:
        _O = _dg.teacher_observations(env, teacher, [_mt("stand_push", 20.0, seed=0)], seconds=20.0)
        _A = np.clip(np.array([teacher(o) for o in _O]), lo, hi)
    ENC = Encoder(np.arange(NS), normalizer=ObsNormalizer.fit(_O), gain=float(os.environ.get("EGAIN", 1.5)))
    if os.environ.get("PROJ", "supervised") == "pca":
        ENC.fit_pca(_O)
    elif os.environ.get("PROJ") == "identity":
        dims = [i for i in range(78) if i not in (6, 7, 8)]
        ENC.W = np.eye(78)[dims]; ENC.b = -ENC.W @ ENC.normalizer(_O).mean(0)
    else:
        ENC.fit_supervised(_O, _A)
    _W, _b = [ENC.W], [ENC.b]
    for k in range(1, DRIVES):
        R, _ = np.linalg.qr(np.random.default_rng(100 + k).standard_normal((NS, NS)))
        _W.append(R @ ENC.W); _b.append(-(R @ ENC.W) @ ENC.normalizer(_O).mean(0))
    ENC.W, ENC.b = np.concatenate(_W), np.concatenate(_b)
    ENC.channels = np.arange(NS * DRIVES)
IN = (NS * DRIVES) if DRIVES else 78 * (2 if MEM else 1)
Wrf = np.random.default_rng(1).standard_normal((hidden, IN)) / np.sqrt(IN) * GAIN
brf = np.random.default_rng(2).standard_normal(hidden) * 0.5


SNOISE = float(os.environ.get("SNOISE", 0))
_SD = None


def corrupt(o):
    if not SNOISE or _SD is None:
        return o
    n = SNOISE * _SD * rng.standard_normal(78)
    n[6:9] = 0.0
    n[55:] = 0.0
    return (o + n).astype(np.float32)


def run(policy, seed, beta):
    task = make_task("stand_push", 20, seed=seed)
    env.reset()
    if policy is not None and hasattr(policy, "__defaults__") and MEM:
        pass
    q = deque([np.zeros(23, np.float32)])
    O, L = [], []
    for s in range(1000):
        t = s * env.dt
        if (f := task.push_fn(t)) is not None:
            env.push(f)
        o = env.observe()
        lab = np.clip(teacher(o), lo, hi)
        a = lab if (policy is None or rng.random() < beta) else policy(corrupt(o))
        O.append(corrupt(o))
        L.append(lab)
        q.append(np.asarray(a, np.float32))
        if env.step(q.popleft()).fell:
            break
    return np.array(O), np.array(L), (env.fell_at if env.fell_at >= 0 else 20.0)


X, Y = [], []
policy = None
for r in range(10):
    beta = 1.0 if r == 0 else 0.5 * 0.6**r
    for e in range(int(os.environ.get("EPIS", 6))):
        O, L, _ = run(policy, 1000 * r + e, beta)
        X.append(O)
        Y.append(L)
    Xa, Ya = np.concatenate(X), np.concatenate(Y)
    if _SD is None:
        _SD = Xa.std(0)
    mu, sd = Xa.mean(0), np.maximum(Xa.std(0), 1e-2)
    ym, ys = Ya.mean(0), np.maximum(Ya.std(0), 1e-2)
    Xt = torch.tensor((Xa - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Ya - ym) / ys, dtype=torch.float32)
    if mode == "rf":
        def lift(Z):
            if MEM:
                Z = np.c_[Z, np.r_[np.zeros((1, Z.shape[1])), np.diff(Z, axis=0)] * 3]
            return np.c_[Z, np.tanh(Z @ Wrf.T + brf)]
        if DRIVES:
            Dall = [np.array([ENC.drive(o) for o in x]) for x in X]
            dmu = np.concatenate(Dall).mean(0); dsd = np.concatenate(Dall).std(0) + 1e-6
            F = np.concatenate([lift((dd - dmu) / dsd) for dd in Dall])
        else:
            F = np.concatenate([lift(np.clip((x - mu) / sd, -5, 5)) for x in X])
        w = None
        if WEIGHT:
            gyro = np.abs(Xa[:, 0:2]).sum(1)
            w = 1.0 + WEIGHT * (gyro > np.percentile(gyro, 80))
        ro = LinearReadout(F.shape[1])
        if w is None:
            cv = ro.fit_cv(F, (Ya - ym) / ys, alphas=np.logspace(-3, 3, 7))
        else:
            cv = ro.fit_cv(F, (Ya - ym) / ys, alphas=np.logspace(-3, 3, 7)); ro.fit(F, (Ya - ym) / ys, alpha=cv["alpha"], weights=w)
        state = {"prev": None}
        def policy(o, ro=ro, mu=mu, sd=sd, ym=ym, ys=ys, dmu=(dmu if DRIVES else None), dsd=(dsd if DRIVES else None)):
            z = (ENC.drive(o) - dmu) / dsd if DRIVES else np.clip((o - mu) / sd, -5, 5)
            if MEM:
                d = np.zeros_like(z) if state["prev"] is None else (z - state["prev"]) * 3
                state["prev"] = z
                z = np.r_[z, d]
            return ro(np.r_[z, np.tanh(z @ Wrf.T + brf)]) * ys + ym
        ev = [run(policy, 100 + s, 0.0)[2] for s in range(5)]
        print(f"{TAG} rf={hidden} round {r} samples {len(Xa)} val_r2 {cv['val_r2']:.3f} alpha {cv['alpha']:.3g} eval {np.mean(ev):.1f}s full {sum(e >= 20 for e in ev)}/5", flush=True)
        continue
    net = torch.nn.Sequential(torch.nn.Linear(78, hidden), torch.nn.ELU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ELU(), torch.nn.Linear(hidden, 23))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for it in range(3000):
        idx = torch.randint(0, len(Xt), (512,))
        loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def policy(o, net=net, mu=mu, sd=sd, ym=ym, ys=ys):
        with torch.no_grad():
            return net(torch.tensor((o - mu) / sd, dtype=torch.float32)).numpy() * ys + ym

    ev = [run(policy, 100 + s, 0.0)[2] for s in range(5)]
    print(f"hidden={hidden} round {r} samples {len(Xa)} loss {loss.item():.3f} eval {np.mean(ev):.1f}s "
          f"full {sum(e >= 20 for e in ev)}/5", flush=True)
