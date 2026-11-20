"""Can ANY student imitate the teacher on stand_push with 1-step latency? DAgger with an MLP on obs."""
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
Wrf = np.random.default_rng(1).standard_normal((hidden, 78)) / np.sqrt(78) * 1.5
brf = np.random.default_rng(2).standard_normal(hidden) * 0.5


def run(policy, seed, beta):
    task = make_task("stand_push", 20, seed=seed)
    env.reset()
    q = deque([np.zeros(23, np.float32)])
    O, L = [], []
    for s in range(1000):
        t = s * env.dt
        if (f := task.push_fn(t)) is not None:
            env.push(f)
        o = env.observe()
        lab = np.clip(teacher(o), lo, hi)
        a = lab if (policy is None or rng.random() < beta) else policy(o)
        O.append(o)
        L.append(lab)
        q.append(np.asarray(a, np.float32))
        if env.step(q.popleft()).fell:
            break
    return np.array(O), np.array(L), (env.fell_at if env.fell_at >= 0 else 20.0)


X, Y = [], []
policy = None
for r in range(10):
    beta = 1.0 if r == 0 else 0.5 * 0.6**r
    for e in range(6):
        O, L, _ = run(policy, 1000 * r + e, beta)
        X.append(O)
        Y.append(L)
    Xa, Ya = np.concatenate(X), np.concatenate(Y)
    mu, sd = Xa.mean(0), np.maximum(Xa.std(0), 1e-2)
    ym, ys = Ya.mean(0), np.maximum(Ya.std(0), 1e-2)
    Xt = torch.tensor((Xa - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Ya - ym) / ys, dtype=torch.float32)
    if mode == "rf":
        F = np.c_[(Xa - mu) / sd, np.tanh(np.clip((Xa - mu) / sd, -5, 5) @ Wrf.T + brf)]
        ro = LinearReadout(F.shape[1]); cv = ro.fit_cv(F, (Ya - ym) / ys, alphas=np.logspace(-3, 3, 7))
        def policy(o, ro=ro, mu=mu, sd=sd, ym=ym, ys=ys):
            z = np.clip((o - mu) / sd, -5, 5)
            return ro(np.r_[z, np.tanh(z @ Wrf.T + brf)]) * ys + ym
        ev = [run(policy, 100 + s, 0.0)[2] for s in range(5)]
        print(f"rf={hidden} round {r} samples {len(Xa)} val_r2 {cv['val_r2']:.3f} alpha {cv['alpha']:.3g} eval {np.mean(ev):.1f}s full {sum(e >= 20 for e in ev)}/5", flush=True)
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
