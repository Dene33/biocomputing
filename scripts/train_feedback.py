"""DishBrain-style sessions: does feedback-shaped plasticity in the culture improve the robot?

Protocol ``scratch``: a fixed motor map decoder (opposing channel groups per joint), learning from nothing.
Protocol ``adapt``: start from a trained reservoir readout (``--readout``), perturb the robot or the culture, and see
whether the session recovers performance. The readout is frozen; only the culture can change.

Conditions (``--condition``):
    dishbrain   predictable stimulation after good outcomes, unpredictable after falls; STDP on
    random      the same amount of feedback at random; STDP on
    silent      no feedback; STDP on
    frozen      DishBrain feedback; plasticity off
    rstdp       reward-modulated STDP with a global reward (in-silico upper bound, not deliverable on hardware)

    uv run scripts/train_feedback.py --protocol scratch --condition dishbrain --minutes 10
    uv run scripts/train_feedback.py --protocol adapt --readout outputs/reservoir/stand_push_mea_mcs60_s0 \
        --perturb weak_knees --condition dishbrain
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from biocompute.culture.aging import age
from biocompute.culture.plasticity import STDP, RewardSTDP, rest
from biocompute.interface.decoders import MotorMapDecoder
from biocompute.interface.feedback import Feedback
from biocompute.robot.asimov import AsimovEnv, Perturbation
from biocompute.robot.teacher import Teacher
from biocompute.train import dagger
from biocompute.train.feedback import blocks, session
from biocompute.train.setup import Setup, build
from biocompute.train.tasks import make_task

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", default="scratch", choices=["scratch", "adapt"])
parser.add_argument("--condition", default="dishbrain", choices=["dishbrain", "random", "silent", "frozen", "rstdp"])
parser.add_argument("--task", default=None, help="Default: the readout's task (adapt) or stand_push (scratch)")
parser.add_argument("--readout", default=None, help="adapt: a train_reservoir.py output folder")
parser.add_argument("--perturb", default="none", choices=["none", "weak_knees", "slippery", "cell_loss"])
parser.add_argument("--minutes", type=float, default=10.0)
parser.add_argument("--rest-minutes", type=float, default=45.0, help="Rest before the persistence re-test")
parser.add_argument("--seconds", type=float, default=20.0, help="Episode length")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--threads", type=int, default=3)
parser.add_argument("--out", default="outputs/feedback")
args = parser.parse_args()
torch.set_num_threads(args.threads)

perturbation = Perturbation()
if args.perturb == "weak_knees":
    perturbation.effort_scale = {".*knee.*": 0.6}
elif args.perturb == "slippery":
    perturbation.foot_friction = 0.3
env, teacher = AsimovEnv(perturbation=perturbation), Teacher()
if args.task is None:
    args.task = (json.loads((Path(args.readout) / "summary.json").read_text())["args"]["task"]
                 if args.protocol == "adapt" and args.readout else "stand_push")
task = make_task(args.task, args.seconds, seed=args.seed)

if args.protocol == "adapt":
    if not args.readout:
        parser.error("--protocol adapt needs --readout")
    trained = json.loads((Path(args.readout) / "summary.json").read_text())
    setup = Setup(**{k: tuple(v) if isinstance(v, list) else v for k, v in trained["setup"].items()})
    clean_env = AsimovEnv()
    obs = dagger.teacher_observations(clean_env, teacher, [make_task(trained["args"]["task"], args.seconds)],
                                      seconds=args.seconds)
    ctrl = build(setup, obs)
    with open(Path(args.readout) / "readout.pkl", "rb") as f:
        ctrl.readout = pickle.load(f)
else:
    setup = Setup(seed=args.seed)
    obs = dagger.teacher_observations(env, teacher, [task], seconds=args.seconds)
    ctrl = build(setup, obs)
    record = np.setdiff1d(ctrl.device.mea.alive_channels, np.r_[ctrl.encoder.channels, ctrl.feedback_channels])
    decoder = MotorMapDecoder(record, seed=args.seed, gain=0.5)
    ctrl.readout = None
    # The fixed decoder reads spikes, not features: wrap sense() so act() returns its output.
    sense = ctrl.sense

    def act(obs_, _sense=sense):
        _sense(obs_)
        return decoder(ctrl.last_block), np.zeros(1, np.float32)

    ctrl.act = act
    _reset = ctrl.reset_readout_state
    ctrl.reset_readout_state = lambda: (_reset(), decoder.reset())

culture = ctrl.device.culture
dagger.warm_up(ctrl)
if args.perturb == "cell_loss":
    age(culture, days=10.0, seed=args.seed)  # about 18% of neurons lost, excitability drifted

baseline = dagger.evaluate(env, ctrl, [task], seeds=[0, 1])  # before the session: no plasticity, no feedback

if args.condition == "rstdp":
    culture.plasticity = RewardSTDP()
elif args.condition != "frozen":
    culture.plasticity = STDP()
mode = {"random": "random", "silent": "off"}.get(args.condition, "dishbrain")
fb = Feedback(ctrl.feedback_channels, mode=mode, seed=args.seed)

name = f"{args.protocol}_{args.task}_{args.perturb}_{args.condition}_s{args.seed}"
out = Path(args.out) / name
out.mkdir(parents=True, exist_ok=True)
print(name, flush=True)

rows = session(env, ctrl, fb, task, args.minutes, walking=args.task.startswith("walk"),
               log=lambda r: print(json.dumps(r), flush=True))
ctrl.feedback = None
if culture.plasticity is not None:
    culture.plasticity.enabled = False
after = dagger.evaluate(env, ctrl, [task], seeds=[0, 1])
rest(culture, args.rest_minutes * 60.0, culture.plasticity)
after_rest = dagger.evaluate(env, ctrl, [task], seeds=[0, 1])

summary = {
    "name": name,
    "args": vars(args),
    "feedback_counts": fb.counts,
    "before": baseline,
    "blocks": blocks(rows),
    "after": after,
    "after_rest": after_rest,
    "episodes": rows,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("before", "blocks", "after", "after_rest", "feedback_counts")}, indent=2))
