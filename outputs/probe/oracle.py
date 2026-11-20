"""Is the missing nonlinear information inside the culture (oracle: every neuron) or lost at the electrodes?
One well, identity encoding; records MEA features and the filtered spike counts of 8000 random neurons."""
import pickle
import sys
import time

import numpy as np

from biocompute.interface.decoders import SpikeFeatures
from biocompute.train.setup import Setup, build

d = pickle.load(open("outputs/probe/teacher_data_20k.pkl", "rb"))
O, Y = d["obs"], d["y"]
mea = sys.argv[1] if len(sys.argv) > 1 else "grid256"
setup = Setup(culture="gpu_20k", mea=mea, n_stim=75, projection="identity", encoder_gain=0.5,
              slots_ms=(0.0, 5.0, 10.0, 15.0), mea_overrides={"stim_length_um": 120.0})
ctrl = build(setup, O, Y)
ctrl.device.step(1000.0)
pick = np.random.default_rng(0).choice(ctrl.device.culture.n, 8000, replace=False)
orc = SpikeFeatures(np.arange(8000), bins=1, taus_ms=(40.0, 160.0))
F, Q = [], []
t = time.time()
for o in O:
    F.append(ctrl.sense(o))
    Q.append(orc.from_counts(ctrl.device.oracle_counts[pick][None]))
print("ms/step", (time.time() - t) / len(O) * 1000)
np.savez_compressed(f"outputs/probe/oracle_{mea}.npz", F=np.array(F, np.float32), Q=np.array(Q, np.float32))
