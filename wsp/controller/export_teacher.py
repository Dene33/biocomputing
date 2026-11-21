"""One-off: the teacher's ONNX weights → teacher_weights.npz, so controllers can start from them in numpy.

    uv run --with onnx python wsp/controller/export_teacher.py
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from biocompute import paths

m = onnx.load(str(Path(paths.POLICY_DIR) / "policy.onnx"))
print([(n.op_type, list(n.input), list(n.output)) for n in m.graph.node])
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
for k, v in W.items():
    print(k, v.shape)
np.savez(Path(__file__).parent / "teacher_weights.npz", **W)
