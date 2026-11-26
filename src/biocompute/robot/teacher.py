from __future__ import annotations

from pathlib import Path

import numpy as np

from biocompute import paths


class Teacher:
    def __init__(self, policy_dir: str | Path = paths.POLICY_DIR):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(Path(policy_dir) / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input: np.asarray(obs, dtype=np.float32)[None]})[0]
        return out[0].astype(np.float32)
