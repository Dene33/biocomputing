"""The experiment scripts' bookkeeping and analysis functions, on made-up rows (no simulation)."""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CP = load("cartpole_replication")


def test_job_key_changes_with_every_setting():
    """A finished job or checkpoint is reused only by a run asking for exactly its settings."""
    base = {"scale": 0.0, "rule": "additive", "eta": 0.0, "w_max_inh": 30.0, "floor": 0.5, "seconds": 600.0,
            "rest_s": 20.0}
    k = CP.job_key("drift_x0", base)
    assert k.startswith("drift_x0_") and CP.job_key("drift_x0", dict(reversed(base.items()))) == k
    for field, other in (("floor", 1.0), ("seconds", 300.0), ("rest_s", 2.0), ("rule", "multiplicative")):
        assert CP.job_key("drift_x0", base | {field: other}) != k
