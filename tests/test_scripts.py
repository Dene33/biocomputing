"""The experiment scripts' bookkeeping and analysis functions, on made-up rows (no simulation)."""

import importlib.util
from pathlib import Path

import numpy as np

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


def drift_row(seed, scale, homeo, minutes, shift=0.0, rate=1.0, rule="additive"):
    """A drift arm: own gains 2, cross 0.5, minute-to-minute noise shared by the arms of one culture, and ``shift``
    added to one cross gain over the last 3 minutes."""
    rng = np.random.default_rng([seed, minutes])
    g = 2.0 * np.eye(2) + 0.5 * (1 - np.eye(2)) + rng.normal(0.0, 0.1, (minutes, 2, 2))
    g[-3:, 0, 1] += shift
    return {"seed": seed, "scale": scale, "homeo": homeo, "eta": 0.0, "rule": rule, "seconds": 60.0 * minutes,
            "gain_60s": g.tolist(), "rate_ratio": rate}


def test_pathway_reading_needs_full_length_arms():
    """Under 3 minutes the SE is NaN and nothing can count as moved; the reading must not pass on that."""
    def arms(minutes, shift):
        return [drift_row(s, x, h, minutes, shift if x else 0.0)
                for s in (0, 1, 2) for x, h in ((0.0, 0.0), (1.0, 1e-5))]

    assert CP.pathways_vs_off(arms(10, 0.0))["reading"]
    moved = CP.pathways_vs_off(arms(10, 5.0))
    assert moved["moved_at_eta"] == 3 and not moved["reading"]
    short = CP.pathways_vs_off(arms(2, 5.0))
    assert short["moved_at_eta"] is None and not short["reading"]
    assert not short["per_culture"]["0"]["x1_homeo1e-05"]["tested"]
    assert CP.pathways_vs_off([])["moved_at_eta"] is None


def test_gate_b1_needs_full_length_arms_and_additive_stdp():
    def rows(minutes, rule="additive"):
        return [drift_row(s, x, h, minutes, rate=r, rule=rule if x else "additive")
                for s in (0, 1, 2) for x, h, r in ((0.0, 0.0, 1.0), (0.0, 1e-5, 1.0), (1.0, 1e-5, 1.2), (1.0, 0.0, 5.0))]

    full = CP.drift_summary(rows(10), "additive")
    assert full["gate_B1"]["tested"] and full["gate_B1"]["pass"]
    short = CP.gate_b1(rows(2))
    assert not short["tested"] and not short["pass"]
    other = CP.drift_summary(rows(10, "multiplicative"), "multiplicative")
    assert "gate_B1" not in other and other["stdp_rule"] == "multiplicative"
