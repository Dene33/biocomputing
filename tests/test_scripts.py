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
        arms = ((0.0, 0.0, 1.0), (0.0, 1e-5, 1.0), (1.0, 1e-5, 1.2), (1.0, 0.0, 5.0))  # scale, scaling eta, rate
        return [drift_row(s, x, h, minutes, rate=r, rule=rule if x else "additive")
                for s in (0, 1, 2) for x, h, r in arms]

    full = CP.drift_summary(rows(10), "additive")
    assert full["gate_B1"]["tested"] and full["gate_B1"]["pass"]
    short = CP.gate_b1(rows(2))
    assert not short["tested"] and not short["pass"]
    other = CP.drift_summary(rows(10, "multiplicative"), "multiplicative")
    assert "gate_B1" not in other and other["stdp_rule"] == "multiplicative"


RD = load("real_drift")


def test_ratios_against_a_silent_or_negative_baseline_are_nan():
    """Preferred gains of -0.05 over the first 3 minutes gave detectable_vs_pref ~1e9 (divided by 1e-9)."""
    g = np.full((RD.MIN_MINUTES, 2, 2), 0.3)
    g[:3] = [[-0.1, -0.05], [-0.1, -0.05]]
    w = RD.window_stats(g, 0.0, np.ones(RD.MIN_MINUTES))
    assert np.isnan(w["detectable_vs_pref"]) and np.isnan(w["rate_ratio_first30"]) and w["rate_ratio_first3"] == 1.0


def null_run(start, rate):
    return {"mixed_encoders": False, "encoders": [1, 2], "start_time": start, "exp_type": "none", "train_pulses": 0,
            "gain_run": np.eye(2), "rate_hz": rate, "end_offset_s": 900.0}


def test_hours_leave_out_a_session_whose_first_null_run_is_silent():
    """Its ratios were all inf -> None, so a summary of the JSON counted a session with no numbers."""
    times = ["2023-11-24T10:00:00", "2023-11-24T12:00:00", "2023-11-24T14:00:00"]
    hours, _, silent = RD.sessions({"live": [null_run(t, r) for t, r in zip(times, (2.0, 3.0, 1.0))],
                                    "silent": [null_run(t, r) for t, r in zip(times, (0.0, 3.0, 1.0))]})
    assert [h["exp"] for h in hours] == ["live"] and silent == ["silent"]
    s = RD.hours_summary(hours, silent)
    assert s["sessions"] == 1 and s["span_h"] == [4.0, 4.0]
    assert s["rate_vs_first"]["median"] == 1.0 and s["rate_vs_first_max"] == 1.5  # later runs: 1.5x and 0.5x
    assert s["sessions_in_band_every_run"] == 0  # 0.5x is outside 2/3-1.5x


def test_time_gaps_do_not_depend_on_the_machine_zone(monkeypatch):
    """The logs' times carry no zone; read in Berlin's, a gap across its DST switch came out an hour short."""
    import time

    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    try:
        assert RD.hours_apart("2024-03-31T01:30:00", "2024-03-31T03:30:00") == 2.0
    finally:
        monkeypatch.undo()
        time.tzset()


def test_closed_loop_files_are_keyed_by_their_settings():
    """A1 and A2 named episodes by seed alone, so a rerun with another --edit or --seconds reused the old ones."""
    clv, cls, o = load("closed_loop_vestibular"), load("closed_loop_sessions"), Path("out")
    sz = {"data": "d.pkl", "replay_steps": 0, "seconds": 20.0, "edit": "gain"}
    f = clv.episode_file(o, "s12", "culture_edit", 9000, sz)
    for k, v in (("edit", "affine"), ("seconds", 10.0), ("data", "e.pkl"), ("replay_steps", 300)):
        assert clv.episode_file(o, "s12", "culture_edit", 9000, sz | {k: v}) != f
    assert clv.setup_stem("s12", "live", sz | {"edit": "affine", "seconds": 1.0}) == clv.setup_stem("s12", "live", sz)
    s2 = {"data": "d.pkl", "a1": "a1", "recal_steps": 3000, "seconds": 20.0}
    assert cls.a1_setup(s2, "s32") == Path("a1") / "setups" / f"{clv.setup_stem('s32', 'live', sz)}.pkl"
    e = cls.episode_file(o, "s32", "day1", "stale", 9200, s2)
    for k, v in (("a1", "b"), ("recal_steps", 300), ("seconds", 10.0), ("data", "e.pkl")):
        assert cls.episode_file(o, "s32", "day1", "stale", 9200, s2 | {k: v}) != e


def test_no_atomic_float_sums_outside_the_reviewed_places():
    """On the GPU, index_add_ and its kin add with atomics in a varying order, so a run that sums floats with them
    doesn't repeat (the 5 s total-input step did this). Sums go through ordered ops (``excitatory_input``, the
    delivery matrix); a new use must be reviewed and listed here with its reason."""
    import re

    reviewed = {"src/biocompute/culture/network.py": "the CPU branch of substep (no atomics on the CPU)",
                "src/biocompute/mea/electrodes.py": "integer spike counts, exact in any order"}
    pat = re.compile(r"index_add_|scatter_add|scatter_reduce|index_reduce|accumulate=True|bincount\([^)]*weights")
    root = SCRIPTS.parent
    found = {str(p.relative_to(root)) for p in [*root.glob("src/**/*.py"), *SCRIPTS.glob("*.py")]
             if pat.search(p.read_text())}
    assert found <= set(reviewed), found - set(reviewed)
