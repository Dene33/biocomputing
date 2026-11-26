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
                "src/biocompute/mea/electrodes.py": "integer spike counts, exact in any order",
                "scripts/takeover_rerun.py": "numpy on the CPU (bincount with weights): sums in a fixed order"}
    pat = re.compile(r"index_add_|scatter_add|scatter_reduce|index_reduce|accumulate=True|bincount\([^)]*weights")
    root = SCRIPTS.parent
    found = {str(p.relative_to(root)) for p in [*root.glob("src/**/*.py"), *SCRIPTS.glob("*.py")]
             if pat.search(p.read_text())}
    assert found <= set(reviewed), found - set(reviewed)


def test_a2c_refit_edit_recovers_the_channel_and_scores_the_rest():
    """A2c keeps the decoder and refits only its affine edit: the fit must recover decoded = a + g * true around
    standing, and the error it leaves on the unseen steps must be ~0 for a channel that doesn't change."""
    from biocompute.eval.dish_channel import IMU_SPREAD

    cls = load("closed_loop_sessions")
    rng = np.random.default_rng(0)
    stand = np.array([0.0, 0.002, 0.0, -0.063, -0.005, -0.998])
    T = stand + rng.normal(0, 1, (3000, 6)) * np.r_[IMU_SPREAD[:5], 0]
    g, a = np.array([0.7, 0.8, 0.9, 0.6, 0.75]), np.array([-0.03, 0.01, 0.09, -0.03, 0.0])
    P = T.copy()
    P[:, :5] = a + g * T[:, :5] + rng.normal(0, 0.2, (3000, 5)) * IMU_SPREAD[:5]
    e = cls.refit_edit(T, P, stand, 500)
    assert e["steps"] == 500 and 0 < e["in_range"] <= 500
    assert np.allclose(e["gain"], g, atol=0.05) and np.allclose(e["intercept"], a, atol=0.01)
    assert e["rest_in_range"] > 2000 and np.abs(e["rest_bias"]).max() < 0.1
    assert "rest_bias" not in cls.refit_edit(T, P, stand, 3000)  # nothing left to score on
    P[:, 0] = 0.01 * T[:, 0]  # a dead channel: no usable edit
    assert cls.refit_edit(T, P, stand, 500)["gain"] is None


def test_a2c_reproduction_check_is_exact():
    import torch

    cls = load("closed_loop_sessions")
    x = {"w": torch.arange(4.0), "a": np.array([1.0, np.nan]), "q": [(np.arange(2), 3)], "t": 1.5}
    y = {"w": torch.arange(4.0), "a": np.array([1.0, np.nan]), "q": [(np.arange(2), 3)], "t": 1.5}
    assert cls.same(x, y)
    y["a"] = np.array([np.nextafter(1.0, 2.0), np.nan])
    assert not cls.same(x, y)
    assert not cls.same(x, x | {"w": torch.arange(4.0).double()})  # dtype counts
    assert not cls.same(x, {k: v for k, v in x.items() if k != "t"})


def test_gate_a2c_needs_every_condition_day0_included():
    cls = load("closed_loop_sessions")
    seeds = list(range(40))

    def tab(full_by_cond):
        by, fulls = {}, {}
        for cond, k in full_by_cond.items():
            for arm in ("refit60", "recal"):
                f = {s: s < (k if arm == "refit60" else 36) for s in seeds}
                fulls[("s32", cond, arm)] = f
                by[("s32", cond, arm)] = {"full": sum(f.values()), "n": len(f)}
        return by, fulls

    ok = {c: 35 for c in cls.CONDITIONS}
    assert cls.gate_a2c(*tab(ok), 40)["pass"]
    assert not cls.gate_a2c(*tab(ok | {"day0": 33}), 40)["pass"]
    assert not cls.gate_a2c(*tab({c: 40 for c in cls.AGED}), 40)["pass"]  # day 0 missing: not complete
    g = cls.gate_a2c(*tab(ok | {"newdish": 38}), 40)
    assert g["per_condition"]["newdish"]["only_refit60_full"] == [36, 37]
    assert g["per_condition"]["newdish"]["only_recal_full"] == []


def _takeover_dump(rng, radial=True, n=3000):
    """A made-up culture on a 16 x 16 grid at 100 um: E->E weights grow along the direction away from electrode
    136 (row 8, column 8) when ``radial``; two inputs whose pulses evoke a wave from their own electrode."""
    gy, gx = np.mgrid[0:16, 0:16]
    exy = np.c_[gx.ravel() * 100.0 - 750, gy.ravel() * 100.0 - 750].astype(np.float32)
    pos = np.c_[rng.uniform(-800, 800, (n, 2)), np.zeros(n)].astype(np.float32)
    pre, post = rng.integers(0, n, 40 * n), rng.integers(0, n, 40 * n)
    keep = (pre != post) & (np.linalg.norm(pos[pre, :2] - pos[post, :2], axis=1) < 300)
    pre, post = pre[keep].astype(np.int32), post[keep].astype(np.int32)
    v, r = pos[post, :2] - pos[pre, :2], pos[pre, :2] - exy[136]
    cos = (v * r).sum(1) / np.maximum(np.linalg.norm(v, axis=1) * np.linalg.norm(r, axis=1), 1e-9)
    w0 = np.full(len(pre), 1.0, np.float32)
    w1 = w0 + (0.5 * cos if radial else 0) + rng.normal(0, 0.05, len(pre)).astype(np.float32)
    pt, pc, st, sc = [], [], [], []
    t = 0.0
    while t < 4000.0:
        e = 136 if rng.random() < 0.5 else 155
        pt.append(t), pc.append(e)
        d = np.linalg.norm(exy - exy[e], axis=1)
        for ch in np.flatnonzero(rng.random(256) < 0.9):
            st.append(t + 1 + d[ch] / 50.0), sc.append(ch)  # a wave at 50 um/ms from the pulsed electrode
        for ch in np.flatnonzero(d < 300):  # and a few extra spikes near it
            st.extend([t + 2 + d[ch] / 50.0, t + 3 + d[ch] / 50.0]), sc.extend([ch, ch])
        t += 60.0
    return {"syn_exc": np.ones(len(pre), bool), "excitatory": np.ones(n, bool), "pre": pre, "post": post, "pos": pos,
            "w_start": w0, "w_end": w1.astype(np.float32), "electrode_xy": exy, "spike_t_ms": np.array(st),
            "spike_ch": np.array(sc, np.int16), "pulse_t_ms": np.array(pt), "pulse_ch": np.array(pc, np.int16),
            "t_start_ms": np.array(0.0), "t_end_ms": np.array(4000.0)}


def test_takeover_reading_finds_a_planted_direction_and_only_there():
    TR = load("takeover_rerun")
    rng = np.random.default_rng(1)
    d = _takeover_dump(rng)
    res, maps = TR.analyse_arm(d, 136, 155, minute_s=1.0)
    assert res["winner"]["z"] > 10 and res["winner"]["score"] > res["null"]["p95"]
    assert res["winner"]["score"] > res["loser"]["score"]
    flat, _ = TR.analyse_arm(_takeover_dump(np.random.default_rng(2), radial=False), 136, 155, minute_s=1.0)
    assert abs(flat["winner"]["z"]) < 3
    # The latency field of the winner's waves runs outward, as the planted weights do.
    c, n_el = TR.field_cosine(maps["latency"][0, 0], maps["weights"], 100.0)
    assert n_el > 100 and c > 0.5
    # Inputs at different electrodes: decodable from every electrode and in most of the area.
    assert min(res["decode_all"]) > 0.9 and min(res["local_area"]) > 0.5
    rd = TR.reading({"takeover_0": res, "control_0": flat})
    assert rd["per_culture"]["0"]["confirmed"] and not rd["confirmed"]  # cultures 1 and 2 missing
    # After-run maps: both scores peak at the planted centre and fall with distance from it; nothing in the flat one.
    for k in ("plain", "within"):
        m = res["after_run"][k]
        assert m["rank_of_winner"] <= 3 and m["peak_dist_to_winner_um"] <= 150 and m["corr_with_dist_to_winner"] < -0.5
    assert res["after_run"]["within"]["winner"] > 10 * res["after_run"]["within"]["winner_se"]
    assert abs(flat["after_run"]["within"]["winner"]) < 3 * flat["after_run"]["within"]["winner_se"]


def test_burst_spread_recovers_planted_waves_on_wagenaars_layout():
    BS = load("burst_spread")
    xy = BS.wagenaar_xy()
    assert np.isnan(xy[14]).all() and np.isfinite(np.delete(xy, 14, 0)).all()  # 59 electrodes, 14 the reference
    assert tuple(xy[23]) == (200.0, 0.0) and tuple(xy[53]) == (1200.0, 1400.0)
    ok = np.flatnonzero(np.isfinite(xy).all(1))
    P, centre, rng = xy[ok], xy[ok].mean(0), np.random.default_rng(0)
    noise = rng.normal(0, 1.0, len(ok))
    plane = BS.classify(P, P[:, 0] / 50.0 + noise, centre, rng)  # 50 mm/s along x
    assert plane["class"] == "plane" and abs(plane["speed_plane_mm_s"] - 50) < 5
    src = BS.classify(P, np.linalg.norm(P - P[7], axis=1) / 100.0 + noise, centre, rng)  # 100 mm/s from electrode
    assert src["class"] == "source" and abs(src["speed_source_mm_s"] - 100) < 15
    assert np.hypot(*(np.array(src["site"]) - P[7])) <= 200
    shuffled = [BS.classify(P, rng.permutation(P[:, 0] / 50.0), centre, rng)["class"] for _ in range(20)]
    assert shuffled.count("complex") >= 15


def test_burst_truth_v2_finds_the_wave_start_despite_early_background_spikes():
    BS = load("burst_spread")
    rng = np.random.default_rng(0)
    n, dt_ms = 20000, 0.5
    pos = np.c_[rng.uniform(-2000, 2000, (n, 2)), np.zeros(n)].astype(np.float32)
    start = np.array([300.0, -200.0])
    # One burst at 1.0 s: every neuron spikes twice as a wave from ``start`` passes (100 um/ms), plus background.
    d = np.linalg.norm(pos[:, :2] - start, axis=1)
    tw = 1.0 + d / 100.0 / 1000.0
    ts = np.concatenate([tw + rng.normal(0, 0.002, n), tw + 0.005 + rng.normal(0, 0.002, n),
                         rng.uniform(0.85, 1.1, n // 4)])
    ids = np.concatenate([np.arange(n), np.arange(n), rng.integers(0, n, n // 4)])
    o = np.argsort(ts, kind="stable")
    z = {"dt_ms": np.array(dt_ms), "n_step": np.round(ts[o] * 1000 / dt_ms).astype(np.int64), "n_id": ids[o],
         "pos": pos}
    gy, gx = np.mgrid[0:8, 0:8]
    exy = np.c_[gx.ravel() * 200.0 - 700, gy.ravel() * 200.0 - 700]
    (tr,) = BS.truth_v2(z, [{"start_s": 1.0, "end_s": 1.05}], exy)
    assert tr["inside"] and np.hypot(*(np.array(tr["site"]) - start)) <= 150
    assert tr["class"] == "source" and 70 <= tr["speed_source_mm_s"] <= 140


def test_burst_spread_per_culture_skips_recordings_without_bursts():
    BS = load("burst_spread")
    full = {"n": 4, "class_share": {"source": 0.5, "plane": 0.25, "rotation": 0.25, "complex": 0.0},
            "speed_source_mm_s_median": 40.0, "top_start_share": 0.25, "starts_for_80pct": 3}
    per = {"a": {"culture": "1-1", "summary": full}, "b": {"culture": "1-1", "summary": {"n": 0}},
           "c": {"culture": "2-1", "summary": {"n": 0}}}
    pc = BS.per_culture(per)
    assert list(pc) == ["1-1"] and pc["1-1"]["speed"] == 40.0 and pc["1-1"]["source"] == 0.5


def test_burst_onsets_from_burstlets_and_from_smoothed_rates():
    BS = load("burst_spread")
    # Three electrodes join a burst at 0, 20 and 40 ms after its start (1.0 s), each with 6 spikes 5 ms apart.
    t = np.concatenate([1.0 + d + np.arange(6) * 0.005 for d in (0.0, 0.02, 0.04)])
    ch = np.repeat([0, 1, 2], 6)
    burst = (1.0, 1.065, {0, 1, 2})
    bl = [(1.0 + d, 1.025 + d, c) for c, d in zip((0, 1, 2), (0.0, 0.02, 0.04))]
    on0 = BS.onsets(t, ch, burst, bl, 0)
    assert np.allclose([on0[c] for c in (0, 1, 2)], [0, 20, 40])
    on5 = BS.onsets(t, ch, burst, bl, 5)
    assert np.allclose(np.diff([on5[c] for c in (0, 1, 2)]), [20, 20], atol=1.01)


def weak_row(seed, scale, rate=1.0, shift=0.0, minutes=10, cross=0.5, homeo=0.0, rule="additive"):
    """A drift arm as weak_synapses.py reads it: per-minute rate against rest and per-minute gains (own 2, cross
    ``cross``, noise shared by one culture's arms), ``shift`` added to one cross gain over the last 3 minutes."""
    r = drift_row(seed, scale, homeo, minutes, shift, rule=rule)
    g = np.array(r["gain_60s"]) + (cross - 0.5) * (1 - np.eye(2))
    return r | {"gain_60s": g.tolist(), "rate_ratio_60s": [1.05] * (minutes - 3) + [1.05 * rate] * 3}


def test_weak_synapse_variants_scale_count_weights_and_cap_together():
    WS = load("weak_synapses")
    o = WS.overrides("s400")
    assert o["synapses_per_neuron"] == 400.0 and o["length_um"] == 225.0 and o["delay_buckets"] == 16
    assert np.isclose(o["w_exc"] * 4, 2.2) and np.isclose(o["w_inh"] * 4, -6.0) and np.isclose(o["w_max"] * 4, 10.0)
    assert set(WS.overrides("s190")) == {"synapses_per_neuron", "w_exc", "w_inh", "w_max"}  # reach as now


def test_weak_synapse_gate_needs_calm_kept_and_separate_cultures():
    WS = load("weak_synapses")
    ok = [weak_row(s, x) for s in range(5) for x in (0.0, 1.0)]
    g = WS.drift_gate(ok)
    assert g["passed"] == 5 and g["pass"]
    hot = [weak_row(s, x, rate=3.0 if x and s < 3 else 1.0) for s in range(5) for x in (0.0, 1.0)]
    g = WS.drift_gate(hot)  # three cultures run hot: 2 of 5 left
    assert g["passed"] == 2 and not g["pass"] and g["per_culture"]["0"]["calm_ratio_last3"] == 3.0
    moved = [weak_row(s, x, shift=5.0 if x and s < 3 else 0.0) for s in range(5) for x in (0.0, 1.0)]
    g = WS.drift_gate(moved)
    assert g["passed"] == 2 and not g["per_culture"]["0"]["pathways_kept"]
    mixed = [weak_row(s, x, cross=1.5) for s in range(5) for x in (0.0, 1.0)]
    assert WS.drift_gate(mixed)["passed"] == 0  # no separate pathway to keep without plasticity
    missing = [r for r in ok if not (r["seed"] in (0, 1, 2) and r["scale"])]  # no encoders: arm absent
    assert WS.drift_gate(missing)["passed"] == 2
    short = [weak_row(s, x, minutes=2) for s in range(5) for x in (0.0, 1.0)]
    assert WS.drift_gate(short)["passed"] == 0
    # Only the no-plasticity arm and STDP x1 with the 5 s step count; scaling and weight-dependent arms are ignored.
    other = ok + [weak_row(s, 1.0, rate=5.0, homeo=1e-5) for s in range(5)] + [
        weak_row(s, 1.0, rate=5.0, rule="multiplicative") for s in range(5)]
    assert WS.drift_gate(other)["passed"] == 5


def test_scorecard_jobs_keep_their_names_unless_the_culture_changes():
    RS = load("realism_scorecard")
    sz = {"warm_s": 10.0, "record_s": 600.0, "evoked": True, "evoked_reps": 25, "smoke": False}
    assert RS.job_name("live", 0, sz) == "live_s0_ed5a2d31"  # the stored scorecard's recordings
    assert RS.job_name("live", 0, sz | {"tag": "_s190"}) == "live_s0_ed5a2d31"
    assert RS.job_name("live", 0, sz | {"culture": {"w_exc": 1.1}}) != "live_s0_ed5a2d31"


def test_configure_returns_none_without_two_eligible_pairs():
    k = 6
    first = np.full((k, k), 0.5)
    np.fill_diagonal(first, 0.0)
    cand, rate = np.arange(10, 10 + k), np.arange(k, 0, -1.0)
    assert CP.configure(cand, rate, first, np.ones(k)) is None  # every encoder bursts on all its pulses
    cfg = CP.configure(cand, rate, first, np.zeros(k))
    assert len(set(cfg["enc"] + cfg["dec"])) == 4


DD = load("day_drift")


def test_finalspark_column_names_are_read_as_text():
    """pandas keeps a values block's column names as a protocol-0 pickled list; they are parsed, never unpickled."""
    assert load("finalspark_data")._names(b"(lp0\nVa1\np1\naVa2\np2\naVd1\np3\na.") == ["a1", "a2", "d1"]
    assert load("finalspark_data")._names(b"(lp0\nVmax_voltage_uv\np1\na.") == ["max_voltage_uv"]


def test_day_rates_count_only_usable_time_and_clean_events():
    """Two days; electrode 0 fires 1 Hz on day 0 and 2 Hz on day 1. Day 1 holds a stimulation (excluded 10 min
    before to 60 min after) and artifact-sized events; neither enters its rate."""
    S = DD.S
    t0 = 1_000_000 * S
    day0 = t0 + np.arange(0, 86400, 1.0) * S
    day1 = t0 + 86400 * S + np.arange(0, 86400, 0.5) * S
    stim = t0 + 86400 * S + 3600 * S * 12
    art = stim + np.arange(5) * S // 1000  # five 2 mV events right after the stimulation
    t = np.r_[day0, day1, art].astype(np.int64)
    amp = np.r_[np.full(len(day0) + len(day1), -40.0), np.full(5, -2000.0)]
    o = np.argsort(t, kind="stable")
    dr = DD.day_rates(t[o], np.zeros(len(t), np.int64), amp[o], np.array([stim], np.int64))
    assert dr["n_days"] == 2 and dr["valid"].all()
    assert dr["use_s"][0] == 86400 and dr["use_s"][1] == 86400 - 71 * 60  # 10 + 60 min and the stimulation's minute
    assert abs(dr["rate"][0, 0] - 1.0) < 1e-9 and abs(dr["rate"][1, 0] - 2.0) < 1e-3


def test_lag_changes_and_the_reading():
    rate = np.zeros((4, 32))
    rate[:, 0] = [1.0, 2.0, 2.0, 0.5]  # electrode 0: x2, x1, x0.25 day to day
    rate[:, 8:16] = 0.05  # a silent group is left out
    ch = DD.lag_changes(rate, np.ones(4, bool), (1,))
    assert ch["1"]["electrode"]["n"] == 3 and ch["1"]["electrode"]["median"] == 1.0
    assert ch["1"]["electrode"]["signed_median"] == 0.0
    assert ch["1"]["group"]["n"] == 3  # group 0 only (mean 1/8 Hz and up), not the silent one
    real = {"spont": {L: {"electrode": {"n": 50, "p25": 0.4, "median": 0.8, "p75": 1.2},
                          "group": {"n": 50, "p25": 0.4, "median": 0.8, "p75": 1.2}} for L in ("1", "3", "7")},
            "evoked": {"classes": {c: {"electrode": {"n": 30, "p25": 0.3, "median": 0.5, "p75": 0.9},
                                       "group": {"n": 3, "p25": 0.3, "median": 0.5, "p75": 0.9}}
                                   for c in ("1d", "7d")}}}
    gentle = {"n": 20, "median": 0.1}
    model = {"spont": {L: {"electrode": gentle, "group": gentle} for L in ("1", "3", "7")},
             "evoked": {d: {"electrode": {"n": 20, "median": 0.5}, "group": gentle} for d in ("1", "7")}}
    c = DD.compare(real, model)
    assert c["verdicts"]["spont_electrode"] == "too gentle" and c["verdicts"]["evoked_electrode"] == "in range"
    assert c["decision"].startswith("part 2")
    for L in ("1", "3", "7"):
        model["spont"][L] = {"electrode": {"n": 20, "median": 0.8}, "group": {"n": 5, "median": 0.8}}
    model["spont"]["7"]["electrode"] = gentle  # too gentle at one lag of three: still in range
    c = DD.compare(real, model)
    assert c["verdicts"]["spont_electrode"] == "in range" and c["decision"].startswith("aging.py")


def test_evoked_changes_use_shared_settings_and_lag_classes():
    S = DD.S
    resp = lambda x: {"all": x, "odd": x, "even": x}  # noqa: E731
    base = np.zeros(32)
    base[:4] = 2.0
    sw = [{"t_first": 0, "delivered": True, "settings": {(k,): resp(base) for k in range(12)}},
          {"t_first": 86400 * S, "delivered": True, "settings": {(k,): resp(base * 0.5) for k in range(12)}},
          {"t_first": 7 * 86400 * S, "delivered": False, "settings": {(k,): resp(base) for k in range(12)}}]
    ev = DD.evoked_changes(sw, 0)
    assert len(ev["pairs"]) == 1 and ev["pairs"][0]["lag_days"] == 1.0  # the undelivered sweep is left out
    assert ev["classes"]["1d"]["electrode"]["median"] == 1.0 and ev["classes"]["7d"]["electrode"]["n"] == 0
    assert ev["classes"]["1d"]["group"]["n"] == 1 and ev["noise_floor_electrode"]["median"] == 0.0


def test_day_drift_doses_interpolate_and_flag_unreachable_targets():
    pt = lambda x, y: {"shift": x, "evoked_electrode_abs": y, "evoked_electrode_signed": -y,  # noqa: E731
                       "evoked_culture": -y, "spont_culture": -2 * y}
    pts = [pt(0.0, 0.1), pt(-1.0, 0.3), pt(-2.0, 0.7), pt(1.0, 0.2), pt(2.0, 0.25)]
    d = DD.dose_for(pts, -1, 0.5)
    assert d["reached"] and abs(d["shift"] + 1.5) < 1e-9 and abs(d["spont_culture"] + 1.0) < 1e-9
    up = DD.dose_for(pts, 1, 0.5)
    assert not up["reached"] and up["shift"] == 2.0  # the largest tried, flagged
    assert DD.dose_for(pts, -1, 0.05)["shift"] == 0.0  # already there without a shift
    assert DD.dose_for([], 1, 0.5) == {"shift": None, "reached": False}  # a lag without sessions