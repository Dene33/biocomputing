import numpy as np
import torch

from biocompute.culture.network import OrganoidCulture, make_config
from biocompute.device import SimulatedBiocomputer
from biocompute.mea.electrodes import make_mea_config


def izhikevich_single(a, b, c, d, current, ms=500.0, dt=0.5):
    v, u, spikes = -65.0, b * -65.0, []
    for s in range(int(ms / dt)):
        for _ in range(2):
            v += dt / 2 * (0.04 * v * v + 5 * v + 140 - u + current)
        u += dt * a * (b * v - u)
        if v >= 30:
            spikes.append(s * dt)
            v, u = c, u + d
    return np.array(spikes)


def test_neuron_types_fire_as_izhikevich_describes():
    rs = izhikevich_single(0.02, 0.2, -65, 8, 10)
    fs = izhikevich_single(0.1, 0.2, -65, 2, 10)
    ch = izhikevich_single(0.02, 0.2, -50, 2, 10)
    assert len(fs) > 2 * len(rs)
    isi_rs = np.diff(rs)
    assert isi_rs[-1] > isi_rs[0]
    assert np.diff(ch).min() < 10 and np.diff(ch).max() > 2 * np.diff(ch).min()


def small_culture(**kw):
    return OrganoidCulture(make_config("cpu_2k", n_neurons=400, synapses_per_neuron=30, radius_um=500, **kw))


def test_same_seed_same_spikes():
    a, b = small_culture(seed=3), small_culture(seed=3)
    ra, rb = a.run(200.0), b.run(200.0)
    assert torch.equal(ra, rb)
    c = small_culture(seed=4)
    assert not torch.equal(ra, c.run(200.0))


def test_short_term_depression_depletes_and_recovers():
    c = small_culture()
    spikes = torch.zeros(c.n, dtype=torch.bool)
    spikes[0] = True
    for _ in range(5):
        c.stp_x = c.stp_x - c.cfg.stp_u * c.stp_x * spikes
    depleted = c.stp_x[0].item()
    assert depleted < 0.4
    for _ in range(int(3 * c.cfg.stp_tau_rec_ms / c.cfg.dt_ms)):
        c.substep()
    assert c.stp_x[0].item() > depleted


def test_stimulation_reaches_only_nearby_neurons():
    dev = SimulatedBiocomputer(
        make_config("cpu_2k", n_neurons=1000, radius_um=800, background_rate_hz=0.0, noise_std=0.0),
        make_mea_config("mcs60"),
        keep_raster=True,
    )
    dev.culture.w.zero_()
    ch = 27
    dev.stimulate([ch], [0.0], [1.0])
    dev.step(10.0)
    fired = dev.last_raster.any(0).cpu()
    dist = (dev.culture.pos[:, :2] - torch.tensor(dev.mea.xy[ch])).norm(dim=1)
    assert fired.sum() > 0
    assert dist[fired].max() < 250.0


def test_recording_is_blanked_after_a_pulse():
    dev = SimulatedBiocomputer(make_config("cpu_2k", n_neurons=600, radius_um=800), make_mea_config("mcs60"))
    dev.mea.cfg.false_positive_hz = 5000.0
    dev.stimulate(np.arange(10), np.zeros(10), np.full(10, 0.5))
    block = dev.step(10.0)
    assert len(block.times_ms) > 0
    assert block.times_ms.min() >= dev.mea.cfg.blank_ms - 1e-9


def test_rate_limit_drops_excess_pulses():
    dev = SimulatedBiocomputer(make_config("cpu_2k", n_neurons=300, radius_um=800), make_mea_config("mcs60"))
    dev.stimulate([5] * 4, [0.0, 1.0, 2.0, 5.0], [0.5] * 4)
    dev.step(10.0)
    assert dev.pulses_delivered == 2 and dev.pulses_dropped == 2


def test_gpu_sparse_delivery_matches_per_synapse_reference():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no GPU")
    c = OrganoidCulture(make_config("cpu_2k", n_neurons=1500, device="cuda", wells=2, seed=5))
    c.run(50.0)
    c.history = torch.rand_like(c.history) * (torch.rand_like(c.history) < 0.1)
    k = c.cfg.delay_buckets + 1
    released = c.history[(c.head - c.delay + 1) % k, c.pre]
    ref_e = torch.zeros(c.n, device="cuda").index_add_(0, c.post[c.syn_exc], (c.w * released)[c.syn_exc])
    ref_i = torch.zeros(c.n, device="cuda").index_add_(0, c.post[~c.syn_exc], (c.w * released)[~c.syn_exc])
    slots = [(c.head - d + 1) % k for d in range(1, k)]
    out = (c._delivery_matrix() @ c.history[slots].reshape(-1, 1)).squeeze(1)
    assert torch.allclose(out[: c.n], ref_e, atol=1e-3) and torch.allclose(out[c.n :], ref_i, atol=1e-3)
    c.w.mul_(0.5)
    out2 = (c._delivery_matrix() @ c.history[slots].reshape(-1, 1)).squeeze(1)
    assert torch.allclose(out2[: c.n], ref_e * 0.5, atol=1e-3)
    assert not ((c.pre < c.n_per_well) ^ (c.post < c.n_per_well)).any()


def test_saved_state_resumes_spike_for_spike():
    from biocompute.culture.plasticity import STDP

    def make():
        dev = SimulatedBiocomputer(make_config("cpu_2k", n_neurons=400, synapses_per_neuron=30, radius_um=500, seed=5),
                                   make_mea_config("mcs60", seed=5))
        dev.culture.plasticity = STDP(istdp_eta=0.01, w_max_inh=30.0)
        return dev

    a = make()
    a.stimulate([10, 20], [5.0, 50.0], [1.0, 1.0])
    a.step(300.0)
    st, pst = a.state(), a.culture.plasticity.state()
    a.stimulate([10], [7.0], [1.0])
    ref = a.step(300.0)
    b = make()
    b.load_state(st)
    b.culture.plasticity.load_state(b.culture, pst)
    b.stimulate([10], [7.0], [1.0])
    out = b.step(300.0)
    assert np.array_equal(ref.channels, out.channels) and np.array_equal(ref.times_ms, out.times_ms)
    assert torch.equal(a.culture.w, b.culture.w)


def test_inhibitory_plasticity_adds_inhibition_to_neurons_above_target():
    from biocompute.culture.plasticity import STDP

    c = small_culture(seed=6)
    c.plasticity = STDP(a_plus=0.0, a_minus=0.0, istdp_eta=0.01, target_hz=0.1, w_max_inh=30.0)
    raster = c.run(2000.0)
    rate = raster.float().sum(0) / 2.0
    inh = ~c.syn_exc
    dw = (c.w - c.w0)[inh]
    target_rate = rate[c.post[inh]]
    busy = target_rate > 5 * 0.1
    assert busy.any() and dw[busy].mean() > 0
    assert torch.equal(c.w[c.syn_exc], c.w0[c.syn_exc]) or (c.w - c.w0)[c.syn_exc].abs().max() < 1e-3


def _plastic_device(device="cpu", **stdp):
    from biocompute.culture.plasticity import STDP

    dev = SimulatedBiocomputer(make_config("cpu_2k", n_neurons=400, synapses_per_neuron=30, radius_um=500, seed=5,
                                           device=device), make_mea_config("mcs60", seed=5))
    dev.culture.plasticity = STDP(**stdp)
    return dev


def test_sparse_stdp_matches_dense_bit_for_bit():
    for kw in ({"istdp_eta": 0.01, "w_max_inh": 30.0}, {"multiplicative": True}, {}):
        runs = []
        for sparse in (False, True):
            dev = _plastic_device(sparse=sparse, tau_forget_s=float("inf"), **kw)
            ch, n = [], 0.0
            for _ in range(20):
                dev.stimulate(list(range(10, 60, 5)), [0.0] * 10, [1.0] * 10)
                b = dev.step(20.0)
                ch.append(b.channels)
                n += dev.oracle_counts
            runs.append((np.concatenate(ch), None, n, dev.culture.w.clone()))
        (c0, _, n0, w0), (c1, _, n1, w1) = runs
        assert n0.sum() > 100 and (w0 != dev.culture.w0).sum() > 100, kw
        assert np.array_equal(c0, c1) and np.array_equal(n0, n1), kw
        assert torch.equal(w0, w1), kw


def test_batched_forgetting_matches_decay_every_substep():
    ws = []
    for sparse in (False, True):
        dev = _plastic_device(sparse=sparse, a_plus=0.0, a_minus=0.0, tau_forget_s=1.0)
        c = dev.culture
        c.w.copy_((c.w0 * 0.5).clamp(max=c.cfg.w_max))
        dev.step(500.0)
        ws.append(c.w.clone())
    assert torch.allclose(ws[0], ws[1], rtol=1e-5, atol=1e-6)


def test_gpu_delivery_matrix_patched_in_place_equals_rebuild():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    dev = _plastic_device(device="cuda", istdp_eta=0.01, w_max_inh=30.0)
    dev.stimulate([10, 20], [5.0, 50.0], [1.0, 1.0])
    dev.step(200.0)
    c = dev.culture
    patched = c._delivery_matrix().values().clone()
    c._A_cache = (None, None, c._A_cache[2])
    assert torch.equal(patched, c._delivery_matrix().values())


def test_guard_catches_weights_bounded_below_their_cap(tmp_path):
    import pytest

    from biocompute.culture.plasticity import STDP
    from biocompute.runguard import Guard, GuardFail

    c = small_culture(seed=6)
    stdp = STDP(istdp_eta=0.01, w_max_inh=30.0)
    g = Guard("t", tmp_path)
    inh = (~c.syn_exc).nonzero(as_tuple=True)[0]
    c.w[inh] = torch.minimum(c.w0[inh] * 3.0, torch.full_like(c.w0[inh], 30.0))
    g.culture(c, stdp, rate_hz=5.0)
    c.w[inh] = torch.minimum(c.w0[inh] * 3.0, torch.full_like(c.w0[inh], c.cfg.w_max))
    with pytest.raises(GuardFail):
        g.culture(c, stdp)
    c.w[inh] = c.w0[inh]
    c.v[0] = float("nan")
    with pytest.raises(GuardFail):
        g.culture(c, stdp)


def test_synaptic_scaling_scales_only_ee_weights_onto_neurons_off_target():
    from biocompute.culture.plasticity import SynapticScaling

    c = small_culture(seed=7)
    s = SynapticScaling(eta=1e-3, every_ms=1.2, tau_rate_ms=1e9)
    s.set_targets(c, torch.full((c.n,), 5.0))
    s.on_substep(c, torch.zeros(c.n, dtype=torch.bool))
    q = c.n // 4
    s.rate[:q], s.rate[q:2 * q] = 15.0, 5005.0
    s.on_substep(c, torch.zeros(c.n, dtype=torch.bool))
    ee = c.syn_exc & c.excitatory[c.post]
    for lo, rate in ((0, 15.0), (q, 5005.0)):
        m = ee & (c.post >= lo) & (c.post < lo + q)
        assert m.any()
        torch.testing.assert_close(c.w[m], c.w0[m] * float(np.exp(1e-3 * 1.0 * (5.0 - rate))))
    at = ee & (c.post >= 2 * q)
    torch.testing.assert_close(c.w[at], c.w0[at], rtol=1e-6, atol=0.0)
    assert torch.equal(c.w[~ee], c.w0[~ee])


def test_synaptic_scaling_rate_estimate_tracks_firing():
    from biocompute.culture.plasticity import SynapticScaling

    c = small_culture(seed=7)
    s = SynapticScaling(eta=0.0, tau_rate_ms=500.0)
    s.set_targets(c, torch.zeros(c.n))
    fire = torch.zeros(c.n, dtype=torch.bool)
    for k in range(10000):
        fire[:50] = k % 100 == 0
        s.on_substep(c, fire)
    assert torch.allclose(s.rate[:50], torch.tensor(20.0), rtol=0.1)
    assert float(s.rate[50:].abs().max()) == 0.0
    assert torch.equal(c.w, c.w0)


def test_chain_of_stdp_and_scaling_resumes_exactly():
    import pytest

    from biocompute.culture.plasticity import STDP, Chain, SynapticScaling

    def rules():
        return Chain(STDP(), SynapticScaling(eta=1e-4, every_ms=20.0))

    a = small_culture(seed=8)
    a.plasticity = rules()
    a.plasticity.rules[1].set_targets(a, torch.full((a.n,), 0.5))
    a.run(190.0)
    cst, pst = a.state(), a.plasticity.state()
    a.run(200.0)
    b = small_culture(seed=8)
    b.load_state(cst)
    b.plasticity = rules()
    b.plasticity.load_state(b, pst)
    b.run(200.0)
    assert torch.equal(a.w, b.w)
    stdp_only = small_culture(seed=8)
    stdp_only.plasticity = STDP()
    stdp_only.run(390.0)
    assert not torch.equal(a.w, stdp_only.w)
    with pytest.raises(ValueError):
        Chain(STDP()).load_state(b, pst)


def test_excitatory_input_is_a_fixed_order_sum():
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        c = OrganoidCulture(make_config("cpu_2k", n_neurons=1500, device=device, wells=2, seed=5))
        e = c.syn_exc
        ref = torch.zeros(c.n, dtype=torch.float64, device=device).index_add_(0, c.post[e], c.w[e].double())
        got = c.excitatory_input()
        assert torch.allclose(got.double(), ref, rtol=1e-6, atol=0.0)
        assert all(torch.equal(got, c.excitatory_input()) for _ in range(10))


def test_gpu_plastic_run_repeats_bit_for_bit():
    import pytest

    from biocompute.culture.plasticity import STDP, Chain, SynapticScaling

    if not torch.cuda.is_available():
        pytest.skip("no GPU")

    def run():
        dev = _plastic_device(device="cuda")
        c = dev.culture
        c.plasticity = Chain(STDP(), SynapticScaling(eta=1e-3, every_ms=10.0))
        w_in0 = c.excitatory_input()
        for i in range(8):
            dev.stimulate([10, 20], [5.0, 30.0], [1.0, 1.0])
            dev.step(50.0)
            w_in = c.excitatory_input()
            f = torch.where(w_in > 0, w_in0 / w_in.clamp(min=1e-6), torch.ones_like(w_in))
            c.w[c.syn_exc] *= f[c.post[c.syn_exc]]
        return c.w.clone(), c.v.clone()

    (w1, v1), (w2, v2) = run(), run()
    assert torch.equal(w1, w2) and torch.equal(v1, v2)
