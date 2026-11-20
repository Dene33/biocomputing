import numpy as np
import torch

from biocompute.culture.network import OrganoidCulture, make_config
from biocompute.device import SimulatedBiocomputer
from biocompute.mea.electrodes import make_mea_config


def izhikevich_single(a, b, c, d, current, ms=500.0, dt=0.5):
    """Reference single-neuron integration, as in the culture (two half-steps for v)."""
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
    rs = izhikevich_single(0.02, 0.2, -65, 8, 10)  # regular spiking: adapts
    fs = izhikevich_single(0.1, 0.2, -65, 2, 10)  # fast spiking: high rate, no adaptation
    ch = izhikevich_single(0.02, 0.2, -50, 2, 10)  # chattering: bursts with short intra-burst intervals
    assert len(fs) > 2 * len(rs)
    isi_rs = np.diff(rs)
    assert isi_rs[-1] > isi_rs[0]  # adaptation
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
    dev.culture.w.zero_()  # no network: only the direct response
    ch = 27
    dev.stimulate([ch], [0.0], [1.0])
    dev.step(10.0)
    fired = dev.last_raster.any(0).cpu()
    dist = (dev.culture.pos[:, :2] - torch.tensor(dev.mea.xy[ch])).norm(dim=1)
    assert fired.sum() > 0
    assert dist[fired].max() < 250.0


def test_recording_is_blanked_after_a_pulse():
    dev = SimulatedBiocomputer(make_config("cpu_2k", n_neurons=600, radius_um=800), make_mea_config("mcs60"))
    dev.mea.cfg.false_positive_hz = 5000.0  # plenty of detections everywhere
    dev.stimulate(np.arange(10), np.zeros(10), np.full(10, 0.5))
    block = dev.step(10.0)
    assert len(block.times_ms) > 0
    assert block.times_ms.min() >= dev.mea.cfg.blank_ms - 1e-9


def test_rate_limit_drops_excess_pulses():
    dev = SimulatedBiocomputer(make_config("cpu_2k", n_neurons=300, radius_um=800), make_mea_config("mcs60"))
    dev.stimulate([5] * 4, [0.0, 1.0, 2.0, 5.0], [0.5] * 4)  # 200 Hz limit: one every 5 ms
    dev.step(10.0)
    assert dev.pulses_delivered == 2 and dev.pulses_dropped == 2
