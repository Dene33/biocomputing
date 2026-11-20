"""Build a biocomputer + controller from a few settings, the same way for training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from biocompute.culture.network import make_config
from biocompute.device import SimulatedBiocomputer
from biocompute.interface.decoders import SpikeFeatures
from biocompute.interface.encoders import Encoder, ObsNormalizer
from biocompute.loop import CultureController


@dataclass
class Setup:
    culture: str = "cpu_2k"
    culture_overrides: dict = field(default_factory=dict)
    mea: str = "mcs60"
    mea_overrides: dict = field(default_factory=dict)
    n_stim: int = 32
    n_feedback: int = 8
    encoding: str = "amplitude"
    projection: str = "pca"
    slots_ms: tuple = (0.0,)
    bins: int = 2
    taus_ms: tuple = (40.0, 160.0)
    source: str = "mea"
    """``mea``, ``oracle``, ``none`` (no culture) or ``dead`` (MEA readout, recurrent synapses removed)."""
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build(setup: Setup, obs_samples: np.ndarray) -> CultureController:
    from biocompute.mea.electrodes import make_mea_config

    rng = np.random.default_rng(setup.seed)
    if setup.source == "none":
        channels = np.arange(setup.n_stim)
        device = None
    else:
        device = SimulatedBiocomputer(
            make_config(setup.culture, seed=setup.seed, **setup.culture_overrides),
            make_mea_config(setup.mea, seed=setup.seed, **setup.mea_overrides),
        )
        if setup.source == "dead":
            device.culture.w.zero_()
            device.culture.w0.zero_()
        channels = rng.permutation(device.mea.alive_channels)
    stim = channels[: setup.n_stim]
    encoder = Encoder(stim, mode=setup.encoding, slots_ms=setup.slots_ms, normalizer=ObsNormalizer.fit(obs_samples),
                      seed=setup.seed)
    if setup.projection == "pca":
        encoder.fit_pca(obs_samples)
    features = None
    if device is not None:
        features = SpikeFeatures(device.mea.alive_channels, bins=setup.bins, taus_ms=setup.taus_ms)
    source = {"dead": "mea"}.get(setup.source, setup.source)
    ctrl = CultureController(device, encoder, features, source=source)
    ctrl.feedback_channels = channels[setup.n_stim : setup.n_stim + setup.n_feedback]
    return ctrl
