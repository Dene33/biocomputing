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
    encoder_gain: float = 1.5
    projection: str = "pca"
    slots_ms: tuple = (0.0,)
    bins: int = 2
    taus_ms: tuple = (40.0, 160.0)
    wells: int = 1
    diverse_wells: bool = False
    efference: bool = False
    select_dims: tuple = ()
    select_spread: float = 1.0
    silent_inputs: bool = False
    source: str = "mea"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build(setup: Setup, obs_samples: np.ndarray, action_samples: np.ndarray | None = None) -> CultureController:
    from biocompute.mea.electrodes import make_mea_config

    rng = np.random.default_rng(setup.seed)
    if setup.source == "none":
        channels = np.arange(setup.n_stim)
        device = None
    else:
        device = SimulatedBiocomputer(
            make_config(setup.culture, seed=setup.seed, wells=setup.wells, **setup.culture_overrides),
            make_mea_config(setup.mea, seed=setup.seed, wells=setup.wells, **setup.mea_overrides),
        )
        if setup.source == "dead":
            device.culture.w.zero_()
            device.culture.w0.zero_()
        per = device.mea.channels_per_well
        alive = device.mea.alive_channels
        wells = [rng.permutation(alive[(alive >= k * per) & (alive < (k + 1) * per)]) for k in range(setup.wells)]
        channels = wells[0]
    stim = channels[: setup.n_stim]
    encoder = Encoder(stim, mode=setup.encoding, slots_ms=setup.slots_ms, normalizer=ObsNormalizer.fit(obs_samples),
                      gain=setup.encoder_gain, seed=setup.seed)
    if setup.projection == "pca":
        encoder.fit_pca(obs_samples)
    elif setup.projection == "identity":
        skip = (6, 7, 8) + (tuple(range(55, 78)) if setup.efference else ())
        encoder.fit_identity(obs_samples, skip=skip)
    elif setup.projection == "select":
        encoder.fit_select(obs_samples, setup.select_dims, spread=setup.select_spread)
    elif setup.projection == "axis":
        encoder.fit_axis(obs_samples, setup.select_dims, spread=setup.select_spread)
    elif setup.projection == "supervised":
        if action_samples is None:
            raise ValueError("projection='supervised' needs action_samples.")
        encoder.fit_supervised(obs_samples, action_samples)
    if device is not None and setup.wells > 1:
        encoder.channels = np.concatenate([w[: setup.n_stim] for w in wells])
        W, b = [encoder.W], [encoder.b]
        for k in range(1, setup.wells):
            if setup.diverse_wells:
                R, _ = np.linalg.qr(np.random.default_rng(setup.seed + 100 + k).standard_normal((len(stim), len(stim))))
                Wk = R @ encoder.W
                W.append(Wk)
                b.append(-Wk @ encoder.normalizer(obs_samples).mean(0))
            else:
                W.append(encoder.W)
                b.append(encoder.b)
        encoder.W = np.concatenate(W)
        encoder.b = np.concatenate(b)
        encoder.phase = np.zeros(len(encoder.channels))
        channels = np.concatenate([w[: setup.n_stim + setup.n_feedback] for w in wells])
    features = None
    if device is not None:
        features = SpikeFeatures(device.mea.alive_channels, bins=setup.bins, taus_ms=setup.taus_ms)
    source = {"dead": "mea"}.get(setup.source, setup.source)
    ctrl = CultureController(device, encoder, features, source=source, efference=setup.efference)
    ctrl.stimulate_inputs = not setup.silent_inputs
    ctrl.feedback_channels = channels[setup.n_stim : setup.n_stim + setup.n_feedback]
    return ctrl
