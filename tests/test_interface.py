import numpy as np

from biocompute.device import SpikeBlock
from biocompute.interface.decoders import LinearReadout, MotorMapDecoder, SpikeFeatures, r2_per_output
from biocompute.interface.encoders import Encoder, ObsNormalizer
from biocompute.interface.feedback import Feedback


def test_encoder_shapes_and_ranges():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((500, 78))
    enc = Encoder(np.arange(16), normalizer=ObsNormalizer.fit(obs))
    ch, t, a = enc.pulses(obs[0], 20.0)
    assert len(ch) == 16 and np.all(t == 0) and np.all((a > 0) & (a <= 1))
    enc.fit_pca(obs)
    drive = np.array([enc.drive(o) for o in obs])
    assert drive.shape == (500, 16) and drive.std(0).min() > 0.05
    rate = Encoder(np.arange(8), mode="rate", slots_ms=(0, 5, 10, 15))
    ch, t, a = rate.pulses(np.zeros(78), 20.0)  # drive 0.5 → about two pulses per channel per step
    assert 12 <= len(ch) <= 20 and set(np.unique(t)) <= {0, 5, 10, 15}


def test_features_and_ridge_readout():
    block = SpikeBlock(np.array([0, 0, 2]), np.array([1.0, 15.0, 15.0]), 20.0, 4)
    feats = SpikeFeatures(np.arange(4), bins=2, taus_ms=(40.0,))
    x = feats(block)
    assert x.shape == (feats.dim,) and x[0] == 1.0 and x[4] == 1.0 and x[6] == 1.0
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 10))
    Y = X @ rng.standard_normal((10, 23)) + 0.01 * rng.standard_normal((2000, 23))
    ro = LinearReadout(10)
    ro.fit(X, Y, alpha=1e-3)
    assert r2_per_output(Y, ro.predict(X)).min() > 0.99
    assert np.allclose(ro(X[0]), ro.predict(X[:1])[0], atol=1e-5)


def test_motor_map_and_feedback():
    dec = MotorMapDecoder(np.arange(60), seed=0)
    a = dec(SpikeBlock(np.repeat(dec.plus[0], 5), np.zeros(5 * len(dec.plus[0])), 20.0, 60))
    assert a[0] > 0 and np.all(a[1:] == 0)
    fb = Feedback(np.arange(8))
    fb.good()
    ch, t, _ = fb.pulses(20.0)
    assert len(ch) == 16 and set(t) == {0.0, 10.0}  # 100 Hz on all 8 together
    fb.bad()
    assert fb.busy


def test_cross_validated_ridge_resists_overfitting():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((600, 400))  # many more features than are useful
    Y = X[:, :3] @ rng.standard_normal((3, 23)) + 0.5 * rng.standard_normal((600, 23))
    ro = LinearReadout(400)
    cv = ro.fit_cv(X, Y)
    assert cv["alpha"] > 0.1 and 0.3 < cv["val_r2"] < 0.95


def test_standardized_targets_weight_every_joint_equally():
    """A joint with a huge spread (an arm pinned past its limit) must not swamp a small balance joint."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3000, 20))
    Y = np.c_[100.0 * (X[:, :1] + rng.standard_normal((3000, 1))), 0.05 * X[:, 1:2]]
    ro = LinearReadout(20, out_dim=2)
    ro.fit_cv(X, Y, standardize_y=True)
    assert r2_per_output(Y, ro.predict(X))[1] > 0.98
    assert np.allclose(ro(X[0]), ro.predict(X[:1])[0], atol=1e-4)
