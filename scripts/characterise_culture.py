"""Spontaneous and evoked activity of a digital organoid, compared with what organoids on MEAs show.

    uv run scripts/characterise_culture.py --preset cpu_2k --seconds 30
"""

import argparse
import json
from pathlib import Path

import numpy as np

from biocompute.culture.network import make_config
from biocompute.device import SimulatedBiocomputer
from biocompute.eval.metrics import culture_stats
from biocompute.mea.electrodes import make_mea_config

# Ranges reported for cortical organoids and dissociated cultures on MEAs (see README, "Culture realism").
TARGETS = {
    "mean_rate_hz": (0.1, 10.0),
    "burst_rate_per_min": (1.0, 30.0),
    "frac_spikes_in_bursts": (0.2, 0.95),
}

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="cpu_2k")
parser.add_argument("--mea", default="mcs60")
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="outputs/culture")
parser.add_argument("--set", nargs="*", default=[], help="Culture overrides, key=value")
args = parser.parse_args()

overrides = {k: type(getattr(make_config(args.preset), k))(v) for k, v in (s.split("=") for s in args.set)}
dev = SimulatedBiocomputer(make_config(args.preset, seed=args.seed, **overrides), make_mea_config(args.mea),
                           keep_raster=True)
c = dev.culture
print(f"{c.n} neurons, {c.n_synapses} synapses ({c.n_synapses / c.n:.0f}/neuron), {dev.n_channels} channels, "
      f"{(dev.mea.detect.sum(1) > 0.5).sum().item()} channels see a neuron")

rasters, detections = [], []
for _ in range(int(args.seconds * 1000 / 100)):
    block = dev.step(100.0)
    rasters.append(dev.last_raster.cpu().numpy())
    detections.append(block.counts(1)[0])
raster = np.concatenate(rasters)
stats = culture_stats(raster, c.cfg.dt_ms)
stats["mea_rate_per_channel_hz"] = float(np.sum(detections) / args.seconds / dev.n_channels)

# Evoked: one pulse on a central channel, 20 repeats, spikes in the 50 ms after vs before.
ch = int(np.argmin(np.linalg.norm(dev.mea.xy, axis=1)))
evoked, base = [], []
for _ in range(20):
    pre = dev.step(50.0).counts(1)[0].sum()
    dev.stimulate([ch], [0.0], [1.0])
    post = dev.step(50.0).counts(1)[0].sum()
    dev.step(400.0)
    base.append(pre)
    evoked.append(post)
stats["evoked_mea_spikes_50ms"] = float(np.mean(evoked))
stats["baseline_mea_spikes_50ms"] = float(np.mean(base))

print(json.dumps(stats, indent=2))
for key, (lo, hi) in TARGETS.items():
    ok = lo <= stats[key] <= hi
    print(f"{'ok ' if ok else 'OUT'} {key} = {stats[key]:.3g} (target {lo}–{hi})")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.preset}_seed{args.seed}.json").write_text(json.dumps(stats, indent=2))
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seg = raster[: int(10_000 / c.cfg.dt_ms)]
    t, n = np.nonzero(seg)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 1])
    ax[0].scatter(t * c.cfg.dt_ms / 1000, n, s=0.2, c="k")
    ax[0].set_ylabel("neuron")
    pop = seg.reshape(-1, 20, seg.shape[1]).sum((1, 2))
    ax[1].plot(np.arange(len(pop)) * 20 * c.cfg.dt_ms / 1000, pop)
    ax[1].set_ylabel("spikes / 10 ms")
    ax[1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out / f"{args.preset}_seed{args.seed}_raster.png", dpi=120)
    print(f"raster: {out / f'{args.preset}_seed{args.seed}_raster.png'}")
except ImportError:
    pass
