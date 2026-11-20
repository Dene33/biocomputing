# Biocomputing: a digital organoid driving Asimov 1

We want to control a humanoid robot (Menlo Research's Asimov 1) with a biocomputer: living neurons grown on a multi-electrode array (MEA), as offered by Cortical Labs (CL1), FinalSpark (Neuroplatform) or MaxWell Biosystems. We have no culture yet, so this project simulates one. It is a **digital organoid** that the robot reaches only as it would reach real hardware: by stimulating electrodes and reading the spikes they detect.

The culture drives **all 23 joints of Asimov 1 at 50 Hz** in MuJoCo. Two ways to make that work are built and compared:

- **(A) Reservoir.** The culture is a fixed dynamical system. A linear readout from electrode activity to joint actions is trained to imitate the trained walking policy, by DAgger and ridge regression (as in Brainoware, *Nature Electronics* 2023).
- **(B) DishBrain feedback.** The decoder is fixed and only the culture learns, through STDP shaped by feedback. Predictable stimulation follows good outcomes and unpredictable stimulation follows falls (Kagan et al., *Neuron* 2022).

## Layout

| Path | What |
| --- | --- |
| `src/biocompute/device.py` | `Biocomputer` protocol (`stimulate`, `step` → `SpikeBlock`) and `SimulatedBiocomputer`. A real CL1/FinalSpark adapter implements the same protocol. |
| `src/biocompute/culture/network.py` | `OrganoidCulture`: Izhikevich neurons (80 % excitatory RS/IB/CH, 20 % inhibitory FS/LTS), a flattened organoid lying on the array, distance-dependent sparse connectivity with axonal delays, Tsodyks–Markram short-term depression, Poisson background. `wells=K`: K independent organoids, as on a multi-well plate. Torch, CPU or GPU (synaptic delivery as one sparse matrix–vector product per 0.5 ms substep), seeded. |
| `src/biocompute/culture/plasticity.py` | STDP, reward-modulated STDP (in-silico upper bound), forgetting towards baseline weights, rest |
| `src/biocompute/culture/aging.py` | Excitability drift and cell loss over days |
| `src/biocompute/mea/electrodes.py` | MEA: multi-unit detection with distance-dependent probability, false positives, dead channels, stimulation that falls off with distance, rate and amplitude limits, artifact blanking. Presets `mcs60` (60 channels, 200 µm), `grid256`, `hd1024` (MaxWell-like). One grid per well. |
| `src/biocompute/interface/` | Encoders (observation → pulses; amplitude or rate coding; identity, PCA, supervised or random projection), spike features, ridge readout (`LinearReadout`, GPU for thousands of features), trained nonlinear decoder (`MLPReadout`), the fixed motor-map decoder, DishBrain feedback |
| `src/biocompute/robot/asimov.py` | `AsimovEnv`: Asimov 1 stepped one 20 ms control step at a time, built on cyclotron's `sim2sim` |
| `src/biocompute/robot/teacher.py` | The trained walking policy (ONNX): the teacher and the ceiling |
| `src/biocompute/loop.py` | The closed loop. The culture needs 20 ms of its own time to answer, so an action applies one step late. |
| `src/biocompute/train/` | Tasks, DAgger (serial, and `parallel.py`: several copies of the same dish in worker processes), feedback sessions |
| `scripts/` | `characterise_culture.py`, `baselines.py`, `train_reservoir.py`, `train_dagger.py` (parallel DAgger, ridge or MLP decoder), `eval_readout.py` (many seeds, ablations), `train_feedback.py`, `evaluate.py`, `view.py`, `train_expert.py` / `program_expert.py` (stance-only balance experts: they cannot pass `stand_push`) |

## Setup

```sh
uv sync
uv run pytest
```

The robot side is reused, not copied. The following paths are read from sibling checkouts, and each can be overridden with an environment variable:

| Variable | Default | What |
| --- | --- | --- |
| `CYCLOTRON_DIR` | `../cyclotron` | Menlo's training repo, tested at `285ed7d`. Only `source/cyclotron/cyclotron/sim2sim.py` is imported, which needs no Isaac Sim. |
| `ASIMOV_POLICY_DIR` | `../denebots/site/learn-sim/policies/menlo-reference` | Teacher `policy.onnx` and its `env.yaml` (78 inputs, 23 outputs) |
| `ASIMOV_XML` | `$CYCLOTRON_DIR/third_party/asimov-1/sim-model/xmls/asimov_1.xml` | Asimov 1 MJCF |

`tests/test_robot.py` checks that the stepwise `AsimovEnv` driven by the teacher reproduces `cyclotron.sim2sim.play_velocity` exactly.

## Running

```sh
uv run scripts/characterise_culture.py --seconds 30             # spontaneous + evoked activity, raster PNG
uv run scripts/baselines.py                                     # teacher (ceiling) and zero action (floor) per task
uv run scripts/train_reservoir.py --task stand_push --source mea --mea hd1024
uv run scripts/train_reservoir.py --task stand_push --source none   # control: no culture (linear policy)
uv run scripts/train_reservoir.py --task stand_push --source dead   # control: culture without recurrent synapses
uv run scripts/train_reservoir.py --task stand_push --source oracle # control: read every neuron
uv run scripts/train_feedback.py --protocol scratch --task stand_push --condition dishbrain --minutes 10
uv run scripts/train_feedback.py --protocol adapt --readout outputs/reservoir/<run> --perturb weak_knees
uv run scripts/evaluate.py                                      # outputs/report.md
uv run scripts/view.py outputs/reservoir/<run>                  # live MuJoCo viewer
uv run scripts/view.py outputs/reservoir/<run> --video run.mp4  # robot + MEA heatmap + raster
```

Tasks (`src/biocompute/train/tasks.py`):

| Task | What it asks |
| --- | --- |
| `stand` | Stand for 20 s |
| `stand_push` | Stand while 120–200 N pushes arrive every 3 s |
| `walk` | Walk at 0.4 m/s |
| `walk_schedule` | Walk, turn and stop on a schedule |

**Zero action stands on flat ground**: holding the default pose with PD keeps Asimov 1 up for 20 s. So `stand` alone proves nothing, and `stand_push` is the first rung where a controller has to do something.

## Culture realism

The `cpu_2k` preset is tuned so that spontaneous activity looks like a cortical organoid or culture on an MEA:

- sparse firing of about 0.6 Hz on average, with interneurons faster than excitatory cells;
- network bursts every ~2–3 s at irregular intervals (CV ≈ 0.3);
- most spikes inside bursts (~75 %).

It also responds to single-electrode stimulation. Bursts end because short-term depression uses up synaptic resources, as in culture models. These are plausible ranges, not a fit to one dataset: reports for organoids (e.g. Trujillo et al., *Cell Stem Cell* 2019) and dissociated cultures (Wagenaar et al., *BMC Neuroscience* 2006) vary with age and protocol. The realism dials are neuron count, electrode count and pitch, detection radius, noise, dead channels, stimulation limits, latency, forgetting time constant and aging.

## Status and findings so far (20 November 2026)

Built: everything in the layout above, with 11 passing tests. Teacher parity with cyclotron is exact.

**Baselines** (`scripts/baselines.py`, 5 seeds, 20 s, 1-step latency):

| Task | Teacher | Zero action |
| --- | --- | --- |
| `stand` | 20 s, 5/5 | 20 s, 5/5 |
| `stand_push` | 20 s, 5/5 | 4.6 s, 0/5 |
| `walk` | 6.1 m | 0 m |

**Reservoir on `stand_push`** (one seed, `outputs/report.md`): no readout beats the zero-action floor.

| Readout | Survived |
| --- | --- |
| 60-ch MEA | 3.0 s |
| 256-ch | 1.9 s |
| 1024-ch | 2.3 s |
| Every neuron (oracle) | 2.4 s |
| Dead culture | 2.0 s |
| No culture | 2.3 s |

What this says:

1. **Electrode bandwidth is real but not the only limit.** 60 electrodes detect ~8 spikes per 20 ms step, at a realistic rate. More channels raise train R² from 0.73 to 0.95, but closed-loop control does not improve, even when the readout sees every neuron.
2. **A linear readout cannot imitate this policy where it matters.** Even a linear policy on the raw 78 observations stands, but it falls after a 120–200 N push, and with one step of latency it cannot walk. The reservoir must supply the missing nonlinearity, and this 2000-neuron culture with a ridge readout does not.
3. **DAgger data from falls hurts.** Validation R² turns negative as rounds add the states the student visits while falling.

**DishBrain sessions:** both protocols run end to end (2-minute smoke tests only). Plain STDP runs away: mean excitatory weight goes from 2.2 to 3.8 in 2 simulated minutes. Synaptic scaling is needed before any M3 result means anything.

**Next steps, most promising first:**

- **Residual control:** the culture adds a bounded correction to the teacher, or to a simple PD/balance controller, rather than producing all 23 targets. This is honest about what a real culture can carry, and it fits the "small adaptive layer" view.
- **A smaller action space:** a few gait synergies (PCA of the teacher's actions), rather than 23 independent joints.
- **Stronger reservoir conditions:** the `gpu_20k` preset, input-driven rather than burst-dominated regimes (stimulation quiets bursting), and longer feature memory.
- **Homeostatic scaling for STDP**, then the full M3 comparison over ≥5 seeds.

## Limits of the simulation

- A point-neuron network is not an organoid. It has no glia, no 3D layered structure, no neuromodulators, no metabolic limits, and no variability between organoids beyond the seed.
- Stimulation is modelled as a direct current into nearby neurons for one substep; real biphasic pulses mostly recruit axons.
- Rest is fast-forwarded: only weight decay acts, with no spontaneous activity.
- Findings here are statements about this model. Each one is a hypothesis to test on a real culture, not a prediction.
