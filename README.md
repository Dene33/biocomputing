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
| `scripts/` (experiments) | `reflex_oracle.py`, `blind_teacher.py`, `no_imu_student.py`, `vestibular_probe.py`, `trigger_probe.py`, `adaptation_slot.py`, `struggle_signal.py`, `coach_probe.py`, `dish_regimes.py`, `burst_probe.py`, `cartpole_replication.py`, `cebra_probe.py`, `bl1_drift.py` (with `bl1_checks.py`, `bl1_rep_analysis.py`; runs in bl1's own venv): one experiment each; results in [docs/route-to-computation.md](docs/route-to-computation.md) and the commit log |
| `scripts/launch.py`, `scripts/watch.py`, `src/biocompute/runguard.py` | Guarded long runs: smoke test, detached systemd unit, in-run checks, watchdog |
| `src/biocompute/eval/dish_channel.py` | A dish as a statistical channel (delay, update rate, noise, gain), the stand-in for the culture in Phases 2 and 3.1 |
| `wsp/controller/` | Weight-space programming of the robot-side controller (Phase 2). `track_a.py` needs `teacher_weights.npz`, written from the teacher's ONNX by `export_teacher.py` |

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

# The current best design (17/20 on stand_push): 6 wells as the sensory path, MLP decoder with efference copy
uv run scripts/train_dagger.py --culture gpu_20k --mea grid256 --wells 6 --diverse-wells \
    --projection identity --n-stim 52 --encoder-gain 0.5 --efference --readout mlp \
    --mea-set stim_length_um=120 --slots 0 5 10 15 --workers 4 --threads 1 --rounds 16 --episodes 12
uv run scripts/train_dagger.py --source none --projection identity --n-stim 52 --encoder-gain 0.5 \
    --efference --readout mlp --workers 6 --rounds 14 --episodes 12           # control: no culture
uv run scripts/eval_readout.py outputs/dagger/<run> --seeds 20                # fresh push seeds
uv run scripts/eval_readout.py outputs/dagger/<run> --control silent          # ablations: silent | dead | zero
uv run scripts/reflex_oracle.py                                               # stance + recovery primitives, oracle choice
```

Long runs go through `scripts/launch.py`: it smoke-tests the script (`--smoke`, the whole pipeline at tiny scale),
then starts it as a detached systemd unit, so an editor crash can't kill it. `scripts/watch.py` follows the log and
stops the unit on an error, a failed in-run check (`biocompute.runguard.Guard`) or a stall:

```sh
uv run scripts/launch.py --unit bio-X --log outputs/X/run.log -- scripts/foo.py <args>
uv run scripts/watch.py --unit bio-X --log outputs/X/run.log
```

`train_dagger.py` checkpoints every round and `cartpole_replication.py` saves when stopped; both resume when the same
command is run again.

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

## Status and findings (23 November 2026)

Built: everything in the layout above, with 19 passing tests. Teacher parity with cyclotron is exact.

**Baselines** (`scripts/baselines.py`, 5 seeds, 20 s, 1-step latency):

| Task | Teacher | Zero action |
| --- | --- | --- |
| `stand` | 20 s, 5/5 | 20 s, 5/5 |
| `stand_push` | 20 s, 5/5 | 4.6 s, 0/5 |
| `walk` | 6.1 m | 0 m |

### Result: the culture keeps Asimov 1 standing under pushes, as its only sensory path

Six independent organoids (wells; 20k neurons and 256 electrodes each) receive the robot's 52 sensory signals as
stimulation. A trained decoder reads their spikes, plus its own last command (efference copy), and drives all 23
joints. `stand_push`, fresh push seeds (`outputs/report.md`):

| Controller | Survived | Full 20 s |
| --- | --- | --- |
| Teacher (ceiling) | 20.0 s | 20/20 |
| **Culture + MLP decoder + efference copy** | **18.5 s** | **17/20** |
| Same decoder, no culture (the encoded signals fed in directly) | 17.5 s | 15/20 |
| Culture with no sensory stimulation | 1.6 s | 0/10 |
| Culture without recurrent synapses (decoder not retrained) | 14.5 s | 6/10 |
| Zero action (floor) | 4.3 s | 0/10 |

What this says:

1. **The dish is necessary as a channel.** Silence it and the robot falls faster than standing still.
2. **The dish computes nothing extra.** No culture does as well as the culture. The decoder does the computing; the
   culture carries the state (decodable at R² ≈ 0.7). This is a biological sensory interface with a BCI-style
   decoder, not reservoir computing.
3. **The dead-culture drop is not evidence of computation.** The decoder was trained on the living culture's responses;
   a fair test retrains it on the dead one.

What it took (each step was necessary):

- **Labels.** The teacher pins the shoulder-roll joints far past their limits (actions of −15 and +10), which
  dominated the fits and R². Labels are clipped to the joint range and standardised per joint.
- **Encoding.** One stimulation channel per z-scored observation dimension (`identity`), sigmoid gain 0.5. The
  command (constant) and the last action (the decoder knows it) are not stimulated. PCA whitening, the old default,
  inflates near-constant directions and broke even an ideal reservoir; gain 1.5 saturated on pushes.
- **Culture.** Strong distributed stimulation keeps it input-driven rather than bursting. Stimulation reach is 120 µm
  (axonal activation; the preset is 60). Wells average independent noise (decodable R² 0.38 → 0.58 from 1 → 6
  wells), each well getting a different random mix of the inputs.
- **Decoder.** The efference copy matters most: the teacher tolerates state noise of 0.5–0.7 × its spread when its
  last action is exact, and only ~0.2 × otherwise. An MLP (2 × 256); push recoveries weighted × 5; 16 DAgger rounds;
  the final decoder averages the last 4.
- **Speed.** Synaptic delivery as one sparse matrix–vector product per substep on the GPU, one host sync per control
  step, ridge on the GPU: 4 wells went from 182 to 26 ms per step.

### Negative results

- **A linear readout of the culture cannot do it.** The part of the action not explained by the last action needs
  nonlinear features of the state: R² 0.64 with random nonlinear features of the observation, 0.25 linear, 0.21–0.23
  from the culture, 0.17 even reading every neuron of a well. More wells, filters, bins and encodings did not close it.
- **Standing still cannot pass `stand_push`.** Zero action falls from a single 160 N push in every direction. Linear
  balance controllers found by random search (`train_expert.py`) or written by hand (`program_expert.py`) plateau at
  ~5 s. Recovery needs steps.
- **Recorded recovery steps don't work open loop** (`reflex_oracle.py`), even when an oracle picks the right one:
  4.7 s. A closed-loop recovery triggered within 200 ms of each push survives 20/20, so a 1-bit trigger suffices if
  the recovery itself sees the state.
- **Few synergies are not enough.** The teacher's leg actions projected onto their top 3–8 principal components fall.
- **Earlier reservoir runs** (single culture, ridge readout, PCA encoding) fell before the first push; DAgger data from
  falls made validation R² negative. **DishBrain sessions** run end to end, but plain STDP runs away (mean excitatory
  weight 2.2 → 3.8 in 2 simulated minutes); synaptic scaling is needed first.

**Phase 1.1, the blind-teacher sweep** (`scripts/blind_teacher.py`, 40 seeds). Remove the teacher's IMU and it falls
in 1.0 s (zero action: 4.8 s). It tolerates IMU noise up to 1.0 × spread, 60 ms of added delay, 10 Hz updates and 8 levels,
each alone. Combined, the spec for a vestibular dish is ≤ 40 ms, ≥ 25 Hz and error ≤ 0.5 × spread (39/40). Gate G1 passes.
DAgger students retrained without the IMU fall within 2.3 s (0/40), even with memory of past joint states. With it, the
same recipe gets 40/40, and IMU plus joint positions alone 35/40 (`scripts/no_imu_student.py`). The vestibular channel
is necessary.
The table is in [docs/next-steps.md](docs/next-steps.md#11-blind-teacher-sweep).

**Phase 3.1, the vestibular probe** (`scripts/vestibular_probe.py`). Only the 6 IMU values are stimulated, open loop.
One well with 12 electrodes (CL1-sized) carries them with errors of 0.53–0.80 × spread and shrinks them to ~63%. Six
wells with 32 electrodes each: 0.27–0.58 × spread. As dish models in closed loop, 1 × 32, 6 × 12 and 6 × 32 keep the
unchanged teacher up in 58–60/60 runs; 1 × 12 in 20/60.

**Phase 2, weight-space programming** (`wsp/controller/`). Scaling the teacher's first-layer IMU weights by 1/gain
brings the CL1-sized channel to 58/60 (validation) and 40/40 (hidden test; unchanged teacher 16/40). These are
statistical channel models; the closed loop with the culture itself is Phase 3.3.

**Route finding** ([docs/route-to-computation.md](docs/route-to-computation.md)). Phase 4 tested in simulation and
the routes ranked against real hardware. No stimulation regime makes the dish compute. It keeps less of the recent IMU
than its own stimulation pattern and detects pushes worse than a filter (AUC ≤ 0.75 vs 0.86). Slow NMDA excitation
doesn't change that, and neither does a stronger readout: CEBRA finds less in the dish than a linear one
(`cebra_probe.py`). There is a job for a dish: weak knees and a 20 kg payload each need their own fix (0/20 → 19/20),
and each fix is fatal for the other change. But our model culture can't be coached to choose (chance), because every
probe sets off the same network burst. The flaw is the model's plasticity: under repeated stimulation it runs away
(the cart-pole organoid protocol, replicated step by step, can't be tested on it), and one time-boxed fix with
inhibitory plasticity brought rates near rest but didn't keep the inputs' pathways apart. An outside model, bl1
(github.com/m9h/bl1, run as shipped and with two of its five bugs fixed, 5 cultures), fails the same drift test:
its per-neuron synaptic scaling holds rates on its own, but with STDP on, pathways move within minutes
(`bl1_drift.py`). Ranked routes: coached adaptation first (fits every platform, cart-pole
precedent; a real dish decides), push detection tested open loop alongside it, the inner-ear relay as hardware testbed.

**Next** (24 November 2026). We have no real culture and no physical robot for now, so the work runs in two
simulated tracks ([Order of work without a dish](docs/route-to-computation.md#order-of-work-without-a-dish)).
Track A puts the simulated culture in the robot's loop (Phase 3.3 of [docs/next-steps.md](docs/next-steps.md)).
Track B tries to make the culture model good enough to ask learning questions of, behind gates fixed in advance.
Real-dish and real-robot steps are parked. Ideas behind it: [docs/organoid-control-ideas.md](docs/organoid-control-ideas.md).

## Limits of the simulation

- A point-neuron network is not an organoid. It has no glia, no 3D layered structure, no neuromodulators, no metabolic limits, and no variability between organoids beyond the seed.
- Stimulation is modelled as a direct current into nearby neurons for one substep; real biphasic pulses mostly recruit axons.
- Rest is fast-forwarded: only weight decay acts, with no spontaneous activity.
- Training ran 4 identical copies of one dish in parallel, and the culture is stationary while it controls (no
  plasticity, no drift). A real dish is unique and changes over hours.
- The current design stimulates 52 electrodes per well at 200 Hz in 6 wells, more than today's platforms offer
  (CL1 has ~59 electrodes in all; MaxTwo has 32 stimulation units per well).
- The 120 µm stimulation reach and graded recruitment are assumptions. The model is not fitted to any real dish.
- Findings here are statements about this model. Each one is a hypothesis to test on a real culture, not a prediction.
