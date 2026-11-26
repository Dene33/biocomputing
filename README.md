# Biocomputing: a digital organoid driving Asimov 1

We want to control a humanoid robot (Menlo Research's Asimov 1) with a biocomputer: living neurons grown on a multi-electrode array (MEA), as offered by Cortical Labs (CL1), FinalSpark (Neuroplatform) or MaxWell Biosystems. We have no culture yet, so this project simulates one. It is a **digital organoid** that the robot reaches only as it would reach real hardware: by stimulating electrodes and reading the spikes they detect.

The culture drives **all 23 joints of Asimov 1 at 50 Hz** in MuJoCo. Two ways to make that work are built and compared:

- **(A) Reservoir.** The culture is a fixed dynamical system. A linear readout from electrode activity to joint actions is trained to imitate the trained walking policy, by DAgger and ridge regression (as in Brainoware, *Nature Electronics* 2023).
- **(B) DishBrain feedback.** The decoder is fixed and only the culture learns, through STDP shaped by feedback. Predictable stimulation follows good outcomes and unpredictable stimulation follows falls (Kagan et al., *Neuron* 2022).

## Layout

| Path | What |
| --- | --- |
| `src/biocompute/device.py` | `Biocomputer` protocol (`stimulate`, `step` → `SpikeBlock`) and `SimulatedBiocomputer`. A real CL1/FinalSpark adapter implements the same protocol. |
| `src/biocompute/culture/network.py` | `OrganoidCulture`: Izhikevich neurons (Izhikevich 2003; 80 % excitatory RS/IB/CH, 20 % inhibitory FS/LTS), a flattened organoid lying on the array, distance-dependent sparse connectivity with axonal delays, Tsodyks–Markram short-term depression, Poisson background, optional slow NMDA-type excitation (`nmda_ratio`, gated by the magnesium block, Jahr & Stevens 1990). `wells=K`: K independent organoids, as on a multi-well plate. Torch, CPU or GPU (synaptic delivery as one sparse matrix–vector product per 0.5 ms substep), seeded. |
| `src/biocompute/culture/plasticity.py` | `STDP` (additive, or weight-dependent with `multiplicative=True`; optional inhibitory plasticity with `istdp_eta`, Vogels et al. 2011), `RewardSTDP` (eligibility trace plus a global reward, Izhikevich 2007: an in-silico upper bound, since no such signal can be delivered through electrodes), forgetting towards baseline weights, `SynapticScaling` (per-neuron homeostatic scaling, Turrigiano 2008, bl1's rule with two fixes: each neuron's own resting rate as target, and the exact exponential step instead of bl1's Euler step, which goes negative far above target), `Chain` (several rules in order), rest |
| `src/biocompute/culture/aging.py` | Excitability drift and cell loss over days |
| `src/biocompute/mea/electrodes.py` | MEA: multi-unit detection with distance-dependent probability, false positives, dead channels, stimulation that falls off with distance, rate and amplitude limits, artifact blanking. Presets `mcs60` (60 channels, 200 µm), `grid256`, `hd1024` (MaxWell-like). One grid per well. |
| `src/biocompute/interface/` | Encoders (observation → pulses; amplitude or rate coding; identity, PCA, supervised or random projection), spike features, ridge readout (`LinearReadout`, GPU for thousands of features), trained nonlinear decoder (`MLPReadout`), the fixed motor-map decoder, DishBrain feedback (Kagan et al. 2022: predictable stimulation, 100 Hz for 100 ms on every sensory electrode, for staying up; unpredictable, 5 Hz for 4 s at random electrodes, for a fall) |
| `src/biocompute/robot/asimov.py` | `AsimovEnv`: Asimov 1 stepped one 20 ms control step at a time, built on cyclotron's `sim2sim` |
| `src/biocompute/robot/teacher.py` | The trained walking policy (ONNX): the teacher and the ceiling |
| `src/biocompute/loop.py` | The closed loop. The culture needs 20 ms of its own time to answer, so an action applies one step late. |
| `src/biocompute/train/` | Tasks, DAgger (serial, and `parallel.py`: several copies of the same dish in worker processes), feedback sessions |
| `scripts/` | `characterise_culture.py`, `baselines.py`, `train_reservoir.py`, `train_dagger.py` (parallel DAgger, ridge or MLP decoder), `eval_readout.py` (many seeds, ablations), `train_feedback.py`, `evaluate.py`, `view.py`, `train_expert.py` / `program_expert.py` (stance-only balance experts: they cannot pass `stand_push`), `run_suite.sh` (the reservoir comparison: every readout source and MEA size in parallel) |
| `scripts/` (experiments) | `reflex_oracle.py`, `blind_teacher.py`, `no_imu_student.py`, `vestibular_probe.py`, `trigger_probe.py`, `adaptation_slot.py`, `struggle_signal.py`, `coach_probe.py`, `dish_regimes.py`, `burst_probe.py`, `cartpole_replication.py`, `cebra_probe.py`, `bl1_drift.py` (with `bl1_checks.py`, `bl1_rep_analysis.py`; runs in bl1's own venv), `closed_loop_vestibular.py` (with `closed_loop_falls.py`), `closed_loop_sessions.py`, `public_data_targets.py`, `real_drift.py`, `realism_scorecard.py` (with `burst_spread.py`), `takeover_rerun.py`, `weak_synapses.py`, `day_drift.py`, `real_pathways.py`, `organoid_scorecard.py` (the last three on FinalSpark's exports, read by `finalspark_data.py`): one experiment each; results in [docs/route-to-computation.md](docs/route-to-computation.md) and the commit log |
| `scripts/launch.py`, `scripts/watch.py`, `src/biocompute/runguard.py` | Guarded long runs: smoke test, detached systemd unit, in-run checks, watchdog, result files named by their settings (`job_key`) |
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

**Data from FinalSpark.** The whole-life datasets fs369 and fs437 come from FinalSpark's Neuroplatform
(finalspark.com; Jordan et al. 2024, *Front. Artif. Intell.* 7:1376042). They are confidential and used here for
research with FinalSpark cited as the data source. Neither the datasets nor the per-day, per-session or
per-electrode tables derived from them are in this repository (`outputs/public_data/raw/finalspark/` and the derived
outputs are gitignored, and a test fails if one is tracked). Only summary results appear here and in the commit log.

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
stops the unit on an error, a failed in-run check (`biocompute.runguard.Guard`) or a stall. Its exit code is the
alarm: 3 for an error or failed check, 4 for a stall (no log output or heartbeat for `--stall-min`), 0 for a clean
end, 2 for an unclean one. `Guard.culture` checks that the state is finite, weights stay within [0, cap] and don't
pile up at any other value, and electrode rates are plausible:

```sh
uv run scripts/launch.py --unit bio-X --log outputs/X/run.log -- scripts/foo.py <args>
uv run scripts/watch.py --unit bio-X --log outputs/X/run.log
```

`train_dagger.py` checkpoints every round and `cartpole_replication.py` saves when stopped; both resume when the same
command is run again. Finished jobs and checkpoints are named by every setting they depend on
(`biocompute.runguard.job_key`), so a rerun with a setting changed computes anew instead of reusing other settings'
results; code changes are not in the name, so after one, run into a fresh `--out`. Runs repeat bit for bit: the same
seed gives the same result on the GPU, also across a stop and resume. Sums go through ordered operations
(`OrganoidCulture.excitatory_input`, the delivery matrix), and a test fails on any new atomic float sum.

The code carries no docstrings or prose comments: facts live in the code, the commit messages and these .md
files, and no new .md file is added. A test fails on any docstring, on any comment other than `# noqa` or a reviewed
one-line warning (`REVIEWED_COMMENTS` in `tests/test_scripts.py`), and on any .md file beyond the existing ones.

The commands behind the committed experiment results. Each run's numbers are in its commit, cited in the route doc's
results table. Each script's original description (design, gates, controls, results) was taken out of the code
after commit d3c85c4 and stays readable with `git show d3c85c4:<path>`. Long runs use the `launch.py` / `watch.py` pair above with the same unit and log, and
`systemctl --user stop <unit>` stops one gracefully; relaunching the same command resumes it.

```sh
uv run scripts/blind_teacher.py; uv run scripts/no_imu_student.py                    # Phase 1.1, Gate G1
uv run scripts/vestibular_probe.py; uv run scripts/trigger_probe.py collect           # Phase 3.1, push trigger
uv run scripts/trigger_probe.py run --n-stim 32 --wells 6 --encoding axis
uv run scripts/adaptation_slot.py; uv run scripts/struggle_signal.py                  # Phase 4 sims
uv run scripts/coach_probe.py; uv run scripts/coach_probe.py --dose --seeds 0 1
uv run scripts/dish_regimes.py; uv run scripts/dish_regimes.py --nmda 0.4 --live-only
uv run scripts/burst_probe.py
uv run scripts/launch.py --unit bio-cebra --log outputs/cebra_probe/run.log --with cebra -- scripts/cebra_probe.py
uv run scripts/cartpole_replication.py                                                # cart-pole protocol
uv run scripts/launch.py --unit bio-cp --log outputs/cartpole_replication/run.log -- \
    scripts/cartpole_replication.py --drift --drift-scales 1 --drift-pairs 0.01:30 --drift-seconds 600
uv run scripts/launch.py --unit bio-bl1-rep --log outputs/bl1_drift/rep/run.log -- \
    --project /mnt/datadisk/repos/bl1 --no-sync python scripts/bl1_drift.py --variant shipped fixed \
    --seed 42 43 44 45 46 --pulse-ms 20 0.5 --controls --workers 3 --root outputs/bl1_drift/rep
uv run --project /mnt/datadisk/repos/bl1 --no-sync python scripts/bl1_checks.py      # the port against bl1's code
python3 scripts/bl1_rep_analysis.py
# Track A
uv run scripts/launch.py --unit bio-a1 --log outputs/closed_loop_vestibular/run.log -- \
    scripts/closed_loop_vestibular.py --workers 3                                     # A1, run 1
#   A1b: the same command plus --edit affine --seed-base 9100 --out outputs/closed_loop_vestibular_affine
uv run scripts/closed_loop_falls.py --traces                                          # A1b falls (GPU, ~3 min)
uv run scripts/launch.py --unit bio-a2 --log outputs/closed_loop_sessions/run.log -- \
    scripts/closed_loop_sessions.py --workers 3                                       # A2
uv run scripts/launch.py --unit bio-a2c --log outputs/closed_loop_sessions_refit/run.log -- \
    scripts/closed_loop_sessions.py --arms refit60 refit30 refit10 recal ideal_imu zero_action \
    --seed-base 9300 --out outputs/closed_loop_sessions_refit --workers 3              # A2c
uv run scripts/launch.py --unit bio-a2d --log outputs/day_drift/run.log -- python -I scripts/day_drift.py --workers 3
uv run scripts/launch.py --unit bio-a2d-cal --log outputs/day_drift/run_calibrate.log -- \
    python -I scripts/day_drift.py --calibrate --workers 3                            # A2d part 2
uv run scripts/launch.py --unit bio-a2d-run --log outputs/closed_loop_sessions_a2d/run.log -- \
    scripts/closed_loop_sessions.py --conds day0 day1_down day1_up day3_down day3_up day7_down day7_up \
    day7_stress --arms refit60 stale ideal_imu zero_action --seed-base 9400 \
    --out outputs/closed_loop_sessions_a2d --workers 3                                # A2d
# Track B and public data
uv run python -I scripts/public_data_targets.py fetch analyze                        # B0 (downloads are gitignored)
uv run scripts/launch.py --unit bio-b1 --log outputs/drift_scaling/run.log -- \
    scripts/cartpole_replication.py --drift --drift-seeds 0 1 2 --drift-scales 0 1 \
    --drift-homeo 0 1e-5 1e-4 --drift-seconds 600 --workers 4 --out outputs/drift_scaling        # B1
uv run scripts/launch.py --unit bio-b1m --log outputs/drift_scaling/run_multiplicative.log -- \
    scripts/cartpole_replication.py --drift --stdp-rule multiplicative --drift-seeds 0 1 2 \
    --drift-scales 0 1 --drift-homeo 0 1e-5 1e-4 --drift-seconds 600 --workers 4 --out outputs/drift_scaling   # the check
uv run scripts/launch.py --unit bio-realdrift --log outputs/real_drift/run.log -- python -I scripts/real_drift.py
uv run scripts/launch.py --unit bio-takeover --log outputs/takeover_rerun/run.log -- scripts/takeover_rerun.py --workers 3
uv run scripts/launch.py --unit bio-real --log outputs/realism_scorecard/run.log -- scripts/realism_scorecard.py --workers 3
uv run python -I scripts/burst_spread.py model; uv run python -I scripts/burst_spread.py wagenaar
uv run scripts/launch.py --unit bio-weak --log outputs/weak_synapses/run_realism.log -- \
    scripts/weak_synapses.py --part realism --workers 3                               # step 4 (part 2 never ran)
uv run python -I scripts/real_pathways.py; uv run python -I scripts/real_pathways.py --fs fs437
uv run python -I scripts/organoid_scorecard.py                                       # organoid column, part A
uv run scripts/launch.py --unit bio-orgsc --log outputs/organoid_scorecard/run.log -- \
    python -I scripts/organoid_scorecard.py --evoked --workers 3                      # part B
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
- network bursts every ~2–3 s (interval CV ≈ 0.3; measured against real cultures, the `gpu_20k` culture bursts
  every 1.4 s and too regularly: see the realism scorecard under Next);
- most spikes inside bursts (~75 %).

Against FinalSpark's organoids (26 November, `scripts/organoid_scorecard.py`, scored on organoid-sized blocks of
8 electrodes) the `gpu_20k` culture is in range on none of five measures: it fires 5.3 Hz against 1.6, bursts 39 times
a minute against 0.85, for 41 ms against 752, regularly (CV 0.32 against 1.08), with 80% of spikes in bursts against
35%. A single pulse drives only the electrode it is given on; in the organoids it reaches 2–3 neighbours (11–17 of 56
electrode pairs respond, about 1.4 spikes each).

It also responds to single-electrode stimulation. Bursts end because short-term depression uses up synaptic resources, as in culture models. These are plausible ranges, not a fit to one dataset: reports for organoids (e.g. Trujillo et al., *Cell Stem Cell* 2019) and dissociated cultures (Wagenaar et al., *BMC Neuroscience* 2006) vary with age and protocol. The realism dials are neuron count, electrode count and pitch, detection radius, noise, dead channels, stimulation limits, latency, forgetting time constant and aging.

## Status and findings (26 November 2026)

Built: everything in the layout above, with 59 passing tests. Teacher parity with cyclotron is exact.

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
| Teacher (ceiling; stored runs on other seeds: `outputs/blind_teacher/results.json`) | 20.0 s | 40/40 |
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

**Phase 3.3, the culture in the loop** (track A1, `scripts/closed_loop_vestibular.py`, 40 fresh seeds per arm). The
simulated culture now carries the IMU inside the robot's loop, with its decoder and weight edit fixed beforehand.
On a CL1-sized well (12 stimulation electrodes) the gate (≥ 34/40) is not reached. The planned 1/gain edit gives
0/40, because the robot stands leaning 3.6° forward and the decoder's offset becomes a false lean. An edit that
also removes the offset gives 23/40. The other 17 fall after pushes from the front: the relay misreads the pitch rate
during the recovery, and the forward lean grows past the encoder's input limit (`closed_loop_falls.py`). A well with
32 electrodes reaches 40/40. Silencing the dish is fatal (0/40), and a dead culture does as well or better
(1 × 12: 32/40), so the dish is a necessary relay that computes nothing.

**Sessions and drift** (track A2, `scripts/closed_loop_sessions.py`, 1 × 32). A new dish breaks the old decoder
completely (0/40), and one 60 s open-loop calibration plus one weight edit brings it back to 36/40. Aging the
culture by up to 7 days doesn't break the old decoder (36–39/40), so the gate (recalibration ≥ 34/40 in every
session) was tested only on the new dish. It fails by one episode at day 3 (33/40). The 60 s decoder is noisier than
the one fitted on 5 minutes (36/40 even at day 0), so the calibration's length is the weak point, not the aging.
A2c (25 November) keeps the old decoder and refits only its edit (gain and offset per IMU channel) from the session's
replay: from 60 s it holds 37–40/40 in every session, the new dish 37/40, against 30–36/40 for the recalibrated
decoder on the same seeds. 30 s is as good; 10 s leaves two new-dish gains loose (36/40). A session therefore needs
two numbers per channel, not a new decoder (shown on one new dish).
A2d (26 November) asks whether that holds when the culture changes as much as real organoids do. FinalSpark's
whole-life recordings (`scripts/day_drift.py`) change far more from day to day than our `aging.py`: each electrode's
evoked response by ×1.13 in a day and ×1.6 in a week (`aging.py`: ×1.04 and ×1.16), and organoids' spontaneous rates
fall ×2.9 in a week where the model's rise. With a shift of every neuron's excitability calibrated to the real evoked
change (1, 3 and 7 days, up and down, plus a stress dose), the 60 s edit still holds 35–40/40 in every session.
Without it, the old decoder breaks only where the culture became more excitable (day 7: 19/40).
Public data for track B (`scripts/public_data_targets.py`): 81 recordings of 29 dense cortical cultures from
Wagenaar et al. 2006, and the cart-pole organoids' Zenodo record, give the ranges the model must match. The
organoids' game logs also give the rate half of a real drift test (`scripts/real_drift.py`): under the input pulses
alone, real rates don't climb. Whether real pathways stay put can't be read from them. FinalSpark's whole-life
exports (received 26 November; data: FinalSpark, confidential, see Setup) add organoids with spike times
and every stimulation, but they can't answer it either (`scripts/real_pathways.py`): no stimulation repeats in a
usable way. Two logged sessions never reached the array at all, so every use checks for the stimulation artifact.

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
(`bl1_drift.py`). The same scaling in our model (`SynapticScaling`, track B1) holds rates at rest on its own, but
with our STDP on it doesn't stop the runaway (6–10× rest at bl1's rate). With weight-dependent STDP and ten times
bl1's rate it brings rates to the edge of the band (1.3–1.5×), yet one input still takes over both outputs in every
culture. Ranked routes: coached adaptation first (fits every platform, cart-pole precedent; a real dish decides),
push detection tested open loop alongside it, the inner-ear relay as hardware testbed.

**Next** (26 November 2026). We have no real culture and no physical robot for now, so the work runs in two
simulated tracks ([Order of work without a dish](docs/route-to-computation.md#order-of-work-without-a-dish)).
Track A puts the simulated culture in the robot's loop (Phase 3.3 of [docs/next-steps.md](docs/next-steps.md));
its steps A1, A2, A2c and A2d have run (above).
Track B tried to make the culture model good enough to ask learning questions of, behind gates fixed in advance.
It closed after its first step, a rate controller (B1): rates can be brought to the edge of the band, pathways
can't, so learning questions wait for a dish. On 25 November a code review of track B's scripts found 13 bugs; all
are fixed with tests, every affected run was repeated, and no verdict changed.
Real-dish and real-robot steps are parked. Ideas behind it: [docs/organoid-control-ideas.md](docs/organoid-control-ideas.md).
Four recent papers suggest a cheaper session recalibration for track A2 (keep the decoder, refit only the edit) and
a test of why track B's pathways mix. The notes are in [docs/literature-notes.md](docs/literature-notes.md). The
simulated steps they lead to, in order: A2c (refit only the edit; passed), a takeover rerun with a weight dump, and
a realism scorecard of the model ([Order](docs/route-to-computation.md#order)). The scorecard (`scripts/realism_scorecard.py`)
puts the model in range on 3 of 6 measures against 29 Wagenaar cultures: rates, burst duration and the share of
spikes in bursts are realistic, but bursts come ten times too often (45 per minute against 4.2), too regularly
(interval CV 0.32 against 0.87), and a pulse sets off one three times too often (20% against 6.5%; that measure doesn't subtract chance, and at
the model's burst rate a 200 ms window holds a burst onset 12–14% of the time by chance, so much of the 20% may be
spontaneous: not tested). The takeover rerun
confirmed why one input takes over: links pointing away from the winning input strengthen and those pointing back
weaken, centred exactly on that input in all three cultures, so its pulses come to set off activity over the whole
array while the other input's stay local (`scripts/takeover_rerun.py`). Track B was reopened for one round, many weak
synapses: the same total input per neuron spread over 1.9, 4 or 10 times as many synapses (`scripts/weak_synapses.py`).
It failed on 26 November, because the model goes silent: there was not one network burst in 10 minutes in 15 of 15
cultures, and the electrode rate fell to 0.54-0.56 Hz (real: 0.7-10.3). Track B is closed for good.

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
