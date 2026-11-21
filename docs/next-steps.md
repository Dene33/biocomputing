# Next steps: from a sensory relay to an organoid that matters

21 November 2026. Context: [README, Status and findings](../README.md#status-and-findings-21-november-2026) and
[organoid-control-ideas.md](organoid-control-ideas.md).

## Where we are

**Task.** Asimov 1 stands for 20 s while 120–200 N pushes hit the pelvis every 3 s (`stand_push`).

**Current best design.**
- Six simulated organoids (wells) are the robot's only sensory path: 52 signals in, as stimulation.
- A trained MLP decoder reads their spikes, plus its own last command (efference copy), and drives all 23 joints.

**Result.**

| Configuration | Survival | Full 20 s |
|---|---|---|
| Culture + decoder | 18.5 s | **17/20** |
| Same decoder, no culture | 17.5 s | 15/20 |
| Dish silenced | 1.6 s | 0/10 |

**What's wrong with it.**
1. **It needs too much hardware:** 312 stimulation electrodes at 200 Hz, more than CL1 or MaxTwo offer.
2. **Fidelity is low:** 52 signals carried at R² ≈ 0.7.
3. **The dish computes nothing:** no culture does as well. The decoder does the work.

## Aim

A design where the dish:
1. carries **few signals at high fidelity**;
2. fits **real hardware** (≤ 32 stimulation channels per well);
3. is **necessary**: silence it and the robot falls;
4. eventually **computes** something a linear filter of its inputs can't.

**Lead candidate: the organoid as vestibular organ.**
- Only balance signals go through the dish: the 6 IMU values (gyro and gravity direction), or a derived estimate such as the capture point (where to step to stop a fall).
- Joint encoders and the efference copy go straight to a robot-side controller.
- That controller is **written with weight-space programming (WSP)**: an agent writes its weights in code, guided by one validation score. The method is from [csinva/weight-space-programming-autoresearch](https://github.com/csinva/weight-space-programming-autoresearch).

## Phase 1: write the spec for the dish (CPU, ~1–2 h)

### 1.1 Blind-teacher sweep

Run the teacher on `stand_push` (20+ seeds) while degrading **only its IMU inputs**: observation indices 0–5, gyro and gravity. Joint positions, joint velocities and the last action stay exact.

| Degradation | Levels |
|---|---|
| Removed | gyro and/or gravity set to their standing values |
| Noise | 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 × each signal's spread; independent and temporally correlated |
| Delay | 1–5 control steps (20–100 ms) |
| Reduced rate | held for 2, 3, 5 steps (25, 17, 10 Hz) |
| Quantisation | 2, 4, 8 levels per signal |

For comparison, run the reverse: joints degraded, IMU exact.

**Output:** `outputs/blind_teacher/results.json` and a tolerance table: the accuracy, latency and rate the dish must deliver.

**Result** (`scripts/blind_teacher.py`, 40 fresh seeds, 20 s, on top of the loop's 1-step action latency; "tolerated"
means ≥ 34/40 full runs, the 17/20 bar of Phase 3.3). Teacher 20.0 s, 40/40; zero action 4.8 s, 0/40.

The spread over teacher `stand_push` rollouts is small: gyro 0.11–0.15 rad/s (6–8 °/s), gravity x/y 0.019–0.026
(1.1–1.5° of tilt).

| The dish must deliver | IMU (dish) | Breaks at | Joints (reverse test) |
|---|---|---|---|
| Presence | removed: 1.0 s, 0/40 (gravity alone 1.6 s, gyro alone 10.7 s) | — | removed: 6.5 s, 0/40 (positions alone 9.7 s; velocities alone 36/40) |
| Accuracy, white noise | ≤ 1.0 × spread (37/40) | 1.5 × (32/40) | ≤ 1.5 × (36/40) |
| Accuracy, noise with τ = 200 ms / 1 s | ≤ 1.0 × (38/40, 36/40) | 1.5 × (33, 31/40); 2.0 × (16, 10/40) | ≤ 1.0 × (36, 36/40) |
| Latency added | ≤ 3 steps, 60 ms (39/40) | 4 (28/40), 5 (1/40) | ≤ 2 steps (38/40); 3 (30/40) |
| Rate (sample and hold) | ≥ 10 Hz (38/40) | 5 Hz (11/40) | ≥ 10 Hz (39/40) |
| Quantisation | ≥ 8 levels (35/40) | 4 (25/40) | ≥ 4 levels (38/40) |

- **The gyro sets the latency limit.** 100 ms of delay: gyro alone 20/40, gravity alone 35/40. Gravity tolerates 200 ms
  (35/40).
- **Combined degradations compound.** A dish delivers all of them together:

| IMU: delay + rate + noise (τ = 200 ms) + quantisation | Full 20 s |
|---|---|
| 40 ms + noise 0.5 × | 40/40 |
| 40 ms + 25 Hz + noise 0.5 × | **39/40** |
| 40 ms + 25 Hz + noise 1.0 × | 29/40 |
| 60 ms + 25 Hz + noise 0.5 × | 30/40 |
| 40 ms + 25 Hz + noise 0.5 × + 8 levels | 14/40 |
| 20 ms + 17 Hz + noise 0.7 × + 8 levels | 6/40 |

**Spec for the vestibular channel:** ≤ 40 ms added latency, ≥ 25 Hz, error ≤ 0.5 × spread (decoding R² ≥ 0.75 if the error
is unbiased), graded output. Latency costs more than noise, so the decoder must not smooth over long windows.

### Gate G1

- **The teacher falls without IMU, but tolerates a usable degradation** (e.g. ≥ 0.3 × noise, or ≥ 2 steps of delay): go to the vestibular design.
- **Joint encoders alone suffice:** the IMU isn't privileged. Either hide more from the robot side (e.g. joint velocities) until the dish's signal is necessary, or switch to the trigger or integrator designs in [organoid-control-ideas.md](organoid-control-ideas.md).

**Verdict: pass, with one caveat.**
- Without the IMU the teacher falls faster than zero action (1.0 s against 4.8 s).
- It tolerates usable degradation: 1.0 × noise, 60 ms of delay, 10 Hz, each alone.
- **Caveat:** this shows the fixed teacher *uses* the IMU, not that a controller retrained without it would fall. With
  both feet flat, leg kinematics give the pelvis tilt.
- The joints aren't dispensable either (removed: 6.5 s), but joint velocities are (36/40 without them).

**Caveat checked: students retrained without the IMU** (`scripts/no_imu_student.py`).
- Recipe: the 17/20 decoder's, fed the robot state directly with no culture. MLP 2 × 256 plus a linear path from the
  last action, clipped labels, recoveries weighted × 5, 16 DAgger rounds × 16 episodes, ensemble of the last 4.
- Test: 40 fresh seeds (the sweep's).

| Student inputs (always + last action) | Survived | Full 20 s | IMU removed at test |
|---|---|---|---|
| IMU + joint pos + joint vel (control) | 20.0 s | **40/40** | 1.4 s, 0/40 |
| IMU + joint pos (the vestibular design) | 19.1 s | **35/40** | 1.4 s, 0/40 |
| Joint pos + joint vel | 2.3 s | 0/40 | — |
| Joint pos | 1.5 s | 0/40 | — |
| Joint pos + vel + their last 4 steps (memory) | 2.1 s | 0/40 | — |

- **Without the IMU no student learns to stand**, even with memory. None survived past 4.9 s in any round, worse than
  zero action (4.8 s).
- **With it, the recipe matches the teacher.** Joint velocities can go: IMU plus joint positions reaches 35/40.
- So for imitation-trained controllers **the IMU is necessary**. A silenced vestibular dish should be fatal, as Phase
  3.3 requires.
- **Not ruled out:** a controller built some other way (RL, or written by hand) that balances on kinematics alone.
  Phase 2 track (b) should check that its controllers still fall without the dish signal.

## Phase 2: weight-space programming on the robot side (CPU + agent loop)

### 2.1 Harness

Borrow csinva's *method*, not their transformer code.

| File | Role |
|---|---|
| `wsp/controller/program.md` | Task, rules, the loop (edit → run → read score → log → repeat, ≥ 20 iterations) |
| `wsp/controller/controller.py` | `write_weights()`: a small numpy controller, [joint positions, joint velocities, last action, vestibular signal] → 23 actions |
| `wsp/controller/eval_controller.py` | Runs `stand_push` in MuJoCo with the vestibular signal corrupted by the dish's noise model (from 1.1, later from Phase 3). Prints **one score**: mean survival on validation seeds |
| `wsp/controller/results.csv` | One row per attempt |

**Evaluation.**
- Validation seeds visible to the agent; a separate **hidden test set**, scored only at the end.
- ~30–60 s per evaluation on 8 CPU workers.

**Rule.** No gradient training inside the agent loop, so the controllers stay readable. DAgger-trained students are the baselines, not part of the search.

### 2.2 Two search tracks

- **(a) Edit the teacher's weights.** This is WSP in the original sense. Adapt the 208k-parameter teacher (78 → 512 → 256 → 128 → 23) so it works when its IMU inputs come from the dish (noisy, delayed or slower).
- **(b) Write a transparent controller from scratch.** Slower, but it shows *which features of the state* balance needs. We couldn't answer that by analysis: additive per-signal features explain 0.49 of the hard part of the action, random nonlinear features 0.64.

### 2.3 Baselines

- The teacher with the same IMU degradation (from 1.1).
- A DAgger MLP student trained with the same IMU noise.

**Cost:** API credits, roughly 30–50 agent iterations per track.

## Phase 3: the vestibular organoid (GPU, ~1–2 days)

### 3.1 Open-loop probe

Uses the existing pipeline (`outputs/probe/encoding.py`).
- Encode only the 6 IMU values, each spread over several electrodes: staggered thresholds, or a moving "bump" across the grid.
- Measure decoding R² for 1, 2 and 6 wells.
- Budget: **≤ 32 stimulation channels per well** (MaxTwo-compatible); also a CL1-sized variant with ~12.
- Compare against the Phase 1 spec.

**Result** (`scripts/vestibular_probe.py`; `outputs/vestibular_probe/`).
- **Setup.** 19k steps of teacher `stand_push` observations, with DART action noise, replayed open loop. Only
  `obs[0:6]` is stimulated (4 pulse slots per 20 ms step, 120 µm reach, gain 0.5).
- **Decoder.** Ridge from spikes to the IMU, 5-fold contiguous cross-validation.
- **Encodings.** `axis`: each electrode tuned to one IMU dimension, sign alternating, thresholds staggered.
  `select`: random directions, random thresholds.
- **Scoring.** Errors in units of the teacher's spread, on in-range steps: every IMU value within 3 spreads of
  standing, 35% of the replay. `grav_z` barely moves, so it is rebuilt from x and y.

| `axis`, decoder with 40 ms trace | Gyro error (× spread) | Gravity x, y error | Gain (shrinkage) |
|---|---|---|---|
| 1 well × 12 electrodes (CL1-sized) | 0.56–0.70 | 0.53–0.80 | 0.61–0.70 |
| 1 well × 32 (MaxTwo-sized) | 0.46–0.63 | 0.47–0.71 | 0.84–0.94 |
| 6 wells × 12 | 0.37–0.55 | 0.41–0.67 | 0.79–0.99 |
| 6 wells × 32 | 0.27–0.45 | 0.33–0.58 | 0.95–1.09 |

- **Decoder memory.** Spike counts of the current step alone are about 1.5 × worse. The 160 ms trace adds nothing.
  The gyro decodes best at lag 0, so the 40 ms trace costs no measurable delay.
- **Encoding.** `select` is within ±0.05 of `axis`.
- **Ceiling.** Decoding the stimulation pattern itself gives 0.05–0.2 × spread, so most of the error comes from the
  culture, not the encoding.
- **Noise correlation time:** 2–3 steps for the gyro, 4–8 for gravity.

**The measured channels in closed loop** (`wsp/controller/dish_models/`). Each configuration becomes a dish model
(per-dimension gain plus correlated noise, no added delay), and the teacher runs through it. Validation: 60 seeds.

| Dish | Teacher unchanged | Track (a): IMU inputs × 1/gain |
|---|---|---|
| 6 wells × 32 | 20.0 s, 60/60 | — |
| 6 wells × 12 | 19.9 s, 59/60 | — |
| 1 well × 32 | 19.6 s, 58/60 | — |
| 1 well × 12 (CL1) | 12.7 s, 20/60 | **19.8 s, 58/60** |
| 1 well × 12, **hidden test** (40 seeds) | 13.7 s, 16/40 | **20.0 s, 40/40** |

Zero action on the hidden test: 4.4 s, 0/40. With the dish silenced, every teacher variant falls in 1.0 s.

- A single CL1-sized well shrinks the IMU to ~63% of its size. One weight edit undoes it: scale the teacher's
  first-layer IMU columns by 1/gain.
- **Caveat:** these channels are statistical summaries of open-loop replays, with a decoder fitted afterwards. The
  real test is Phase 3.3: the culture in the loop, the decoder fixed in advance.

### 3.2 Weight-space programming on the dish side

- **Interpretable decoder:** spikes → IMU estimate.
- **Encoding model:** stimulation pattern → electrode responses. This is our first readable digital twin of the dish, the same kind of model the WSP paper builds for fMRI and ECoG.
- Score: held-out R². Ridge regression is the baseline.

### 3.3 Closed loop

The dish plus the Phase 2 controller, on 20+ fresh seeds. Ablations:
- dish silenced;
- ideal IMU (no culture);
- dead culture **with its own retrained decoder** (fixes the confound in today's dead-culture result).

**Success:** ≥ 17/20 survive, silenced dish 0/N, ≤ 32 stimulation channels per well.

## Phase 4: make the dish compute (the real claim)

### 4.1 Vestibular integrator

Decode what the robot's sensors lack: **linear velocity** or the **capture point**. Both need memory over time. The robot side then uses it, e.g. through a capture-point stepping controller.

It counts as computation only if:
- the dish beats a linear filter of the same inputs; and
- the live culture beats the dead one, each with its own decoder.

### 4.2 Adaptation (the DishBrain arm)

- Weaken a knee or add a payload.
- Freeze the decoder; let feedback stimulation drive plasticity.
- Compare plasticity on vs off.
- Needs homeostatic synaptic scaling first: plain STDP runs away.

## Phase 5: hardware readiness (in parallel, when access exists)

1. **Channel test on a real dish.** Replay recorded IMU sequences open-loop, record, fit the decoder offline, compare against the Phase 1 spec. It needs no real-time loop, so it could even run remotely on FinalSpark.
2. **Calibrate the simulator** to the real dish with the Phase 3.2 encoding model: stimulation reach, gain, noise.
3. **Device adapter** (CL API or MaxLab) implementing the `Biocomputer` protocol in `src/biocompute/device.py`.

## Rules throughout

- **Evaluation.** 20+ fresh seeds per claim; fixed validation seeds, plus hidden test seeds for anything an agent optimises.
- **Baselines.** Every result is reported against the teacher, zero action (default pose), no-culture and silenced-dish.
- **Claims.** "Computation" only when the dish beats the no-culture control.
- **Runs.** Long runs are detached (`setsid nohup …`) and checkpoint every round.

## Order and decision points

1. **Phase 1.1**, then Gate G1. Short, CPU.
2. **Phase 2** (CPU and agent) **and Phase 3.1** (GPU) in parallel.
3. **Phase 3.2–3.3** once both deliver.
4. **Phase 4** after the vestibular closed loop works.
5. **Phase 5** whenever hardware access appears. The channel test can start at once.
