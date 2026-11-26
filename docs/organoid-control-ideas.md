# How organoids could control Asimov 1: ideas (20 November 2026)

## Where we stand

- **Direct joint control is the hardest possible job.** Keeping Asimov 1 up under 120–200 N pushes needs a precise, nonlinear 23-joint stepping policy, every 20 ms. A realistic culture's spikes don't carry the nonlinear features of the state that this needs. On the part of the action the last command doesn't explain, the culture reaches R² 0.21–0.23, a noise-free random-feature reservoir 0.64, and even every neuron of a well only 0.17.
- **What works in simulation:** the culture as the only sensory channel, read by a trained MLP decoder that also gets an efference copy of its own last command. It's honest, but the dish relays information rather than computing.
- **Stepping is required.** With zero action the robot falls from a single 160 N push in every direction, within 2 s. Stance-only controllers (ankle/hip, linear in tilt) plateau at ~5 s.
- **What cultures have demonstrably done** is low-rate pattern work:
  - telling a few stimulus patterns apart, and acting on them (DishBrain Pong);
  - separating two hidden sources (Isomura);
  - classifying speech features (Brainoware);
  - integrating over time, i.e. fading memory, which is the reservoir property.

  Nothing like continuous high-dimensional motor control at 50 Hz has been shown.

## Design principles for any idea

1. **Match the information rate.** A recovery decision ("step, which way, how far") is ~4–5 bits per push. Continuous 23-D targets at 50 Hz are hundreds of times more. Give the organoid jobs on the low-rate side.
2. **Ablation must hurt.** Remove or silence the dish, and the robot must fail.
3. **The no-culture control must not trivially win.** If the same decoder on the raw stimulation pattern matches the culture, the culture is a wire. Claim computation only when the culture beats that control: through memory, adaptation, or learned selectivity.
4. **Give the dish a privileged sense.** Route some information *only* through the organoid (e.g. vestibular/IMU signals), so the robot-side controller can't solve the task without it.
5. **Budget latency.** A push lasts 0.1 s, and the robot falls 0.5–2 s later. Reflex decisions have ~150–300 ms; continuous control has 20 ms.
6. **Play to biology's strengths:** adaptation, memory, pattern discrimination. Not precision, bandwidth or stationarity.

## Idea families

| Family | Ideas |
|---|---|
| **A. Change what the organoid outputs** | Discrete reflex choices; low-dimensional intention over learned skills; gain/stiffness modulation; residual corrections |
| **B. Give it a computation reservoirs are good at** | State estimation by temporal integration (CoM velocity, capture point); push detection/prediction; rhythm (CPG-like) entrainment |
| **C. Make the dish better at its job** | Topographic/population coding of a few variables; multi-well ensembles; HD-MEA spike sorting; optical interfaces (optogenetics plus calcium/voltage imaging); assembloids (cortico-spinal-muscle); pharmacological regime tuning; unsupervised pre-conditioning |
| **D. Change how it learns** | Closed-loop decoder adaptation (BCI-style); co-adaptation of decoder and culture; RL/ES fine-tuning on survival rather than imitation; twin pretraining, then fine-tuning on the real dish; curricula (stand, small pushes, big pushes) |

## Shortlist, best first

### 1. Reflex arc: the organoid detects the push and picks the recovery step

- **Robot side.** A stance controller (PD to the default pose) plus a library of K recovery-step primitives, e.g. 8 directions × 2 sizes. They're cut from teacher recovery episodes and replayed with light joint feedback. They must *not* recover on their own in the wrong direction, so the choice matters.
- **Organoid.** Gets IMU and joint signals, and outputs "which primitive, now", or "none". Decoded DishBrain-style: argmax over K+1 channel groups, or a linear classifier.
- **Why plausible.** ~4 bits per push, ~150–300 ms to decide. This is pattern discrimination, the thing cultures have done.
- **Ablation.** Without the dish, stance only, the robot falls at 160 N (measured). A wrong choice also falls.
- **Cheap sim tests** ((a) ran on 21 November, 38e96bf; see the result below):
  - (a) Build the primitives; measure survival with *oracle* choices plus injected error rate and delay. That gives the required accuracy and latency.
  - (b) Open loop: push-direction accuracy from culture features against time since the push, with the existing probe pipeline.
- **Hardware.** Fits CL1 (few electrodes, sub-ms loop).
- **Risk.** Brittle primitives under back-to-back pushes. Mitigation: feedback-stabilised primitives with a short re-decision window.

### 2. The organoid as vestibular integrator: it estimates CoM velocity / capture point

- **Key fact.** The robot's observation has **no base linear velocity**: gyro, gravity, joints and last action only. Velocity has to be inferred from history, which is a fading-memory task.
- **Robot side.** An analytic capture-point stepping controller: place the foot at x + v/ω.
- **Organoid.** Reservoir plus linear readout to a 2-D capture point or velocity. That's reservoir computing's home ground: low-dimensional output, temporal integration. In our features, the slow traces already carried most of the information.
- **Ablation.** A zero or stale estimate means wrong foot placement and a fall.
- **Honest claim.** A Kalman filter does this too. The culture only "computes" if it beats a linear filter of the raw signals. The organoid does a real, necessary computation, not a superior one.
- **Cheap sim tests:**
  - (a) Capture-point controller with *true* velocity: does it survive 160–200 N?
  - (b) Open loop: velocity-estimate R² from culture features against a linear-filter baseline.

### 3. High-level intention over a learned skill space (cortex over spinal cord)

- **Robot side.** A low-level policy trained in simulation with proprioception plus a 2–4-D latent z ("lean/step that way"). It never gets the IMU, so it can't balance alone (principle 4).
- **Organoid.** Gets the IMU, outputs z at 5–10 Hz.
- **Ablation.** Frozen or random z means a fall.
- **Cost.** Needs RL training of a latent-conditioned policy, which is heavy. This is the most scalable architecture, and it extends to walking with commands.

### 4. Adaptive layer (cerebellum-like) for perturbations

- **Robot side.** Runs a competent baseline. A perturbation (weak knee, payload, slippery floor) degrades it.
- **Organoid.** Adds bounded residual corrections or gain changes. The decoder is frozen; the culture learns from feedback stimulation (DishBrain predictable/unpredictable, or error signals like climbing fibres).
- **Why.** It's the biocomputing-native story: online adaptation by plasticity. It is the DishBrain arm of this repo.
- **Test.** Recovery after the perturbation with the decoder frozen, against a culture without plasticity.
- **Risk.** The simulated plasticity is crude, so the simulation can only rehearse the protocol, not predict the outcome.

### 5. Sensory channel plus trained decoder (current design): baseline and fallback

- Keep it as the reference.
- Improve it with more wells and HD-MEA spike sorting.
- Frame it as a "biological sensory interface".

### Longer term: make the dish compute more

- **Unsupervised pre-conditioning (Isomura-style).** Stimulate with the robot's latent causes (push directions, tilt states) until neurons become selective to them. Then a linear readout suffices, and the culture has *learned* a representation the raw input lacks.
- **Topographic coding.** Encode 2-D tilt or push direction as a moving "bump" across the electrode grid: high SNR, low-dimensional, and natural for local circuits.
- **Optical interfaces.** Optogenetic stimulation plus calcium/voltage imaging read thousands of neurons. That's the bandwidth fix.
- **Assembloids.** Cortical–spinal–muscle assembloids give the tissue a motor-like structure.
- **Regime control.** Background distributed stimulation and pharmacology to keep the culture input-driven rather than bursting.

## Comparison

| Idea | Organoid's job | Info rate | Ablation hurts | Culture computes something | Sim test cost | Hardware |
|---|---|---|---|---|---|---|
| 1 Reflex arc | Classify push and pick a step | ~4 bits per push | Yes | Pattern discrimination | Low | CL1 |
| 2 Vestibular integrator | Estimate velocity / capture point | 2 analog values at 50 Hz | Yes | Temporal integration | Low–medium | MaxTwo / CL1 |
| 3 Intention over skills | 2–4-D latent at 5–10 Hz | Low | Yes, by design | Sensorimotor mapping | High (RL) | Any closed-loop |
| 4 Adaptive layer | Learn corrections online | Very low, slow | Under perturbation | Plasticity | Medium | Any; FinalSpark possible (slow) |
| 5 Sensory channel | Transmit the state | High | Yes | No (relay) | Done | MaxTwo (multi-well) |

## Suggested order once the current run finishes

1. **Idea 1, test (a):** recovery primitives plus an oracle selector with injected errors. MuJoCo only, minutes. It sets the accuracy and latency the dish needs.
2. **Idea 2, test (a):** a capture-point stepping controller with true velocity. MuJoCo only.
3. If either passes, run its open-loop culture test (b) on the existing probe pipeline, then a closed-loop run.
4. Keep idea 4 as the DishBrain-arm experiment, and idea 5 as the baseline every result is compared against.

## Result: reflex-arc oracle test (21 November 2026, `scripts/reflex_oracle.py`)

20 seeds of `stand_push`, 20 s each. The library holds 16 primitives: 8 directions × 140/180 N, each the teacher's 1.5 s recovery from stance.

| Controller | Survival | Full 20 s |
|---|---|---|
| Stance only | 4.5 s | 0/20 |
| Teacher always on | 20.0 s | 20/20 |
| Gated teacher (stance; closed-loop recovery for 1.5 s after each detected push) | 20.0 s | 20/20 |
| Gated teacher, 200 ms detection delay | 19.3 s | 19/20 |
| Open-loop replay, perfect oracle choice, 0–300 ms delay | 4.6–4.7 s | 0/20 |
| Open-loop replay, 10 / 25 / 100 % wrong picks | 4.5 / 4.5 / 4.2 s | 0/20 |

- **A 1-bit trigger with a 200 ms deadline is enough, if the recovery is closed-loop.** Open-loop step replay fails even with perfect choices: a recovery step has to react to the state, it can't be recorded once.
- **But then the trigger is trivial** (a gyro threshold would do), and the closed-loop recovery does the work. To make the dish *necessary*, the robot-side controller must lack information that only the dish provides (principle 4).
- **Next cheap test: the "blind teacher".** Zero, degrade, delay or down-sample the teacher's IMU inputs (gyro, gravity) while its joint encoders stay exact. If it needs the IMU but tolerates a noisy or low-rate version, the cleanest design is **the organoid as vestibular organ**:
  - only the 6 IMU dimensions go through the dish, instead of 52 sensory dimensions;
  - the joint encoders go straight to the controller;
  - that's ~9× more electrodes per dimension, so much higher fidelity;
  - the dish stays necessary, because no IMU means no balance.

## Weight-space programming (csinva/weight-space-programming-autoresearch)

**What it is.** An agent (Claude Code) *writes* a model's weights in code, with no training. It loops: edit, run, read a single validation score, log, repeat for at least 20 iterations. It has built interpretable transformers and brain **encoding models** (fMRI, ECoG). Paper: arXiv 2611.18114.

Where it could help here, best first:

1. **An interpretable digital twin of the dish.** Write an encoding model from stimulation pattern to per-electrode response. This is the same kind of model the paper builds for fMRI/ECoG. Validation: held-out R² of evoked responses. It works on our simulated culture now (probe data exists), and on a real dish's channel-test recordings later. Bonus: the twin is readable (which electrodes respond to what, and through which nonlinearity), which SBI-fitted biophysical models don't give.
2. **Understanding the teacher.** Have the agent write a transparent approximation of the teacher's balance policy. That would identify *which* nonlinear features of the state the policy needs: the features we found the culture can't supply (additive per-dimension codes reach 0.49 innovation R², random features 0.64). This tells us exactly what to ask the organoid to compute, or what to give the robot side.
3. **Robot-side controllers that consume organoid signals.** For example, a closed-loop recovery controller conditioned on a dish-provided push direction or vestibular estimate, written rather than trained by RL. Validation: closed-loop survival, cheap in MuJoCo.
4. **Interpretable decoders.** Spikes to trigger, capture point or actions. Low value: plain training already works.

Limits:
- It can't add information the dish doesn't carry.
- The repo's scaffolding is transformer- and sequence-oriented, so borrow the *method* (program.md loop, `write_weights`, single score) rather than the code.
- Each search costs API credits and agent time.
