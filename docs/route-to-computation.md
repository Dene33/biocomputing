# The route to a biocomputer that computes and drives the real robot

22 November 2026. Builds on [next-steps.md](next-steps.md) and the README's status. Each claim cites the script and
commit that produced it. Simulated results are statements about our model; each is a hypothesis for a real dish.

## The bar

A route has to clear all five:

1. **Necessary.** Silence the dish and the robot does worse.
2. **Computes.** The dish adds something its input does not already contain. It must beat a simple stand-in fed the
   same input (a filter of the input, or the stimulation pattern itself) and a control culture (dead, frozen, or
   given random feedback). A trained decoder must not do the work: readouts stay simple and fixed.
3. **Fits real hardware.** Electrode count, stimulation rate, loop latency, culture lifetime and drift.
4. **Drives the real robot.** In Asimov's loop, safely.
5. **Has precedent in real cultures**, so the simulation is not the only evidence.

## Real hardware

| Platform | Electrodes | Closed-loop latency | Fits |
|---|---|---|---|
| Cortical Labs CL1 | ~59 usable (third-party figure; ~800k neurons) | sub-millisecond (CL API) | local fast loop; every route |
| MaxWell MaxTwo / MaxOne | 26,400 per well; MaxOne: 1,020 recording, 32 stimulation channels | not published | local loop; used for the cart-pole organoids |
| FinalSpark Neuroplatform | 4 organoids × 8 electrodes per MEA; organoids live 100+ days | remote API over the internet; not published | open loop and slow loops only |

The robot's balance loop runs at 50 Hz. A dish in that loop may add at most ~40 ms (Phase 1 spec). Anything slower
(seconds) can run remotely.

## What real cultures are known to compute

- **Learning from feedback.** DishBrain (Kagan et al., Neuron 2022): Pong improved within 5 minutes of play, and not
  without feedback. Cortical organoids on a MaxWell MEA (Robbins et al., Cell Reports 2026; press coverage only)
  learned to balance a simulated cart-pole. Between episodes, an algorithm chose which neurons got short
  high-frequency training bursts: 46% proficient, against 2.3% with no feedback and 4.4% with random feedback.
  Learning was short-term: gone after 45 minutes of rest.
- **Adaptation and deviance detection.** Cultures adapt to repeated stimuli (Eytan et al. 2003). A 2025 study on CMOS
  arrays found true deviance responses, NMDA-dependent, 11–150 ms after the stimulus.
- **Fading memory and nonlinear dynamics** (Brainoware, Cai et al., Nature Electronics 2023).
- **Encoding.** In closed loop, burst stimulation decoded by time to first spike worked best (96%). Rate coding fell
  from 86% open loop to 62% closed loop (arXiv 2607.13644).

## What our simulation says

| Test | Result | Script, commit |
|---|---|---|
| Robot without its balance sense | falls in 1.0 s; students retrained without it 0/40 | `blind_teacher.py` 64f452d, `no_imu_student.py` 364348e |
| Balance sense through a hardware-sized dish | 1 well × 32: 58/60 unchanged; 1 × 12 (CL1): 40/40 after one weight edit | `vestibular_probe.py`, WSP, fa16306 |
| Dish detecting pushes (computation?) | live dish worse than its own input, no better than dead | `trigger_probe.py` 31e110e |
| Memory beyond the input, by stimulation regime | none. In every regime the dish holds less of the recent IMU than the stimulation pattern itself (IMU 120 ms ago: dish R² 0.02–0.13, stimulation 0.37), and the live culture holds less than the dead one. Push detection AUC 0.51–0.72 (dead 0.61–0.75) against 0.86 for a filter of the IMU. Sparser stimulation does not free the synapses: resources 0.57 (200 Hz) vs 0.61 (50 Hz), so the depression comes from the culture's own bursts | `dish_regimes.py` |
| Slow NMDA-type excitation | no help. With NMDA at 0.4 of excitation, the live culture's push AUC is 0.72 (50 Hz amplitude coding, was 0.72) and 0.68 (200 Hz, was 0.63). That is above the dead culture at 200 Hz (0.61), but below the dead culture at 50 Hz (0.75) and the filter (0.86). Memory of the IMU 60 ms ago is unchanged (0.11 vs 0.11, 0.14 vs 0.15) | `dish_regimes.py --nmda 0.4 --live-only` |
| Is it the readout? CEBRA on the same dish data | no. Contrastive embeddings (CEBRA, Nature 2023), label-guided or self-supervised, current sample only, 3 folds: the live dish never beats the dead one or its own stimulation pattern (IMU now, R²: input 0.67, live 0.13–0.16, dead 0.17–0.21; push AUC 0.81 vs 0.52 vs 0.52–0.53). On the dish CEBRA finds less than a ridge readout (live 0.36–0.37), and without a label it finds nothing about the IMU (shuffled-label level): the dish's main structure is unrelated to its input | `cebra_probe.py` |
| A job the frozen robot can't do | weak knees: knee bias −0.6 rescues 0/20 → 19/20; 20 kg payload: IMU gain 1.5 rescues 0/20 → 19/20; each fix is fatal for the other change; no single setting works for both | `adaptation_slot.py` c34be6c |
| Can the robot sense which change happened? | payload: yes, at once (knee tracking error 0.113 vs 0.048); weak knees: no, until pushes near the failure point | `struggle_signal.py` 7cc4a57 |
| Can a culture be coached to choose the fix? | not our model culture. Correct choices in the last 50 of 150 trials: adaptive coaching 0.51, random 0.43, none 0.50 (3 cultures each; chance 0.5). Each context's score drifts by ±10–25 spikes on its own. Dose-response: 400 pairings move the probe response by 20–25 spikes (~20× its standard error), but both contexts move together, in a direction that depends on the culture. The trained pathway's own share is ±3–6 spikes, with either sign. The model culture answers every probe with the same network burst, so it can't hold a context-specific mapping. Without homeostasis its weights double in 100 s of activity | `coach_probe.py`, `--dose` |
| Does suppressing bursts make pairing specific? | no. Background stimulation, stronger inhibition, sparser wiring or more noise cut burstiness, but the trained pathway's own share stays inconsistent: −3.2 / −9.2 in the base culture, from +3.2 / −2.8 to −0.1 / −10.3 in the others (2 cultures; target negative). The site of the probe was already decodable at 75–99% | `burst_probe.py` fb08bf9 |
| The cart-pole organoid protocol, step by step | can't be tested on this model. Electrode selection by first-order connectivity works: specific pairs (own 0.80–1.00, crossed 0–0.04). Once the decoder is calibrated under stimulation, one of three cultures balances the pole with no training at all (90th percentile 34.6 s, bar 20.85 s): the selected pairs act as a reflex. Training never beat no training. With plasticity on, the culture runs away under the task's stimulation: rates rise 2–7× (6 → 13–45 Hz per electrode) within 20 s to 10 min, and the input with the slightly stronger response takes over both outputs (cross gain up to 24.6 spikes per pulse vs 2–3 for its own). That holds for every strength tried (×1 to ×0.03) and also for weight-dependent STDP. Scaling total input doesn't stop it: weights shift onto inputs that fire together | `cartpole_replication.py` a51c67a |
| Does inhibitory plasticity fix the model? | no, not within one time-boxed attempt. Vogels-type inhibitory plasticity, each neuron's target its resting rate, cuts the runaway from 45 to ~9 Hz per electrode, but rates settle at 1.5–1.7× rest (gate 1.5×) while inhibitory weights keep climbing (4–9× their start, rarely at the cap), and the inputs' evoked responses fade to 0.5–1.5 spikes per pulse. No setting is specific and steady (5 settings, 10 min each). Untested guess: the rule also strengthens inhibitory→inhibitory synapses, partly silencing the inhibition | `cartpole_replication.py --drift` 6cd83c4 |

## The routes

Ranked by how likely each is to give a culture that does real work for the real robot, and how cheaply a real dish
can settle it.

| Rank | Route | What the dish does | Loop | Sim evidence | Real-culture precedent | Platforms |
|---|---|---|---|---|---|---|
| 1 | Coached adaptation | picks one of a few vetted body settings, learning from falls | seconds to minutes | the job exists; our model culture can't be coached | cart-pole organoids, DishBrain | CL1, MaxWell, FinalSpark |
| 2 | Push / surprise detection | flags pushes in the IMU stream | 20–40 ms | no, in every regime tested | deviance detection (open loop) | open-loop test: any; in the loop: CL1, MaxWell |
| 3 | Inner ear relay | carries the IMU to the controller | 20 ms | necessary and hardware-sized, computes nothing | sensory encoding | CL1, MaxWell |
| – | Whole controller | encoder → dish → decoder | 20 ms | the decoder does the work | – | rejected |

### 1. Coached adaptation: do first, on a real dish

**What.** The teacher keeps running the balance loop untouched. The dish chooses one setting: knee bias −0.6 (weak
knees) or IMU gain 1.5 (heavy payload). It learns from the robot's falls or struggle, between push bouts, the way
the cart-pole organoids were coached.

**For.**
- The job is real. Two body changes the frozen controller can't survive, each with a fix that is fatal for the
  other change and harmless on a healthy robot (20/20).
- It fits every platform. One decision every few seconds on 8–16 electrodes, so loop latency hardly matters and a
  remote API (FinalSpark) is fine.
- It is the closest thing to what real cultures have already done: learning from feedback (Robbins et al. 2026,
  Kagan et al. 2022).
- Robot risk is bounded. The dish can only choose among settings each verified on the nominal robot.

**Against.**
- Our simulated culture cannot be coached: chance in every mode. Training moves its responses a lot, but to both
  contexts alike, because every probe sets off the same network burst. This is a limit of the model: the cart-pole
  organoids did learn a specific mapping, and on our model their protocol fails too. Its plasticity turns any
  repeated input into a trigger for the whole network within minutes. So the simulation neither supports nor refutes the route, and a real dish
  has to decide it.
- Memory is short in real cultures: the cart-pole learning was gone after 45 minutes of rest. Expect to re-coach
  each session.
- Weak knees are invisible until the robot nearly falls (`struggle_signal.py`), so the real robot learns in a
  harness.
- Honest framing: a counter could do this job. The claim is "a culture learns from the robot's feedback and its
  choice keeps the robot standing", not "the culture beats silicon". That is the bar the field has reached so far
  (cart-pole), now on a real robot.

**Gates.**
- **B1, real dish, robot in simulation.** Two context electrode groups, two output groups, outcomes drawn from the
  measured success table (`coach_probe.py` protocol). Pass: adaptive coaching beats random and no feedback in correct
  choices over the last 50 trials by ≥ 20 points, in ≥ 4 cultures. Fail: try the cart-pole training rule (an
  algorithm choosing which electrodes get bursts); fail again: stop the route.
  Take two things from the cart-pole protocol from the start (`cartpole_replication.py`):
  - Choose the context and output electrodes by first-order connectivity (spike within 10 ms of a pulse), not by a
    fixed layout.
  - Calibrate the decoder on its rates under the task's stimulation. Raw counts from electrodes firing tens of Hz
    pin the choice to one side, and so do resting rates.
- **B2, real dish.** No context signal; the condition flips unannounced every 30 trials. Pass: ≥ 70% correct in the
  last 15 trials of each block, controls at chance.
- **B3, real Asimov in a harness.** Weak knees emulated by limiting knee torque to 45% in the motor drivers
  (reversible, no hardware change); payload 20 kg, added in steps (10 kg needs no fix). Pushes of 80–120 N. The dish
  sets the correction between push bouts. Pass: fewer harness catches than the fixed default setting, following the
  simulation's 0/20 → 19/20.
- **B4.** Harness slack, then off.

**Hardware.** Any of CL1, MaxWell, FinalSpark. 16 electrodes (4 per group); with FinalSpark's 8 per organoid, 2 per
group. Patterned high-frequency stimulation for the training bursts. Sessions of ≥ 1 hour (150 trials take 10–40
minutes). Latency: seconds. Robot side: gantry or harness, a torque-limit setting in the drivers, a payload mount at
the pelvis.

### 2. Push and surprise detection: test open loop, alongside route 1

**What.** The dish flags pushes and other surprises in the IMU stream sooner or better than a filter would. The robot
uses the flag, for example to raise its reflex gain for half a second.

**Evidence.** In our model, no. The best live culture detects pushes no better than the best dead one (AUC 0.72
vs 0.75), and every culture is worse than a squared filter of its own input (0.86). Each holds less of the recent
past than the stimulation pattern does. Slow NMDA-type excitation lifts the 200 Hz regime from 0.63 to 0.68, still
short of both. Real cultures do show deviance responses, NMDA-dependent, 11–150 ms after the stimulus. Our model may
simply lack the mechanism, so a real dish should decide. Testing it is cheap: open loop, on recorded data.

**Gates.**
- **C1, real dish, open loop.** Replay the 133 recorded pushes (`outputs/trigger_probe/data.pkl`) as stimulation
  (amplitude code, 50 Hz) and record. Score with a fixed linear readout. Pass: AUC above the squared IMU filter
  (0.86) *and* above the same culture with NMDA receptors blocked (APV) and with fast excitation blocked (CNQX + APV).
  If the live culture beats the blocked one but not the filter, it computes but adds nothing: park the route.
  A nonlinear readout as a second check is fine (CEBRA's single-sample model with a shuffled-label control; it is
  Apache-2.0, but EPFL holds a US patent on the method, US12499131B2, so ask EPFL before commercial use), never instead of the
  fixed linear one: in our model it found less, not more.
- **C2, local loop.** Adds ≤ 40 ms. The dish's flag improves stand_push survival over the same controller fed the
  filter's flag.

**Hardware.** C1: any platform with timed stimulation, plus a wet lab for the blockers (FinalSpark's remote service
may not offer pharmacology). C2: local CL1 or MaxWell only.

### 3. Inner ear relay: base layer and hardware testbed, local only

**What.** The dish carries the IMU signal from the robot to the controller.

**Status.** Necessary: without its balance sense the robot falls in 1 s. Hardware-sized: a CL1-sized channel
(12 electrodes) gives 40/40 after one weight edit. But it computes nothing: a dead culture relays as well. By our bar
it is not a computation route. Keep it for two jobs: proving a real dish can sit in a 50 Hz loop (latency, drift,
per-session recalibration by one weight edit), and carrying route 2's signal if C1 passes.

**Gates.**
- **A1, open loop.** Replay recorded IMU and measure the real channel (gain, noise, delay, hold). Fit `DishModel`,
  edit the controller's weights (WSP track a). Pass: ≥ 38/40 on held-out simulated seeds.
- **A2, closed loop in a harness.** The real IMU runs in parallel as a fallback. Switch to it if the dish's estimate
  leaves 3 spreads for more than 100 ms.
- **A3.** Free standing, with pushes.

**Hardware.** Local CL1 (≥ 12 electrodes) or MaxWell (32 stimulation channels), ≤ 40 ms added. Not FinalSpark: a
remote loop is too slow.

### Rejected: the dish as the whole controller

Encoder → dish → decoder controlled the robot, and still did without the dish: the trained decoder did the work.
On 32–59 electrodes no reservoir carries a 23-joint policy.

## Order of work

1. **No new hardware.** Write the B1, C1 and A1 protocols as stimulation schedules plus analysis scripts for the
   CL API. Run no more simulated learning experiments until the culture model reproduces a published learning effect
   (the cart-pole gap of 46% vs 2–4%). Until then, its negative learning results say nothing about real dishes. First
   it must answer two electrode groups with different responses, keep a change in one pathway specific to that
   pathway, and hold its weights steady without an added scaling step. The cart-pole replication names the missing
   piece: the model has nothing that holds firing *rates* in place under stimulation. One time-boxed attempt to add
   it (inhibitory plasticity, 2026-11-23) failed the drift test (`cartpole_replication.py --drift`: rates and
   pathway specificity steady for 10 minutes of task stimulation): rates fell close to the gate, specificity never
   held. A further attempt is a modelling project of its own (inhibitory plasticity on I→E synapses only,
   intrinsic excitability homeostasis); until then, learning questions go to a real dish.
2. **First real-dish session**, one culture: A1 (channel, ~20 min), C1 (open-loop pushes, ~20 min, then blockers at
   the end), B1 (coaching, ~1 hour, before the blockers). Each decides its route on its own.
3. **If B1 passes:** B2 on the dish; build the harness, torque limit and payload mount; B3.
4. **If C1 passes:** A2, then C2 on a local loop.
5. **If both fail:** the dish can't yet do useful work for this robot. Report the negative result with its
   controls and pick the next job from the adaptation slot (`adaptation_slot.py`) with a richer choice.
