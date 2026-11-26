# The route to a biocomputer that computes and drives the real robot

22 November 2026. Builds on [next-steps.md](next-steps.md) and the README's status. Each claim cites the script and
commit that produced it. Simulated results are statements about our model; each is a hypothesis for a real dish.

**Status, 26 November 2026.** The routes below are ranked for real hardware, and every one of their gates needs a
dish. We have no culture and no physical robot for now, so the routes are parked. Both simulated tracks in
[Order of work without a dish](#order-of-work-without-a-dish) have run: in track A, A1 and A2 missed their gates
narrowly and A2c and A2d passed; track B closed after its first step. What waits is listed under [Parked until dish access](#parked-until-dish-access).
On 25 November a code review of track B's scripts found 13 bugs (among them a scaling step that could turn weights
negative, cached results reused across settings, and a yardstick that differed between arms). All are fixed with
tests, every drift arm was rerun, and no verdict changed; the numbers below are from the rerun. Runs now repeat bit
for bit; a single run of one culture is still one path of a chaotic network. Leads from four recent papers are under
[Leads from recent papers](#leads-from-recent-papers); all three have run since: A2c passed, the takeover's direction
explanation was confirmed, and a realism scorecard put the model in range on 3 of 6 measures. Track B was then
reopened for one round, many weak synapses (step 4), and it failed its first part. With the same total input spread
over 1.9 to 10 times as many synapses, the model stops bursting altogether. Track B is closed for good.

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
  without feedback. Mouse cortical organoids on a MaxWell MaxOne MEA (Robbins et al., Cell Reports 45:116984, 2026)
  learned to balance a simulated cart-pole. Between episodes, an algorithm chose which neurons got short
  high-frequency training bursts. Proficient cycles: 23.1% with adaptive training (46.4% when it ran continuously),
  against 2.3% with no training and 4.5% with random training (from the paper, as recorded in
  `cartpole_replication.py`). Learning was short-term: gone after 45 minutes of rest (from press coverage).
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
| Memory beyond the input, by stimulation regime | none. In every regime the dish holds less of the recent IMU than the stimulation pattern itself (IMU 120 ms ago: dish R² 0.02–0.13, stimulation 0.37), and the live culture holds less than the dead one. Push detection AUC 0.51–0.72 (dead 0.61–0.75) against 0.86 for a filter of the IMU. Sparser stimulation does not free the synapses: resources 0.57 (200 Hz) vs 0.61 (50 Hz), so the depression comes from the culture's own bursts | `dish_regimes.py` 71ceefd |
| Slow NMDA-type excitation | no help. With NMDA at 0.4 of excitation, the live culture's push AUC is 0.72 (50 Hz amplitude coding, was 0.72) and 0.68 (200 Hz, was 0.63). That is above the dead culture at 200 Hz (0.61), but below the dead culture at 50 Hz (0.75) and the filter (0.86). Memory of the IMU 60 ms ago is unchanged (0.11 vs 0.11, 0.14 vs 0.15) | `dish_regimes.py --nmda 0.4 --live-only` 71ceefd |
| Is it the readout? CEBRA on the same dish data | no. Contrastive embeddings (CEBRA, Nature 2023), label-guided or self-supervised, current sample only, 3 folds: the live dish never beats the dead one or its own stimulation pattern (IMU now, R²: input 0.67, live 0.13–0.16, dead 0.17–0.21; push AUC 0.81 vs 0.52 vs 0.52–0.53). On the dish CEBRA finds less than a ridge readout (live 0.36–0.37), and without a label it finds nothing about the IMU (shuffled-label level): the dish's main structure is unrelated to its input | `cebra_probe.py` 1b18290 |
| A job the frozen robot can't do | weak knees: knee bias −0.6 rescues 0/20 → 19/20; 20 kg payload: IMU gain 1.5 rescues 0/20 → 19/20; each fix is fatal for the other change; no single setting works for both | `adaptation_slot.py` c34be6c |
| Can the robot sense which change happened? | payload: yes, at once (knee tracking error 0.113 vs 0.048); weak knees: no, until pushes near the failure point | `struggle_signal.py` 7cc4a57 |
| Can a culture be coached to choose the fix? | not our model culture. Correct choices in the last 50 of 150 trials: adaptive coaching 0.51, random 0.43, none 0.50 (3 cultures each; chance 0.5). Each context's score drifts by ±10–25 spikes on its own. Dose-response: 400 pairings move the probe response by 20–25 spikes (~20× its standard error), but both contexts move together, in a direction that depends on the culture. The trained pathway's own share is ±3–6 spikes, with either sign. The model culture answers every probe with the same network burst, so it can't hold a context-specific mapping. Without homeostasis its weights double in 100 s of activity | `coach_probe.py`, `--dose` 5e5cf89 |
| Does suppressing bursts make pairing specific? | no. Background stimulation, stronger inhibition, sparser wiring or more noise cut burstiness, but the trained pathway's own share stays inconsistent: −3.2 / −9.2 in the base culture, from +3.2 / −2.8 to −0.1 / −10.3 in the others (2 cultures; target negative). The site of the probe was already decodable at 75–99% | `burst_probe.py` fb08bf9 |
| The cart-pole organoid protocol, step by step | can't be tested on this model. Electrode selection by first-order connectivity works: specific pairs (own 0.80–1.00, crossed 0–0.04). Once the decoder is calibrated under stimulation, one of three cultures balances the pole with no training at all (90th percentile 34.6 s, bar 20.85 s): the selected pairs act as a reflex. Training never beat no training. With plasticity on, the culture runs away under the task's stimulation: rates rise 2–7× (6 → 13–45 Hz per electrode) within 20 s to 10 min, and the input with the slightly stronger response takes over both outputs (cross gain up to 24.6 spikes per pulse vs 2–3 for its own). That holds for every strength tried (×1 to ×0.03) and also for weight-dependent STDP. Scaling total input doesn't stop it: weights shift onto inputs that fire together | `cartpole_replication.py` a51c67a |
| Does inhibitory plasticity fix the model? | no, not within one time-boxed attempt. Vogels-type inhibitory plasticity, each neuron's target its resting rate, cuts the runaway from 45 to ~9 Hz per electrode, but rates settle at 1.5–1.7× rest (gate 1.5×) while inhibitory weights keep climbing (4–9× their start, rarely at the cap), and the inputs' evoked responses fade to 0.5–1.5 spikes per pulse. No setting is specific and steady (5 settings, 10 min each). Untested guess: the rule also strengthens inhibitory→inhibitory synapses, partly silencing the inhibition | `cartpole_replication.py --drift` 9355cbf |
| Does an outside model hold up? bl1 (github.com/m9h/bl1: JAX, Wagenaar-calibrated, STDP + homeostatic scaling) | no. Same drift test on bl1's own code, as shipped and with two of its five bugs fixed, 5 cultures, 600 s, two pulse lengths. Nothing runs away (its STDP is depression-dominated), but the gate is never passed except by running down. Per-neuron synaptic scaling alone holds the non-stimulated electrodes at their resting rate (1.00–1.01× in every culture). With STDP on it holds neither rates (realistic 0.5 ms pulses: 1.25× the no-plasticity arm, still climbing) nor pathways (each decoder's own gain moves ~2 spikes per pulse within 10 minutes, 5 of 10 pathways beyond 2 SE). A potentiation-dominated control shows the test catches upward drift. The port is spike-for-spike bl1's code (`bl1_checks.py`) | `bl1_drift.py`, `bl1_checks.py`, `bl1_rep_analysis.py` 1e3f994 |
| The culture in the robot's loop (track A1): only the IMU through the dish, decoder and weight edit fixed beforehand | not on a CL1-sized well. 1 well × 12 electrodes: 0/40 with the planned 1/gain edit, 23/40 with an edit that also removes the decoder's offset (gate ≥ 34/40). 1 × 32: 40/40. Silenced dish 0/40, ideal IMU 40/40. A dead culture does as well or better (1 × 12: 32/40), so the dish relays and computes nothing. The planned edit failed because the robot stands leaning 3.6° forward, not upright: the decoder's offset turned into a false lean of 2.35 spreads. Phase 3.1's channel model had no offset and couldn't show it. The remaining 1 × 12 falls follow pushes from the front: the relay reads the pitch rate too high during the recovery, the forward lean grows past the encoder's input limit (4 replay SDs), and past it the teacher's view of the tilt stops changing | `closed_loop_vestibular.py`, `closed_loop_falls.py` daa1fec |
| Sessions and drift (track A2): age the culture 1, 3, 7 days or swap in a new dish; stale decoder vs one recalibrated from 60 s of open-loop replay plus one edit (1 × 32) | not passed, by one episode. A new dish breaks the stale relay (0/40), and recalibration restores 36/40. Aging up to 7 days leaves the stale relay at 36–39/40, so the gate was tested only on the new dish. Recalibrated: 33/40 at day 3, 36–37/40 elsewhere, 36/40 even at day 0. The 60 s decoder reads grav_x with noise 0.84–0.89 spreads against 0.53–0.55 for the one fitted on 312 s, so the calibration's length is the limit | `closed_loop_sessions.py` b20145b |
| Sessions with only the edit refitted (track A2c): A1's decoder kept, gain and offset per IMU channel refitted from 60, 30 or 10 s of the session's replay, fresh seeds (1 × 32) | passed. Edit from 60 s: 37–40/40 in every session, new dish 37/40; recalibrated decoder on the same seeds 30–36/40 (new dish 30/40). Pooled 190/200 vs 171/200; 28 seeds stand only with the edit, 9 only with the recalibration. The old decoder (312 s of fitting) is less noisy than a 60 s one (grav_x 0.98 vs 1.61 spreads on the new dish); the edit removes its 1–2.5-spread offsets. 10 s pins the offsets (≤ 0.09 spread) but leaves two new-dish gains 22–29% low (36/40); 30 s pins every gain within 5%. One new dish (one culture seed) | `closed_loop_sessions.py --arms refit60 …` 37fcbd3 |
| Does per-neuron synaptic scaling hold rates? (track B1: bl1's rule, our model, 3 cultures, 600 s) | not with our STDP on. Scaling alone keeps every electrode at rest (0.99–1.04× in the last minute, 0.99–1.07× in every minute) without muting the inputs, and brings neurons above 1.5× their own target from 1–4% to ~1%. With STDP at bl1's rate (gate) it runs away like the old total-input step: 6.3–10.2× rest against 5.5–7.8×, E→E weights up 1.4–1.8×. Ten times faster (exploratory): 1.7–2.1× rest from minute 5 on, but the pathways mix (in two cultures a cross gain climbs from 2–3 to 4.7–6.6 spikes per pulse and one decoder follows the wrong encoder). Likely reason: additive STDP against multiplicative scaling only balances at a standing rate error | `cartpole_replication.py --drift --drift-homeo`, `SynapticScaling` 273b94e, rerun 51fd354 |
| Does weight-dependent STDP let the scaling hold? (the check before closing track B, 3 cultures, 600 s) | rates nearly at ten times bl1's rate, pathways no. At bl1's rate (gate): 2.2–5.0× rest, no better than with the old step (2.2–4.1×). Ten times faster: 1.34–1.53× rest in the last minute, in band in 28 of 30 minutes, inputs still answered (own gains 2.7–6.0 spikes per pulse), the closest any setting came; one culture ends 0.03 over the band (all three inside on the 24 November run, 1.32–1.45×). And 2 of 4 pathways move beyond 2 SE of the no-plasticity arm in every culture, and by the end one input drives both outputs (4.0–6.5 spikes per pulse from the wrong input against 1.0–3.4 from the right one). Scaling alone: 1 of 24 moved (the noise floor). Numbers from the rerun of 25 November after the code review's fixes. Single arms are sensitive: the exp factor's slight change to the weights moved them and flipped some pathway verdicts, since the network's chaos grows any change into another path | `cartpole_replication.py --drift --stdp-rule multiplicative`, `pathways_vs_off` 273b94e, rerun 51fd354 |
| Why does one input take over? (the takeover rerun: one takeover arm per culture with a weight dump, 3 cultures, 600 s) | links pointing away from the winning input strengthen, links pointing back weaken: away − toward 4.3–6.0 weight units, about 500× the tilt under scaling alone, confirmed in all three cultures by the pre-registered reading. The effect is centred exactly on the winning input (peak of a map over all 256 electrode positions) and fades with distance; not a few hubs. The winner's pulses come to set off activity over the whole array, the loser's stay local; the inputs stay decodable (0.94–1.00 in two cultures), so the readout is captured, the input not erased. Whether orientation causes the takeover or grows with it is not settled | `takeover_rerun.py` 4d2f52b |
| Does the model's spontaneous activity look real? (realism scorecard: B0's yardstick on 5 cultures × 600 s, mcs60 array, against 29 Wagenaar cultures; and burst spread) | half. In range: electrode rate (5.0 Hz vs 2.8 [0.7–10.3]), burst duration (103 ms vs 556 [79–1788]), share of spikes in bursts (0.85). Out of range in every culture: bursts ten times too often (45 per minute vs 4.2 [1.2–15.4]), like clockwork (interval CV 0.32 vs 0.87 [0.56–1.99]), and a pulse sets off a burst 20% of the time vs 6.5% [0–16%]. Burst spread from electrodes isn't validated on the model (speed recovered, start site and pattern not); read with that limit, real bursts are messier and start at fewer sites, at the same speed (30 vs 31–41 mm/s) | `realism_scorecard.py`, `burst_spread.py` 010d6d1 |
| Do many weaker synapses make the model realistic? (step 4: the same total input per neuron spread over 1.9, 4 or 10 times as many synapses, 5 cultures each, 600 s, mcs60 array) | no, the model goes silent. There was not one network burst in 10 minutes in any of the 15 cultures. The electrode rate is 0.54–0.56 Hz, below the real 0.7–10.3, and neurons fire 18–28 times less than now. Only the evoked share is in range (0–4%). With no bursts, that measure probably counts small responses (not checked). Halving each synapse at the same reach is already enough. Each neuron's total input was checked within 1%, and the unchanged culture repeated its stored rows bit for bit. The stability part didn't run, and track B is closed for good | `weak_synapses.py` d55805f |
| Sessions under day-to-day change of real size (track A2d): A2c's kept decoder and 60 s edit refit, on seven sessions shifted so each electrode's evoked change matches FinalSpark's fs369 organoids over 1, 3 and 7 days (1 × 32) | passed. Real organoids' day-to-day change is 2.4–4.5× `aging.py`'s (median log2 change), which is too gentle on every measure and moves the spontaneous rate the wrong way. Edit from 60 s: 35–40/40 in every session; the stale relay drops only on the more excitable day-7 culture (19/40), and the edit removes its offsets (≤ 0.09 spread). The two largest targets were not reached (day 7 up 0.35 of 0.66, stress 1.21 of 1.99). One culture, one aging seed, coherent change only | `day_drift.py` e729e72, `closed_loop_sessions.py` eb61e58 |
| Does the model look like an organoid? (organoid column: 8-electrode blocks of the scorecard recordings against FinalSpark's organoids, 64 group-days; and fs369's pathway map protocol) | no. In range on 0 of 5 spontaneous measures: bursts 46× too often and 18× too short (39 per minute vs 0.85, 41 ms vs 752). Under the map protocol no model pathway responds (0 of 56; real groups 20–30%): a pulse drives only its own electrode | `organoid_scorecard.py` 9433ad7 |

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

**Status.** Necessary: without its balance sense the robot falls in 1 s. Hardware-sized: in Phase 3.1's channel model a
CL1-sized channel (12 electrodes) gives 40/40 after one weight edit; with the simulated culture itself in the loop
(A1) 1 × 12 reached only 23/40 and 1 × 32 40/40. But it computes nothing: a dead culture relays as well. By our bar
it is not a computation route. Keep it for two jobs: proving a real dish can sit in a 50 Hz loop (latency, drift,
per-session recalibration by one weight edit), and carrying route 2's signal if C1 passes. In simulation the
session protocol holds: the old decoder plus a refitted edit (gain and offset per channel, from 30–60 s of replay)
keeps 37–40/40 after aging and on a new dish (A2c), and 35–40/40 under day-to-day change of real size (A2d).

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

## Order of work without a dish

24 November 2026. No real culture and no physical robot are available for now, so everything runs in simulation (the
robot in MuJoCo). Two tracks; each step has a gate fixed in advance. Everything that needs a dish or the robot is
under [Parked until dish access](#parked-until-dish-access). The track steps (A1, A2, B0–B4) are not the real-dish
gates of the same names in the routes above.

### Track A: the simulated culture in the robot's loop

This is Phase 3.3 of [next-steps.md](next-steps.md#33-closed-loop), and route 3 rehearsed in simulation.

- **A1. Closed loop** (`closed_loop_vestibular.py`).
  - Only the 6 IMU values go through the dish. Joint states and the last action go straight to the controller.
  - Everything is fixed before the first closed-loop seed. The decoder (ridge, 40 ms trace) is fitted on open-loop
    replays of teacher episodes, and the 1/gain weight edit comes from held-out replay. Plasticity is off: a relay
    doesn't need it, and the model's plasticity is its known flaw.
  - Main configuration: 1 well × 12 electrodes (CL1-sized). Comparison: 1 × 32.
  - Six arms, 40 fresh seeds each: culture with the edited teacher; culture with the unchanged teacher; dish silenced;
    ideal IMU; dead culture (recurrent synapses removed) with its own refitted decoder and edit; zero action.
  - **Gate:** the culture arm reaches ≥ 34/40 full 20 s runs, and the silenced dish 0/40. The dead culture will
    probably match the live one. If so, we report it plainly: the dish relays and computes nothing.
  - **Result (24 November 2026): not passed on 1 × 12.** 0/40 with the 1/gain edit, 23/40 with an affine edit
    (designed after the first run, scored on fresh seeds). 1 × 32 passes: 40/40, silenced 0/40. The dead culture
    does as well or better. Details in the simulation table above and in `closed_loop_vestibular.py`.
- **A2. Sessions and drift.**
  - Between "sessions", age the culture (`aging.py`: 1, 3 and 7 days), and add a fresh culture seed to stand for a
    new dish.
  - Compare the stale decoder with one recalibrated from a short open-loop replay (~60 s) plus only the 1/gain edit.
  - **Gate:** after aging, the recalibrated relay reaches ≥ 34/40 where the stale one drops. This tests route 3's
    claim that a session needs only one weight edit.
  - **Result (24 November 2026): not passed, by one episode** (`closed_loop_sessions.py`, 1 × 32, affine edit
    because A1 showed the 1/gain edit fails). The stale relay drops only on a new dish (0/40), where 60 s of
    calibration and one edit restore 36/40. Aging up to 7 days leaves the stale relay at 36–39/40. The recalibrated
    relay reaches 33/40 at day 3 and 36–37/40 elsewhere, 36/40 even at day 0: 60 s is the weak point, not aging.
  - **A2c (25 November 2026): passed.** Keep the old decoder and refit only the affine edit from the session's
    replay. With 60 s: 37–40/40 in every session, the new dish 37/40, against 30–36/40 for the recalibrated decoder
    on the same seeds. 30 s is as good; 10 s pins the offsets but leaves two new-dish gains loose (36/40). A session
    needs two numbers per IMU channel, not a new decoder. Shown on one new dish.
  - **A2d (26 November 2026): passed, under day-to-day change of real size.** FinalSpark's whole-life organoid
    recordings change far more from day to day than `aging.py` does. In fs369 (21 days) each electrode's evoked
    response moves by a median ×1.13 in a day and ×1.6 in a week (75th percentile ×4), and spontaneous rates fall
    ×2.9 per organoid in a week; `aging.py` moves the model's evoked response by ×1.04 and ×1.16 and its spontaneous
    rate up, not down (`day_drift.py`, too gentle on all four measures). A shift of every neuron's excitability,
    calibrated to the real evoked change, gives seven sessions (1, 3 and 7 days, up and down, and a stress dose).
    The 60 s edit keeps the relay at 35–40/40 in all of them. The kept decoder alone breaks only where the culture
    gets more excitable (day 7 up: 19/40); coherent declines leave it standing, even at ×0.43. Limits: coherent
    change only, one culture; the two largest targets weren't reached.

### Track B: a culture model good enough for learning questions

Why: until the model reproduces a published learning effect, its learning results say nothing about real dishes. The
cart-pole replication named the missing piece: nothing holds firing rates in place under stimulation. One time-boxed
fix with inhibitory plasticity failed the drift test (rates near the gate, specificity never steady), and bl1
(1e3f994) fails it too. bl1's per-neuron synaptic scaling holds rates on its own, so it is the piece worth porting,
but with STDP on its pathways still move. So the track is time-boxed, and every step has a gate; a failed gate ends
the track with a write-up, and learning questions wait for a dish.

- **B0. Public data** (CPU, alongside A1). Recordings of real cultures from other labs:
  - Wagenaar et al. 2006, dissociated cortical cultures (spontaneous, plus stimulation sets if published);
  - FinalSpark's whole-life exports (data: FinalSpark, cited as the source; confidential, so neither they nor the
    per-day, per-session or per-electrode tables derived from them are committed), received 26 November: fs369 (one MEA for 21.7 days, four organoid groups of 8
    electrodes, mostly spontaneous, with four whole-array stimulation sweeps and one mapping session) and fs437 (a
    dying MEA, 6 days, stimulated by many clients). Spike events with times, every stimulation, 3 ms raw snippets.
    Read with h5py only (`scripts/finalspark_data.py`; two logged sessions never reached the array, so every use
    checks for the stimulation artifact). They gave A2d's day-to-day targets (`day_drift.py`) and an organoid column
    for the realism scorecard (`organoid_scorecard.py`), and showed that the pathway half of the drift test can't be
    read from them either (`real_pathways.py`; parked item 5). Also found 26 November: a public CC-BY record of
    spontaneous spikes from human forebrain organoids on FinalSpark's platform (TU Munich, Zenodo
    10.5281/zenodo.22755004, v3: 8 electrodes per organoid, 600-1800 s each, ages 75-367 days, no stimulation); not
    fetched;
  - the Robbins et al. Zenodo record (10.5281/zenodo.17684862), if it holds recordings.

  Extract per-electrode rates, burst rate, burst duration, inter-burst interval, recruitment, and the stability of
  evoked responses if there is stimulation data.
- **B1. Rate controller.** Per-neuron synaptic scaling: bl1's rule with its target bug fixed. Each neuron's rate is
  a 10 s moving average, and its E→E weights scale multiplicatively toward its own resting rate.
  **Gate:** scaling alone holds rates at rest. In the drift test's rate half (3 cultures), with our STDP on, the rate
  stays ≤ 1.5× rest over the last minute.
  **Result (24 November 2026, rerun 25 November after the code review's fixes): not passed.** Scaling alone holds every
  culture at rest (0.99–1.04×). With STDP on, at bl1's rate it doesn't slow the runaway (6.3–10.2× rest, the old step
  5.5–7.8×). Ten times faster, the rate settles near 2× while the pathways mix. Details in the simulation table above
  and in `cartpole_replication.py`. A labelled check of B2's first candidate, weight-dependent STDP with the same
  scaling, didn't change the verdict (2.2–5.0× rest at bl1's rate).
- **B2. Stable pathways.** One round, then one refinement round. Each candidate is combined with B1:
  weight-dependent STDP, inhibitory plasticity on I→E synapses only, a BCM-style sliding threshold, intrinsic
  excitability homeostasis, heterosynaptic normalisation.
  **Gate:** the full drift gate (rates and pathway specificity steady for 10 minutes) passes in ≥ 3 of 5 cultures.
- **B3. Reproduce published effects.** This is the gate for any simulated learning result.
  - Spontaneous statistics fall within the ranges of B0's data.
  - Stimulus-specific adaptation (Eytan et al. 2003): with one site stimulated often and one rarely, the rare site's
    responses stay relatively larger.
  - The cart-pole gap (`cartpole_replication.py`), in ≥ 3 cultures: adaptive training ≥ 15% proficient, no training
    and random training ≤ 5%, a frozen culture at the no-training level. The paper reports 23.1% against 2.3% and
    4.5%.
- **B4** (only if B3 passes). Rerun coaching in simulation (`coach_probe.py`) with route 1's bar: adaptive coaching
  beats random and no feedback by ≥ 20 points in ≥ 4 cultures. This reopens Phase 4.2 in simulation, then 4.1.

**Track B closed (24 November 2026).** B1 failed, and the one check run before closing didn't rescue it. Four brakes
have now been tried on the model's runaway: a total-input step, inhibitory plasticity, bl1's per-neuron scaling (in
bl1's model and in ours), and that scaling with weight-dependent STDP. The last, at ten times bl1's rate, comes closest
to holding rates (1.34–1.53× rest in the last minute, in band in 28 of 30 minutes; one culture ends just over). But in
every culture one input still ends up driving both outputs within minutes. Holding rates doesn't keep pathways apart:
under repeated stimulation our STDP hands the network to whichever input drives it. So simulated learning questions wait
for a dish, as the track's rule says. B2's other candidates are untried: inhibitory plasticity on I→E only, a BCM-style
threshold, intrinsic excitability, heterosynaptic normalisation. The model still serves where plasticity is off, as
track A's relay. Public recordings settle only the rate half: in the cart-pole organoids' game logs, real rates don't
climb under the same kind of input pulses (`real_drift.py`; parked item 5 below). What real pathways do still needs a
dish. One more mismatch they show: stimulation sets off a network burst in about 20% of trials in our model (the
replication's selected encoders: 16–24%), against 6.5% in the organoids (IQR 2.7–10%). The realism scorecard
(25 November) confirms it in five cultures (20–21%) and adds two more: spontaneous bursts come ten times too often
(45 per minute against 4.2) and too regularly (interval CV 0.32 against 0.87). Against FinalSpark's organoids
(26 November, `organoid_scorecard.py`, on 8-electrode blocks) the model is in range on none of five spontaneous
measures (bursts 39 per minute against 0.85, 41 ms long against 752). Their mapping session shows what the model
lacks on the evoked side: a pulse reaches 2–3 neighbouring electrodes (11–17 of 56 pairs respond) and sets off a burst
2.5–5 points above chance; in the model it drives only its own electrode and sets off nothing beyond chance. Its 20%
figure above comes from a measure that doesn't subtract chance; at its burst rate a 200 ms window holds a burst onset
12–14% of the time anyway, so much of it may be spontaneous (not tested).

**Why one input takes over (takeover rerun, 25 November 2026; a diagnostic, the track stays closed).** One takeover
arm per culture was rerun with a weight dump and every electrode's spikes (`takeover_rerun.py`; each arm repeated
its stored row bit for bit). The pre-registered direction reading is confirmed in all three cultures. E→E links that
point away from the winning input gain 3.1–4.9 on average, links pointing back lose 1.1–1.35, about 500 times the
tilt under scaling alone. Mapped over all 256 electrode positions, the effect is centred exactly on the winning
input in every culture and fades with distance from it. It is not a few hub neurons: the top 1% hold 2.7–4.1% of the
gain. From the electrodes alone, the winner's pulses set off activity over the whole array (first-spike maps whose
direction matches the weight change, cosine 0.84–0.87), while the loser's stay local. The network still tells the
two inputs apart (decoding 0.94–1.00 in two cultures), so the takeover captures the readout rather than erasing the
input. Not settled: whether the orientation causes the takeover or grows with it, since both happen in the first
minute.

**Many weak synapses (step 4, 25–26 November 2026): failed; track B is closed for good.** The user reopened the
track for this one round. Each neuron got 1.9, 4 or 10 times as many synapses, each as much weaker. Each neuron's
total input stayed within 1% of now, and the unchanged culture repeated its stored results bit for bit
(`weak_synapses.py`). None of the 15 cultures (3 variants × 5) burst even once in 10 minutes: the busiest 10 ms held
at most 0.36% of the neurons. The electrode rate fell to 0.54–0.56 Hz (real: 0.7–10.3 Hz), so 1 of 6 measures was
in range: the evoked share, 0–4%. Halving each synapse at today's reach is already enough. The pre-registered
expectation, that the cultures would keep bursting, less often and still too regularly, was wrong.

A reading, not tested: in this model, bursts start when strong single inputs coincide, and smaller jolts at the same
mean never reach ignition. A model that bursts with fine-grained input would need other machinery. The stability part
was not run, as pre-registered.

The round departed from the wording above in two ways, both fixed before it ran:
- 1,000 synapses per neuron don't fit at the model's 150 µm reach (about 194 at most), so the round was a dose series
  that widened the reach only as needed (×1.9 at 150 µm, ×4 at 225 µm, ×10 at 370 µm).
- The stability arm was to be STDP with the 5 s total-input step, not B1's scaling. That scaling pulls each neuron
  toward its resting rate within seconds, so a culture whose rate rises under stimulation would fail through the
  brake.

### Order

1. A1 on the GPU, with B0 alongside (CPU and network). Done.
2. A2. Done. A2c (item 4.1) settles its open point.
3. B1: not passed (24 November 2026), and a check of weight-dependent STDP didn't change it. Track B is closed;
   B2, B3 and B4 not run.
4. Next, set on 25 November 2026 after the reading of four papers ([literature-notes.md](literature-notes.md)).
   Each step is pre-registered, guarded and committed as its own milestone:
   1. **A2c** (track A). Keep the old decoder and refit only the affine edit, from 10, 30 and 60 s of replay, on
      fresh seeds. Gate: ≥ 34/40 in every session with the 60 s edit. If it passes, route 3's session protocol
      becomes "refit the edit". If it fails, keep refitting the decoder, from a longer calibration.
      **Passed (25 November): 37–40/40 in every session; the protocol is "refit the edit", from 30–60 s.**
   2. **Takeover rerun** (track B, a diagnostic; the track stays closed). One takeover arm per culture plus the
      scaling-alone control, with a final weight dump and every electrode's spike times. It tests the direction
      hypothesis, and the electrode-level direction maps a real dish would need.
      **Confirmed (25 November) in all three cultures: the weight change points outward from the winning input. The
      track stays closed unless the user reopens it for step 4.**
   3. **Realism scorecard.** B0's yardstick (`spike_stats`) on the model's spontaneous activity against the
      Wagenaar ranges, and burst spread (start sites, speed, pattern), model first.
      **Run (25 November): 3 of 6 measures in range.** Bursts come ten times too often and too regularly, and a
      pulse sets off a burst three times too often. The burst-spread method wasn't validated on the model.
   4. Only if step 2 confirms the hypothesis and track B is reopened by decision: **many weak synapses**, one
      time-boxed round. Gate: about 6.5% of pulses set off a burst, spontaneous statistics in range, and the full
      drift gate in ≥ 3 of 5 cultures. A failure ends track B for good.
      **Failed (25–26 November): with the same total input, the model stops bursting (0 bursts per minute in 15 of
      15 cultures, electrode rate 0.54–0.56 Hz). Track B is closed for good.** It ran as a dose series (×1.9, ×4,
      ×10) because 1,000 synapses don't fit at today's reach (see the track B paragraph).
   5. Optional: **A1c**, a wider encoder input limit on the 12-electrode well, as a labelled run.
5. FinalSpark's whole-life exports (26 November 2026), each step pre-registered:
   1. **A2d** (track A): how much real organoids change from day to day, against `aging.py`, then A2c rerun under a
      calibrated change of that size. **`aging.py` too gentle on all four measures; A2d passed (35–40/40).**
   2. **Real pathways** (parked item 5): **not answerable from these records** (see item 5 under Parked).
   3. **Organoid column of the realism scorecard** (`organoid_scorecard.py`): see the track B paragraph.

   The notes' A1 state test (proposal 3) is dropped: it explains a relay flaw but changes no decision.

One GPU job at a time (8 GB card). Every run is guarded: a smoke test, in-run checks, a watchdog, 40 fresh seeds and
the standard baselines.

### Readings fixed in advance

Each step was pre-registered in its script's docstring before its first seed or number. The docstrings were taken
out of the code after commit d3c85c4, and their full text stays readable with `git show d3c85c4:scripts/<script>.py`.
That text covers design, gates, positive controls, expectations and results. Gates and readings not already given
above:
- **A2d** (`closed_loop_sessions.py`). Gate: refit60 ≥ 34/40 full 20 s runs in every session, day 0 included. It is
  tested only where stale drops below 34/40. Controls: day 0 equals A2's stored session, aging is on its nominal
  dose, the shift is applied exactly, and no weight moves.
- **Day-to-day drift** (`day_drift.py`).
  - Reading: for each measure and lag, the model is too gentle if its median |log2 change| lies below the real 25th
    percentile, and too strong if above the real 75th.
  - Verdicts: spontaneous change (electrode and group level) at ≥ 2 of 3 lags; evoked change per electrode at both
    lags. The evoked change per group is reported, not used.
  - Any verdict out of range leads to part 2: a calibrated change, then A2c rerun under it (A2d).
  - Part 2 doses: per lag and direction, the smallest |shift| that reaches the target, interpolated linearly between
    calibration points. If no point reaches it, the largest shift tried, flagged.
- **B1** (`cartpole_replication.py`).
  - Gate: at eta 1e-5, in each of 3 cultures, the last-minute rate per electrode stays within 2/3–1.5× rest, both
    with scaling alone and with STDP ×1 plus scaling.
  - The lower bound is ours: calm by muting doesn't count.
  - Controls: the no-plasticity arm stays in the band, and the old 5 s step runs away.
  - Only full 10-minute arms count.
- **Drift gate** (`cartpole_replication.py --drift`, `bl1_drift.py`; fixed before the inhibitory-plasticity sweep):
  - rate per electrode in the last minute ≤ 1.5× rest;
  - each decoder's own gain ≥ 2× its cross gain, with the cross gain ≤ 1 spike per pulse.
- **Real drift** (`real_drift.py`, the cart-pole organoids' "none" runs).
  - Real tissue holds if ≤ 10% of pathways move, the median rate ratio (last 3 min / first 30 s) is within 2/3–1.5,
    and the median 2 SE is no larger than the median preferred gain.
  - It drifts if ≥ 25% move or the ratio exceeds 1.5.
  - Otherwise it is in between. Without the power, the test is inconclusive.
- **Takeover rerun** (`takeover_rerun.py`).
  - Measure: per E→E synapse, the cosine between its direction and the direction from a centre point to its
    presynaptic neuron. "Away" means > 0.5 and "toward" < −0.5. Score = mean Δw(away) − mean Δw(toward).
  - Confirmed if, in each of 3 cultures, centred on the winning encoder:
    1. the score is > 2 SE;
    2. it beats the scaling-alone arm's score by > 2 SE;
    3. it beats both the score around the losing encoder and the 95th percentile around every position ≥ 500 µm
       from both encoders.
- **Realism scorecard** (`realism_scorecard.py`): a measure is in range when the median of the five cultures lies
  inside the real cultures' 5–95% range.
- **Burst spread** (`burst_spread.py`).
  - Wagenaar's recordings are read as validated only after the method recovers the model's own spread, on bursts
    starting under the array: median start-site error ≤ 200 µm, pattern class right in ≥ 70%, and median speed ratio
    within 0.67–1.5.
  - Otherwise the Wagenaar run is reported as unvalidated.
- **Many weak synapses** (`weak_synapses.py`).
  - Part 1: a variant passes when all six gated scorecard measures are in range.
  - Part 2: in ≥ 3 of 5 cultures, the arm with STDP ×1 and the 5 s total-input step must meet all three:
    - it is calm: last-3-minute rate within 2/3–1.5× the no-plasticity arm's;
    - no encoder → decoder gain moves against the no-plasticity arm;
    - the no-plasticity arm is separate.
  - Control: the same rule fails the stored runs of the unchanged culture.
- **Real pathways** (`real_pathways.py`).
  - The test is inconclusive with < 20 responding pathway-pairs or no power.
  - Pathways hold if the share moved exceeds the null by ≤ 10 points, and drift if by ≥ 25.
  - Amendment for single-input series: inconclusive with < 2 responding outputs or no power; hold if no output
    moved; drift if at least half moved, all in one direction.
- **Organoid column** (`organoid_scorecard.py`).
  - Part A: as the realism scorecard, on 8-electrode blocks, against FinalSpark's 5–95% range.
  - Part B, like real if both hold:
    - the responding share and the median gain lie within a factor 2 of the real groups' range;
    - the burst excess is at most max(2× the larger real excess, that excess + 2 points).

## Leads from recent papers

25 November 2026. Four papers, read against these results in [literature-notes.md](literature-notes.md). Nothing
new was run for them, and none gives a route past the bar. What they give:

- **A cheaper session recalibration for track A2**, backed by stored A2 numbers. On a new dish the old decoder's
  output is shifted by 1–2.5 spreads on the gyros, but it is no noisier than on day 0. After the edit, its grav_x
  error is 0.98 spreads, against 1.70 for the decoder refitted from 60 s. Keeping the decoder and refitting only
  the edit (two numbers per channel) is route 3's claim taken literally, and A2 never tested it.
- **A testable explanation of track B's takeover.** Activity spreading from a stimulated site gives STDP a
  direction, and the winning input drives outputs up to 0.94 mm away at 20–36 spikes per pulse. One rerun with a
  weight dump can confirm or refute it. Maps of which way evoked activity travels across the electrodes could show
  the same without weights, which is the only way to test a real dish.
- **Methods for the real-dish drift test**: co-firing corrected for rate with a proper null, decoding across time,
  response geometry compared between minutes, and those direction maps.

The order of work they lead to is item 4 under [Order](#order).

## Parked until dish access

These wait until a real culture or the robot is available. Then the ranking of the routes above applies again.

1. **First real-dish session**, one culture: A1 (channel, ~20 min), C1 (open-loop pushes, ~20 min, then blockers at
   the end), B1 (coaching, ~1 hour, before the blockers). Each decides its route on its own. Under the blockers,
   measure A1's channel once more: in the model, a dish with its links cut relays better (1 × 12: 32 vs 23/40).
2. **What follows it.** If B1 passes: B2 on the dish, then B3 and B4 on the robot. If C1 passes: A2 and A3, then C2
   on a local loop. If both fail: report the negative result with its controls and pick the next job from the
   adaptation slot (`adaptation_slot.py`) with a richer choice.
3. **Protocol scripts** for B1, C1 and A1 as stimulation schedules plus analysis, for the CL API and FinalSpark's
   Neuroplatform.
4. **Robot hardware:** a harness or gantry, the knee torque limit in the motor drivers, a payload mount at the
   pelvis.
5. **The drift test on a real dish**: whether a real culture keeps its rates and pathways steady under the task's
   stimulation, which is what blocks our model. Public data answers the rate half (`real_drift.py`, the cart-pole
   organoids' game logs, 98 runs without training): over 15 minutes of input pulses alone, real rates settle slightly
   (last 3 minutes 0.77× the first 30 s, 95th percentile 1.18×), and over hours they wander by up to ~1.5–2×. The
   pathway half can't be read from them: the smallest change they could show (2 SE, 0.43 spikes per pulse) is
   larger than the gains themselves (median 0.08), and with only 200 ms counts in a closed loop, whether those gains
   are evoked spikes can't be checked. FinalSpark's exports (26 November, `real_pathways.py`) have spike times but
   don't answer it either: no stimulation setting repeats in fs369's mapping session; its one repeated single-input
   series and a 30-minute whole-array session show no stimulation artifact, so they never reached the array; and the
   series fs437 did deliver to a live organoid evoke nothing. They do show that real pathways are specific: in the
   two live organoids 11 and 17 of 56 electrode pairs respond (about 1.4 spikes each), and the outputs' strongest
   inputs differ (4 and 6 different inputs).
