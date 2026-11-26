# Reading notes: what four recent papers mean for this project

25 November 2026. Four papers read against the repo's results. Nothing new was simulated for these notes. Numbers
marked **stored** come from result files already on disk, named where they appear. Everything marked **expected** is
reasoning, not a result.

**In short.** None of the four papers gives a dish a way past the "computes" test in
[the bar](route-to-computation.md#the-bar). What they describe (memory of recent input, prediction, steady codes,
computing by colliding waves), a filter of the input also provides. They help in three other ways:

1. **Track A2 can probably pass with a cheaper recalibration** (paper 3, stored A2 numbers). A new dish breaks the
   old decoder by shifting its output by 1–2.5 spreads, not by making it noisier. Refitting only the weight edit
   (two numbers per IMU channel) may succeed where refitting the whole decoder fell one episode short.
2. **A testable explanation of why track B closed** (paper 1). Activity spreading out from a stimulated site gives
   STDP a direction, and the input whose spread wins takes over. The winning input drives outputs up to 0.94 mm
   away. One rerun with a weight dump can test this. Paper 4's direction maps would show the same thing from the
   electrodes alone, which is the only way a real dish can be tested.
3. **Methods for the parked real-dish drift test** (papers 2–4): co-firing corrected for rate with a proper null,
   decoding across time, response geometry compared between minutes, and maps of which way evoked activity travels.

| Paper | What it shows, in its own system | What we take |
|---|---|---|
| Muller et al., *Neuron* 2026: traveling waves in cortex | Wiring that falls off with distance plus slow axons make activity travel; sparse waves change how strongly neurons answer input | Takeover hypothesis; our model sits in the "dense" regime; its delays are too short to hold memory; A1 state test; burst spread as a realism target |
| Verzhbinsky et al., *Nature Neuroscience* 2026 (article s41593-026-02403-z): co-ripples in human cortex | Neurons fire together more often during coincident ~90 Hz ripples, at any distance | Co-firing measure (STTC), duplicate-unit merging and per-culture statistics for the real-dish pathway test; three traps |
| Vishne, Gerber, Knight & Deouell, *Cell Reports* 42:112752, 2023: sustained visual representations | Responses fade by ~80% while an image stays on, yet a fixed linear readout decodes it the whole time | A2 refit-the-edit arm; decoding across time in the drift runs; geometry reliability; analysis methods |
| Xu, Long, Feng & Gong, *Nature Human Behaviour* 2023 (doi 10.1038/s41562-023-01626-5): rotating waves in human fMRI | Slow brain-wide activity forms many rotating waves; where they sit and which way they turn depends on the task | The takeover test from electrode spike times alone; pattern classes and nulls for burst spread; five traps |

## 1. Traveling waves: Muller et al., *Neuron* 2026

**What it says.** Two ingredients make activity travel across cortex as waves: connections whose likelihood falls with
distance, and slow conduction along axons (0.1–0.6 m/s).
- In real cortex, with many weak synapses, waves are **sparse**. Few neurons fire as a wave passes, but the wave changes
  how strongly they answer input. Between wave phases, background firing about doubles and responses to a stimulus
  change by 10–20%.
- Small networks with strong synapses give **dense** waves instead, in which most neurons fire.
- The claims about memory and prediction come from network models with delays of tens of milliseconds. The same
  networks lose them when the wiring is random (Benigno et al. 2023).

Our model has both ingredients: wiring that falls off over 150 µm, and conduction at 300 µm/ms = 0.3 m/s
(`culture/network.py`, `CultureConfig`). So the review applies to it directly.

### A mechanism for the takeover that closed track B (hypothesis, untested)

Track B closed because, under repeated stimulation, one input ends up driving both outputs
([route doc](route-to-computation.md#track-b-a-culture-model-good-enough-for-learning-questions)). The proposed
mechanism, step by step:

- **Without a consistent order, our STDP weakens synapses.** The drift runs use a_plus 0.01 and a_minus 0.012, both
  with τ 20 ms (`cartpole_replication.make_stdp`), so depression outweighs potentiation (0.24 vs 0.20) when
  pre- and postsynaptic spikes come in random order (Song, Miller & Abbott 2000). Net strengthening needs a
  consistent "sender fires before receiver".
- **Spreading activity supplies that order.** Neurons near the stimulated electrode fire first. Synapses pointing away
  from it strengthen on every pulse, and synapses pointing back weaken.
- **That loop picks a winner.** The input whose spread covers the other input's area more often turns the synapses
  there its way. Its spread then becomes more reliable, which turns more synapses: positive feedback, winner takes
  all.
- **Rate brakes can't stop it.** Synaptic scaling multiplies all of a neuron's inputs by one factor, so it keeps
  their direction: rates are held, pathways aren't. That is track B's closing finding. Weight-dependent STDP limits
  how large weights get, not which way they point.

**Stored evidence that fits.** The table lists each takeover from the last minute of the drift runs
(`outputs/drift_scaling/drift_scaling.json`, `drift_multiplicative_scaling.json`), with electrode positions on the
256-electrode grid at 100 µm pitch (`DRIFT_CONFIGS`). The last column is how likely a direct synapse is at that
distance, relative to zero distance (exp(−d / 150 µm)).

| Culture | Winning input → other input's output | Distance | Largest cross gain, spikes per pulse (arm) | Direct synapse, relative |
|---|---|---|---|---|
| 0 | 136 → 139 | 0.30 mm | 25.4 (additive, scaling 1e-5) | 13% |
| 1 | 129 → 35 | 0.63 mm | 20.1 (weight-dependent, scaling 1e-5) | 1.5% |
| 1 | 34 → 113 | 0.51 mm | 4.7 (additive, scaling 1e-4) | 3% |
| 2 | 100 → 233 | 0.94 mm | 35.8 (additive, scaling 1e-5) | 0.2% |
| 2 | 217 → 116 | 0.78 mm | 8.6 (weight-dependent, scaling 1e-5) | 0.5% |

Distance doesn't cap the takeover: it reaches 20–36 spikes per pulse at 0.63–0.94 mm, as large as the 25 at
0.30 mm. At 0.94 mm a direct synapse is about 0.2% as likely as at zero distance, so this takeover travels through
the network, not along direct synapses.

**Checked and not supported:** "the burstier input wins". When the electrodes were chosen, the two inputs set off a
burst on 6 and 6 (culture 0), 6 and 4 (culture 1, electrodes 129 and 34), and 6 and 5 (culture 2, 217 and 100) of
25 pulses (`outputs/cartpole_replication/run_task.log`). In cultures 1 and 2 the input that wins most arms is the
*less* bursty one, and in both cultures the winner flips between arms.

**Test (proposal 2 below).** The drift runs empty their checkpoint folder when they finish, so no weights are stored.
1. Rerun one takeover arm per culture and dump the weights at the end. Runs repeat bit for bit, so the rerun follows
   the stored path; its per-minute gains must equal the stored rows.
2. Score each excitatory synapse by its direction relative to the winning input: +1 if it points straight away,
   −1 if it points straight back.

The hypothesis predicts that synapses pointing away gained and those pointing back lost. In the scaling-alone arm,
where weights move with no timing rule, there should be no such pattern. If the pattern is absent, the hypothesis is
wrong and the takeover goes through something else, for example a few hub neurons. If it is present, B2's candidate
brakes, which all act on rates or on single synapses, miss the actual problem.

**Result (25 November 2026): present, in all three cultures** (`takeover_rerun.py`; each rerun equals its stored row
bit for bit). Away synapses gain 3.1–4.9, those pointing back lose 1.1–1.35; the score is 4.3–6.0 against
0.009 under scaling alone. Mapped over all 256 electrode positions, it peaks exactly at the winning input in each
culture and fades with distance (r = −0.99). Hubs don't explain it: the top 1% of neurons hold 3–4% of the gain.
Scaling alone also tilts the score past 2 SE (z 17–18, about 1/500 of the size), so "no such pattern" holds only in
size, not in significance. Open: whether the orientation causes the takeover or grows with it; both happen in the
first minute.

### Our model is in the dense regime

The review warns that if neuron count, synapses per neuron and synapse strength don't match real tissue, waves turn
dense and force neurons to fire together. Our cultures have 60–100 strong synapses per neuron. The symptoms we
measured:
- every probe sets off the same network burst (`coach_probe.py`);
- a pulse sets off a burst in 16–24% of trials, against 6.5% in the cart-pole organoids
  (`public_data_targets.py`).

An untried lever: more synapses per neuron, each weaker, with the same total input. `burst_probe.py`'s "sparse"
condition went the other way (100 → 40 synapses). Suppressing bursts didn't make pairing specific there, so this is
a lever to test, not a known fix. The review's scaling argument concerns living cortex, not cultures. The target
would be about 6.5% of pulses setting off a burst, with spontaneous activity inside the Wagenaar ranges. Cost
(arithmetic): gpu_20k at 1,000 synapses per neuron holds roughly 1–2 GB of synapse data, which fits the 8 GB card,
but plastic runs would be several times slower.

**Result (26 November 2026): the lever doesn't work in this model; the cultures go silent** (`weak_synapses.py`, step
4; the track was reopened for this round). 1,000 synapses don't fit at the model's 150 µm reach (about 194 at most),
so the round was a dose series:
- ×1.9 synapses at the same reach, ×4 at 225 µm and ×10 at 370 µm;
- each synapse weaker by the same factor;
- each neuron's total input within 1% of now.

In all 15 cultures (3 × 5), not one network burst occurred in 10 minutes. The electrode rate was 0.54–0.56 Hz
(real 0.7–10.3), and neurons fired 18–28 times less. The evoked share came out 0–4%, which is in range. Halving each
synapse is already enough. The model doesn't get sparse; it stops. A reading, not tested: its bursts need strong single inputs. Track B is
closed for good.

### The model can't hold memory in its delays

Each synapse's delay is 0.5 ms plus distance ÷ 300 µm/ms, capped at 4 ms (8 buckets of 0.5 ms). With wiring that
falls off over 150 µm:
- the average connection is about 300 µm long, so about 1.5 ms of delay;
- under 1% of synapses reach the cap;
- the average delay is shorter than the 5 ms synaptic time constant.

The review's memory mechanism needs delays of tens of milliseconds, long compared with that time constant, so our
model excludes it by construction. That fits `dish_regimes.py`: the dish's R² for the IMU 120 ms earlier is only
0.02–0.13. Longer delays are cheap, since they cost spike-history memory and no extra work per synapse. But they
wouldn't pass the bar: a filter of the input also remembers and predicts, and for a smooth 6-value IMU a linear
predictor is close to the best possible.

### Track A1: why the live culture relays worse than the dead one

The same input gets a bigger or smaller response depending on the network's state when it arrives. For a relay, that
is noise. A dead culture, with no recurrent synapses, has no such state. In A1 on a 1 × 12 well, the live culture
reached 23/40 and the dead one 32/40 (`closed_loop_vestibular.py`).

In our model the state is mostly synaptic resources, which sit at about 0.6 and are drained by the culture's own
bursts (`dish_regimes.py`). A1's remaining falls follow the relay reading the pitch rate too high during push
recoveries. Hypothesis (untested): large inputs pull extra spikes out of the network.

**Test (proposal 3):**
1. Replay the 17 failing seeds with spike counts across the whole well logged at every step.
2. Relate the decoding error to the activity just before each step. Expect a relation in the live culture and none
   in the dead one.
3. Train the decoder in one state (high resources) and test it in the other (paper 3's decoding across time,
   applied to states).

If the relation shows up, give the decoder one extra input: the earlier activity multiplied by the current counts.
That would still be a relay, not computation.

### A realism target from data already on disk

The Wagenaar dense recordings (`outputs/public_data/raw/`) have spike times and electrode numbers at 200 µm pitch.
That is enough to measure how bursts spread: where they start, how stable the start sites are, and how long a burst
takes to cross the array. The model has never been compared on this. The review's methods give the cautions:
- **Use spike timing, not narrowly filtered signals.** Narrow filters create waves that aren't there.
- **Compare against a null with shuffled electrode positions.**
- **Remember coarse sampling.** It can make separate clusters firing in turn look like one wave (Orsher et al. 2024).

Electrode positions must come from the dataset's documentation, not be assumed. Bursts may also start off the array:
the gpu_20k culture is 4 mm across, and the 256-electrode grid covers only the central 1.5 mm. Check the method on the
model first, where every neuron's position and spike is known, then apply it to Wagenaar (proposal 4).

**Result (25 November 2026): the method is not validated** (`burst_spread.py`). On the model it recovers the wave
speed (electrode against true speed 0.95–0.99), but not the start site to one pitch (578 µm with the pre-registered
truth, 239 µm with a labelled, sturdier truth; gate 200) nor the pattern class (51–52% agreement; gate 70%). Most
model bursts start off the array: 67–82% by the two truths. Applied to Wagenaar anyway and read with those limits,
real bursts are messier than the model's (41% fit no simple pattern, against 2–15%) and start at fewer favoured
sites (13 electrodes cover 80% of starts, against 31–36), while the speed matches (real 30 mm/s [IQR 21–45], model
31–41).

### For the parked real-dish work

- **A learning signature that needs no task decoder.** After a stimulus was shown many times, spontaneous waves came
  to look like the waves that stimulus evoked (Han, Caporale & Dan 2008, anaesthetised rodent visual cortex). Our
  runaway would pass this for the wrong reason, so use it only together with the drift gate.
- **Probing that accounts for state.** In coaching, log the time since the last burst at every probe, and either use
  it as a covariate or avoid probing in the depleted window after a burst.
- **Topographic coding**, already in [organoid-control-ideas.md](organoid-control-ideas.md), is supported only
  where wiring falls off with distance and delays are long. That gives it a control: the same network with shuffled
  connections should lose the benefit.

**What doesn't transfer:**
- The evidence is from living cortex, mostly awake primates.
- The tempo differs: Wagenaar's cultures burst about every 12 s (median), against about 10 waves per second in the
  primate recordings.
- The review's section on effects of the electric field changes nothing for us.

## 2. Co-ripples: Verzhbinsky et al., *Nature Neuroscience* 2026

**What it says.** In 35 epilepsy patients doing a memory task, the authors recorded ripples: brief ~90 Hz bursts,
about 72 ms long, about 0.5 per second per channel. When ripples coincide at two sites ("co-ripples"), neurons at
those sites fire within 25 ms of each other about 30% more often, at any distance up to 220 mm. Co-rippling rises
with memory load. Neuron pairs that fired together when an image was learned fire together again when it is probed
(0.29% vs 0.14% of pair-trials, and more often on fast answers).

**Worth borrowing for the real-dish pathway test** (parked item 5; it needs spike times):
- **STTC (spike time tiling coefficient):** a co-firing score corrected for firing rate. Under stimulation our rates
  climb, and raw co-firing would climb with them. Set its window to our read window.
- **The rate-only null as a floor:** expected co-firing = rate A × rate B × 2 × window.
- **Duplicate units:** merge units with more than 30% of their spikes within 1 ms of each other, and never pair two
  units from the same electrode. This matters on high-density arrays (MaxWell, our `hd1024`), where one neuron shows
  up on neighbouring electrodes.
- **Statistics per culture.** Their per-patient test is the trustworthy one, and per-culture is the level our gates
  already use.

**Traps not to copy** (reasoning from the paper's own numbers):
1. **Long-range co-ripples look like chance.** Two independent ripple trains at 0.52 per second, with ripples of
   ~72 ms, overlap by at least 25 ms by chance in 0.52 × (72 + 72 − 2 × 25) ms ≈ 5% of ripples. Between electrode
   bundles they report a median of 5% (4–6%), flat with distance, which is what chance predicts. Within a bundle,
   13% is above chance. The paper gives no chance level. Chance overlap also grows with the product of the two
   sites' ripple rates, so a ~3% rise in each site's rate under memory load would by itself produce their +6% in
   co-rippling. They don't report per-site numbers by load.
2. **"Beyond rate" is weaker than stated.** STTC and their null correct for average rates, not for rates that rise
   and fall together. Both neurons fire more at the centre of their own ripple, and requiring 25 ms of overlap lines
   those peaks up. That alone beats an average-rate null; even their periods without ripples sit above it
   (+0.038 Hz). A fairer null pairs neuron A's spikes from one co-ripple with neuron B's from another, both aligned
   on the ripple centres. We would need the same, because responses locked to pulses create false coupling in the
   same way.
3. **Their "reinstatement" would pass by reflex in a dish.** The same image activates the same image-selective
   neurons, so they fire together again whether or not anything was remembered. Their label shuffle shows the
   pattern is image-specific, not that it is memory. In a dish, the same pulse gives the same response. Our bar
   already handles this: history has to change the response beyond what the input explains.
4. **Their pooled p-values overstate certainty.** The pooled tests treat each neuron pair and trial as independent,
   though pairs share neurons and patients (hence p = 0 and 1e-16). Trust the per-patient result.

**Not useful:** the public dataset (DANDI 000673) is human brain recordings, not cultures. The ripple detector needs
raw voltage, and FinalSpark delivers spike events only. The link to STDP in their discussion is a suggestion, not a
measurement.

## 3. Sustained representations: Vishne et al., *Cell Reports* 2023

**What it says.** Ten epilepsy patients with electrodes on the surface of the cortex viewed images for 300, 900 or
1,500 ms. The measure is high-frequency activity (70–150 Hz), a stand-in for local firing.
- **Single electrodes fade.** By 800–900 ms, responses are down 82% from their peak, about five-fold, and the share
  of variance explained by image category is down 77%.
- **The population still decodes, for as long as the image is shown.** In visual cortex (occipital and ventral
  temporal), a linear classifier tells faces from watches at AUC 99.8% (ventral temporal) and 92.4% (occipital),
  averaged over 100–900 ms. Decoding ends 230–450 ms after the image goes off, so it tracks the image's duration.
- **The readout direction stays fixed.** A classifier trained at one moment works at all the others ("temporal
  generalization"). The direction that separates categories stays put while response size falls; the authors call
  it a stable "experience subspace".
- **Frontal and parietal cortex hold it only briefly.** There, categories are decodable 150–600 ms after onset,
  whatever the duration.
- **Single images are reproducible too.** The pattern for one image repeats across showings in ventral temporal
  cortex, even after category structure is removed (representational similarity analysis: item and geometry
  reliability, each against a null with image identities shuffled).
- **Methods:**
  - shrinkage LDA (Ledoit–Wolf), scored by AUC;
  - five-fold cross-validation repeated five times, with balanced classes;
  - repeats of one image averaged, so an image is never in both training and test;
  - permutation tests corrected by max-statistic or clusters, plus point-by-point FDR wherever timing is claimed
    (cluster tests can't date an effect);
  - electrode counts matched between regions (36 electrodes, 1,000 random subsets);
  - the central decoding results also shown in single patients.
- **Availability:** data on OSF (10.17605/OSF.IO/4HXPW), code on Zenodo (10.5281/zenodo.8051237), CC BY.

### Track A2: a new dish shifts the decoder's output, it doesn't scramble it (stored)

The paper's point, in our terms: a fixed linear readout survives large changes in response size as long as the
pattern's direction holds. Only scale and offset then need correcting.

A2 (`closed_loop_sessions.py`, 1 well × 32) tested two ways to start a session:
- **stale:** keep the old decoder and its weight edit;
- **recalibrated:** refit both decoder and edit from 60 s of open-loop replay.

It never tested the middle way: **keep the old decoder and refit only the edit.** That is the literal form of route
3's claim, "per-session recalibration by one weight edit".

Stored numbers (`outputs/closed_loop_sessions/results.json`): open loop, on held-out steps near the standing
posture, in teacher spreads. "Error after the edit" is noise around the fitted line divided by the gain, which is
what the teacher sees once the edit rescales the channel.

| | gyro_x | gyro_y | gyro_z | grav_x | grav_y |
|---|---|---|---|---|---|
| Day 0, stale decoder: error after the edit | 0.53 | 0.52 | 0.62 | 0.69 | 0.49 |
| New dish, stale decoder: error after the edit | 0.68 | 0.73 | 0.70 | **0.98** | 0.59 |
| New dish, stale decoder: offset | −1.01 | −1.03 | +2.49 | −0.41 | −0.14 |
| New dish, 60 s decoder: error after the edit | 0.62 | 0.75 | 0.75 | **1.70** | 0.71 |

Reading:
- **Error.** Once scale and offset are corrected, the stale decoder's error on the new dish is close to the 60 s
  decoder's on the gyros. On grav_x, the channel that limits the 60 s decoder, it is well below (0.98 vs 1.70).
- **What breaks it.** The offset: 1–2.5 spreads on the gyros, the same failure as A1's first run.
- **Closed loop agrees.** On the new dish the stale decoder's noise is 0.41–0.61 spreads, against 0.40–0.89 for the
  60 s decoder, and its offsets are −1.05, −0.84, +2.60 and −1.04 on four channels. Those episodes fell within
  3.3 s, so these figures are less certain.
- **One difference against it.** The stale decoder's noise is slower: it stays correlated for 1.2–1.7 steps,
  against 0.8–1.05 for the 60 s decoder.

Why this happens in our model (reasoning): the new dish sits on the same electrodes with the same encoder, and the
dish mostly relays what stimulation drives directly (a dead culture relays as well). So the electrodes' response
pattern carries over, and what changes is the baseline firing under each electrode.

On a real new organoid, the cells under each electrode differ far more, so this need not hold between dishes.
Between sessions of one dish, which is route 3's case, it decides the protocol, and it can be measured: train the
decoder in one session and test it in the next. If it holds, refit only the edit; if not, refit the decoder.

**Proposal A2c (proposal 1 below).** A labelled extra arm, on fresh seeds:
- **What:** for each A2 condition, the stale decoder with an affine edit (gain and offset per channel) refitted on
  the same 60 s of replay the recalibrated arm used, scored on held-out steps as before.
- **Shorter replays too:** also fit the edit from 10 s and 30 s. Two numbers per channel need far less data than a
  ridge decoder.
- **Gate, fixed in advance:** ≥ 34/40 in every condition.
- **Expected:** at least the recalibrated arm's 33–37/40, and more on the new dish, where grav_x error drops from
  1.70 to 0.98. It may not reach day-0 stale's 39/40 (grav_x 0.98 vs 0.69).
- **Result (25 November 2026): passed.** From 60 s: 37–40/40 in every condition, the new dish 37/40; the recalibrated
  arm on the same seeds 30–36/40. Closed loop, grav_x noise after the edit: 0.98 on the new dish (as predicted from
  the stored sessions), 1.61 for the recalibrated decoder. 30 s is as good; 10 s pins the offsets but leaves two
  new-dish gains 22–29% low (36/40). Details in `closed_loop_sessions.py`. The caveat above stands: in the model a
  new dish is the same network type under the same electrodes, so the decoder's transfer between dishes is a model
  result; between sessions of one dish, route 3's case, it is the protocol to test first.

### Track B: does the takeover destroy information, or only capture the readout?

The paper cites work proposing that single neurons' tuning can drift while a population readout stays stable, if the
drift runs along directions the readout ignores (Rule et al. 2020; Rule & O'Leary 2022; not read for these notes).
Our drift gate reads two fixed decoder electrodes, as the cart-pole protocol does, so any drift at those electrodes
counts against it. That gate stays as it is.

A second measure would show what kind of failure the takeover is. Decode from all electrodes which input was
pulsed: train in minute 1, test in every later minute.
- **Decoding stays high while the two decoders mix:** the inputs stay separate in the network, and only the two
  readout electrodes are captured.
- **Decoding falls to chance:** the network answers both inputs alike, the picture from `coach_probe.py`.

Before any training, `burst_probe.py` decoded the probed site at 75–99%. This costs nothing extra: it rides on
proposal 2's rerun by logging all electrodes' counts per pulse.

### For the real-dish drift test (parked item 5)

- **Geometry reliability.** Stimulate four or more sites and compare the dissimilarities between their responses in
  minute 1 with those in minute N, against a null with the sites shuffled. This needs no decoder and no chosen
  electrodes. With two sites, as in our drift test, the geometry is a single number, so use decoding across time
  instead.
- **Decoder across sessions**, to choose between refitting the edit and refitting the decoder (above).

### Methods to adopt

- Score by AUC, as `trigger_probe.py` already does. For many electrodes and few trials, use shrinkage LDA;
  `coach_probe.py` and `burst_probe.py` used nearest centroids.
- Never let repeats of one stimulus fall in both training and test. Contiguous folds over time, as in
  `vestibular_probe.py`, already do this for streams.
- Compare conditions at equal electrode counts, by subsampling.
- Cluster tests don't date an effect. A claim about *when* (for example "the runaway starts within the first
  minute") needs point-by-point correction.
- Report per culture, never pooled into one decoder across cultures. The paper pools electrodes across patients and
  checks single patients afterwards; our gates are already per culture.
- Choose electrodes on data independent of the test. They chose by responsiveness, blind to category and time.

### A model gap it points to

The paper's limitations cite Mongillo, Barak & Tsodyks 2008: memory held in synapses by short-term facilitation,
without sustained firing. Our model has short-term depression only (`culture/network.py`); facilitation is untried.
It would make the model more realistic. It would not be a route, because a filter of the input holds memory too.

**What doesn't transfer.** This is human cortex, recorded as high-frequency power summed over millimetres of tissue,
framed around conscious perception. It says nothing about cultures or learning.

## 4. Rotating waves in human fMRI: Xu, Long, Feng & Gong, *Nature Human Behaviour* 2023

**What it says.** The data are human brain scans (fMRI, Human Connectome Project): 100 people at rest, and three
other groups of 100 doing language or working-memory tasks, with one image every 0.72 s at 2 mm resolution. Only slow
fluctuations are kept (0.01–0.1 Hz), and a spatial band-pass keeps patterns 70–138 mm across.
- **How they find the waves.** Each voxel's phase comes from a Hilbert transform. The phase map's gradient gives the
  direction activity travels at each point (a "phase vector field"), and its curl marks rotating patterns ("brain
  spirals") around points where all phases meet.
- **At rest.** About 16 spirals at once per hemisphere, 4–31 mm in radius.
  - They turn at 0.36 rad/s on average (one turn in about 17 s).
  - Their centres drift at about 2 mm/s, and further than a random walk would (exponent 1.5).
  - They gather on borders between functional regions.
  - When two meet, they annihilate (51%), partly annihilate (46%) or repel (2.6%).
- **During tasks.** The rotation direction at a given place depends on the task.
  - A template classifier given only the spirals' centres and rotation directions tells four language conditions
    apart 48–50% of the time (chance 25%, null 24%). Amplitudes at 19 random voxels give 30%.
  - Working memory: stimulus type 44% (chance 25%), memory load 67% and right vs wrong answers 59% (chance 50% for
    both).
- **Proposed routing rules.** Activity passes between two spirals turning opposite ways (a "gate"), and is diverted
  between two turning the same way (a "wall"). Arrays of spirals route activity up the cortical hierarchy while
  listening and down while answering. A 30 × 30 lattice of coupled phase oscillators reproduces the collisions, and
  reversing spirals by hand there reverses the flow.
- **Controls.** The null randomises the Fourier phases in space and time together, which keeps both autocorrelations.
  Spatial filtering creates spirals even in that null, so a spiral counts only where the filtered and unfiltered
  phases agree (within 30°) more often than in 95% of null cases. Spirals also appear in unfiltered signals averaged
  over trials.
- **Availability:** MATLAB code at github.com/BrainDynamicsUSYD/BrainVortexToolbox; the data are from the HCP (human
  scans, of no use as culture targets).

### What transfers: little of the biology

This is whole human cortex, seen through a slow blood-flow signal: waves centimetres across, turning once in about
17 s. Our model cultures are 2–4 mm across and work in milliseconds. The paper cites rotating waves in intact cortex and in
slices of rodent neocortex (Huang et al., *Neuron* 2010), none in dissociated cultures or organoids. I have not checked
whether cultures show them.

Whether our model could hold a rotating wave is a measurable question, not one to assume. The condition, standard for
excitable media: a wave keeps circling only if one lap takes longer than the tissue needs to recover. Our model's
synapses recover over 2 s (`stp_tau_rec_ms`), but how fast its bursts travel has never been measured. Proposal 4
measures the speed, after which the condition can be checked.

Measured since (25 November 2026, `burst_spread.py`): bursts travel at about 40–50 mm/s, so a lap around a circle
1 mm across takes about 60–80 ms, far less than the 2 s recovery. By the condition above (reasoning, not
simulated), the model can't keep a wave circling; its few "rotating" bursts (6–9%) are single sweeps.

### Usable: the takeover test from electrodes alone

A real dish's weights can't be read, so proposal 2's weight dump has no dish version. If the takeover hypothesis
(paper 1) holds, it should also show in the electrodes' spike times:
- **Direction field.** For each pulse, take each electrode's first-spike latency. Its gradient over the grid gives the
  direction the evoked activity travels at each point: the paper's phase vector field, built from spike timing as
  paper 1 advises. A Hilbert phase suits ongoing oscillations, not single evoked bursts.
- **Where the inputs still differ.** Average the direction field per input. Then, for each 3 × 3 patch of electrodes,
  ask whether single pulses match their own input's average better than the other input's (the paper's local
  direction classifier, Fig. 8c). The result is a map of where on the array the two inputs' spread still differs.
- **Prediction.** Over a takeover run, the area where the inputs differ shrinks. Around the losing input's output
  electrode, the direction turns to point away from the winning input.

Proposal 2's rerun can log every electrode's spike times and check this against the weights. If the electrode measure
follows the weights, it becomes the pathway test a real dish needs (parked item 5), which public data couldn't answer.

**Result (25 November 2026).** The direction field follows the weights: the winner's first-spike gradient lines up
with the weight change at mean cosine 0.84–0.87 by minute 10, over all 256 electrodes. The area prediction was wrong
in direction: the area where the inputs differ grows from 12–17% (scaling alone) to the whole array, because the
winner's pulses come to set off activity everywhere while the loser's stay local (≥ 30% response on ≤ 6
electrodes), so the loser has no direction field to turn. For a real dish the sign of a takeover is therefore one
input's first-spike map spreading over the array while the other's stays local.

The grid matters: gradients need neighbours on every side (their smallest spiral core is 3 × 3 points).
- Our 16 × 16 grid at 100 µm and high-density arrays (MaxWell) are fine.
- An 8 × 8 grid at 200 µm (our `mcs60`) is coarse.
- FinalSpark's 8 electrodes per organoid can't do it.

### Usable: patterns and nulls for proposal 4

- **Classify each burst's spread on the grid:** spreading out from a source, a plane wave, or rotating (curl and its
  sign).
- **Agreement test.** Smoothing creates rotations even in noise. Keep a pattern only if it is also there before
  smoothing, beyond the null.
- **Use a different null.** Theirs keeps both autocorrelations but makes the signal Gaussian-like, and bursty spike
  rates aren't (reasoning). Use nulls that keep each electrode's own spikes (shift each electrode's train in time, or
  shuffle electrode positions), or an amplitude-adjusted surrogate.
- **Don't hunt for spirals first.** Rotating waves and the gate/wall rules have no known dish counterpart: report them
  if found.

### Traps, from its methods

1. **Scale chosen on the outcome.** The spatial filter band was chosen where task classification scored best, and
   the accuracy is then reported at that band as the "maximum accuracy". Fix analysis settings before looking, as our
   gates do. For proposal 4, choose the smoothing scale on the model.
2. **An unfair baseline.** Spiral features summarise the whole cortex's phase map and sit at informative places, while
   the amplitude baseline reads 19 random voxels per time step. So "spirals carry more than amplitude" isn't shown. A
   fair stand-in gets the same input, as our bar requires.
3. **Pseudo-replication.** Error bars and t-tests are over 100 random re-splits of the same trials (n = 100
   iterations), and the same subjects' trials fall in both training and test. The honest unit is the held-out
   subject; for us, the held-out culture.
4. **Uncorrected maps.** Contrast maps are thresholded at one-tailed p < 0.05 per voxel, with no correction for
   multiple comparisons (Fig. 5 caption).
5. **A circle in PCA says nothing about spirals** (reasoning). A plane wave sin(k·x − ωt) equals
   sin(k·x)·cos ωt − cos(k·x)·sin ωt: two fixed maps weighted by cos ωt and sin ωt. So PCA finds exactly two
   components and traces a circle, and so does any wave crossing an array. Their check (a simulated spiral gives a
   circle) shows a spiral is enough, not that one is needed. For us: circles in PCA or CEBRA embeddings of MEA data
   are expected whenever a burst crosses the array, and are no sign of computation.

### No route past the bar

The discussion points to computing by colliding waves (Gong & van Leeuwen 2009, not read for these notes). Reasoning:
a collision computes a function of where and when the inputs arrived within a short window. That information is in
the input, so a short filter of the input can match it. Like the memory and prediction in papers 1 and 3, it doesn't
pass the "computes" test.

## Proposals, in order

None has been run. Each needs a go-ahead and follows the usual guarded-run rules (smoke test, in-run checks,
watchdog, fresh seeds, standard baselines). The order adopted afterwards is item 4 of the route doc's
[Order](route-to-computation.md#order):
1. proposal 1 (A2c);
2. proposal 2;
3. a realism scorecard that includes proposal 4;
4. proposal 5, only by decision;
5. A1c (a wider encoder input limit), optional.

Proposal 3 is dropped.

| # | What | Track | Needs | Reading fixed in advance |
|---|---|---|---|---|
| 1 | **A2c:** stale decoder, only the edit refitted (from 60 s; also 10 s and 30 s) | A | A2's session setups (`outputs/closed_loop_sessions/sessions/`), 5 conditions × 40 fresh seeds per replay length | ≥ 34/40 in every condition |
| 2 | **Takeover rerun** with a final weight dump and every electrode's spike times: one takeover arm per culture plus the scaling-alone arm as control (6 arms × 600 s; the weight-dependent check's 9 arms took 2–2.5 h) | B (diagnostic) | — | Rerun equals the stored rows. Synapses pointing away from the winner gain more than those pointing back, by > 2 SE in all 3 cultures and not in the control. Reported alongside: decoding of the pulsed input across minutes (paper 3), and the electrodes' direction maps compared with the weights (paper 4) |
| 3 | **A1 state test:** replay the 17 failing 1 × 12 seeds with whole-well logging. *Dropped (25 November): it explains a relay flaw but changes no decision* | A | A1b setups | Decoding error tracks prior activity in the live culture, not in the dead one |
| 4 | **Burst spread:** start sites, their stability, crossing times (the wave speed) and pattern class (source, plane, rotating), against nulls that keep each electrode's spikes | B0 | Model first, then the Wagenaar spike files | The method recovers the model's known spread before it is applied to Wagenaar |
| 5 | **Many weak synapses**, same total input. *Failed (26 November): the cultures go silent (0 bursts per minute); track B closed for good* | B, only if reopened | Drift reruns, slower | ~6.5% of pulses set off a burst; spontaneous activity inside the Wagenaar ranges |
| parked | Real dish: STTC with an event-shift null, duplicate-unit merging, geometry reliability over time, decoder tested across sessions, time since the last burst as a covariate, and the takeover test from direction maps if proposal 2 validates it | dish | A culture | — |
