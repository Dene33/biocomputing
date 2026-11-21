# WSP log

Dish model: `dish_model.json` (placeholder: 40 ms + 25 Hz + noise 1.0 × spread, τ 200 ms). Score = mean survival
(s) over the validation seeds. Iterations 0–13 used 20 seeds. Seeds survive either 20 s or ~5 s, so the standard
error there was ~1 s, not the 0.3 s first assumed. From the re-scoring on, 60 seeds (standard error ~0.5 s).

## Track (a): edit the teacher

| Iter | Hypothesis / change | Score | Full | Kept |
|---|---|---|---|---|
| 0 | Teacher unchanged (baseline) | 17.79 | 16/20 | — |
| 1 | Less gyro gain (0.5) averages out noise | 9.23 | 0/20 | no |
| 2 | Gyro gain 0.75 | 16.42 | 10/20 | no |
| 3 | Gravity gain 0.7 | 13.76 | 7/20 | no |
| 4 | More gyro gain (1.25) offsets the delay (more damping) | 18.40 | 15/20 | yes |
| 5 | Gyro gain 1.5 | **18.69** | 17/20 | yes |
| 6 | Gyro gain 2.0 | 14.29 | 9/20 | no |
| 7 | Gyro 1.5 + gravity 1.25 | 17.90 | 14/20 | no |
| 8 | Rotate gravity 3 steps ahead with the gyro (dg/dt = g × ω) | 16.64 | 11/20 | no: right sign offline (error 0.0068 → 0.0047 on true signals), but it injects gyro noise |
| 9 | Same, 6 steps | 9.32 | 2/20 | no |
| 10 | Sign check, −3 steps | 17.94 | 17/20 | no |
| — | Gyro from leg kinematics (−J_rot q̇ of a planted foot), offline check | R² < 0 | — | no: the teacher steps almost constantly, no foot stays planted |

| 11 | Gyro gain 1.5 on x, y only | 18.48 | 16/20 | no (same, simpler without) |
| 12 | Gravity lead 0.5: extrapolate between the dish's samples | 17.69 | 15/20 | no |
| 13 | Gravity lead 1.0 | 19.31 | 18/20 | re-scored |

Re-scored on 60 seeds:

| Controller | Score | Full |
|---|---|---|
| Zero action | 4.74 | 0/60 |
| Teacher unchanged | 17.77 | 46/60 |
| **Gyro gain 1.5 (current track_a.py)** | **18.76** | **50/60** |
| Gravity lead 1.0 only | 18.19 | 46/60 |
| Gyro 1.5 + gravity lead 1.0 | 18.41 | 47/60 |

### Measured dish (Phase 3.1), from here on

`dish_model.json` is now the measured CL1-sized channel (1 well × 12 electrodes: gain ~0.63, noise 0.53–0.80 ×
spread, correlation 30–70 ms). The placeholder is kept as `dish_models/placeholder_spec.json`.

| Iter | Hypothesis / change | Score | Full | Kept |
|---|---|---|---|---|
| 14 | Teacher unchanged on the CL1 dish | 12.75 | 20/60 | baseline |
| 15 | Gyro 1.5 (from the placeholder) on the CL1 dish | 15.41 | 32/60 | — |
| 16 | IMU columns × 1/gain (gyro 1.5–1.65, gravity 1.43–1.59) | **19.82** | 58/60 | yes |
| 17 | Same, gyro × 1.5 more on top | 19.63 | 58/60 | no |
| test | Iteration 16, **hidden test** seeds | **20.00** | **40/40** | teacher unchanged: 13.68, 16/40 |

Notes. Observation joints are in slot order `[0,1,6,7,12,13,14,18,19, 2,3,8,9,15,16,20,21, 4,5,10,11,17,22]`
(MuJoCo order indices); joint velocities are ×0.1, gyro ×0.25.

## Track (b): from scratch

| Iter | Hypothesis / change | Score | Full | Kept |
|---|---|---|---|---|
| 0 | Zero action (PD to default pose) | 4.34 | 0/20 | — |
| 1 | Ankle pitch P on forward tilt, K = +8 (60 seeds) | 3.83 | 0/60 | no |
| 2 | Same, K = −8 | 4.07 | 0/60 | no |

Stance reflexes are worse than standing still, as in the earlier linear experts (README): `stand_push` needs steps,
so track (b) needs a stepping controller (e.g. capture point → foot placement → leg inverse kinematics).
