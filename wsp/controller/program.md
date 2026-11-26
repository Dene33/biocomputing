# Weight-space programming: the robot-side controller

Method from [csinva/weight-space-programming-autoresearch](https://github.com/csinva/weight-space-programming-autoresearch):
an agent writes a model's weights in code, guided by one validation score. Plan: `docs/next-steps.md`, Phase 2.

## Task

Asimov 1 on `stand_push`: stand for 20 s while 120–200 N pushes hit the pelvis every 3 s. The controller drives
all 23 joints. It sees the joint encoders and its own last command exactly. Its balance signal `vest` (gyro ×0.25
and projected gravity, `obs[0:6]`) comes through the **dish model** in `dish_model.json`: since Phase 3.1 the measured CL1-sized
channel, with a gain, noise and correlation time per channel and no added delay or hold (`delay: 0, hold: 1`).

## Files

| File | Role |
|---|---|
| `track_a.py` | Track (a): the teacher's weights (78 → 512 → 256 → 128 → 23, ELU), edited |
| `track_b.py` | Track (b): a transparent controller written from scratch |
| `eval_controller.py` | Runs 60 validation seeds; prints `score:` (mean survival, s) and logs a row in `results.csv` |
| `dish_model.json` | The channel the vestibular signal passes through |
| `teacher_weights.npz` | The teacher's weights (from `uv run --with onnx python wsp/controller/export_teacher.py`) |
| `log.md` | One line per iteration: hypothesis, change, score |

A controller file defines `write_weights() -> dict` (every parameter, written in code), `act(W, jpos, jvel,
last_action, vest) -> (23,)` and optionally `reset(W)` (clears filter state at each episode start). Inputs are in the
teacher's observation units: `jpos = obs[9:32]`, `jvel = obs[32:55]`, `last_action = obs[55:78]`, `vest` = the dish's
estimate of `obs[0:6]`.

## Rules

1. **Weights are written in code.** No gradient descent, regression fits or other training inside the loop. Fixed
   arithmetic is fine: filters, predictors, rescaled or hand-set weights, features derived from physics.
2. **numpy only** in `act`; it runs once per 20 ms step.
3. **The dish must stay necessary.** `silenced` (the dish off) must stay below 5 s, roughly zero action. A controller
   that balances without `vest` gets no credit, however high its score.
4. **Validation seeds only.** `--test` (hidden seeds 8000–8039) runs once per track, at the end.
5. **Log every attempt**, including failures, with `--note`.

## Loop (≥ 20 iterations per track)

1. Read `log.md` and the last rows of `results.csv`.
2. State a hypothesis, then edit the controller.
3. `uv run python wsp/controller/eval_controller.py wsp/controller/track_X.py --note "iter N: <change>"`
4. Keep the change if the score rises by more than ~0.6 s (the standard error at 60 seeds is ~0.5 s: seeds survive either 20 s or ~5 s); otherwise revert.
5. Add a line to `log.md`.

## Baselines (same dish model, validation seeds)

- Teacher unchanged: track (a) iteration 0.
- Zero action (default pose): track (b) iteration 0.
- DAgger MLP student trained with the same dish noise: `scripts/no_imu_student.py`-style, run separately.
