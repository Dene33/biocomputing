# Results

## Parallel DAgger (scripts/train_dagger.py)

Final: 10 fresh push seeds. Ablations (scripts/eval_readout.py, 20 fresh seeds): `none` = as trained, `silent` = no sensory stimulation reaches the culture, `dead` = culture without recurrent synapses, `zero` = default pose.

| run | source | wells | encoding | decoder | rounds | best round (s) | final | ablations |
|---|---|---|---|---|---|---|---|---|
| stand_push_mea_gpu_20k_grid256_w6_s0_mlp_eff | mea | 6 | identity | mlp + efference | 16 | 20.0 | 18.8 s, 9/10 | dead: 14.5 s, 6/10; none: 18.5 s, 17/20; silent: 1.6 s, 0/10; zero: 4.3 s, 0/10 |
| stand_push_none_gpu_20k_hd1024_w1_s0_mlp_eff | none | 1 | identity | mlp + efference | 14 | 20.0 | 18.9 s, 9/10 | none: 17.5 s, 15/20 |

## Reservoir (DAgger + ridge readout)

| task | readout source | MEA | seed | train R² | validation R² | survived (s) | full episodes | distance (m) |
|---|---|---|---|---|---|---|---|---|
| stand_push | dead | mcs60 | 0 | 0.773 | -0.673 | 1.96 | 0/5 | -0.6 |
| stand_push | mea | grid256 | 0 | 0.94 | -0.075 | 1.89 | 0/5 | 0.1 |
| stand_push | mea | hd1024 | 0 | 0.95 | -0.409 | 2.34 | 0/5 | -0.57 |
| stand_push | mea | mcs60 | 0 | 0.734 | -0.969 | 2.96 | 0/5 | -0.38 |
| stand_push | none | mcs60 | 0 | 0.738 | -0.522 | 2.26 | 0/5 | -0.12 |
| stand_push | oracle | mcs60 | 0 | 0.96 | -0.306 | 2.4 | 0/5 | -0.35 |

## DishBrain feedback sessions

| protocol | task | perturbation | condition | seed | before (s) | first block (s) | last block (s) | after (s) | after rest (s) |
|---|---|---|---|---|---|---|---|---|---|
| scratch | stand_push | none | dishbrain | 0 | 5.91 | 5.07 | 4.77 | 3.62 | 7.32 |
