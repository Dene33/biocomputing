cd /mnt/datadisk/repos/biocomputing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
R=outputs/dagger/stand_push_mea_gpu_20k_grid256_w6_s0_mlp_eff
uv run scripts/eval_readout.py $R --seeds 20 --control none
uv run scripts/eval_readout.py $R --seeds 10 --control silent
uv run scripts/eval_readout.py $R --seeds 10 --control dead
uv run scripts/eval_readout.py $R --seeds 10 --control zero
uv run scripts/eval_readout.py outputs/dagger/stand_push_none_gpu_20k_hd1024_w1_s0_mlp_eff --seeds 20 --control none --workers 8
