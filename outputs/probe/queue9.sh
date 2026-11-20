cd /mnt/datadisk/repos/biocomputing
export OMP_NUM_THREADS=8 PROBE_DATA=teacher_data_20k.pkl
P="uv run python outputs/probe/encoding.py run"
$P id_w6 projection=identity n_stim=75 encoder_gain=0.5 slots_ms=[0,5,10,15] 'mea_overrides={"stim_length_um":120.0}' mea=grid256 wells=6 diverse_wells=true
