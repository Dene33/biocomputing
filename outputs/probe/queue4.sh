cd /mnt/datadisk/repos/biocomputing
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
P="uv run python outputs/probe/encoding.py run"
S="projection=supervised"
($P stim200_wexc1.2 $S slots_ms=[0,5,10,15] 'mea_overrides={"stim_length_um":200.0}' 'culture_overrides={"w_exc":1.2}';
 $P stim120_wexc1.2_1slot $S slots_ms=[0] 'mea_overrides={"stim_length_um":120.0}' 'culture_overrides={"w_exc":1.2}';
 $P stim300c40_wexc1.2 $S slots_ms=[0,5,10,15] 'mea_overrides={"stim_length_um":300.0,"stim_current":40.0}' 'culture_overrides={"w_exc":1.2}') &
($P stim120_wexc1.2_bins4 $S slots_ms=[0,5,10,15] bins=4 'mea_overrides={"stim_length_um":120.0}' 'culture_overrides={"w_exc":1.2}';
 $P stim200_wexc0.8 $S slots_ms=[0,5,10,15] 'mea_overrides={"stim_length_um":200.0}' 'culture_overrides={"w_exc":0.8}';
 $P stim200_wexc1.2_2slots $S slots_ms=[0,10] 'mea_overrides={"stim_length_um":200.0}' 'culture_overrides={"w_exc":1.2}') &
wait
