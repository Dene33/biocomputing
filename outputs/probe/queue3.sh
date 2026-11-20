cd /mnt/datadisk/repos/biocomputing
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
P="uv run python outputs/probe/encoding.py run"
A="projection=supervised slots_ms=[0,5,10,15]"
($P stim120 $A 'mea_overrides={"stim_length_um":120.0}'; $P stim200_c60 $A 'mea_overrides={"stim_length_um":200.0,"stim_current":60.0}') &
($P stim120_wexc1.2 $A 'mea_overrides={"stim_length_um":120.0}' 'culture_overrides={"w_exc":1.2}'; $P base_again $A) &
wait
