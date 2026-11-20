cd /mnt/datadisk/repos/biocomputing
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
P="uv run python outputs/probe/encoding.py run"
S='projection=supervised slots_ms=[0,5,10,15]'
M='mea_overrides={"stim_length_um":120.0}'
$P real_w4_same $S "$M" mea=grid256 wells=4
$P real_w4_diverse $S "$M" mea=grid256 wells=4 diverse_wells=true
$P real_w6_diverse $S "$M" mea=grid256 wells=6 diverse_wells=true
