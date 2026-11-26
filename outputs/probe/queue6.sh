cd /mnt/datadisk/repos/biocomputing
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
P="uv run python outputs/probe/encoding.py run"
S='projection=supervised slots_ms=[0,5,10,15]'
M='mea_overrides={"stim_length_um":120.0}'
$P real_g256_w4_gain1.5 $S "$M" mea=grid256 wells=4
$P real_g256_w4_gain0.5 $S "$M" mea=grid256 wells=4 encoder_gain=0.5
$P real_g256_w8_gain0.5 $S "$M" mea=grid256 wells=8 encoder_gain=0.5
$P real_g256_w4_gain0.3 $S "$M" mea=grid256 wells=4 encoder_gain=0.3
