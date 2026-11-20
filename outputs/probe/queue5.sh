cd /mnt/datadisk/repos/biocomputing
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
P="uv run python outputs/probe/encoding.py run"
S='projection=supervised slots_ms=[0,5,10,15]'
M='mea_overrides={"stim_length_um":120.0}'
C='culture_overrides={"w_exc":1.2}'
$P g256_w1 $S "$M" "$C" mea=grid256 wells=1
$P g256_w4 $S "$M" "$C" mea=grid256 wells=4
$P g256_w8_lite $S "$M" "$C" mea=grid256 wells=8 bins=1 'taus_ms=[60.0]'
$P g256_w4_gain0.5 $S "$M" "$C" mea=grid256 wells=4 encoder_gain=0.5
$P g256_w4_gain0.3 $S "$M" "$C" mea=grid256 wells=4 encoder_gain=0.3
$P hd_w1_gain0.5 $S "$M" "$C" encoder_gain=0.5
