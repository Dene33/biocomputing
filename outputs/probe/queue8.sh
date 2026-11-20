cd /mnt/datadisk/repos/biocomputing
# wait for any running probe (one GPU job at a time)
while pgrep -f "outputs/probe/encoding.py run" >/dev/null; do sleep 10; done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PROBE_DATA=teacher_data_20k.pkl
P="uv run python outputs/probe/encoding.py run"
S='projection=supervised slots_ms=[0,5,10,15]'
M='mea_overrides={"stim_length_um":120.0}'
$P big_w4_diverse $S "$M" mea=grid256 wells=4 diverse_wells=true
$P big_w8_diverse $S "$M" mea=grid256 wells=8 diverse_wells=true
