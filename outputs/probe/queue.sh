cd /mnt/datadisk/repos/biocomputing
P="uv run python outputs/probe/encoding.py run"
$P pca_s0
$P sup_s0 projection=supervised
$P sup_4slots projection=supervised slots_ms=[0,5,10,15]
$P sup_64stim projection=supervised n_stim=64
$P sup_64stim_4slots projection=supervised n_stim=64 slots_ms=[0,5,10,15]
$P sup_rate4 projection=supervised encoding=rate slots_ms=[0,5,10,15]
