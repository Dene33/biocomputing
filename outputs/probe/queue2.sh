cd /mnt/datadisk/repos/biocomputing
P="uv run python outputs/probe/encoding.py run"
A="projection=supervised slots_ms=[0,5,10,15]"
($P wexc1.6 $A 'culture_overrides={"w_exc":1.6}'; $P wexc1.2 $A 'culture_overrides={"w_exc":1.2}'; $P wexc0.8 $A 'culture_overrides={"w_exc":0.8}') &
($P winh-10 $A 'culture_overrides={"w_inh":-10.0}'; $P winh-15 $A 'culture_overrides={"w_inh":-15.0}'; $P stpu0.5 $A 'culture_overrides={"stp_u":0.5}') &
($P wexc1.6_winh-10 $A 'culture_overrides={"w_exc":1.6,"w_inh":-10.0}'; $P dead $A source=dead; $P oracle $A source=oracle) &
wait
