"""Collect every reservoir and feedback run into one table: outputs/report.md and outputs/report.json.

    uv run scripts/evaluate.py
"""

import json
from pathlib import Path

root = Path("outputs")
rows = []
for path in sorted((root / "reservoir").glob("*/summary.json")):
    s = json.loads(path.read_text())
    last = s["history"][-1]
    rows.append({
        "kind": "reservoir", "task": s["args"]["task"], "source": s["args"]["source"], "mea": s["args"]["mea"],
        "seed": s["args"]["seed"], "train_r2": round(last["train_r2"], 3), "val_r2": round(last.get("val_r2", float("nan")), 3),
        "survived_s": round(s["eval_survived_s_mean"], 2),
        "full": f"{s['eval_full_episodes']}/{len(s['eval'])}", "distance_m": round(s["eval_distance_x_m_mean"], 2),
    })
fb_rows = []
for path in sorted((root / "feedback").glob("*/summary.json")):
    s = json.loads(path.read_text())
    a = s["args"]

    def mean(evals, key="survived_s"):
        return round(sum(e[key] for e in evals) / len(evals), 2)

    b = s["blocks"]
    fb_rows.append({
        "protocol": a["protocol"], "task": a["task"], "perturb": a["perturb"], "condition": a["condition"],
        "seed": a["seed"], "before_s": mean(s["before"]), "first_block_s": round(b[0]["survived_s"], 2),
        "last_block_s": round(b[-1]["survived_s"], 2), "after_s": mean(s["after"]),
        "after_rest_s": mean(s["after_rest"]),
    })

lines = ["# Results", "", "## Reservoir (DAgger + ridge readout)", "",
         "| task | readout source | MEA | seed | train R² | validation R² | survived (s) | full episodes | distance (m) |",
         "|---|---|---|---|---|---|---|---|---|"]
lines += [f"| {r['task']} | {r['source']} | {r['mea']} | {r['seed']} | {r['train_r2']} | {r['val_r2']} | {r['survived_s']} | "
          f"{r['full']} | {r['distance_m']} |" for r in rows]
lines += ["", "## DishBrain feedback sessions", "",
          "| protocol | task | perturbation | condition | seed | before (s) | first block (s) | last block (s) | "
          "after (s) | after rest (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
lines += [f"| {r['protocol']} | {r['task']} | {r['perturb']} | {r['condition']} | {r['seed']} | {r['before_s']} | "
          f"{r['first_block_s']} | {r['last_block_s']} | {r['after_s']} | {r['after_rest_s']} |" for r in fb_rows]
(root / "report.md").write_text("\n".join(lines) + "\n")
(root / "report.json").write_text(json.dumps({"reservoir": rows, "feedback": fb_rows}, indent=2))
print("\n".join(lines))
