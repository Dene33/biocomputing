from __future__ import annotations

import os
import sys
from pathlib import Path

REPOS = Path(__file__).resolve().parents[3]

CYCLOTRON_DIR = Path(os.environ.get("CYCLOTRON_DIR", REPOS / "cyclotron")).expanduser()

POLICY_DIR = Path(
    os.environ.get("ASIMOV_POLICY_DIR", REPOS / "denebots" / "site" / "learn-sim" / "policies" / "menlo-reference")
).expanduser()

ASIMOV_XML = Path(
    os.environ.get("ASIMOV_XML", CYCLOTRON_DIR / "third_party" / "asimov-1" / "sim-model" / "xmls" / "asimov_1.xml")
).expanduser()


def import_sim2sim():
    source = CYCLOTRON_DIR / "source" / "cyclotron"
    if not (source / "cyclotron" / "sim2sim.py").is_file():
        raise FileNotFoundError(f"cyclotron's sim2sim.py not found under {source}. Set CYCLOTRON_DIR.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from cyclotron import sim2sim

    return sim2sim
