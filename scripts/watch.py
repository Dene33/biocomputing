import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

BAD = re.compile(r"Traceback|GUARD FAIL|\b[A-Z]\w*Error\b|out of memory|\bKilled\b|\bNaN\b")


def unit_state(unit):
    out = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "Result"],
                         capture_output=True, text=True).stdout
    kv = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
    return kv.get("ActiveState", "unknown"), kv.get("Result", "unknown")


def last_activity(log: Path):
    times = [log.stat().st_mtime] if log.exists() else [0.0]
    times += [p.stat().st_mtime for p in log.parent.rglob("heartbeat/*.json")]
    return max(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--stall-min", type=float, default=15.0)
    ap.add_argument("--every-s", type=float, default=20.0)
    ap.add_argument("--from-start", action="store_true", help="also scan the log's existing lines")
    a = ap.parse_args()
    log = Path(a.log)
    pos = 0 if a.from_start or not log.exists() else log.stat().st_size

    def finish(code, why):
        text = log.read_text(errors="replace") if log.exists() else ""
        print(f"{why}\n--- last lines of {log} ---\n" + "\n".join(text.splitlines()[-20:]), flush=True)
        sys.exit(code)

    while True:
        state, result = unit_state(a.unit)
        if log.exists() and log.stat().st_size > pos:
            with open(log, errors="replace") as f:
                f.seek(pos)
                new = f.read()
                pos = f.tell()
            bad = [ln for ln in new.splitlines() if BAD.search(ln)]
            if bad:
                if state in ("active", "activating"):
                    subprocess.run(["systemctl", "--user", "stop", a.unit])
                finish(3, f"STOPPED {a.unit}: " + bad[0][:300])
        if state not in ("active", "activating", "deactivating"):
            finish(0 if result == "success" else 2, f"ENDED {a.unit}: {state}, result {result}")
        idle_min = (time.time() - last_activity(log)) / 60
        if idle_min > a.stall_min:
            subprocess.run(["systemctl", "--user", "stop", a.unit])
            finish(4, f"STOPPED {a.unit}: no log output or heartbeat for {idle_min:.0f} min")
        time.sleep(a.every_s)


if __name__ == "__main__":
    main()
