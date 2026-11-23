"""Launch a long run safely: smoke test, detached launch, start-up check.

    uv run scripts/launch.py --unit bio-x --log outputs/x/run.log [--with cebra] -- scripts/foo.py --arg ...

1. Smoke: the same script and arguments plus ``--smoke`` (every long script runs its whole pipeline, result writing
   included, at tiny scale into <out>/smoke), in the foreground, with a timeout. Any failure (exit code, Traceback,
   GUARD FAIL) stops here. Reports wall time, CPU cores used and GPU busy share, for choosing workers: one core
   pegged with the GPU idle means a single process is launch-bound and more workers (or batching) will help; the GPU
   busy throughout means extra processes will only queue.
2. Launch: ``systemd-run --user --unit=<unit> -p TimeoutStopSec=300``, output appended to the log (folder created
   first). Resumable scripts save on stop; relaunching the same command resumes.
3. Start-up check: the unit is still active after ``--verify-s`` and its log is clean.

Then: ``uv run scripts/watch.py --unit <unit> --log <log>`` in the background stops the unit at the first failure.
"""

import argparse
import os
import re
import resource
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

BAD = re.compile(r"Traceback|GUARD FAIL|\b[A-Z]\w*Error\b|out of memory|\bKilled\b|\bNaN\b")


def gpu_busy():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return float(out.split()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--with", dest="with_", nargs="*", default=[], help="extra packages for uv run --with")
    ap.add_argument("--no-smoke", action="store_true", help="only for scripts without --smoke (say why)")
    ap.add_argument("--smoke-timeout-s", type=float, default=900.0)
    ap.add_argument("--verify-s", type=float, default=60.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- script.py args...")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("no command after --")
    uv = ["uv", "run"] + [x for p in a.with_ for x in ("--with", p)]
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    active = subprocess.run(["systemctl", "--user", "is-active", a.unit], capture_output=True, text=True).stdout.strip()
    if active in ("active", "activating"):
        sys.exit(f"unit {a.unit} is already {active}")
    subprocess.run(["systemctl", "--user", "reset-failed", a.unit], capture_output=True)

    if not a.no_smoke:
        smoke_log = log.with_name(log.stem + ".smoke.log")
        print(f"smoke: {' '.join(uv + cmd + ['--smoke'])}  (log {smoke_log})", flush=True)
        samples, done = [], threading.Event()

        def sample():
            while not done.wait(1.0):
                g = gpu_busy()
                if g is not None:
                    samples.append(g)

        threading.Thread(target=sample, daemon=True).start()
        r0, t0 = resource.getrusage(resource.RUSAGE_CHILDREN), time.time()
        with open(smoke_log, "w") as f:
            try:
                code = subprocess.run(uv + cmd + ["--smoke"], stdout=f, stderr=subprocess.STDOUT,
                                      timeout=a.smoke_timeout_s).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        done.set()
        wall = time.time() - t0
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cores = (r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) / max(wall, 1e-9)
        text = smoke_log.read_text(errors="replace")
        bad = [ln for ln in text.splitlines() if BAD.search(ln)]
        gpu = f"{sum(samples) / len(samples):.0f}% (max {max(samples):.0f}%)" if samples else "n/a"
        print(f"smoke: exit {code}, {wall:.0f} s, CPU cores used {cores:.1f}, GPU busy {gpu}", flush=True)
        if code != 0 or bad:
            print("SMOKE FAILED; last lines:\n" + "\n".join(text.splitlines()[-25:]), flush=True)
            sys.exit(1)

    inner = " ".join(shlex.quote(x) for x in uv + cmd) + f" >> {shlex.quote(str(log))} 2>&1"
    env = [f"--setenv={k}={os.environ[k]}" for k in ("PATH", "HOME") if k in os.environ]
    subprocess.run(["systemd-run", "--user", f"--unit={a.unit}", "-p", "TimeoutStopSec=300",
                    f"--working-directory={os.getcwd()}", *env, "bash", "-c", inner], check=True)
    size0 = log.stat().st_size if log.exists() else 0
    time.sleep(a.verify_s)
    state = subprocess.run(["systemctl", "--user", "is-active", a.unit], capture_output=True, text=True).stdout.strip()
    new = log.read_text(errors="replace")[size0:] if log.exists() else ""
    bad = [ln for ln in new.splitlines() if BAD.search(ln)]
    if bad or state not in ("active", "activating"):
        print(f"START FAILED: unit {state}; last lines:\n" + "\n".join(new.splitlines()[-25:]), flush=True)
        subprocess.run(["systemctl", "--user", "stop", a.unit])
        sys.exit(1)
    print(f"running: unit {a.unit}, log {log}\n"
          f"  watch:  uv run scripts/watch.py --unit {a.unit} --log {log}\n"
          f"  stop:   systemctl --user stop {a.unit}   (resumable scripts save first)\n"
          f"  resume: the same launch command", flush=True)


if __name__ == "__main__":
    main()
