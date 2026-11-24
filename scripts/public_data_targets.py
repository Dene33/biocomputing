"""Track B, step B0 (docs/route-to-computation.md, "Order of work without a dish"): targets for the culture model,
from public recordings of real cultures made by other labs.

    uv run python -I scripts/public_data_targets.py fetch      # downloads into outputs/public_data/raw/ (gitignored)
    uv run python -I scripts/public_data_targets.py analyze    # outputs/public_data/targets.json
    uv run python -I scripts/public_data_targets.py --smoke fetch analyze   # 2 recordings, into <out>/smoke

Standalone (numpy only), so it runs under ``python -I``: downloaded files are untrusted data.

Sources, terms and provenance
-----------------------------
1. Wagenaar, Pine & Potter 2006, "An extremely rich repertoire of bursting patterns during the development of
   cortical cultures", BMC Neuroscience 7:11. Dissociated rat cortical cultures on 59-electrode MCS arrays, 25 kHz,
   half-hour recordings of spontaneous activity, daily over five weeks. Terms (the dataset's help page): downloads
   are on condition that publications using the data cite the article; no guarantees of quality. The lab's pages
   (potterlab.bme.gatech.edu/development-data, formerly neurodatasharing.bme.gatech.edu) return 404 as of 24 Nov 2026,
   so the files come from the Internet Archive's copy (captured 8 Feb 2026): the "simple-text" daily spontaneous
   recordings of the *dense* cultures, ``<batch>-<culture>-<DIV>.spk.txt.bz2``, two columns (time in s, channel
   0-59). Only these 527 files are archived; the stimulation sets and the sparse/small cultures are not. We take up to
   three recordings per culture between DIV 21 and 35 (the days nearest 24, 28 and 32), when bursting is mature.
2. Robbins et al. 2026, "Goal-directed learning in cortical organoids", Cell Reports 45:116984: Zenodo record
   10.5281/zenodo.17684862, CC-BY 4.0. ``aggregated_experiment_data.pkl`` (md5 af99f1f859136ca4a6dad47a1aa7c86d):
   processed HD-MEA data from mouse cortical organoids (spike-sorted units' rates, ISIs, burstiness; evoked
   connectivity and the share of stimulations that set off a network burst; cart-pole runs). Unpickling untrusted data
   can run code, so it is loaded with a restricted unpickler that allows only the three numpy constructors the file
   references (checked with pickletools before the first load) and plain containers.
3. FinalSpark's free dataset needs a request form (finalspark.com/neuroplatform); not fetched.

Measures (one function, ``spike_stats``, so the model can be scored the same way in B3a)
---------------------------------------------------------------------------------------
- Active electrode: >= 0.1 Hz. Per-electrode rate over active electrodes (median, mean).
- Burstlets, per electrode (Wagenaar et al. 2005/2006): >= 4 spikes whose inter-spike intervals are all
  <= min(100 ms, 1 / (4 x the electrode's mean rate)).
- Network bursts: burstlets overlapping in time are merged; a merged event counts if it spans >= 3 electrodes.
  Burst rate (per min), duration (first to last burstlet spike), inter-burst interval (start to start) and its CV,
  recruitment (share of active electrodes in the burst), share of spikes inside network bursts.
Per culture: the median over its recordings. Targets: the spread across cultures (median, 25-75% and 5-95%).
B3(a) passes when the model's values fall inside the 5-95% range.

Results (24 Nov 2026; outputs/public_data/targets.json, manifest.json)
---------------------------------------------------------------------
Wagenaar dense cultures: 81 recordings (173 MB) from 29 cultures, DIV 21-32; one archived file (5-3-24) is
truncated in the Archive and was skipped. Across cultures, median [5-95%]:
- active electrodes 56 [51-59] of 59; per-electrode rate, median electrode 2.8 Hz [0.7-10.3], mean 4.5 Hz [1.8-11.1]
- network bursts 4.2 per min [1.2-15.4], lasting 556 ms [79-1788]
- inter-burst interval 12.0 s [0.85-41.5], CV 0.87 [0.56-1.99]: irregular
- recruitment 0.59 [0.08-1.00] of the active electrodes; share of spikes inside bursts 0.63 [0.10-0.93]
Robbins organoids: 38 experiments, 16 organoids (spike-sorted units, so rates are per unit, not per electrode):
- unit rate 4.8 Hz [2.5-13.8] (median per experiment), median ISI 69 ms [5-245]; the burstiness index is saturated
  at 1.0 and says nothing
- evoked: a stimulation sets off a network burst in 6.5% of trials [0-16%] (``burst_percent`` is a fraction);
  each stimulated unit's best first-order target fires within the window in 22% of trials [8-98%]
- cart-pole, top-decile balance time per run: adaptive 51.8 s (n 229), random 36.4 s (67), none 31.0 s (130)
Not yet measured on our model (B3a uses ``spike_stats`` on its electrodes). Two quoted figures already suggest
where it may fall outside: the README's cpu_2k bursts come every 2-3 s with CV ~0.3 (below the 0.56-1.99 range),
and coach_probe found that every probe sets off a network burst (real organoids: 6.5%).
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import pickle
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

CDX = ("http://web.archive.org/cdx/search/cdx?url=potterlab.bme.gatech.edu/development-data/simple-text/daily/spont/"
       "dense/*&output=txt&filter=statuscode:200&collapse=urlkey&fl=original,timestamp,length")
ROBBINS = "https://zenodo.org/api/records/17684862/files/aggregated_experiment_data.pkl/content"
ROBBINS_MD5 = "af99f1f859136ca4a6dad47a1aa7c86d"
DAYS = (21, 35)
PICK = (24, 28, 32)
N_CH = 60
UA = {"User-Agent": "research-data-fetch/1.0"}


def get(url: str, tries: int = 3, timeout: float = 120.0) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # the Archive is flaky: back off and retry
            if k == tries - 1:
                raise
            print(json.dumps({"retry": url, "error": str(e)[:120]}), flush=True)
            time.sleep(5 * (k + 1))


def choose(rows):
    """Up to three recordings per culture in DIV 21-35, the days nearest 24, 28 and 32."""
    by = {}
    for url, ts in rows:
        m = re.search(r"/(\d+)-(\d+)-(\d+)\.spk\.txt\.bz2$", url)
        if m and DAYS[0] <= int(m[3]) <= DAYS[1]:
            by.setdefault((int(m[1]), int(m[2])), {})[int(m[3])] = (url, ts)
    out = []
    for (b, c), days in sorted(by.items()):
        picked = []
        for d in PICK:
            best = min((x for x in days if x not in picked), key=lambda x: abs(x - d), default=None)
            if best is not None and abs(best - d) <= 3:
                picked.append(best)
        out += [(b, c, d, *days[d]) for d in sorted(picked)]
    return out


def fetch(raw: Path, manifest: Path, smoke: bool) -> None:
    w = raw / "wagenaar_dense"
    w.mkdir(parents=True, exist_ok=True)
    cdx = raw / "wagenaar_cdx.txt"  # the Archive's index answers 503 at times: fetch it once, keep it
    if not cdx.exists():
        cdx.write_bytes(get(CDX))
    rows = [ln.split()[:2] for ln in cdx.read_text().splitlines() if ln.startswith("http")]
    sel = choose(rows)[: 2 if smoke else None]
    man = json.loads(manifest.read_text()) if manifest.exists() else {"wagenaar": {}, "robbins": {}}
    for b, c, d, url, ts in sel:
        name = f"{b}-{c}-{d}.spk.txt.bz2"
        p = w / name
        if name in man.setdefault("failed", {}):
            continue
        if not p.exists():
            try:
                data = get(f"http://web.archive.org/web/{ts}id_/{url}")
                bz2.decompress(data)  # a truncated or HTML error page fails here, before anything is written
            except Exception as e:  # e.g. 5-3-24: the archived copy always ends at the same byte
                man["failed"][name] = {"source": url, "archive_timestamp": ts, "error": str(e)[:200]}
                print(json.dumps({"skipped": name, "error": str(e)[:120]}), flush=True)
                manifest.write_text(json.dumps(man, indent=1))
                continue
            p.write_bytes(data)
            time.sleep(1.0)  # be polite to the Archive
        man["wagenaar"][name] = {"batch": b, "culture": c, "div": d, "source": url, "archive_timestamp": ts,
                                 "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(man, indent=1))  # as we go: a crash keeps what was fetched
    r = raw / "robbins_zenodo" / "aggregated_experiment_data.pkl"
    r.parent.mkdir(parents=True, exist_ok=True)
    if not r.exists() and not smoke:
        r.write_bytes(get(ROBBINS, timeout=900))
    if r.exists():
        md5 = hashlib.md5(r.read_bytes()).hexdigest()
        if md5 != ROBBINS_MD5:
            sys.exit(f"Robbins pickle md5 {md5} != Zenodo's {ROBBINS_MD5}")
        man["robbins"] = {"file": r.name, "source": ROBBINS, "doi": "10.5281/zenodo.17684862", "license": "CC-BY-4.0",
                          "md5": md5, "bytes": r.stat().st_size}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man, indent=1))
    print(json.dumps({"wagenaar_files": len(man["wagenaar"]), "robbins": bool(man["robbins"])}), flush=True)


# --- Measures -----------------------------------------------------------------------------------------------------------
def spike_stats(t: np.ndarray, ch: np.ndarray, duration_s: float, n_ch: int, min_rate: float = 0.1,
                min_channels: int = 3) -> dict:
    """Electrode-level spontaneous statistics; see the module docstring. ``t`` in seconds, ``ch`` in 0..n_ch-1."""
    t, ch = np.asarray(t, float), np.asarray(ch, int)
    rates = np.bincount(ch, minlength=n_ch)[:n_ch] / duration_s
    active = np.flatnonzero(rates >= min_rate)
    out = {"duration_s": round(duration_s, 1), "n_active": int(len(active))}
    if len(active) == 0:
        return out
    out["rate_hz_median"] = round(float(np.median(rates[active])), 3)
    out["rate_hz_mean"] = round(float(rates[active].mean()), 3)
    bl = []  # (start, end, channel)
    for c in active:
        ts = np.sort(t[ch == c])
        thr = min(0.1, 1.0 / (4.0 * rates[c]))
        short = np.diff(ts) <= thr
        # runs of >= 3 consecutive short intervals = >= 4 spikes
        edges = np.diff(np.r_[0, short.astype(int), 0])
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s >= 3:
                bl.append((ts[s], ts[e], c))
    bursts = []
    if bl:
        bl.sort()
        cur_s, cur_e, chans = bl[0][0], bl[0][1], {bl[0][2]}
        for s, e, c in bl[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
                chans.add(c)
            else:
                bursts.append((cur_s, cur_e, chans))
                cur_s, cur_e, chans = s, e, {c}
        bursts.append((cur_s, cur_e, chans))
    bursts = [x for x in bursts if len(x[2]) >= min_channels]
    out["n_bursts"] = len(bursts)
    out["burst_rate_per_min"] = round(len(bursts) / duration_s * 60, 3)
    in_active = np.isin(ch, active)
    if bursts:
        starts = np.array([b[0] for b in bursts])
        ends = np.array([b[1] for b in bursts])
        dur = ends - starts
        out["burst_duration_ms_median"] = round(float(np.median(dur) * 1000), 1)
        out["recruitment_median"] = round(float(np.median([len(b[2]) / len(active) for b in bursts])), 3)
        idx = np.searchsorted(starts, t[in_active], side="right") - 1
        inside = (idx >= 0) & (t[in_active] <= ends[np.maximum(idx, 0)])
        out["spikes_in_bursts"] = round(float(inside.mean()), 3)
        if len(bursts) > 2:
            ibi = np.diff(starts)
            out["ibi_s_median"] = round(float(np.median(ibi)), 3)
            out["ibi_cv"] = round(float(ibi.std() / ibi.mean()), 3)
    return out


def load_spk(path: Path):
    a = np.loadtxt(io.StringIO(bz2.decompress(path.read_bytes()).decode()), ndmin=2)
    t, ch = a[:, 0], a[:, 1].astype(int)
    keep = (ch >= 0) & (ch < N_CH)
    return t[keep], ch[keep]


class SafeUnpickler(pickle.Unpickler):
    """Only the numpy constructors the Robbins file uses (pickletools: _frombuffer, dtype, scalar) and their
    numpy-1.x names; anything else is refused before it can run."""

    ALLOWED = {("numpy._core.numeric", "_frombuffer"), ("numpy.core.numeric", "_frombuffer"), ("numpy", "dtype"),
               ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def spread(values) -> dict | None:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(len(v)), "median": round(float(q[2]), 3), "iqr": [round(float(q[1]), 3), round(float(q[3]), 3)],
            "p5_p95": [round(float(q[0]), 3), round(float(q[4]), 3)]}


def analyze_wagenaar(raw: Path, manifest: dict) -> dict:
    per_rec, per_culture = [], {}
    for name, info in sorted(manifest["wagenaar"].items()):
        t, ch = load_spk(raw / "wagenaar_dense" / name)
        dur = float(np.ceil(t.max() / 60.0) * 60.0) if len(t) else 0.0  # half-hour recordings; whole minutes
        s = spike_stats(t, ch, dur, N_CH) | {"file": name, "batch": info["batch"], "culture": info["culture"],
                                             "div": info["div"]}
        per_rec.append(s)
        per_culture.setdefault(f"{info['batch']}-{info['culture']}", []).append(s)
        print(json.dumps({k: s.get(k) for k in ("file", "n_active", "rate_hz_median", "burst_rate_per_min",
                                                 "burst_duration_ms_median", "ibi_cv", "spikes_in_bursts")}),
              flush=True)
    keys = ["n_active", "rate_hz_median", "rate_hz_mean", "burst_rate_per_min", "burst_duration_ms_median",
            "ibi_s_median", "ibi_cv", "recruitment_median", "spikes_in_bursts"]
    cult = {c: {k: (float(np.median([r[k] for r in rs if k in r])) if any(k in r for r in rs) else None)
                for k in keys} for c, rs in per_culture.items()}
    return {"recordings": len(per_rec), "cultures": len(cult),
            "targets_across_cultures": {k: spread([c[k] for c in cult.values()]) for k in keys},
            "per_culture": cult, "per_recording": per_rec}


def analyze_robbins(path: Path) -> dict:
    with open(path, "rb") as f:
        data = SafeUnpickler(f).load()
    rates, burstiness, isi, burst_pct, fo_max, fo_frac, top = [], [], [], [], [], [], {}
    orgs = set()
    for exp_id, e in data.items():
        orgs.add(e["metadata"].get("organoid"))
        b = e.get("baseline", {})
        if b.get("spikes_available"):
            r = np.asarray(b["firing_rates_hz"], float)
            rates.append(float(np.median(r)))
            burstiness.append(float(np.nanmedian(np.asarray(b["burstiness"], float))))
            isi.append(float(np.nanmedian(np.asarray(b["median_isis_ms"], float))))
        c = e.get("causal", {})
        if c.get("causal_available"):
            burst_pct.append(float(np.nanmedian(np.asarray(c["burst_percent"], float))))
            fo = np.asarray(c["first_order_connectivity"], float)
            if fo.size:
                fo_max.append(float(np.nanmedian(np.nanmax(fo, 1))))
                fo_frac.append(float(np.nanmean(fo > 0.5)))
        for run in e.get("cartpole", {}).get("runs", []):
            if run.get("top_decile") is not None:
                top.setdefault(run["exp_type"], []).append(float(run["top_decile"]))
    return {"experiments": len(data), "organoids": len(orgs - {None}),
            "note": "spike-sorted units on an HD-MEA, not electrodes: rates are not directly comparable with "
                    "electrode multi-unit rates",
            "unit_rate_hz_median_per_experiment": spread(rates),
            "burstiness_median_per_experiment": spread(burstiness),
            "median_isi_ms_per_experiment": spread(isi),
            "evoked_burst_percent_median_per_experiment": spread(burst_pct),
            "first_order_connectivity_best_target_median": spread(fo_max),
            "first_order_pairs_above_0p5": spread(fo_frac),
            "cartpole_top_decile_s_by_condition": {k: spread(v) for k, v in top.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["fetch", "analyze"])
    ap.add_argument("--out", default="outputs/public_data")
    ap.add_argument("--smoke", action="store_true", help="2 Wagenaar recordings, Robbins only if already present")
    a = ap.parse_args()
    out = Path(a.out)
    raw = out / "raw"
    if a.smoke:
        out = out / "smoke"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
    manifest = out / "manifest.json"
    if "fetch" in a.steps:
        fetch(raw, manifest, a.smoke)
    if "analyze" in a.steps:
        man = json.loads(manifest.read_text())
        res = {"wagenaar_2006_dense_div21_35": analyze_wagenaar(raw, man)}
        rob = raw / "robbins_zenodo" / "aggregated_experiment_data.pkl"
        if man.get("robbins") and rob.exists():
            res["robbins_2026_organoids"] = analyze_robbins(rob)
        (out / "targets.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res["wagenaar_2006_dense_div21_35"]["targets_across_cultures"]), flush=True)
        if "robbins_2026_organoids" in res:
            print(json.dumps({k: v for k, v in res["robbins_2026_organoids"].items() if k != "note"}), flush=True)


if __name__ == "__main__":
    main()
