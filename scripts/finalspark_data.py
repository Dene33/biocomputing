"""Readers for FinalSpark's whole-life exports (outputs/public_data/raw/finalspark/, gitignored; README in each export).

The exports are pandas HDF5 "frame_table"s. They are read here with h5py only: pandas would unpickle the tables'
attributes, and downloaded files are untrusted. A values block's column names (attribute ``values_block_N_kind``) are
a protocol-0 pickled list of str, parsed as text. Electrode ids are system-wide (MEA k holds 32k..32k+31; fs369 sits
on 64-95) and are returned mod 32; an organoid group is ``electrode // 8`` (the 4 x 8 air-liquid-interface MEA).
Times are int64 ns since the epoch (UTC).

Events: one row per threshold crossing (6 x the median noise SD), with the peak voltage after it. Spontaneous spikes
stay below 300 uV (fs369 p99.99: 282 uV); stimulation artifacts reach mV and end ~14 ms after a stimulation's onset.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "outputs/public_data/raw/finalspark"
FS = {"fs369": RAW / "fs369/fs369data/fs369_package.hdf5", "fs437": RAW / "fs437/fs437_export/fs437_package.hdf5"}
S, MS = 10**9, 10**6  # ns per s, per ms


def _names(kind: bytes) -> list[str]:
    """Column names of a pandas values block: a protocol-0 pickled list of str, read as text (never unpickled).
    Each element is "V<name>" after "(lp0\\n" (the first) or the APPEND opcode "a" (the rest)."""
    return re.findall(r"\na?V([^\n]*)", kind.decode())


def read_table(path: Path, key: str) -> dict:
    """A pandas frame_table as numpy columns, with h5py only. Electrode ids mod 32 (an MEA's own index)."""
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        ds = f[key]["table"]
        for field in ds.dtype.names:
            if field == "index":
                continue
            col = ds.fields(field)[:]
            if field.startswith("values_block_"):
                names = _names(ds.attrs[f"{field}_kind"])
                col = col.reshape(len(col), -1)
                if col.shape[1] != len(names):
                    raise ValueError(f"{key}/{field}: {col.shape[1]} columns, {len(names)} names")
                out |= {n: col[:, j] for j, n in enumerate(names)}
            else:
                out[field] = col
    if "electrode" in out:
        out["electrode"] = out["electrode"] % 32
    return out


def load_record(path: Path, fs: str) -> dict:
    """Events (t, e, amp; sorted by time) and the stimulation log (columns, sorted by time)."""
    ev = read_table(path, f"{fs}_wholelife_events")
    st = read_table(path, f"{fs}_wholelife_stimulations")
    t, e, amp = ev["time_of_event"], ev["electrode"].astype(np.int64), ev["max_voltage_uv"]
    if not (np.diff(t) >= 0).all():
        o = np.argsort(t, kind="stable")
        t, e, amp = t[o], e[o], amp[o]
    o = np.argsort(st["time_of_stim"], kind="stable")
    st = {k: v[o] for k, v in st.items()}
    return {"t": t, "e": e, "amp": amp, "st": st}


def stim_sessions(ts: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    """Index ranges [a, b] into the sorted unique stimulation times, split where they are > gap_s apart."""
    if len(ts) == 0:
        return []
    cut = np.flatnonzero(np.diff(ts) > gap_s * S)
    return list(zip(np.r_[0, cut + 1].tolist(), np.r_[cut, len(ts) - 1].tolist()))


def window_counts(t, e, amp, times, a_ms, b_ms, art_uv) -> np.ndarray:
    """Clean events (|amp| <= art_uv) per electrode in [time + a_ms, time + b_ms) for each time: (len(times), 32)."""
    out = np.zeros((len(times), 32))
    for i, u in enumerate(times):
        lo, hi = np.searchsorted(t, u + int(a_ms * MS)), np.searchsorted(t, u + int(b_ms * MS))
        ok = np.abs(amp[lo:hi]) <= art_uv
        out[i] = np.bincount(e[lo:hi][ok], minlength=32)
    return out


def stim_events(st: dict) -> dict:
    """One entry per stimulation time: the time, the set of electrodes (as a 32-bit mask), the setting (every
    parameter of the log, rounded; None where the rows of one time differ) and the row ranges."""
    keys = sorted(k for k in st if k not in ("time_of_stim", "electrode"))
    P = np.round(np.stack([st[k] for k in keys], 1), 6)
    ut, start, n = np.unique(st["time_of_stim"], return_index=True, return_counts=True)
    bounds = np.r_[start, len(st["time_of_stim"])]
    mask = np.bitwise_or.reduceat(np.left_shift(np.int64(1), st["electrode"].astype(np.int64)), start)
    setting = []
    for i in range(len(ut)):
        rows = P[bounds[i]:bounds[i + 1]]
        setting.append(tuple(rows[0].tolist()) if (rows == rows[0]).all() else None)
    return {"t": ut, "mask": mask, "n": n, "setting": setting, "keys": keys, "start": start}
