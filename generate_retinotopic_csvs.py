#!/usr/bin/env python3
"""
generate_retinotopic_csvs.py

Modified from the standard platform paper script to support:
1. Forced Area Renaming (e.g., 'VISp' -> 'V1_site2')
2. Unit ID Offsetting (to prevent collisions with SDK data)
3. Modulation Index calculation for Passive Viewing (Drifting Gratings)
"""

from __future__ import annotations
import argparse
import os
import math
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import h5py

from common.drifting_gratings import compute_drifting_grating_metrics

# Optional SciPy for paper-matching timescale + fits
try:
    from scipy import signal as scipy_signal
    from scipy.optimize import curve_fit
except Exception:
    scipy_signal = None
    curve_fit = None

def log(msg: str) -> None:
    print(f"[V1_Shim] {msg}")


# ----------------------------
# Legacy math (repo-local)
# ----------------------------
_FUNCTIONS_DIR = os.path.join(os.path.dirname(__file__), "functions")
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

try:
    from time_to_first_spike import compute_first_spike as legacy_compute_first_spike
except Exception as e:
    legacy_compute_first_spike = None
    log(f"Legacy import failed: time_to_first_spike.compute_first_spike ({e.__class__.__name__}): {e}")

try:
    from modulation_index import main as legacy_modulation_index
except Exception as e:
    legacy_modulation_index = None
    log(f"Legacy import failed: modulation_index.main ({e.__class__.__name__}): {e}")

# ----------------------------
# NWB Extraction
# ----------------------------
@dataclass
class NWBExtract:
    units_df: pd.DataFrame
    spikes_by_unit: Dict[int, np.ndarray]
    intervals_tables: Dict[str, pd.DataFrame]

_SKIP_COLS = frozenset({"electrodes", "electrodes_index", "timeseries", "timeseries_index"})
_STR_NUMERIC_COLS = frozenset({
    "temporal_frequency", "orientation", "spatial_frequency", "contrast",
})

def _read_numeric_dset(fid: h5py.h5f.FileID, path: str) -> np.ndarray:
    """Read any HDF5 numeric dataset (float or int, including 80-bit) as float64."""
    dset = h5py.h5d.open(fid, path.encode())
    n = dset.get_space().get_simple_extent_npoints()
    arr = np.empty(n, dtype="<f8")
    mem_type = h5py.h5t.py_create(np.dtype("<f8"), logical=True)
    dset.read(h5py.h5s.ALL, h5py.h5s.ALL, arr, mem_type)
    return arr


def _read_dset(fid: h5py.h5f.FileID, f: h5py.File, path: str, col: str) -> Optional[np.ndarray]:
    """Dispatch: numeric → low-level float64 read; string → high-level decode."""
    try:
        dset_id = h5py.h5d.open(fid, path.encode())
        t = dset_id.get_type()
        if isinstance(t, (h5py.h5t.TypeFloatID, h5py.h5t.TypeIntegerID)):
            return _read_numeric_dset(fid, path)
        else:
            raw = f[path][:]
            decoded = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw])
            if col in _STR_NUMERIC_COLS:
                return pd.to_numeric(pd.Series(decoded), errors="coerce").to_numpy(dtype=float)
            return decoded
    except Exception as e:
        log(f"  Column read failed '{path}': {e.__class__.__name__}: {e}")
        return None


def read_nwb_tables(nwb_path: str) -> NWBExtract:
    log(f"Opening NWB (h5py): {nwb_path}")
    # Explicit read-only mode is required for mounted DANDI assets.
    fid = h5py.h5f.open(nwb_path.encode(), flags=h5py.h5f.ACC_RDONLY)

    with h5py.File(nwb_path, "r") as f:
        u_grp = f["units"]

        # Spike times ragged array
        st_flat = _read_numeric_dset(fid, "/units/spike_times")
        st_idx = _read_numeric_dset(fid, "/units/spike_times_index").astype(np.int64)
        n_units = len(st_idx)
        log(f"Units: {n_units}, total spikes: {len(st_flat)}")

        seg_starts = np.concatenate([[0], st_idx[:-1]])
        spikes_by_unit: Dict[int, np.ndarray] = {}
        for i in range(n_units):
            spikes_by_unit[i] = np.sort(st_flat[seg_starts[i]:st_idx[i]])
        nonempty = sum(1 for s in spikes_by_unit.values() if s.size > 0)
        log(f"Spike arrays parsed: {nonempty} units with spikes")

        # Units table
        units_data: Dict[str, np.ndarray] = {}
        for col in u_grp.keys():
            if col in _SKIP_COLS or col in ("spike_times", "spike_times_index"):
                continue
            arr = _read_dset(fid, f, f"/units/{col}", col)
            if col == "device_name" and (arr is None or len(arr) != n_units):
                raise RuntimeError(
                    f"'device_name' is present in the NWB units table but failed to "
                    f"read cleanly (got {'None' if arr is None else f'{len(arr)} rows, expected {n_units}'}). "
                    f"Refusing to silently fall back to a single site-wide area label "
                    f"for all probes; fix the underlying read failure instead."
                )
            if arr is not None and len(arr) == n_units:
                units_data[col] = arr
        units_df = pd.DataFrame(units_data)
        units_df.index = pd.RangeIndex(n_units)
        log(f"Units table columns: {list(units_df.columns)[:10]}")

        # Interval tables
        intervals: Dict[str, pd.DataFrame] = {}
        if "intervals" in f:
            for name in f["intervals"].keys():
                try:
                    int_grp = f["intervals"][name]
                    int_data: Dict[str, np.ndarray] = {}
                    for col in int_grp.keys():
                        if col in _SKIP_COLS:
                            continue
                        arr = _read_dset(fid, f, f"/intervals/{name}/{col}", col)
                        if arr is not None:
                            int_data[col] = arr
                    df = pd.DataFrame(int_data)
                    intervals[name] = df
                    log(f"Interval '{name}' loaded: {len(df)} rows, cols={list(df.columns)[:8]}")
                except Exception as e:
                    log(f"Interval '{name}' failed ({e.__class__.__name__}): {e}")

        log(f"Available interval tables: {list(intervals.keys())}")
        return NWBExtract(units_df, spikes_by_unit, intervals)


def choose_stim_table(intervals: Dict[str, pd.DataFrame], requested: str) -> Tuple[str, pd.DataFrame]:
    if requested in intervals:
        return requested, intervals[requested]
    # Fallback: pick a grating-like table, then any table
    gratings_like = [k for k in intervals.keys() if "grating" in k.lower() or "drifting" in k.lower()]
    if gratings_like:
        chosen = gratings_like[0]
        log(f"Requested stim table '{requested}' not found; using '{chosen}' (gratings-like fallback)")
        return chosen, intervals[chosen]
    if intervals:
        chosen = list(intervals.keys())[0]
        log(f"Requested stim table '{requested}' not found; using '{chosen}' (first available)")
        return chosen, intervals[chosen]
    raise KeyError(f"No interval tables available to satisfy stim table '{requested}'")

# ----------------------------
# Glue: binning spikes to ms bins
# ----------------------------
def _bin_spikes_trials_1ms_binary(spikes_s: np.ndarray, trial_starts_s: np.ndarray, duration_ms: int) -> np.ndarray:
    """Return (n_trials, duration_ms) in {0,1} with 1ms bins."""
    trial_starts_s = np.asarray(trial_starts_s, dtype=float)
    out = np.zeros((len(trial_starts_s), int(duration_ms)), dtype=np.uint8)
    if len(trial_starts_s) == 0:
        return out
    spikes_s = np.asarray(spikes_s, dtype=float)
    if spikes_s.size == 0:
        return out
    if spikes_s.size >= 2 and np.any(np.diff(spikes_s) < 0):
        spikes_s = np.sort(spikes_s)

    t0 = trial_starts_s
    t1 = trial_starts_s + (duration_ms / 1000.0)
    i0 = np.searchsorted(spikes_s, t0, side="left")
    i1 = np.searchsorted(spikes_s, t1, side="left")
    for tr in range(len(trial_starts_s)):
        seg = spikes_s[i0[tr]:i1[tr]]
        if seg.size == 0:
            continue
        ms = ((seg - t0[tr]) * 1000.0).astype(np.int64)
        ms = ms[(ms >= 0) & (ms < duration_ms)]
        if ms.size:
            out[tr, np.unique(ms)] = 1
    return out


def _bin_spikes_trials_counts(spikes_s: np.ndarray, trial_starts_s: np.ndarray, bin_edges_s: np.ndarray) -> np.ndarray:
    """Return (n_trials, n_bins) spike counts for each trial using absolute edges = start + bin_edges_s."""
    trial_starts_s = np.asarray(trial_starts_s, dtype=float)
    bin_edges_s = np.asarray(bin_edges_s, dtype=float)
    if trial_starts_s.size == 0:
        return np.zeros((0, max(len(bin_edges_s) - 1, 0)), dtype=float)

    spikes_s = np.asarray(spikes_s, dtype=float)
    if spikes_s.size == 0:
        return np.zeros((len(trial_starts_s), max(len(bin_edges_s) - 1, 0)), dtype=float)
    if spikes_s.size >= 2 and np.any(np.diff(spikes_s) < 0):
        spikes_s = np.sort(spikes_s)

    abs_edges = trial_starts_s[:, None] + bin_edges_s[None, :]
    idx = np.searchsorted(spikes_s, abs_edges, side="left")
    counts = np.diff(idx, axis=1)
    return counts.astype(float, copy=False)

# ----------------------------
# Metric: TTFS (legacy; paper window 30-200ms)
# ----------------------------
def compute_ttfs_seconds_legacy(
    spikes_by_unit: Dict[int, np.ndarray],
    flash_df: pd.DataFrame,
    *,
    window_ms: int = 500,
    start_time_ms: int = 30,
    end_time_ms: int = 200,
) -> pd.DataFrame:
    """Compute per-unit TTFS (seconds). Uses legacy compute_first_spike on 1ms-binned spike trains."""
    rows = []
    if legacy_compute_first_spike is None:
        log("TTFS: legacy compute_first_spike not available; filling NaN")
        for uid in spikes_by_unit.keys():
            rows.append({"unit_id": uid, "time_to_first_spike": np.nan})
        return pd.DataFrame(rows)

    if flash_df is None or flash_df.empty or "start_time" not in flash_df.columns:
        log("TTFS: no flash table or missing start_time; filling NaN")
        for uid in spikes_by_unit.keys():
            rows.append({"unit_id": uid, "time_to_first_spike": np.nan})
        return pd.DataFrame(rows)

    starts = flash_df["start_time"].to_numpy(dtype=float)
    total = len(spikes_by_unit)
    for idx, (uid, spikes) in enumerate(spikes_by_unit.items()):
        if (idx + 1) % 200 == 0:
            log(f"TTFS progress: {idx + 1}/{total} units")
        if spikes.size == 0:
            rows.append({"unit_id": uid, "time_to_first_spike": np.nan})
            continue

        x2d = _bin_spikes_trials_1ms_binary(spikes, starts, duration_ms=window_ms)
        x3d = x2d[None, :, :]
        first_ms = legacy_compute_first_spike(x3d, start_time=start_time_ms, end_time=end_time_ms)
        med_ms = float(np.nanmedian(first_ms[0, :])) if np.isfinite(first_ms).any() else np.nan
        rows.append({"unit_id": uid, "time_to_first_spike": (med_ms / 1000.0) if np.isfinite(med_ms) else np.nan})

    return pd.DataFrame(rows)

# ----------------------------
# Metric: Response decay timescale (paper-matching; Figure3/timescale_calculation.py)
# ----------------------------
def _paper_fit_exp(t_ms: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    exponential = lambda t, a, b, c: a * np.exp(-1 / b * t) + c
    params, pcov = curve_fit(
        exponential,
        t_ms,
        y,
        p0=(5, 20, 0.1),
        method="trf",
        bounds=([0, 1, -np.inf], [np.inf, 1000, np.inf]),
        maxfev=1000000000,
    )
    error = np.sqrt(np.diag(pcov))
    return params, error


def _paper_autocorr2d(x: np.ndarray) -> np.ndarray:
    ac_matrix = scipy_signal.correlate(x, x, mode="same")
    ac_matrix = np.delete(ac_matrix, [ac_matrix.shape[0] // 2], axis=0)
    return ac_matrix


def compute_timescale_flash_paper(
    spikes_by_unit: Dict[int, np.ndarray],
    flash_df: pd.DataFrame,
    *,
    bin_edges_s: np.ndarray = np.arange(0, 2.01, 0.01),
    timespan_s: Tuple[float, float] = (0.04, 0.29),
) -> pd.DataFrame:
    """Compute response decay timescale matching Figure3/timescale_calculation.py.

    Outputs columns compatible with downstream merges:
      - autocorr_tau (ms)
      - err_ac
      - spike_count_ac
    """
    rows = []
    if scipy_signal is None or curve_fit is None:
        log("Timescale: SciPy unavailable; filling NaN")
        for uid in spikes_by_unit.keys():
            rows.append({"unit_id": uid, "autocorr_tau": np.nan, "err_ac": np.nan, "spike_count_ac": np.nan})
        return pd.DataFrame(rows)

    if flash_df is None or flash_df.empty or "start_time" not in flash_df.columns:
        log("Timescale: no flash table or missing start_time; filling NaN")
        for uid in spikes_by_unit.keys():
            rows.append({"unit_id": uid, "autocorr_tau": np.nan, "err_ac": np.nan, "spike_count_ac": np.nan})
        return pd.DataFrame(rows)

    starts = flash_df["start_time"].to_numpy(dtype=float)
    left_edges = np.asarray(bin_edges_s[:-1], dtype=float)
    mask = (left_edges >= timespan_s[0]) & (left_edges <= timespan_s[1])
    if not np.any(mask):
        raise ValueError("Timespan mask is empty; check bin_edges_s/timespan_s")

    total = len(spikes_by_unit)
    for idx, (uid, spikes) in enumerate(spikes_by_unit.items()):
        if (idx + 1) % 200 == 0:
            log(f"Timescale progress: {idx + 1}/{total} units")

        if spikes.size == 0 or starts.size == 0:
            rows.append({"unit_id": uid, "autocorr_tau": np.nan, "err_ac": np.nan, "spike_count_ac": np.nan})
            continue

        counts = _bin_spikes_trials_counts(spikes, starts, bin_edges_s)
        spikes_win = counts[:, mask]

        if spikes_win.shape[0] < 3 or spikes_win.shape[1] < 4:
            rows.append({"unit_id": uid, "autocorr_tau": np.nan, "err_ac": np.nan, "spike_count_ac": float(np.sum(spikes_win))})
            continue

        nbins = spikes_win.shape[1]
        try:
            ac_matrix = _paper_autocorr2d(spikes_win)
            if ac_matrix.size == 0:
                raise ValueError("Empty autocorr matrix")

            accg = np.mean(ac_matrix, axis=0)
            accg = accg[nbins // 2 :]
            t_ms = np.linspace(0, nbins / 2 * 10, len(accg))

            params, error = _paper_fit_exp(t_ms, accg)
            timescale = float(params[1])
            err = float(error[1])
            spike_count = float(np.sum(spikes_win))
            rows.append({"unit_id": uid, "autocorr_tau": timescale, "err_ac": err, "spike_count_ac": spike_count})
        except Exception:
            rows.append({"unit_id": uid, "autocorr_tau": np.nan, "err_ac": np.nan, "spike_count_ac": float(np.sum(spikes_win))})

    log(f"Timescale complete: {len(rows)}/{total} units processed")
    return pd.DataFrame(rows)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--site_name", required=True, help="Force area name (e.g. V1_site2)")
    ap.add_argument("--id_offset", type=int, default=1000000, help="Offset for Unit IDs")
    ap.add_argument("--stim_table", default="drifting_gratings")
    ap.add_argument(
        "--only_gratings",
        action="store_true",
        help="Write grating_metrics.csv and skip flash/layer metrics.",
    )
    
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    ex = read_nwb_tables(args.nwb)
    stim_name, stim_df = choose_stim_table(ex.intervals_tables, args.stim_table)
    log(f"Stim table '{stim_name}' selected: {len(stim_df)} rows; columns: {list(stim_df.columns)}")

    # Full-condition drifting-grating metrics.  This file keeps Allen's two
    # temporal-modulation metrics separate and names them canonically.
    log("Computing DG metrics at preferred orientation x TF x SF condition...")
    grating_df = compute_drifting_grating_metrics(
        ex.units_df.index.values, ex.spikes_by_unit, stim_df
    )
    grating_df["unit_id"] = grating_df["unit_id"].astype(int) + args.id_offset
    grating_path = os.path.join(args.out_dir, "grating_metrics.csv")
    grating_df.to_csv(grating_path, index=False)
    log(f"Saved grating metrics: {grating_path} ({len(grating_df)} rows)")
    if args.only_gratings:
        log("Done (grating-only mode)")
        return

    # Compatibility export for code that still expects the old filename.  New
    # analysis code should load grating_metrics.csv and the f1_f0_dg column.
    mod_df = grating_df[["unit_id", "f1_f0_dg"]].rename(
        columns={"f1_f0_dg": "modulation_index"}
    )
    mod_path = os.path.join(args.out_dir, "change_modulation_data.csv")
    mod_df.to_csv(mod_path, index=False)
    log(f"Saved deprecated F1/F0 compatibility export: {mod_path}")

    # Find flash stimulus table (needed for TTFS + paper timescale)
    flash_tables = sorted(k for k in ex.intervals_tables.keys() if 'flash' in k.lower())
    flash_df = ex.intervals_tables[flash_tables[0]] if flash_tables else pd.DataFrame()
    if len(flash_tables) > 1:
        log(f"WARNING: {len(flash_tables)} flash-like tables found {flash_tables}; "
            f"using '{flash_tables[0]}' (alphabetically first)")
    if flash_tables:
        log(f"Using flash table '{flash_tables[0]}' with {len(flash_df)} presentations")
    else:
        log("No flash table found")

    # 2) Timescale (paper; flash-locked)
    log("Computing response decay timescale (paper method)...")
    tau_df = compute_timescale_flash_paper(ex.spikes_by_unit, flash_df)
    tau_df['unit_id'] = tau_df['unit_id'].astype(int) + args.id_offset
    tau_path = os.path.join(args.out_dir, "timescale_metrics.csv")
    tau_df.to_csv(tau_path, index=False)
    log(f"Saved timescale metrics: {tau_path} ({len(tau_df)} rows)")
    
    # 3. Layer Info + Area Renaming (per probe)
    log("Computing Layer Info...")
    layer_df = pd.DataFrame(index=ex.units_df.index)
    layer_df['unit_id'] = ex.units_df.index.astype(int) + args.id_offset
    depth_col = next((c for c in ('probe_vertical_position', 'depth') if c in ex.units_df.columns), None)
    if depth_col:
        layer_df['cortical_depth'] = ex.units_df[depth_col].astype(float)
    else:
        layer_df['cortical_depth'] = np.nan
    layer_df['cortical_layer'] = np.nan
    
    # Map device_name (probe) to area: e.g., probeA -> V1_site2_A
    if 'device_name' in ex.units_df.columns:
        probe_names = ex.units_df['device_name'].astype(str)
        # Extract probe letter (A, B, C, E) from device_name
        probe_letters = probe_names.str.extract(r'[Pp]robe([A-Z])', expand=False)
        # Build area as site_name + probe letter
        layer_df['ecephys_structure_acronym'] = args.site_name + '_' + probe_letters.fillna('unknown')
        log(f"Mapped {probe_letters.nunique()} probes to areas: {layer_df['ecephys_structure_acronym'].unique()}")
    else:
        # Fallback: use site_name for all units if no device_name
        log("No 'device_name' column; using site_name for all units")
        layer_df['ecephys_structure_acronym'] = args.site_name
    
    layer_path = os.path.join(args.out_dir, "layer_info.csv")
    layer_df.to_csv(layer_path, index=False)
    log(f"Saved layer info: {layer_path} ({len(layer_df)} rows)")

    # 4) Latency (TTFS) - legacy code; stored in seconds to match repo data/time_to_first_spike.csv
    log("Computing TTFS (legacy method; 30-200ms)...")
    ttfs_df = compute_ttfs_seconds_legacy(ex.spikes_by_unit, flash_df, window_ms=500, start_time_ms=30, end_time_ms=200)
    ttfs_df['unit_id'] = ttfs_df['unit_id'].astype(int) + args.id_offset
    ttfs_path = os.path.join(args.out_dir, "time_to_first_spike.csv")
    ttfs_df.to_csv(ttfs_path, index=False)
    log(f"Saved latency metrics: {ttfs_path} ({len(ttfs_df)} rows)")
    
    log(f"Done. Processed {args.site_name} -> outputs in '{args.out_dir}'")

if __name__ == "__main__":
    main()
