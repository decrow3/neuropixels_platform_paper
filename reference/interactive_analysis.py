# %% [markdown]
# Interactive Analysis: NWB Data Loading, RF Mapping, and Grating Tuning/OSI
# 
# This script is organized into VS Code interactive cells. Run cell-by-cell (Shift+Enter)
# to explore the dataset. It stitches together the early data-loading steps and adds
# receptive field (RF) mapping and orientation tuning/OSI analyses.
# 
# Assumptions
# - NWB file contains interval tables (under nwbfile.intervals) with stimulus
#   presentations (e.g., 'receptive_field_block_presentations', 'drifting_gratings', etc.).
# - RF table has x/y position columns (default: 'x_position', 'y_position').
# - Grating tables include orientation (degrees), and optionally temporal/spatial frequency.
# - Units table has a ragged 'spike_times' column per unit.
# 
# Notes
# - Edit paths or search logic below to choose your .nwb file.
# - Results are saved to ./results.

# %%
# Imports and configuration
from __future__ import annotations
import os
import sys
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import re
import matplotlib as mpl
from matplotlib import gridspec
import subprocess
import math
try:
    from scipy.optimize import curve_fit  # type: ignore
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    curve_fit = None

from pynwb import NWBHDF5IO

# Plot display behavior: set non-interactive backend unless SHOW_PLOTS requested
SHOW_PLOTS = ("--show" in sys.argv) or (str(os.environ.get("SHOW_PLOTS", "0")).lower() in {"1","true","yes","on"})
if not SHOW_PLOTS:
    # Must set backend before importing pyplot
    mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 (import after backend selection)

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "axes.grid": True,
})

def _maybe_show(fig=None):
    if SHOW_PLOTS:
        plt.show()
    else:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            pass

# Default column names for RF table
RF_POS_X = "x_position"
RF_POS_Y = "y_position"

RESULTS_DIR = os.path.join(".", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# QC thresholds for OSI/tuning analysis (adjust as needed)
OSI_QC = {
    "MIN_TOTAL_SPIKES": 50.0,     # across all grating presentations per unit
    "MIN_MEAN_RATE_HZ": 1.0,      # global_total_spikes / global_total_duration
    "MIN_PRESENTATIONS": 10,      # per-group presentations used for OSI per unit
    "MIN_PREF_VALUE_HZ": 2.0,     # preferred response within group
    "MIN_PARTICIPATION": 0.30,    # fraction of orientations with nonzero response
}

# %%
"""
Discover an NWB file (modify as needed)
Strategy:
- If nwb_file_path is None: search current workspace for .nwb files (recursively) and pick the first.
- If nwb_file_path is a directory: search within that directory for .nwb files.
- If nwb_file_path is a file: use it directly.
"""
# Option A: leave as None to auto-discover, or set to a parent directory to batch-process
nwb_file_path: Optional[str] = '/media/huklaban5/Data/MouseV2/001568/'  # set to parent dir to process all NWBs under it

# Optional CLI override: --nwb <path_to_file>
if "--nwb" in sys.argv:
    try:
        arg_i = sys.argv.index("--nwb") + 1
        if arg_i < len(sys.argv):
            nwb_file_path = sys.argv[arg_i]
    except Exception:
        pass

def _find_nwb_under(root_dir: str) -> Optional[str]:
    cands = [p for p in glob.glob(os.path.join(root_dir, "**", "*.nwb"), recursive=True) if os.path.isfile(p)]
    return cands[0] if cands else None

if nwb_file_path is None:
    # Search the current workspace
    found = _find_nwb_under(".")
    if found is None:
        raise FileNotFoundError("No .nwb files found under current directory. Set nwb_file_path to a valid file.")
    nwb_file_path = found
elif os.path.isdir(nwb_file_path):
    # Search within the provided directory
    batch_files = [p for p in glob.glob(os.path.join(nwb_file_path, "**", "*.nwb"), recursive=True) if os.path.isfile(p)]
    if not batch_files:
        # Fall back to workspace search
        fallback = _find_nwb_under(".")
        if fallback is None:
            raise FileNotFoundError(f"No .nwb files found under provided dir {nwb_file_path!r} or current workspace.")
        nwb_file_path = fallback
    elif len(batch_files) == 1:
        # Exactly one candidate found: proceed with that file
        nwb_file_path = batch_files[0]
    else:
        print(f"Found {len(batch_files)} NWB files under {nwb_file_path}. Processing each sequentially...")
        for one in sorted(batch_files):
            print(f"\n=== Processing: {one} ===")
            try:
                subprocess.run([sys.executable, __file__, "--nwb", one], check=False)
            except Exception as e:
                print(f"Failed processing {one}: {e}")
        # After batch run, exit to avoid running the rest of the single-file pipeline
        sys.exit(0)
else:
    # Assume it's a file path
    if not os.path.isfile(nwb_file_path):
        raise FileNotFoundError(f"Provided nwb_file_path does not exist or is not a file: {nwb_file_path}")

print(f"Using NWB file: {nwb_file_path}")
 

# Derive a dataset label from the NWB filename for output naming
dataset_label = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(str(nwb_file_path)))[0])
SESSION_DIR = os.path.join(RESULTS_DIR, dataset_label)
os.makedirs(SESSION_DIR, exist_ok=True)

# %%
# Load NWB file and list interval tables and units shape
with NWBHDF5IO(nwb_file_path, mode="r", load_namespaces=True) as io:
    nwbfile = io.read()

    # Stimulus interval tables -> pandas
    interval_names = list(nwbfile.intervals.keys())
    stimulus_intervals: Dict[str, pd.DataFrame] = {}
    for name in interval_names:
        try:
            stimulus_intervals[name] = nwbfile.intervals[name].to_dataframe()
        except Exception as e:
            print(f"Skipping {name}: {e}")

    units_df: pd.DataFrame = nwbfile.units.to_dataframe()
    print("Interval tables:")
    for k, df_ in stimulus_intervals.items():
        cols = ", ".join(list(df_.columns)[:8]) + (" ..." if len(df_.columns) > 8 else "")
        print(f"  - {k}: {len(df_)} rows, cols: {cols}")

    print(f"Units: {len(units_df)} units, columns: {list(units_df.columns)[:10]}...")

# Keep handle open for next cells by re-opening on demand when needed

# %%
# Helper: access spike times per unit (list of np.ndarray), plus unit IDs

def load_units_and_spikes(nwb_path: str) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    with NWBHDF5IO(nwb_path, mode="r", load_namespaces=True) as io:
        nf = io.read()
        units_df = nf.units.to_dataframe()
        # Access ragged spike times per row index
        spike_times_col = nf.units["spike_times"]
        spikes_by_unit: List[np.ndarray] = []
        for i in range(len(units_df)):
            try:
                st = np.array(spike_times_col[i])
            except Exception:
                st = np.asarray([])
            spikes_by_unit.append(st)
        return units_df, spikes_by_unit

units_df, spikes_by_unit = load_units_and_spikes(nwb_file_path)
print(f"Loaded {len(units_df)} units with spike time arrays.")

# %%
# RF mapping utilities

def get_rf_table(nwb_path: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns (stim_table, x_col, y_col) for RF block presentations.
    Tries common interval names and column aliases.
    """
    candidates = [
        "receptive_field_block_presentations",
        "rf_presentations",
        "receptive_field_presentations",
    ]
    x_cols = [RF_POS_X, "x", "x_pos", "pos_x"]
    y_cols = [RF_POS_Y, "y", "y_pos", "pos_y"]

    with NWBHDF5IO(nwb_path, mode="r", load_namespaces=True) as io:
        nf = io.read()
        for name in nf.intervals.keys():
            if name in candidates:
                stim = nf.intervals[name].to_dataframe().copy()
                # pick position columns
                found_x = next((c for c in x_cols if c in stim.columns), None)
                found_y = next((c for c in y_cols if c in stim.columns), None)
                if found_x is None or found_y is None:
                    continue
                # standardize
                return stim, found_x, found_y

    raise KeyError("Could not locate an RF interval table with recognized x/y position columns.")


def compute_rf_counts(nwb_path: str,
                      stim_df: pd.DataFrame,
                      spikes_by_unit: List[np.ndarray],
                      x_col: str,
                      y_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (rf_counts, positions_y, positions_x)
      - rf_counts shape: (n_y, n_x, n_units)
    """
    xs = np.sort(stim_df[x_col].unique())
    ys = np.sort(stim_df[y_col].unique())
    n_y, n_x, n_u = len(ys), len(xs), len(spikes_by_unit)
    rf = np.zeros((n_y, n_x, n_u), dtype=float)

    # Iterate presentations, count spikes in [start, stop)
    for _, row in stim_df.iterrows():
        try:
            xi = np.where(xs == row[x_col])[0][0]
            yi = np.where(ys == row[y_col])[0][0]
        except Exception:
            continue
        start_val = row.get("start_time", None)
        stop_val = row.get("stop_time", None)
        if start_val is None or stop_val is None:
            continue
        try:
            start = float(start_val)
            stop = float(stop_val)
        except Exception:
            continue
        for u_idx, st in enumerate(spikes_by_unit):
            if st.size == 0:
                continue
            count = np.sum((st >= start) & (st < stop))
            rf[yi, xi, u_idx] += count

    return rf, ys, xs


def plot_rf_for_unit(rf_counts: np.ndarray,
                     ys: np.ndarray,
                     xs: np.ndarray,
                     unit_index: int,
                     title: Optional[str] = None,
                     cmap: str = "viridis") -> None:
    rf2d = rf_counts[:, :, unit_index]
    fig, ax = plt.subplots()
    im = ax.imshow(rf2d, origin="lower", cmap=cmap,
                   extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                   aspect="auto")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(title or f"RF counts (unit {unit_index})")
    plt.colorbar(im, ax=ax, label="spike count")
    plt.tight_layout()
    _maybe_show(fig)

def compute_rf_counts_single_unit_fast(
    stim_df: pd.DataFrame,
    spike_times: np.ndarray,
    x_col: str,
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast RF spike-count computation for a single unit using vectorized searchsorted.
    Returns (rf2d, positions_y, positions_x)
    """
    # Unique, sorted positions
    xs = np.sort(stim_df[x_col].unique())
    ys = np.sort(stim_df[y_col].unique())

    # Map each presentation to position indices
    x_to_idx = {v: i for i, v in enumerate(xs)}
    y_to_idx = {v: i for i, v in enumerate(ys)}
    xi = stim_df[x_col].map(x_to_idx).to_numpy()
    yi = stim_df[y_col].map(y_to_idx).to_numpy()

    # Start/stop arrays and validity
    starts = np.asarray(pd.to_numeric(stim_df["start_time"], errors="coerce"), dtype=float)
    stops = np.asarray(pd.to_numeric(stim_df["stop_time"], errors="coerce"), dtype=float)
    valid = np.isfinite(starts) & np.isfinite(stops)
    if not np.any(valid):
        return np.zeros((len(ys), len(xs)), float), ys, xs

    xi = xi[valid]
    yi = yi[valid]
    starts = starts[valid]
    stops = stops[valid]

    # Ensure sorted spike times
    st = np.asarray(spike_times, dtype=float)
    if st.size:
        st = np.sort(st)

    # Vectorized counts per presentation via binary search
    idx_start = np.searchsorted(st, starts, side="left")
    idx_stop = np.searchsorted(st, stops, side="left")  # [start, stop)
    counts = (idx_stop - idx_start).astype(float)

    # Accumulate into RF grid
    rf2d = np.zeros((len(ys), len(xs)), dtype=float)
    np.add.at(rf2d, (yi, xi), counts)

    return rf2d, ys, xs

def compute_rf_counts_multiunit_fast(
    stim_df: pd.DataFrame,
    spikes_by_unit: List[np.ndarray],
    x_col: str,
    y_col: str,
    aggregate: str = "mean",  # "mean" across repeats, or "rate" per second
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast RF spike accumulation for all units.
    Returns (rf, ys, xs) with rf shape (n_y, n_x, n_units).
    """
    # Positions and indices (coerce to numeric to avoid category issues)
    xs_vals_arr = np.asarray(pd.to_numeric(stim_df[x_col], errors="coerce"), dtype=float)
    ys_vals_arr = np.asarray(pd.to_numeric(stim_df[y_col], errors="coerce"), dtype=float)
    xs = np.sort(np.unique(xs_vals_arr[~np.isnan(xs_vals_arr)]))
    ys = np.sort(np.unique(ys_vals_arr[~np.isnan(ys_vals_arr)]))

    x_codes = pd.Categorical(xs_vals_arr, categories=list(xs), ordered=True).codes
    y_codes = pd.Categorical(ys_vals_arr, categories=list(ys), ordered=True).codes

    starts = np.asarray(pd.to_numeric(stim_df["start_time"], errors="coerce"), dtype=float)
    stops  = np.asarray(pd.to_numeric(stim_df["stop_time"],  errors="coerce"), dtype=float)

    valid = (x_codes >= 0) & (y_codes >= 0) & np.isfinite(starts) & np.isfinite(stops)
    if not np.any(valid):
        return np.zeros((len(ys), len(xs), len(spikes_by_unit)), float), ys, xs

    xi = x_codes[valid]
    yi = y_codes[valid]
    starts = starts[valid]
    stops = stops[valid]
    durations = stops - starts

    n_y, n_x, n_u = len(ys), len(xs), len(spikes_by_unit)
    rf = np.zeros((n_y, n_x, n_u), dtype=float)

    for u_idx, st in enumerate(spikes_by_unit):
        st = np.asarray(st, dtype=float)
        if st.size == 0:
            continue
        st.sort()
        idx_start = np.searchsorted(st, starts, side="left")
        idx_stop  = np.searchsorted(st, stops,  side="left")  # [start, stop)
        counts = (idx_stop - idx_start).astype(float)
        np.add.at(rf[..., u_idx], (yi, xi), counts)

    if aggregate == "rate":
        dur_grid = np.zeros((n_y, n_x), dtype=float)
        np.add.at(dur_grid, (yi, xi), durations)
        mask = dur_grid > 0
        for u in range(n_u):
            out = np.zeros_like(rf[..., u])
            np.divide(rf[..., u], dur_grid, out=out, where=mask)
            rf[..., u] = out
    elif aggregate == "mean":
        ntrials = np.zeros((n_y, n_x), dtype=float)
        np.add.at(ntrials, (yi, xi), 1.0)
        mask = ntrials > 0
        for u in range(n_u):
            out = np.zeros_like(rf[..., u])
            np.divide(rf[..., u], ntrials, out=out, where=mask)
            rf[..., u] = out

    return rf, ys, xs

def infer_unit_probes(nwb_path: Optional[str], units_df: pd.DataFrame) -> List[str]:
    """Strict probe labels per unit: require units_df['device_name'] and disallow fallbacks.

    Raises KeyError if 'device_name' missing; raises ValueError if any entry is null/NaN.
    """
    if "device_name" not in units_df.columns:
        raise KeyError("Strict probe labeling requires units_df['device_name'] column.")
    raw = units_df["device_name"].tolist()
    if any(pd.isna(v) for v in raw):
        raise ValueError("units_df['device_name'] contains missing values; cannot label probes strictly.")
    return [str(v) for v in raw]

# --- QC and RF plotting helpers mirroring Cells 22–25 ---

def select_high_quality_units(units_df: pd.DataFrame) -> pd.Index:
    """snr > 5, d_prime > 2, rp_contamination < 0.1, default_qc == True (skip missing cols)."""
    mask = pd.Series(True, index=units_df.index)
    if "snr" in units_df.columns:
        mask &= pd.to_numeric(units_df["snr"], errors="coerce") > 5
    if "d_prime" in units_df.columns:
        mask &= pd.to_numeric(units_df["d_prime"], errors="coerce") > 2
    if "rp_contamination" in units_df.columns:
        mask &= pd.to_numeric(units_df["rp_contamination"], errors="coerce") < 0.1
    if "default_qc" in units_df.columns:
        mask &= units_df["default_qc"].astype(bool)
    return units_df.index[mask]

def _unit_scores_from_rf(rf_counts: np.ndarray, mode: str = "max") -> np.ndarray:
    flat = rf_counts.reshape(-1, rf_counts.shape[-1])
    if mode == "sum":
        return np.nansum(flat, axis=0)
    if mode == "mean":
        return np.nanmean(flat, axis=0)
    return np.nanmax(flat, axis=0)


def _estimate_noise_sigma(rf2d: np.ndarray, mode: str = "mad") -> float:
    """Estimate noise sigma from an unsmoothed RF map.

    - Subtracts median, then estimates spread.
    - mode == 'mad' uses 1.4826 * MAD; otherwise falls back to std.
    """
    vals = rf2d[np.isfinite(rf2d)].ravel()
    if vals.size == 0:
        return float("nan")
    vals = vals - np.nanmedian(vals)
    if mode == "mad":
        mad = np.nanmedian(np.abs(vals))
        sigma = 1.4826 * mad
    else:
        sigma = np.nanstd(vals, ddof=0)
    return float(sigma)


def _has_rf_mask(
    rf_counts: np.ndarray,
    smooth_sigma: float = 1.0,
    snr_threshold: float = 3.0,
    noise_mode: str = "mad",
) -> np.ndarray:
    """Boolean mask per unit: smoothed peak stands above unsmoothed noise floor.

    For each unit u:
      - Smooth RF with Gaussian(sigma=smooth_sigma) and take peak value p.
      - Estimate noise sigma from the raw (unsmoothed) RF via MAD or std.
      - Mark unit if p / sigma_noise >= snr_threshold and p > 0.
    """
    n_u = rf_counts.shape[-1]
    has = np.zeros(n_u, dtype=bool)
    for u in range(n_u):
        rf2d = rf_counts[..., u]
        if not np.isfinite(rf2d).any():
            continue
        # Smooth, treating NaNs as 0 contribution
        smoothed = gaussian_filter(np.nan_to_num(rf2d, nan=0.0), sigma=smooth_sigma)
        p = float(np.nanmax(smoothed))
        if not np.isfinite(p) or p <= 0:
            continue
        sigma_n = _estimate_noise_sigma(rf2d, mode=noise_mode)
        if not np.isfinite(sigma_n) or sigma_n <= 0:
            # fallback to std if MAD failed
            sigma_n = float(np.nanstd(rf2d))
        if not np.isfinite(sigma_n) or sigma_n <= 0:
            continue
        snr = p / (sigma_n + 1e-9)
        if snr >= snr_threshold:
            has[u] = True
    return has

def compute_probe_avg_maps(
    rf_counts: np.ndarray,
    units_df: pd.DataFrame,
    ys: np.ndarray,
    xs: np.ndarray,
    normalize_per_unit: bool = False,
    top_k: Optional[int] = None,
    score_mode: str = "max",
    good_units_idx: Optional[pd.Index] = None,
    has_rf_smooth_sigma: float = 1.0,
    has_rf_snr_threshold: float = 3.0,
    has_rf_noise_mode: str = "mad",
) -> Dict[str, np.ndarray]:
    """Average RF per probe using all good units that have RFs.

    Notes:
    - By default, we do NOT normalize individual unit RFs so amplitude differences are preserved.
    - A unit is counted as having an RF if the smoothed peak (Gaussian sigma=has_rf_smooth_sigma)
      exceeds the unsmoothed noise floor (estimated via has_rf_noise_mode) by has_rf_snr_threshold.
    - If top_k is provided, selection is applied within each probe after filtering to valid RFs.
    """
    n_y, n_x, n_u = rf_counts.shape
    assert len(units_df) == n_u, "units_df length must match rf_counts units axis"

    probe_labels = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
    if good_units_idx is None:
        good_units_idx = units_df.index

    # map index -> positional index (row order)
    index_to_pos = pd.Series(np.arange(n_u), index=units_df.index)

    rf_stack = rf_counts.copy()
    # Determine which units actually have an RF by smoothed-peak vs unsmoothed noise
    unit_has_rf = _has_rf_mask(
        rf_stack,
        smooth_sigma=has_rf_smooth_sigma,
        snr_threshold=has_rf_snr_threshold,
        noise_mode=has_rf_noise_mode,
    )
    # Optional per-unit normalization (off by default)
    if normalize_per_unit:
        unit_max_norm = np.nanmax(rf_stack.reshape(-1, n_u), axis=0)
        unit_max_safe = unit_max_norm.copy()
        unit_max_safe[~np.isfinite(unit_max_safe)] = 0.0
        safe = unit_max_safe > 0
        rf_stack[..., safe] = rf_stack[..., safe] / unit_max_safe[safe]

    scores = _unit_scores_from_rf(rf_counts, mode=score_mode)

    avg_maps: Dict[str, np.ndarray] = {}
    for probe, group_idx in units_df.loc[good_units_idx].groupby(probe_labels.loc[good_units_idx]).groups.items():
        pos_ids = index_to_pos.loc[group_idx].to_numpy()
        # Keep only units that have a non-empty RF
        if pos_ids.size:
            pos_ids = pos_ids[unit_has_rf[pos_ids]]
        if pos_ids.size == 0:
            continue
        if top_k is not None and top_k < pos_ids.size:
            local_scores = scores[pos_ids]
            sel = np.argsort(local_scores)[-top_k:]
            pos_ids = pos_ids[sel]
        probe_rf = rf_stack[..., pos_ids]
        if probe_rf.size == 0:
            continue
        avg_maps[str(probe)] = np.nanmean(probe_rf, axis=-1)
    return avg_maps

def plot_probe_unit_panels(
    rf_counts: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    units_df: pd.DataFrame,
    probe_name: str,
    good_units_idx: pd.Index,
    n_show: int = 25,
):
    """Plot first n_show QC units for a probe, normalized per unit (gray colormap)."""
    probe_labels = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
    sel = units_df.index[probe_labels == probe_name].intersection(good_units_idx)
    if len(sel) == 0:
        print(f"No units for probe {probe_name}")
        return
    # positions
    index_to_pos = pd.Series(np.arange(len(units_df)), index=units_df.index)
    pos = index_to_pos.loc[sel].to_numpy()[:n_show]
    if pos.size == 0:
        print(f"No units to display for probe {probe_name}")
        return

    n = pos.size
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))

    for ax, u_pos in zip(axes, pos):
        rf = rf_counts[..., u_pos]
        maxv = np.nanmax(rf)
        if maxv > 0:
            rf = rf / maxv
        im = ax.imshow(rf, cmap="gray", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
        unit_id = units_df.index[u_pos]
        ax.set_title(f"U{unit_id}", fontsize=6)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{probe_name}: first {n} QC units", fontsize=12)
    plt.tight_layout()
    _maybe_show(fig)

def save_probe_qc_units_grid(
    rf_counts: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    units_df: pd.DataFrame,
    probe_name: str,
    good_units_idx: pd.Index,
    out_dir: str,
    n_show: int = 25,
) -> Optional[str]:
    """Save a grid of the first n_show QC units for the given probe; returns filepath or None."""
    os.makedirs(out_dir, exist_ok=True)
    probe_labels = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
    sel = units_df.index[probe_labels == probe_name].intersection(good_units_idx)
    if len(sel) == 0:
        return None
    index_to_pos = pd.Series(np.arange(len(units_df)), index=units_df.index)
    pos = index_to_pos.loc[sel].to_numpy()[:n_show]
    if pos.size == 0:
        return None

    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    for ax, u_pos in zip(axes, pos):
        rf = rf_counts[..., u_pos]
        maxv = np.nanmax(rf)
        if maxv > 0:
            rf = rf / maxv
        ax.imshow(rf, cmap="gray", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
        unit_id = units_df.index[u_pos]
        ax.set_title(f"U{unit_id}", fontsize=6)
        ax.axis("off")
    for ax in axes[len(pos):]:
        ax.axis("off")
    fig.suptitle(f"{probe_name}: first {len(pos)} QC units", fontsize=12)
    plt.tight_layout()
    probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe_name))
    fname = os.path.join(out_dir, f"{dataset_label}_rf_qc_first{n_show}_{probe_safe}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

def save_probe_top_units_grid(
    rf_counts: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    units_df: pd.DataFrame,
    probe_name: str,
    out_dir: str,
    n_top: int = 25,
) -> Optional[str]:
    """Save a grid PNG of the top n_top units (by max RF) for a probe; returns filepath or None."""
    os.makedirs(out_dir, exist_ok=True)
    probe_labels = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
    sel = units_df.index[probe_labels == probe_name]
    if len(sel) == 0:
        return None
    index_to_pos = pd.Series(np.arange(len(units_df)), index=units_df.index)
    pos = index_to_pos.loc[sel].to_numpy()
    if pos.size == 0:
        return None

    # Score by max RF
    scores = np.nanmax(rf_counts.reshape(-1, rf_counts.shape[-1]), axis=0)
    top_idx = pos[np.argsort(scores[pos])[-n_top:]] if pos.size > n_top else pos
    # Sort descending for aesthetics
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    n = top_idx.size
    cols = int(np.ceil(np.sqrt(n_top)))
    rows = int(np.ceil(n_top / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    for ax, u_pos in zip(axes, top_idx):
        rf = rf_counts[..., u_pos]
        maxv = np.nanmax(rf)
        if maxv > 0:
            rf = rf / maxv
        ax.imshow(rf, cmap="gray", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
        unit_id = units_df.index[u_pos]
        ax.set_title(f"U{unit_id}", fontsize=6)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{probe_name}: top {min(n_top, n)} units", fontsize=12)
    plt.tight_layout()
    probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe_name))
    fname = os.path.join(out_dir, f"{dataset_label}_rf_top{n_top}_{probe_safe}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

def plot_probe_avg_maps(avg_maps: Dict[str, np.ndarray], ys: np.ndarray, xs: np.ndarray, title: str = "Average RF per probe"):
    n_probes = len(avg_maps)
    if n_probes == 0:
        print("No average maps to plot.")
        return
    cols = int(np.ceil(np.sqrt(n_probes)))
    rows = int(np.ceil(n_probes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    for ax, (probe, rf_avg) in zip(axes, avg_maps.items()):
        ax.imshow(rf_avg, cmap="inferno", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
        ax.set_title(f"{probe}", fontsize=9)
        ax.axis("off")
    for ax in axes[n_probes:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    _maybe_show(fig)

def save_probe_avg_maps_png(avg_maps: Dict[str, np.ndarray], ys: np.ndarray, xs: np.ndarray, out_dir: str, filename_prefix: Optional[str] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    for probe, rf_avg in avg_maps.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(rf_avg, cmap="inferno", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
        ax.set_title(f"{probe}")
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        plt.tight_layout()
        probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
        prefix = (filename_prefix + "_") if filename_prefix else ""
        fname = os.path.join(out_dir, f"{prefix}rf_avg_{probe_safe}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)

def save_avg_rf_grid_png(avg_maps: Dict[str, np.ndarray], ys: np.ndarray, xs: np.ndarray, out_path: str, title: str = "Average RF per probe") -> Optional[str]:
    """Save a composite grid image of the per-probe average RF maps to the given path."""
    if not avg_maps:
        return None
    n_probes = len(avg_maps)
    cols = int(np.ceil(np.sqrt(n_probes)))
    rows = int(np.ceil(n_probes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    for ax, (probe, rf_avg) in zip(axes, avg_maps.items()):
        ax.imshow(rf_avg, cmap="inferno", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
        ax.set_title(f"{probe}", fontsize=9)
        ax.axis("off")
    for ax in axes[n_probes:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def save_single_probe_avg_map_png(
    probe: str,
    rf_avg: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    out_dir: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(rf_avg, cmap="inferno", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
    ax.set_title(f"{probe}")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    plt.tight_layout()
    probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
    fname = os.path.join(out_dir, f"rf_avg_{probe_safe}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

def save_osi_histograms_png(osi_df: pd.DataFrame, out_dir: str, prefix: str = "gratings") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Overall histograms for classic and vector OSI across all groups/units
    for col in ["osi_classic", "osi_vector"]:
        if col not in osi_df.columns:
            continue
        series = pd.to_numeric(osi_df[col], errors="coerce")
        # Ensure this is a Series and drop NaNs
        vals = pd.Series(series).dropna()
        if vals.empty:
            continue
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(vals, bins=30, color="#1f77b4", alpha=0.85)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(f"Distribution of {col}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{prefix}_{col}_hist.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)

def sanitize_grating_columns(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Coerce orientation, temporal_frequency, spatial_frequency to numeric and validate strictly.

    Prints unique values for present columns. If strict=True, raises on any non-numeric values.
    """
    out = df.copy()
    check_cols = ["orientation", "temporal_frequency", "spatial_frequency"]
    for c in check_cols:
        if c in out.columns:
            numeric = pd.to_numeric(out[c], errors="coerce")
            mask_bad = pd.isna(numeric) & pd.notna(out[c])
            n_bad = int(np.count_nonzero(mask_bad.to_numpy())) if hasattr(mask_bad, "to_numpy") else int(np.count_nonzero(mask_bad))
            if strict and n_bad > 0:
                sample_bad = pd.Series(out.loc[mask_bad, c]).astype(str).head(5).tolist()
                raise ValueError(f"Non-numeric entries in '{c}': count={n_bad}, samples={sample_bad}")
            out[c] = numeric
            uniq = np.sort(out[c].dropna().unique())
            print(f"Unique {c} values ({len(uniq)}): {uniq[:20]}{' ...' if len(uniq) > 20 else ''}")
        else:
            print(f"Column '{c}' not found in gratings table.")
    return out

def attach_probe_to_osi_df(osi_df: pd.DataFrame, units_df: pd.DataFrame, nwb_path: Optional[str]) -> pd.DataFrame:
    """Add a 'probe' column to OSI summary using unit_index to look up the unit's probe label."""
    probes = infer_unit_probes(nwb_path, units_df)
    probes_series = pd.Series(probes)
    # Map unit_index (positional) to probe label
    osi_df = osi_df.copy()
    if "unit_index" in osi_df.columns:
        osi_df["probe"] = osi_df["unit_index"].apply(lambda i: str(probes_series.iloc[int(i)]) if pd.notna(i) else "unknown")
    else:
        osi_df["probe"] = "unknown"
    return osi_df

def compute_pref_group_per_unit(osi_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """For each unit, choose the group (TF/SF combo) with the max pref_value and return those preferences."""
    if not group_cols:
        return pd.DataFrame()
    # Ensure needed columns exist
    cols = [c for c in ["unit_index", "pref_value", "pref_angle"] + group_cols if c in osi_df.columns]
    if "unit_index" not in cols or "pref_value" not in cols:
        return pd.DataFrame()
    df = osi_df[cols].copy()
    # For units with multiple rows (different groups), keep row with max pref_value
    idx = df.groupby("unit_index")["pref_value"].idxmax()
    best = df.loc[idx].reset_index(drop=True)
    # Rename group columns to preferred_* names
    rename_map = {c: f"pref_{c}" for c in group_cols}
    best = best.rename(columns=rename_map)
    return best[[c for c in ["unit_index", "pref_angle", "pref_value"] + list(rename_map.values()) if c in best.columns]]

def save_per_probe_histograms(
    osi_df_with_probe: pd.DataFrame,
    pref_group_df: pd.DataFrame,
    out_dir: str,
    filename_prefix: Optional[str] = None,
    fit_pref_df: Optional[pd.DataFrame] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Per-probe OSI and preferred angle histograms; plus preferred SF/TF if available
    for probe, dfp in osi_df_with_probe.groupby("probe"):
        probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
        # Save into a dedicated subfolder per probe
        probe_dir = os.path.join(out_dir, probe_safe)
        os.makedirs(probe_dir, exist_ok=True)
        prefix = (filename_prefix + "_") if filename_prefix else ""
        # OSI hists
        for col in ["osi_classic", "osi_vector"]:
            if col not in dfp.columns:
                continue
            vals = pd.Series(pd.to_numeric(dfp[col], errors="coerce")).dropna()
            if vals.empty:
                continue
            fig, ax = plt.subplots(figsize=(4,3))
            ax.hist(vals, bins=30, color="#2ca02c", alpha=0.85)
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            ax.set_title(f"{probe} - {col}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = os.path.join(probe_dir, f"{prefix}{col}_hist.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)

        # Preferred angle hist
        if "pref_angle" in dfp.columns:
            angs = pd.Series(pd.to_numeric(dfp["pref_angle"], errors="coerce")).dropna()
            if not angs.empty:
                fig, ax = plt.subplots(figsize=(4,3))
                ax.hist(angs, bins=np.linspace(0, 180, 19), color="#ff7f0e", alpha=0.85)
                ax.set_xlabel("preferred angle (deg)")
                ax.set_ylabel("count")
                ax.set_title(f"{probe} - preferred angle")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fname = os.path.join(probe_dir, f"{prefix}pref_angle_hist.png")
                fig.savefig(fname, dpi=150)
                plt.close(fig)

        # Preferred SF/TF from discrete groups if provided
        if not pref_group_df.empty:
            # Merge preferred group info for units present in this probe slice
            units_in_probe = pd.unique(dfp["unit_index"]) if "unit_index" in dfp.columns else []
            pref_sub = pref_group_df[pref_group_df["unit_index"].isin(units_in_probe)].copy()
            for pcol, label in [("pref_spatial_frequency", "pref_SF"), ("pref_temporal_frequency", "pref_TF")]:
                if pcol in pref_sub.columns:
                    vals = pd.Series(pd.to_numeric(pref_sub[pcol], errors="coerce")).dropna()
                    if vals.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.hist(vals, bins=20, color="#1f77b4", alpha=0.85)
                    ax.set_xlabel(pcol)
                    ax.set_ylabel("count")
                    ax.set_title(f"{probe} - {label}")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    fname = os.path.join(probe_dir, f"{prefix}{pcol}_hist.png")
                    fig.savefig(fname, dpi=150)
                    plt.close(fig)

        # Preferred SF/TF from fitted log-Gaussian peaks (continuous), if provided
        if fit_pref_df is not None and not fit_pref_df.empty and ("probe" in fit_pref_df.columns):
            fit_slice = fit_pref_df[fit_pref_df["probe"].astype(str) == str(probe)].copy()
            for pcol, label in [("sf_pref", "pref_SF_fit"), ("tf_pref", "pref_TF_fit")]:
                if pcol in fit_slice.columns:
                    vals = pd.Series(pd.to_numeric(fit_slice[pcol], errors="coerce")).dropna()
                    # Keep positive, finite values
                    vals = vals[np.isfinite(vals) & (vals > 0)]
                    if vals.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.hist(vals, bins=30, color="#9467bd", alpha=0.85)
                    ax.set_xlabel(label)
                    ax.set_ylabel("count")
                    ax.set_title(f"{probe} - {label}")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    fname = os.path.join(probe_dir, f"{prefix}{pcol}_hist.png")
                    fig.savefig(fname, dpi=150)
                    plt.close(fig)

def save_session_summary_figure(
    avg_maps: Dict[str, np.ndarray],
    rf_counts: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    units_df: pd.DataFrame,
    osi_df_with_probe: pd.DataFrame,
    out_dir: str,
    dataset_label: str,
    good_units_idx: Optional[pd.Index] = None,
    n_small: int = 6,
) -> Optional[str]:
    """Create a single session summary PNG with:
    - Row 1: per-probe average RFs
    - Row 2: global OSI histograms (classic and vector)
    - Rows 3..: small sets of normalized unit RFs (top n_small by max RF) per probe
    Returns path if saved, else None.
    """
    try:
        if not avg_maps or rf_counts.size == 0 or units_df is None or osi_df_with_probe is None:
            return None
        probes = list(avg_maps.keys())
        n_probes = len(probes)
        if n_probes == 0:
            return None

        ncols = max(n_probes, n_small, 2)
        height = 3 + 3 + 2.2 * n_probes
        fig = plt.figure(figsize=(ncols * 2.2, height))
        gs = gridspec.GridSpec(nrows=2 + n_probes, ncols=ncols, figure=fig,
                               height_ratios=[3, 3] + [2.2] * n_probes)

        # Row 0: per-probe average RFs
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        for i, probe in enumerate(probes):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(avg_maps[probe], cmap="inferno", origin="lower",
                      extent=[x_min, x_max, y_min, y_max], aspect="equal")
            ax.set_title(str(probe), fontsize=10)
            ax.axis("off")

        # Row 1: global OSI histograms
        osi_cols = ["osi_classic", "osi_vector"]
        for j, col in enumerate(osi_cols):
            ax = fig.add_subplot(gs[1, j])
            if col in osi_df_with_probe.columns:
                vals = pd.Series(pd.to_numeric(osi_df_with_probe[col], errors="coerce")).dropna()
                if not vals.empty:
                    ax.hist(vals, bins=30, color="#1f77b4" if j == 0 else "#2ca02c", alpha=0.85)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)

        # Rows 2..: per-probe small RFs grids
        probe_labels = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
        index_to_pos = pd.Series(np.arange(len(units_df)), index=units_df.index)
        scores = np.nanmax(rf_counts.reshape(-1, rf_counts.shape[-1]), axis=0)
        if good_units_idx is None:
            good_units_idx = units_df.index
        for r, probe in enumerate(probes):
            row = 2 + r
            ax_row_title = fig.add_subplot(gs[row, 0])
            ax_row_title.axis("off")
            ax_row_title.text(0, 0.5, f"{probe} units", fontsize=10, va="center", ha="left")
            sel_idx = units_df.index[probe_labels == probe].intersection(good_units_idx)
            if len(sel_idx) == 0:
                continue
            pos = index_to_pos.loc[sel_idx].to_numpy()
            # top n_small by score
            if pos.size > n_small:
                top = pos[np.argsort(scores[pos])[-n_small:]]
                top = top[np.argsort(scores[top])[::-1]]
            else:
                top = pos
            for c in range(min(n_small, len(top))):
                ax = fig.add_subplot(gs[row, c + 1])  # shift by 1 to leave space for label
                rf = rf_counts[..., top[c]]
                m = np.nanmax(rf)
                if m > 0:
                    rf = rf / m
                ax.imshow(rf, cmap="gray", origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
                unit_id = units_df.index[top[c]]
                ax.set_title(f"U{unit_id}", fontsize=7)
                ax.axis("off")

        fig.suptitle(f"Session Summary: {dataset_label}", fontsize=14)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dataset_label}_session_summary.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception:
        return None

# --- New: per-unit RF peak CSVs and per-unit tuning CSVs per probe ---

def compute_per_unit_rf_peaks(
    rf_counts: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    units_df: pd.DataFrame,
    good_units_idx: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Compute RF peak location for each unit.
    Returns DataFrame with columns: unit_index, unit_id, probe, peak_y_idx, peak_x_idx, peak_y, peak_x, peak_value, is_qc.
    """
    n_y, n_x, n_u = rf_counts.shape
    probes = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
    qc_set = set(good_units_idx) if good_units_idx is not None else set()
    rows = []
    for u in range(n_u):
        arr = rf_counts[..., u]
        unit_id = units_df.index[u]
        probe = str(probes.loc[unit_id]) if unit_id in probes.index else "unknown"
        peak_y_idx = peak_x_idx = None
        peak_y = peak_x = np.nan
        peak_val = np.nan
        try:
            # Handle all-NaN arrays
            if not np.isfinite(arr).any():
                raise ValueError("all NaN")
            imax = int(np.nanargmax(arr))
            peak_y_idx = int(imax // n_x)
            peak_x_idx = int(imax % n_x)
            peak_val = float(arr[peak_y_idx, peak_x_idx])
            peak_y = float(ys[peak_y_idx])
            peak_x = float(xs[peak_x_idx])
        except Exception:
            pass
        rows.append({
            "unit_index": u,
            "unit_id": unit_id,
            "probe": probe,
            "peak_y_idx": peak_y_idx,
            "peak_x_idx": peak_x_idx,
            "peak_y": peak_y,
            "peak_x": peak_x,
            "peak_value": peak_val,
            "is_qc": bool(unit_id in qc_set),
        })
    return pd.DataFrame(rows)

def save_per_probe_rf_peaks_csv(peaks_df: pd.DataFrame, session_dir: str, dataset_label: str) -> None:
    """Save per-probe RF peaks as CSVs under results/<dataset_label>/<probe>/"""
    for probe, dfp in peaks_df.groupby("probe"):
        probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
        out_dir = os.path.join(session_dir, probe_safe)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{dataset_label}_rf_peaks.csv")
        dfp.sort_values(["is_qc", "peak_value"], ascending=[False, False]).to_csv(out_csv, index=False)

def compute_best_tuning_per_unit(osi_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """From OSI summary (potentially multiple rows per unit for different TF/SF groups),
    select the row with max pref_value per unit. Returns those rows with OSI and group columns.
    """
    if osi_df.empty:
        return pd.DataFrame()
    df = osi_df.copy()
    idx = df.groupby("unit_index")["pref_value"].idxmax()
    best = df.loc[idx].reset_index(drop=True)
    # Keep only relevant columns
    keep = [c for c in ["unit_index", "osi_classic", "osi_vector", "pref_angle", "pref_value"] + group_cols if c in best.columns]
    if len(keep) == 0:
        return pd.DataFrame()
    df_out = pd.DataFrame(best.loc[:, keep].copy())
    return df_out

def save_per_probe_tuning_csv(best_tuning_df: pd.DataFrame, units_df: pd.DataFrame, session_dir: str, dataset_label: str) -> None:
    """Attach probe labels and save per-probe tuning parameters as CSVs."""
    if best_tuning_df.empty:
        return
    probes = pd.Series(infer_unit_probes(nwb_file_path, units_df))
    best = best_tuning_df.copy()
    # Add unit_id and probe from positional unit_index
    best["unit_id"] = best["unit_index"].apply(lambda i: units_df.index[int(i)] if pd.notna(i) else None)
    best["probe"] = best["unit_index"].apply(lambda i: str(probes.iloc[int(i)]) if pd.notna(i) else "unknown")
    for probe, dfp in best.groupby("probe"):
        probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
        out_dir = os.path.join(session_dir, probe_safe)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{dataset_label}_tuning_best.csv")
        dfp.sort_values(["pref_value"], ascending=False).to_csv(out_csv, index=False)

# %%
# Compute and plot RF maps (fast, all units; then pick top or specific unit)
try:
    stim_rf, x_col, y_col = get_rf_table(nwb_file_path)
    rf_counts, pos_y, pos_x = compute_rf_counts_multiunit_fast(
        stim_rf, spikes_by_unit, x_col, y_col, aggregate="mean"
    )
    print(f"RF stack: {rf_counts.shape}  (y, x, units)")

    unit_index = 0  # change as desired
    if rf_counts.shape[-1] > 0:
        plot_rf_for_unit(rf_counts, pos_y, pos_x, unit_index)
    else:
        print("No units available for RF plotting.")
except KeyError as e:
    print(f"RF table not found or missing columns: {e}")

# %%
# Average RFs per probe and per-probe unit panels (QC), mirroring notebook Cells 22–25
try:
    # Ensure rf_counts exists
    if 'rf_counts' not in locals():
        stim_rf, x_col, y_col = get_rf_table(nwb_file_path)
        rf_counts, pos_y, pos_x = compute_rf_counts_multiunit_fast(
            stim_rf, spikes_by_unit, x_col, y_col, aggregate="mean"
        )

    # QC selection and probe labels
    good_idx = select_high_quality_units(units_df)
    print(f"High-quality units: {len(good_idx)} of {len(units_df)}")
    probes_series = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
    print(f"Unique probes found: {sorted(probes_series.unique())}")
    for probe, dfp in units_df.loc[good_idx].groupby(probes_series.loc[good_idx]):
        print(f"  {probe}: {len(dfp)} good units")

    # Plot first 25 QC units for the first probe
    if len(probes_series.unique()) > 0 and len(good_idx) > 0:
        first_probe_idx = units_df.loc[good_idx].groupby(probes_series.loc[good_idx]).size().index[0]
        first_probe = str(first_probe_idx)
        plot_probe_unit_panels(rf_counts, pos_y, pos_x, units_df, first_probe, good_idx, n_show=25)
        # Save QC grid for the first probe in the session/probe folder
        first_probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(first_probe))
        first_probe_dir = os.path.join(SESSION_DIR, first_probe_safe)
        _ = save_probe_qc_units_grid(rf_counts, pos_y, pos_x, units_df, first_probe, good_idx, first_probe_dir, n_show=25)

    # Build average RF per probe (no per-unit normalization; include all QC units with RFs)
    avg_maps = compute_probe_avg_maps(
        rf_counts, units_df, pos_y, pos_x,
        normalize_per_unit=False,
        top_k=None,  # keep None to use all good units with RFs per probe
        score_mode="max",
        good_units_idx=good_idx,
    )
    plot_probe_avg_maps(avg_maps, pos_y, pos_x, title="Average RF per probe (QC units; no normalization)")
    # Save composite Average RF per probe grid PNG in session folder
    _ = save_avg_rf_grid_png(avg_maps, pos_y, pos_x, out_path=os.path.join(SESSION_DIR, f"{dataset_label}_avg_rf_per_probe.png"), title="Average RF per probe")
    # Save per-probe average maps into session/probe subfolders
    for probe, rf_avg in avg_maps.items():
        probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
        probe_dir = os.path.join(SESSION_DIR, probe_safe)
        _ = save_single_probe_avg_map_png(str(probe), rf_avg, pos_y, pos_x, out_dir=probe_dir)

    # Save top-25 grids per probe
    probes_series = pd.Series(infer_unit_probes(nwb_file_path, units_df), index=units_df.index)
    for probe in sorted(probes_series.unique()):
        probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
        probe_dir = os.path.join(SESSION_DIR, probe_safe)
        # Save the first-25 QC grid per probe
        _ = save_probe_qc_units_grid(rf_counts, pos_y, pos_x, units_df, str(probe), good_units_idx=good_idx, out_dir=probe_dir, n_show=25)
        # Save the top-25-by-score grid per probe
        _ = save_probe_top_units_grid(rf_counts, pos_y, pos_x, units_df, str(probe), out_dir=probe_dir, n_top=25)
    # Save per-unit RF peak locations as per-probe CSVs
    try:
        peaks_df = compute_per_unit_rf_peaks(rf_counts, pos_y, pos_x, units_df, good_units_idx=good_idx)
        save_per_probe_rf_peaks_csv(peaks_df, SESSION_DIR, dataset_label)
    except Exception as _e:
        pass
    # Preliminary session summary (RFs, placeholders for OSI)
    _ = save_session_summary_figure(
        avg_maps=avg_maps,
        rf_counts=rf_counts,
        ys=pos_y,
        xs=pos_x,
        units_df=units_df,
        osi_df_with_probe=pd.DataFrame(),
        out_dir=SESSION_DIR,
        dataset_label=dataset_label,
        good_units_idx=good_idx,
        n_small=6,
    )
except Exception as e:
    print(f"Probe RF averaging/QC plotting failed: {e}")

# %%
# Gratings: tuning functions and OSI for each unit
# We search interval tables that contain 'orientation' and compute spike rate per orientation.

def find_grating_table(nwb_path: str) -> pd.DataFrame:
    candidates_substrings = [
        "drifting", "grating", "gratings", "static_grating", "gratings_presentations",
    ]
    with NWBHDF5IO(nwb_path, mode="r", load_namespaces=True) as io:
        nf = io.read()
        for name in nf.intervals.keys():
            if any(sub in name.lower() for sub in candidates_substrings):
                df = nf.intervals[name].to_dataframe().copy()
                if "orientation" in df.columns:
                    return df
    raise KeyError("Could not locate a grating interval table with an 'orientation' column.")


def compute_tuning_and_osi(gratings_df: pd.DataFrame,
                           spikes_by_unit: List[np.ndarray],
                           orientation_col: str = "orientation",
                           rate: bool = True,
                           group_by: Optional[List[str]] = None,
                           orientation_mode: str = "orientation"  # "orientation" in [0,180), or "direction" in [0,360)
                           ) -> Tuple[pd.DataFrame, Dict[Tuple, Dict[int, pd.DataFrame]]]:
    """
    Returns (osi_summary_df, tuning_curves_by_unit)
      - osi_summary_df columns: unit_index, osi_classic, osi_vector, pref_orientation, pref_rate
      - tuning_curves_by_unit: unit_index -> DataFrame with columns [orientation_deg, rate_or_count]
    """
    # Normalize angle
    gdf = gratings_df.copy()
    # Coerce orientation column to numeric to avoid pandas string-formatting errors with modulo
    gdf[orientation_col] = pd.to_numeric(gdf[orientation_col], errors="coerce")
    gdf = gdf.dropna(subset=[orientation_col])
    if orientation_mode == "direction":
        angle_period = 360.0
        angle_factor = 1.0  # single-angle vector
    else:
        angle_period = 180.0
        angle_factor = 2.0  # double-angle vector for orientation
    # Apply modulo using numpy on a float array to avoid pandas operator dispatch
    gdf[orientation_col] = np.mod(gdf[orientation_col].to_numpy(dtype=float), angle_period)

    # Group structure
    if group_by is None:
        group_by = []
    groups = [tuple()]
    if group_by:
        # Coerce potential missing cols
        for col in group_by:
            if col not in gdf.columns:
                gdf[col] = np.nan
        groups = sorted(gdf[group_by].drop_duplicates().itertuples(index=False, name=None))

    # Duration per presentation
    durations = (gdf["stop_time"].astype(float) - gdf["start_time"].astype(float)).to_numpy()

    tuning_by_group_and_unit: Dict[Tuple, Dict[int, pd.DataFrame]] = {}
    records: List[dict] = []

    # Index presentations per orientation for faster counting
    gdf_reset = gdf.reset_index(drop=True)
    angle_arr = gdf_reset[orientation_col].to_numpy()
    # Build per-group presentation index maps
    pres_by_group_and_angle: Dict[Tuple, Dict[float, List[int]]]= {}
    for gi in range(len(gdf_reset)):
        angle = float(angle_arr[gi])
        if group_by:
            # Build group key deterministically, avoiding ambiguous Series -> list conversions
            grp_vals = [gdf_reset.at[gi, col] for col in group_by]
            grp = tuple(grp_vals)
        else:
            grp = tuple()
        pres_by_group_and_angle.setdefault(grp, {}).setdefault(angle, []).append(gi)

    starts = gdf["start_time"].astype(float).to_numpy()
    stops = gdf["stop_time"].astype(float).to_numpy()

    # Global metrics per unit for QC
    n_units = len(spikes_by_unit)
    global_total_spikes = np.zeros(n_units, dtype=float)
    global_total_dur = float(np.nansum(durations)) if durations.size else 0.0
    for u_idx, st_all in enumerate(spikes_by_unit):
        st_all = np.asarray(st_all, dtype=float)
        if st_all.size == 0:
            continue
        st_all.sort()
        idx_start = np.searchsorted(st_all, starts, side="left")
        idx_stop  = np.searchsorted(st_all, stops,  side="left")
        global_total_spikes[u_idx] = float(np.sum(idx_stop - idx_start))
    global_mean_rate = np.divide(global_total_spikes, global_total_dur, out=np.zeros_like(global_total_spikes), where=(global_total_dur > 0))

    for grp in groups:
        # Orientations present in this group
        pres_by_angle = pres_by_group_and_angle.get(grp, {})
        orientations = np.sort(np.array(list(pres_by_angle.keys()), dtype=float))
        if orientations.size == 0:
            continue

        tuning_by_unit: Dict[int, pd.DataFrame] = {}
        for u_idx, st in enumerate(spikes_by_unit):
            if np.asarray(st).size == 0:
                curve = pd.DataFrame({"angle_deg": orientations, "value": np.zeros_like(orientations)})
                tuning_by_unit[u_idx] = curve
                records.append({
                    **{c: grp[i] for i, c in enumerate(group_by)},
                    "unit_index": u_idx, "osi_classic": np.nan, "osi_vector": np.nan,
                    "pref_angle": np.nan, "pref_value": 0.0,
                })
                continue

            st = np.asarray(st, dtype=float)
            st.sort()
            values = []  # rate or count per angle
            for ang in orientations:
                idxs = pres_by_angle.get(float(ang), [])
                if len(idxs) == 0:
                    values.append(0.0)
                    continue
                counts = []
                dur_sum = 0.0
                for i in idxs:
                    start = float(starts[int(i)])
                    stop  = float(stops[int(i)])
                    dur_i = float(durations[int(i)])
                    c = np.sum((st >= start) & (st < stop))
                    counts.append(c)
                    dur_sum += dur_i
                total_count = float(np.sum(counts))
                values.append(total_count / dur_sum if rate and dur_sum > 0 else total_count)

            values = np.array(values, dtype=float)
            curve = pd.DataFrame({"angle_deg": orientations, "value": values})
            tuning_by_unit[u_idx] = curve

            # Preferred angle and OSI/DSI
            pref_idx = int(np.nanargmax(values)) if values.size else 0
            pref_ang = float(orientations[pref_idx]) if values.size else np.nan
            pref_val = float(values[pref_idx]) if values.size else 0.0

            # Classic contrast metric: 90-deg away for orientation; 180-deg away for direction
            if orientation_mode == "direction":
                opp = (pref_ang + 180.0) % 360.0
            else:
                opp = (pref_ang + 90.0) % 180.0
            opp_idx = np.where(np.isclose(orientations, opp))[0]
            if opp_idx.size == 0:
                osi_classic = np.nan
            else:
                opp_val = float(values[opp_idx[0]])
                denom = (pref_val + opp_val)
                osi_classic = (pref_val - opp_val) / denom if denom > 0 else np.nan

            # Vector-based (angle_factor = 2 for orientation, 1 for direction)
            if values.sum() > 0:
                thetas = np.deg2rad(orientations)
                vec = np.sum(values * np.exp(1j * angle_factor * thetas))
                osi_vector = np.abs(vec) / np.sum(values)
            else:
                osi_vector = np.nan

            # Group-level participation metrics
            n_orients = int(orientations.size)
            n_orients_nz = int(np.count_nonzero(values > 0))
            participation = (n_orients_nz / n_orients) if n_orients > 0 else 0.0
            n_presentations_group = int(sum(len(pres_by_angle.get(float(ang), [])) for ang in orientations))

            records.append({
                **{c: grp[i] for i, c in enumerate(group_by)},
                "unit_index": u_idx,
                "osi_classic": osi_classic,
                "osi_vector": float(osi_vector) if np.isfinite(osi_vector) else np.nan,
                "pref_angle": pref_ang,
                "pref_value": pref_val,
                # QC fields (duplicated on each group row for simple filtering)
                "global_total_spikes": float(global_total_spikes[u_idx]),
                "global_total_duration": float(global_total_dur),
                "global_mean_rate_hz": float(global_mean_rate[u_idx]),
                "n_presentations_group": n_presentations_group,
                "n_orientations_group": n_orients,
                "n_orientations_nz": n_orients_nz,
                "participation_ratio": participation,
            })

        tuning_by_group_and_unit[grp] = tuning_by_unit

    osi_df = pd.DataFrame.from_records(records)
    return osi_df, tuning_by_group_and_unit

def filter_osi_df_by_qc(osi_df: pd.DataFrame, qc: Dict[str, float]) -> pd.DataFrame:
    """Filter OSI rows by strict QC thresholds. Returns filtered DataFrame."""
    if osi_df.empty:
        return osi_df
    df = osi_df.copy()
    # Ensure numeric
    for c in [
        "global_total_spikes", "global_total_duration", "global_mean_rate_hz",
        "n_presentations_group", "pref_value", "participation_ratio",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if "global_total_spikes" in df.columns:
        mask &= df["global_total_spikes"] >= float(qc["MIN_TOTAL_SPIKES"])
    if "global_mean_rate_hz" in df.columns:
        mask &= df["global_mean_rate_hz"] >= float(qc["MIN_MEAN_RATE_HZ"])
    if "n_presentations_group" in df.columns:
        mask &= df["n_presentations_group"] >= int(qc["MIN_PRESENTATIONS"])
    if "pref_value" in df.columns:
        mask &= df["pref_value"] >= float(qc["MIN_PREF_VALUE_HZ"])
    if "participation_ratio" in df.columns:
        mask &= df["participation_ratio"] >= float(qc["MIN_PARTICIPATION"])
    filtered = pd.DataFrame(df.loc[mask].reset_index(drop=True))
    try:
        n_units_total = len(df["unit_index"].unique())
        n_units_kept = len(filtered["unit_index"].unique())
        print(
            f"OSI QC: kept {n_units_kept}/{n_units_total} units "
            f"({100.0 * n_units_kept / max(1, n_units_total):.1f}%) and {len(filtered)}/{len(df)} rows."
        )
    except Exception:
        print(f"OSI QC applied. Rows: {len(filtered)}/{len(df)}")
    return filtered

# --- Log-Gaussian fits for SF/TF marginalized tuning ---

def _log_gaussian(xlog: np.ndarray, A: float, mu: float, sigma: float, baseline: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((xlog - mu) / max(sigma, 1e-6)) ** 2) + baseline

def _log_skew_gaussian_piecewise(
    xlog: np.ndarray, A: float, mu: float, sigma_left: float, sigma_right: float, baseline: float
) -> np.ndarray:
    xlog = np.asarray(xlog, float)
    sig_l = max(float(sigma_left), 1e-6)
    sig_r = max(float(sigma_right), 1e-6)
    left = xlog < mu
    z = np.empty_like(xlog, dtype=float)
    z[left] = (xlog[left] - mu) / sig_l
    z[~left] = (xlog[~left] - mu) / sig_r
    return float(A) * np.exp(-0.5 * z * z) + float(baseline)

def _log_skew_gaussian_single(
    xlog: np.ndarray, A: float, mu: float, sigma: float, alpha: float, baseline: float
) -> np.ndarray:
    """Single-parameter skew in log domain using an Azzalini-like form.
    y = baseline + A * exp(-0.5*z^2) * (1 + erf(alpha * z / sqrt(2)))
    Constants are absorbed into A; alpha controls skew sign and magnitude.
    """
    xlog = np.asarray(xlog, float)
    sig = max(float(sigma), 1e-6)
    z = (xlog - float(mu)) / sig
    # standard CDF via erf: Phi(k*z) ~= 0.5*(1+erf(k*z/sqrt(2))) but we use (1+erf(..)) without 0.5 and fold into A
    skew_term = 1.0 + np.vectorize(math.erf)(float(alpha) * z / math.sqrt(2.0))
    return float(A) * np.exp(-0.5 * z * z) * skew_term + float(baseline)

def fit_log_gaussian(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit y(x) with a log-Gaussian in log10(x) domain. Returns dict with params and fit stats.
    If fitting unavailable or insufficient data, returns NaNs with success=0.
    """
    out = {
        "A": np.nan, "mu": np.nan, "sigma": np.nan, "baseline": np.nan,
        "r2": np.nan, "x_pref": np.nan, "success": 0,
        "n_points": int(np.size(y)),
    }
    try:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x = x[mask]
        y = y[mask]
        if x.size < 3 or y.size < 3 or curve_fit is None:
            return out
        xlog = np.log10(x)
        # Initial guesses
        y_min = float(np.nanmin(y)) if np.isfinite(y).any() else 0.0
        y_max = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
        A0 = max(y_max - y_min, 1e-3)
        baseline0 = max(y_min, 0.0)
        # Prefer mode at max y
        try:
            mu0 = float(np.log10(float(x[np.nanargmax(y)])))
        except Exception:
            mu0 = float(np.nanmedian(xlog)) if np.isfinite(xlog).any() else 0.0
        sigma0 = 0.5
        bounds = (
            [0.0, float(np.min(xlog)) - 1.0, 0.05, 0.0],
            [float(10*y_max + 1.0), float(np.max(xlog)) + 1.0, 2.0, float(10*y_max + 1.0)],
        )
        popt, _ = curve_fit(_log_gaussian, xlog, y, p0=[A0, mu0, sigma0, baseline0], bounds=bounds, maxfev=20000)
        A, mu, sigma, baseline = [float(v) for v in popt]
        y_pred = _log_gaussian(xlog, A, mu, sigma, baseline)
        ss_res = float(np.nansum((y - y_pred) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2)) if np.isfinite(y).any() else np.nan
        r2 = 1.0 - ss_res / ss_tot if ss_tot and np.isfinite(ss_tot) and ss_tot > 0 else np.nan
        x_pref = float(10 ** mu)
        out.update({"A": A, "mu": mu, "sigma": sigma, "baseline": baseline, "r2": r2, "x_pref": x_pref, "success": 1})
        return out
    except Exception:
        return out

def fit_log_skew1_gaussian(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out = {
        "A": np.nan, "mu": np.nan, "sigma": np.nan, "alpha": np.nan, "baseline": np.nan,
        "r2": np.nan, "x_pref": np.nan, "success": 0, "n_points": int(np.size(y)),
    }
    try:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x = x[mask]
        y = y[mask]
        if x.size < 4 or y.size < 4 or curve_fit is None:
            return out
        xlog = np.log10(x)
        y_min = float(np.nanmin(y)) if np.isfinite(y).any() else 0.0
        y_max = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
        A0 = max(y_max - y_min, 1e-3)
        baseline0 = max(y_min, 0.0)
        try:
            mu0 = float(np.log10(float(x[np.nanargmax(y)])))
        except Exception:
            mu0 = float(np.nanmedian(xlog)) if np.isfinite(xlog).any() else 0.0
        sigma0 = 0.5
        alpha0 = 0.0
        bounds = (
            [0.0, float(np.min(xlog)) - 1.0, 0.05, -10.0, 0.0],
            [float(10*y_max + 1.0), float(np.max(xlog)) + 1.0, 2.0, 10.0, float(10*y_max + 1.0)],
        )
        popt, _ = curve_fit(
            _log_skew_gaussian_single, xlog, y,
            p0=[A0, mu0, sigma0, alpha0, baseline0], bounds=bounds, maxfev=30000
        )
        A, mu, sigma, alpha, baseline = [float(v) for v in popt]
        yhat = _log_skew_gaussian_single(xlog, A, mu, sigma, alpha, baseline)
        ss_res = float(np.nansum((y - yhat) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2)) if np.isfinite(y).any() else np.nan
        r2 = 1.0 - ss_res / ss_tot if ss_tot and np.isfinite(ss_tot) and ss_tot > 0 else np.nan
        x_pref = float(10 ** mu)
        out.update({
            "A": A, "mu": mu, "sigma": sigma, "alpha": alpha, "baseline": baseline,
            "r2": r2, "x_pref": x_pref, "success": 1,
        })
        return out
    except Exception:
        # fallback to symmetric
        sym = fit_log_gaussian(x, y)
        if sym.get("success"):
            out.update({
                "A": sym.get("A", np.nan), "mu": sym.get("mu", np.nan),
                "sigma": sym.get("sigma", np.nan), "alpha": 0.0, "baseline": sym.get("baseline", np.nan),
                "r2": sym.get("r2", np.nan), "x_pref": sym.get("x_pref", np.nan), "success": 1,
            })
        return out
def fit_log_skew_gaussian(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out = {
        "A": np.nan, "mu": np.nan, "sigma_left": np.nan, "sigma_right": np.nan, "baseline": np.nan,
        "r2": np.nan, "x_pref": np.nan, "success": 0, "n_points": int(np.size(y)),
    }
    try:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x = x[mask]
        y = y[mask]
        if x.size < 4 or y.size < 4 or curve_fit is None:
            return out
        xlog = np.log10(x)
        y_min = float(np.nanmin(y)) if np.isfinite(y).any() else 0.0
        y_max = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
        A0 = max(y_max - y_min, 1e-3)
        baseline0 = max(y_min, 0.0)
        try:
            mu0 = float(np.log10(float(x[np.nanargmax(y)])))
        except Exception:
            mu0 = float(np.nanmedian(xlog)) if np.isfinite(xlog).any() else 0.0
        sigma_l0 = 0.5
        sigma_r0 = 0.5
        bounds = (
            [0.0, float(np.min(xlog)) - 1.0, 0.05, 0.05, 0.0],
            [float(10*y_max + 1.0), float(np.max(xlog)) + 1.0, 2.0, 2.0, float(10*y_max + 1.0)],
        )
        popt, _ = curve_fit(
            _log_skew_gaussian_piecewise, xlog, y,
            p0=[A0, mu0, sigma_l0, sigma_r0, baseline0], bounds=bounds, maxfev=30000
        )
        A, mu, sigma_l, sigma_r, baseline = [float(v) for v in popt]
        yhat = _log_skew_gaussian_piecewise(xlog, A, mu, sigma_l, sigma_r, baseline)
        ss_res = float(np.nansum((y - yhat) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2)) if np.isfinite(y).any() else np.nan
        r2 = 1.0 - ss_res / ss_tot if ss_tot and np.isfinite(ss_tot) and ss_tot > 0 else np.nan
        x_pref = float(10 ** mu)
        out.update({
            "A": A, "mu": mu, "sigma_left": sigma_l, "sigma_right": sigma_r, "baseline": baseline,
            "r2": r2, "x_pref": x_pref, "success": 1,
        })
        return out
    except Exception:
        # fallback to symmetric
        sym = fit_log_gaussian(x, y)
        if sym.get("success"):
            out.update({
                "A": sym.get("A", np.nan), "mu": sym.get("mu", np.nan),
                "sigma_left": sym.get("sigma", np.nan), "sigma_right": sym.get("sigma", np.nan),
                "baseline": sym.get("baseline", np.nan), "r2": sym.get("r2", np.nan),
                "x_pref": sym.get("x_pref", np.nan), "success": 1,
            })
        return out

def compute_sf_tf_loggauss_fits(
    tuning_by_group: Dict[Tuple, Dict[int, pd.DataFrame]],
    osi_df_qc: pd.DataFrame,
    group_cols: List[str],
    units_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute marginalized SF and TF tuning per unit (mean across orientation, then average across the other variable),
    and fit log-Gaussians. Returns per-unit DataFrame with SF/TF fit params and quality metrics.
    Uses only (unit, group) rows that passed QC (osi_df_qc).
    """
    if not group_cols:
        return pd.DataFrame()
    has_sf = "spatial_frequency" in group_cols
    has_tf = "temporal_frequency" in group_cols
    if not has_sf and not has_tf:
        return pd.DataFrame()

    # Allowed groups (those that passed QC rows); store as tuples in group_cols order
    allowed_groups = set()
    if not osi_df_qc.empty:
        for _, row in osi_df_qc.dropna(subset=group_cols, how="any").iterrows():
            allowed_groups.add(tuple(row[c] for c in group_cols))

    n_units = len(units_df)
    probes = pd.Series(infer_unit_probes(None, units_df))
    rows = []

    # Pre-index group keys by SF/TF values
    # Extract available group keys from tuning_by_group
    all_groups = list(tuning_by_group.keys())
    # Clamp ranges for fitted peak preferences (requested bounds)
    SF_MIN, SF_MAX = 0.01, 0.64
    TF_MIN, TF_MAX = 0.5, 32.0

    for u in range(n_units):
        unit_id = units_df.index[u]
        probe = str(probes.iloc[u])
        # Accumulate mean-over-orientation values per group
        group_means: Dict[Tuple, float] = {}
        for grp in all_groups:
            if allowed_groups and (grp not in allowed_groups):
                continue
            curve = tuning_by_group.get(grp, {}).get(u)
            if curve is None or curve.empty or "value" not in curve.columns:
                continue
            val = float(np.nanmean(pd.to_numeric(curve["value"], errors="coerce")))
            group_means[grp] = val

        # Build SF- and TF-marginal arrays
        sf_vals = []
        sf_resp = []
        tf_vals = []
        tf_resp = []
        if has_sf:
            # group index for SF is group_cols.index("spatial_frequency")
            si = group_cols.index("spatial_frequency")
            # For each unique SF, average across TF
            sf_to_vals: Dict[float, List[float]] = {}
            for grp, v in group_means.items():
                try:
                    sfv = float(grp[si])
                except Exception:
                    continue
                sf_to_vals.setdefault(sfv, []).append(v)
            for sfv, lst in sorted(sf_to_vals.items(), key=lambda kv: kv[0]):
                if len(lst) > 0:
                    sf_vals.append(sfv)
                    sf_resp.append(float(np.nanmean(lst)))
        if has_tf:
            ti = group_cols.index("temporal_frequency")
            tf_to_vals: Dict[float, List[float]] = {}
            for grp, v in group_means.items():
                try:
                    tfv = float(grp[ti])
                except Exception:
                    continue
                tf_to_vals.setdefault(tfv, []).append(v)
            for tfv, lst in sorted(tf_to_vals.items(), key=lambda kv: kv[0]):
                if len(lst) > 0:
                    tf_vals.append(tfv)
                    tf_resp.append(float(np.nanmean(lst)))

        # Fit single-parameter log-skew Gaussians (fallbacks to symmetric)
        sf_fit = fit_log_skew1_gaussian(np.array(sf_vals), np.array(sf_resp)) if has_sf else {"success": 0}
        tf_fit = fit_log_skew1_gaussian(np.array(tf_vals), np.array(tf_resp)) if has_tf else {"success": 0}

        # Clamp preferred values to requested bounds
        sf_pref = sf_fit.get("x_pref", np.nan)
        if np.isfinite(sf_pref):
            sf_pref = float(np.clip(sf_pref, SF_MIN, SF_MAX))
        else:
            sf_pref = np.nan
        tf_pref = tf_fit.get("x_pref", np.nan)
        if np.isfinite(tf_pref):
            tf_pref = float(np.clip(tf_pref, TF_MIN, TF_MAX))
        else:
            tf_pref = np.nan

        rows.append({
            "unit_index": u,
            "unit_id": unit_id,
            "probe": probe,
            # SF fits
            "sf_points": int(len(sf_vals)),
            "sf_pref": sf_pref,
            # Skew param alpha and sigma (single-parameter skew)
            "sf_alpha": sf_fit.get("alpha", np.nan),
            "sf_sigma": sf_fit.get("sigma", np.nan),
            "sf_amplitude": sf_fit.get("A", np.nan),
            "sf_baseline": sf_fit.get("baseline", np.nan),
            "sf_r2": sf_fit.get("r2", np.nan),
            # TF fits
            "tf_points": int(len(tf_vals)),
            "tf_pref": tf_pref,
            "tf_alpha": tf_fit.get("alpha", np.nan),
            "tf_sigma": tf_fit.get("sigma", np.nan),
            "tf_amplitude": tf_fit.get("A", np.nan),
            "tf_baseline": tf_fit.get("baseline", np.nan),
            "tf_r2": tf_fit.get("r2", np.nan),
        })

    return pd.DataFrame(rows)

def save_sf_tf_fits(fits_df: pd.DataFrame, session_dir: str, dataset_label: str) -> Tuple[Optional[str], List[str]]:
    if fits_df is None or fits_df.empty:
        return None, []
    os.makedirs(session_dir, exist_ok=True)
    session_csv = os.path.join(session_dir, f"{dataset_label}_sf_tf_fits.csv")
    fits_df.to_csv(session_csv, index=False)
    paths: List[str] = []
    if "probe" in fits_df.columns:
        for probe, dfp in fits_df.groupby("probe"):
            probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
            out_dir = os.path.join(session_dir, probe_safe)
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"{dataset_label}_sf_tf_fits.csv")
            dfp.to_csv(out_csv, index=False)
            paths.append(out_csv)
    return session_csv, paths

# --- Unified per-unit summary (RF peak + best tuning + SF/TF fits) ---

def build_unit_summary_df(
    peaks_df: pd.DataFrame,
    best_tuning_df: pd.DataFrame,
    units_df: pd.DataFrame,
    good_units_idx: Optional[pd.Index] = None,
    fits_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge RF peaks with QC-filtered best tuning per unit. Attach unit-level QC/meta if present."""
    if peaks_df is None or peaks_df.empty or best_tuning_df is None or best_tuning_df.empty:
        return pd.DataFrame()
    base = peaks_df.copy()
    tune = best_tuning_df.copy()
    merged = base.merge(tune, on="unit_index", how="inner", suffixes=("", "_best"))
    # Optional: attach SF/TF fit parameters if provided
    if fits_df is not None and not fits_df.empty:
        keep_fit_cols = [
            c for c in [
                "unit_index", "sf_points", "sf_pref", "sf_sigma", "sf_amplitude", "sf_baseline", "sf_r2",
                "sf_alpha",
                "tf_points", "tf_pref", "tf_sigma", "tf_amplitude", "tf_baseline", "tf_r2",
                "tf_alpha",
            ] if c in fits_df.columns
        ]
        if keep_fit_cols:
            merged = merged.merge(fits_df.loc[:, keep_fit_cols], on="unit_index", how="left")
    # is_qc from provided good_units_idx (preferred) or default_qc column
    if good_units_idx is not None:
        qc_set = set(good_units_idx)
        merged["is_qc"] = merged.get("unit_id", merged["unit_index"]).apply(lambda uid: bool(uid in qc_set))
    elif "default_qc" in units_df.columns:
        qc_map = units_df["default_qc"].astype(bool).reset_index(drop=True)
        merged["is_qc"] = merged["unit_index"].apply(lambda i: bool(qc_map.iloc[int(i)]) if pd.notna(i) else False)
    # Attach common unit-level meta if present
    add_cols = [c for c in ["snr", "d_prime", "rp_contamination", "default_qc"] if c in units_df.columns]
    if add_cols:
        meta = units_df[add_cols].copy().reset_index().rename(columns={"index": "unit_id"})
        merged = merged.merge(meta, on="unit_id", how="left")
    # Order columns
    preferred = [
        "unit_index", "unit_id", "probe", "is_qc",
        "peak_x", "peak_y", "peak_value",
        "osi_classic", "osi_vector", "pref_angle", "pref_value",
        "pref_spatial_frequency", "pref_temporal_frequency",
    ]
    keep = [c for c in preferred if c in merged.columns]
    rest = [c for c in merged.columns if c not in keep]
    return pd.DataFrame(merged.loc[:, keep + rest])

def save_unit_summaries(unit_summary_df: pd.DataFrame, session_dir: str, dataset_label: str) -> Tuple[Optional[str], List[str]]:
    if unit_summary_df is None or unit_summary_df.empty:
        return None, []
    os.makedirs(session_dir, exist_ok=True)
    session_csv = os.path.join(session_dir, f"{dataset_label}_units_summary.csv")
    unit_summary_df.to_csv(session_csv, index=False)
    probe_paths: List[str] = []
    if "probe" in unit_summary_df.columns:
        for probe, dfp in unit_summary_df.groupby("probe"):
            probe_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(probe))
            pdir = os.path.join(session_dir, probe_safe)
            os.makedirs(pdir, exist_ok=True)
            out_csv = os.path.join(pdir, f"{dataset_label}_units_summary.csv")
            dfp.to_csv(out_csv, index=False)
            probe_paths.append(out_csv)
    return session_csv, probe_paths

# %%
# Run grating tuning analysis and preview grouped tuning curves (by TF/SF)
try:
    gratings = find_grating_table(nwb_file_path)
    print(f"Grating table found with {len(gratings)} rows. Columns: {list(gratings.columns)[:12]}...")
    # Strictly sanitize numeric grating columns and print unique values
    gratings = sanitize_grating_columns(gratings, strict=True)

    # Decide grouping columns present
    group_cols = [c for c in ["temporal_frequency", "spatial_frequency"] if c in gratings.columns]
    # Print unique grouped columns as an extra check
    for gc in group_cols:
        uniq = np.sort(gratings[gc].dropna().unique())
        print(f"Unique {gc} values ({len(uniq)}): {uniq[:20]}{' ...' if len(uniq) > 20 else ''}")
    osi_df, tuning_by_group = compute_tuning_and_osi(
        gratings, spikes_by_unit,
        orientation_col="orientation",
        rate=True,
        group_by=group_cols,
        orientation_mode="orientation",  # set to "direction" for 0–360° analysis
    )
    print(osi_df.head())

    # Pick a non-empty group and plot top unit by pref_value
    if not osi_df.empty:
        sort_cols = ["pref_value"]
        if group_cols:
            sort_cols = group_cols + sort_cols
        first_row = osi_df.sort_values(sort_cols, ascending=[True]*len(group_cols) + [False]).iloc[0]
        grp = tuple(first_row[c] for c in group_cols) if group_cols else tuple()
        # Robustly coerce to int in case of dtype/object oddities
        ui_raw = first_row.get("unit_index", None)
        unit_index = 0
        if ui_raw is not None:
            try:
                unit_index = int(ui_raw)  # scalar path
            except Exception:
                try:
                    unit_index = int(float(np.asarray(ui_raw).ravel()[0]))
                except Exception:
                    unit_index = 0
        curve = tuning_by_group.get(grp, {}).get(unit_index)
        if curve is not None and not curve.empty:
            fig, ax = plt.subplots()
            ax.plot(curve["angle_deg"], curve["value"], marker="o")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Firing rate (Hz)")
            subtitle = ", ".join(f"{c}={first_row[c]}" for c in group_cols) if group_cols else ""
            ax.set_title(f"Unit {unit_index}: tuning curve {subtitle}")
            plt.tight_layout()
            _maybe_show(fig)

    # Save full (unfiltered) OSI summary
    out_csv_full = os.path.join(SESSION_DIR, f"{dataset_label}_gratings_osi_summary_full.csv")
    osi_df.to_csv(out_csv_full, index=False)
    print(f"Saved OSI summary (full): {out_csv_full}")

    # Apply QC filters and save QC OSI summary
    osi_df_qc = filter_osi_df_by_qc(osi_df, OSI_QC)
    out_csv_qc = os.path.join(SESSION_DIR, f"{dataset_label}_gratings_osi_summary_qc.csv")
    osi_df_qc.to_csv(out_csv_qc, index=False)
    print(f"Saved OSI summary (QC): {out_csv_qc}")

    # Attach probe labels and save histograms (QC only)
    osi_with_probe = attach_probe_to_osi_df(osi_df_qc, units_df, nwb_file_path)
    save_osi_histograms_png(osi_with_probe, SESSION_DIR, prefix=f"{dataset_label}_gratings")

    # Preferred group per unit (QC)
    pref_group_df = compute_pref_group_per_unit(osi_df_qc, group_cols)

    # Save per-unit best tuning parameters per probe (QC-only)
    try:
        best_tuning_df = compute_best_tuning_per_unit(osi_df_qc, group_cols)
        save_per_probe_tuning_csv(best_tuning_df, units_df, SESSION_DIR, dataset_label)
    except Exception as _e:
        pass

    # Compute and save SF/TF marginalized log-Gaussian fits (QC-only groups)
    try:
        fits_df = compute_sf_tf_loggauss_fits(tuning_by_group, osi_df_qc, group_cols, units_df)
        save_sf_tf_fits(fits_df, SESSION_DIR, dataset_label)
    except Exception as _e:
        fits_df = pd.DataFrame()

    # Now save per-probe histograms including continuous SF/TF prefs if available
    try:
        save_per_probe_histograms(osi_with_probe, pref_group_df, SESSION_DIR, filename_prefix=dataset_label, fit_pref_df=fits_df)
    except Exception as _e:
        pass

    # Build and save unified per-unit summary CSVs (RF peaks + best tuning + SF/TF fits)
    try:
        if 'peaks_df' in locals():
            unit_summary_df = build_unit_summary_df(
                peaks_df=peaks_df,
                best_tuning_df=best_tuning_df if 'best_tuning_df' in locals() else pd.DataFrame(),
                units_df=units_df,
                good_units_idx=good_idx if 'good_idx' in locals() else None,
                fits_df=fits_df if 'fits_df' in locals() else None,
            )
            _ = save_unit_summaries(unit_summary_df, SESSION_DIR, dataset_label)
    except Exception as _e:
        pass

    # Final session summary now that OSI exists
    try:
        _ = save_session_summary_figure(
            avg_maps=avg_maps if 'avg_maps' in locals() else {},
            rf_counts=rf_counts if 'rf_counts' in locals() else np.array([]),
            ys=pos_y if 'pos_y' in locals() else np.array([]),
            xs=pos_x if 'pos_x' in locals() else np.array([]),
            units_df=units_df,
            osi_df_with_probe=osi_with_probe,
            out_dir=SESSION_DIR,
            dataset_label=dataset_label,
            good_units_idx=good_idx if 'good_idx' in locals() else None,
            n_small=6,
        )
    except Exception as _e:
        pass
except KeyError as e:
    print(f"No usable grating table found: {e}")

# %% [markdown]
# Tips
# - Adjust `unit_index` in the RF and tuning cells to inspect specific units.
# - If your dataset uses different interval names/columns, tweak `get_rf_table()`
#   and `find_grating_table()` to match.
# - Orientation vs. direction: The implementation here computes orientation tuning and OSI
#   on [0,180). For direction tuning (0–360°), adapt the normalization and use single-angle
#   vector sums.


# %% Check number of orientation values
if 'gratings' in locals():
    n_orientations = gratings['orientation'].nunique()
    print(f"Number of unique orientations in grating table: {n_orientations}")


# Check number of orientation values in RF stimuli
if 'stim_rf' in locals():
    if 'orientation' in stim_rf.columns:
        n_orientations_rf = stim_rf['orientation'].nunique()
        print(f"Number of unique orientations in RF stimulus table: {n_orientations_rf}")
    else:
        print("No 'orientation' column found in RF stimulus table.")

# Check number of SF/TF values in grating table
if 'gratings' in locals():
    for col in ['spatial_frequency', 'temporal_frequency']:
        if col in gratings.columns:
            n_values = gratings[col].nunique()
            print(f"Number of unique {col} values in grating table: {n_values}")
        else:
            print(f"No '{col}' column found in grating table.")
       