"""
scripts/compute_probe_ccg.py

Compute jitter-corrected CCG feedforwardness scores for all cross-probe
unit pairs within each new V1 session.

For each session and each probe pair (B-C, B-A, B-E, C-A, C-E, A-E):
  · Extract binarised spike trains during drifting-gratings trials (1 ms bins)
  · Run jitter-corrected CCG (jitter window = 25 ms, peak search ±10 ms)
  · Collect peak-offset distribution
  · Compute feedforwardness score:
      FF = (n_positive_offsets − n_negative_offsets) / n_total
    (positive offset → first-named probe fires before second-named probe)

Output: data/processed_data/probe_ccg_results.npz
  peak_offsets : dict  probe_pair_str → list of offsets (all sessions pooled)
  ff_per_session : dict  probe_pair_str → list of per-session FF scores
  session_ids : list of site numbers

Usage:
  PYTHONNOUSERSITE=1 python3 scripts/compute_probe_ccg.py [--test]
  --test : run on one session only for speed check
"""

import h5py
import numpy as np
import pandas as pd
import os
import sys
import glob
import re
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from jitter_ccg import get_ccgjitter_crosspair

# ── Configuration ─────────────────────────────────────────────────────────────
NWB_BASE   = '/media/huklaban5/Data/MouseV2/001568'
CODE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(CODE_DIR, 'data')
OUT_PATH   = os.path.join(DATA_DIR, 'processed_data', 'probe_ccg_results.npz')

SITE_MAP = {          # site_num → (sub_id, id_offset)
    2: ('816305', 2_000_000),
    3: ('810531', 3_000_000),
    4: ('810532', 4_000_000),
    5: ('813810', 5_000_000),
    6: ('815152', 6_000_000),
    7: ('816308', 7_000_000),
    8: ('817334', 8_000_000),
    9: ('817335', 9_000_000),
}

PROBE_ORDER   = ['B', 'C', 'A', 'E']
PROBE_PAIRS   = [(PROBE_ORDER[i], PROBE_ORDER[j])
                 for i in range(len(PROBE_ORDER))
                 for j in range(i + 1, len(PROBE_ORDER))]

JITTER_WIN  = 25    # ms
FR_THRESH   = 2.0   # Hz — skip units below this
PEAK_WIN_MS = 10    # search CCG peak within ±10 ms
BIN_MS      = 1     # 1 ms bins

# ── Low-level NWB readers ─────────────────────────────────────────────────────
def _open(nwb_path):
    return h5py.h5f.open(nwb_path.encode())

def _read_num(fid, path):
    dset = h5py.h5d.open(fid, path.encode())
    n    = dset.get_space().get_simple_extent_npoints()
    arr  = np.empty(n, dtype='<f8')
    mem  = h5py.h5t.py_create(np.dtype('<f8'), logical=True)
    dset.read(h5py.h5s.ALL, h5py.h5s.ALL, arr, mem)
    return arr

def _read_str(f, path):
    raw = f[path][:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw])

# ── Load session data ─────────────────────────────────────────────────────────
def load_session(site_num, nwb_path, apply_qc=True, verbose=True):
    """
    Returns
    -------
    spikes    : ndarray (n_units, n_conditions, n_trials, n_timebins)
    FR        : ndarray (n_units,)  — spikes/s over full session
    probe_ids : ndarray (n_units,)  — probe letter 'A'/'B'/'C'/'E'
    unit_ids  : ndarray (n_units,)  — local unit indices (0-based)
    """
    offset = site_num * 1_000_000
    site_dir = os.path.join(DATA_DIR, f'site{site_num}_processed')

    # load probe labels from layer_info
    lay = pd.read_csv(os.path.join(site_dir, 'layer_info.csv'))
    def _probe(acronym):
        m = re.match(r'V1_site\d+_([ABCE])', str(acronym))
        return m.group(1) if m else None
    lay['probe'] = lay['ecephys_structure_acronym'].map(_probe)
    lay = lay[lay['probe'].notna()].copy()
    lay['local_idx'] = (lay['unit_id'] - offset).astype(int)

    # apply QC filter
    if apply_qc:
        qc_path = os.path.join(site_dir, 'unit_quality.csv')
        if os.path.isfile(qc_path):
            qc = pd.read_csv(qc_path)[['unit_id', 'default_qc']]
            lay = lay.merge(qc, on='unit_id', how='left')
            lay = lay[lay['default_qc'] == True].drop(columns=['default_qc'])

    local_idxs  = lay['local_idx'].values.astype(int)
    probe_labels = lay['probe'].values

    if verbose:
        print(f'  site{site_num}: {len(local_idxs)} units after QC filter', flush=True)
        for p in PROBE_ORDER:
            print(f'    probe {p}: {(probe_labels == p).sum()} units', flush=True)

    # ── Open NWB ─────────────────────────────────────────────────────────────
    fid = _open(nwb_path)
    print('  reading NWB stimulus tables...', flush=True)

    with h5py.File(nwb_path, 'r') as f:
        base = '/intervals/drifting_gratings_field_block_presentations/'
        oris = _read_str(f, base + 'orientation')
        tfs  = _read_str(f, base + 'temporal_frequency')

    starts = _read_num(fid, '/intervals/drifting_gratings_field_block_presentations/start_time')
    stops  = _read_num(fid, '/intervals/drifting_gratings_field_block_presentations/stop_time')

    dur_ms = int(round(np.median(stops - starts) * 1000 / BIN_MS)) * BIN_MS
    n_t    = dur_ms // BIN_MS
    bin_s  = BIN_MS / 1000.0

    cond_keys  = sorted(set(zip(oris, tfs)))
    cond_map   = {k: i for i, k in enumerate(cond_keys)}
    n_cond     = len(cond_keys)
    trial_cond = np.array([cond_map[(o, t)] for o, t in zip(oris, tfs)])
    n_trials_per_cond = np.bincount(trial_cond, minlength=n_cond)
    n_trials_min      = int(n_trials_per_cond.min())

    if verbose:
        print(f'  {n_cond} conditions × {n_trials_min} trials × {n_t} bins', flush=True)

    # repeat index per trial
    rep_idx = np.zeros(len(starts), dtype=np.int32)
    _ctr    = np.zeros(n_cond, dtype=np.int32)
    for t in range(len(starts)):
        ci = trial_cond[t]
        rep_idx[t] = _ctr[ci]; _ctr[ci] += 1
    valid_trial = rep_idx < n_trials_min

    # ── Spike times ───────────────────────────────────────────────────────────
    print('  reading spike times...', flush=True)
    st_flat = _read_num(fid, '/units/spike_times')
    st_idx  = _read_num(fid, '/units/spike_times_index').astype(np.int64)
    n_units_total = len(st_idx)

    # compute overall FR (fast: just count spikes per unit)
    spike_counts = np.diff(np.concatenate([[0], st_idx]))
    session_dur  = float(stops[-1] - starts[0])
    FR_all       = spike_counts / session_dur   # Hz, all units
    FR_sel       = FR_all[local_idxs]

    # ── Filter to units with FR > threshold before allocating spike array ─────
    active_mask  = FR_sel >= FR_THRESH
    active_local = local_idxs[active_mask]
    active_probe = probe_labels[active_mask]
    active_FR    = FR_sel[active_mask]
    n_active     = active_mask.sum()

    if verbose:
        print(f'  {n_active}/{len(local_idxs)} units pass FR>{FR_THRESH}Hz', flush=True)
        mem_mb = n_active * n_cond * n_trials_min * n_t / 1e6
        print(f'  spike array: {n_active}×{n_cond}×{n_trials_min}×{n_t} = {mem_mb:.0f} MB (uint8)', flush=True)

    # ── Binarise spike trains (vectorised per unit via searchsorted) ──────────
    print('  binarising...', flush=True)
    spikes = np.zeros((n_active, n_cond, n_trials_min, n_t), dtype=np.uint8)

    for k, ui in enumerate(active_local):
        sp0 = 0 if ui == 0 else int(st_idx[ui - 1])
        sp1 = int(st_idx[ui])
        if sp1 <= sp0:
            continue
        unit_spikes = st_flat[sp0:sp1]
        tr  = np.searchsorted(starts, unit_spikes, side='right') - 1
        ok  = (tr >= 0) & (tr < len(starts))
        ok[ok] &= unit_spikes[ok] < (starts[tr[ok]] + n_t * bin_s)
        ok[ok] &= valid_trial[tr[ok]]
        if not ok.any():
            continue
        sp   = unit_spikes[ok];  tr_v = tr[ok]
        ci_v = trial_cond[tr_v]; ri_v = rep_idx[tr_v]
        bi_v = np.floor((sp - starts[tr_v]) / bin_s).astype(np.int32)
        bi_v = np.clip(bi_v, 0, n_t - 1)
        spikes[k, ci_v, ri_v, bi_v] = 1

    del st_flat  # free 5+ GB spike-times flat array — no longer needed
    print('  binarisation done', flush=True)
    return spikes.astype(np.float32), active_FR, active_probe, active_local


# ── FF score from offset list ─────────────────────────────────────────────────
def ff_score(offsets, threshold_ms=0):
    """(n_positive - n_negative) / total; positive = first probe fires first."""
    offsets = np.array(offsets)
    n_pos = np.sum(offsets >  threshold_ms)
    n_neg = np.sum(offsets <= threshold_ms)
    total = n_pos + n_neg
    return float(n_pos - n_neg) / total if total > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────
def main(test_mode=False):
    sites = sorted(SITE_MAP.keys())
    if test_mode:
        sites = sites[:1]
        print('=== TEST MODE: running site', sites[0], 'only ===\n')

    pair_labels = [f'{a}-{b}' for a, b in PROBE_PAIRS]
    all_offsets     = {lbl: [] for lbl in pair_labels}
    ff_per_session  = {lbl: [] for lbl in pair_labels}
    session_list    = []

    for site_num in sites:
        sub_id, offset = SITE_MAP[site_num]
        sub_dir = os.path.join(NWB_BASE, f'sub-{sub_id}')
        nwbs = [f for f in os.listdir(sub_dir) if f.endswith('.nwb')]
        if not nwbs:
            print(f'site{site_num}: no NWB, skipping')
            continue
        nwb_path = os.path.join(sub_dir, nwbs[0])
        print(f'\n=== site{site_num} ({sub_id}) ===')
        t0 = time.time()

        spikes, FR, probe_labels, _ = load_session(site_num, nwb_path)

        session_list.append(site_num)

        for (pa, pb) in PROBE_PAIRS:
            lbl = f'{pa}-{pb}'
            idx_a = list(np.where(probe_labels == pa)[0])
            idx_b = list(np.where(probe_labels == pb)[0])

            print(f'  pair {lbl}: {len(idx_a)} × {len(idx_b)} units', flush=True)

            if not idx_a or not idx_b:
                ff_per_session[lbl].append(np.nan)
                continue

            offsets, n_tested = get_ccgjitter_crosspair(
                spikes, FR, idx_a, idx_b,
                jitterwindow=JITTER_WIN, fr_thresh=FR_THRESH
            )

            all_offsets[lbl].extend(offsets)
            ff = ff_score(offsets)
            ff_per_session[lbl].append(ff)
            print(f'    {len(offsets)}/{n_tested} pairs evaluated  FF={ff:+.3f}')

        print(f'  elapsed: {time.time()-t0:.0f}s')

    # save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH,
             pair_labels    = pair_labels,
             session_ids    = session_list,
             all_offsets    = all_offsets,
             ff_per_session = ff_per_session)
    print(f'\nSaved → {OUT_PATH}')

    # quick summary
    print('\n=== FF scores (mean ± std across sessions) ===')
    for lbl in pair_labels:
        vals = [v for v in ff_per_session[lbl] if not np.isnan(v)]
        if vals:
            print(f'  {lbl}: {np.mean(vals):+.3f} ± {np.std(vals):.3f}  (n={len(vals)} sessions)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()
    main(test_mode=args.test)
