"""
scripts/jitter_ccg.py

Cross-correlogram with PSTH predictor correction — sparse + transposed approach.

Instead of the jitter predictor (which requires iterating over all time bins),
this version uses the PSTH cross-correlogram as the stimulus-locked predictor:

  ccg_corr[i, j, k] = ccg_raw[i, j, k] - n_trials × psth_ccg[i, j, k]

where psth_ccg[i, j, k] = sum_cond sum_t PSTH_i[cond, t] × PSTH_j[cond, t+k]
and    PSTH[cond, t]   = mean over trials of the binarised spike train.

This removes the stimulus-evoked co-fluctuation between units (the main
confounder for inter-probe CCG) without iterating over 1.5M time bins.

The raw CCG is computed using the SPARSE positions of unit i's spikes
(~7500 per unit) and a TRANSPOSED (L_flat, n_b) group-b matrix so each
row gather (one time step, all group-b units) is contiguous in memory.

Reference:
  Jia, Tanabe & Kohn (2013) Neuron 77(4), 762-774.
"""

import numpy as np


def get_ccgjitter_crosspair(spikes, FR, group_a, group_b,
                            jitterwindow=25, fr_thresh=2.0, max_lag=10):
    """
    PSTH-corrected CCG for all cross-group unit pairs.

    Parameters
    ----------
    spikes       : ndarray (n_units, n_conditions, n_trials, n_timebins)
                   float32 binarised spike trains (1 ms bins)
    FR           : ndarray (n_units,) — firing rate in Hz
    group_a, group_b : list of int
    jitterwindow : int — (unused, kept for API compatibility)
    fr_thresh    : float Hz
    max_lag      : int — search CCG peak within ±max_lag ms

    Returns
    -------
    peak_offsets   : list of float — CCG peak lag (ms) per pair.
                     Positive = group_a fires before group_b.
    n_pairs_tested : int
    """
    group_a = [i for i in group_a if FR[i] >= fr_thresh]
    group_b = [j for j in group_b if FR[j] >= fr_thresh]
    if not group_a or not group_b:
        return [], 0

    n_cond, n_trials, n_t = spikes.shape[1], spikes.shape[2], spikes.shape[3]
    n_b = len(group_b)
    n_a = len(group_a)
    L   = n_cond * n_trials * n_t

    lags  = np.arange(-max_lag, max_lag + 1)          # (n_lag,)
    n_lag = len(lags)
    n_ct  = n_cond * n_trials
    # Triangle normalisation: n_ct × (n_t − |k|) pairs per lag k
    theta = (n_ct * (n_t - np.abs(lags))).astype(np.float64)

    # ── PSTH: (n_units, n_cond, n_t) — mean over trials ─────────────────────
    PSTH_a = np.stack([spikes[i].mean(axis=1) for i in group_a])  # (n_a, n_cond, n_t)
    PSTH_b = np.stack([spikes[j].mean(axis=1) for j in group_b])  # (n_b, n_cond, n_t)

    # ── PSTH cross-CCG: (n_a, n_b, n_lag) — computed once for ALL pairs ──────
    # psth_ccg_all[i, j, k] = n_trials × sum_c sum_t PSTH_a[i,c,t] × PSTH_b[j,c,t+k]
    print(f'    computing PSTH CCG for {n_a}×{n_b} pairs × {n_lag} lags ...', flush=True)
    psth_ccg_all = np.zeros((n_a, n_b, n_lag), dtype=np.float64)

    for ki, k in enumerate(lags):
        for c in range(n_cond):
            # (n_a, n_t−|k|) @ (n_t−|k|, n_b) = (n_a, n_b)
            if k >= 0:
                psth_ccg_all[:, :, ki] += PSTH_a[:, c, :n_t - k] @ PSTH_b[:, c, k:].T
            else:
                psth_ccg_all[:, :, ki] += PSTH_a[:, c, -k:] @ PSTH_b[:, c, :n_t + k].T

    psth_ccg_all *= n_trials   # PSTH is mean/trial; scale back

    # ── group_b spike trains in transposed layout: (L, n_b) ──────────────────
    # Contiguous rows → cache-friendly gather when indexing by sparse spike positions
    print(f'    loading {n_b} group_b units into transposed matrix...', flush=True)
    sp_b_T = np.empty((L, n_b), dtype=np.float32)
    for jj, j in enumerate(group_b):
        sp_b_T[:, jj] = spikes[j].ravel()

    # ── Outer loop: sparse raw CCG for each group_a unit ─────────────────────
    peak_offsets = []

    for ii, i in enumerate(group_a):
        if ii % 20 == 0:
            print(f'    group_a unit {ii + 1}/{n_a} ...', flush=True)

        sp_i  = spikes[i].ravel()                    # (L,) float32
        i_pos = np.flatnonzero(sp_i)                 # spike positions (~7500)
        i_bin = (i_pos % (n_trials * n_t)) % n_t     # bin within trial

        ccg_raw = np.zeros((n_b, n_lag), dtype=np.float64)

        for ki, k in enumerate(lags):
            # Only spikes where the lagged position stays within the same trial
            valid   = (i_bin + k >= 0) & (i_bin + k < n_t)
            pos_j_v = i_pos[valid] + k
            if len(pos_j_v) > 0:
                # Gather: (n_valid, n_b) contiguous rows → sum → (n_b,)
                ccg_raw[:, ki] = sp_b_T[pos_j_v].sum(axis=0)

        # Normalise corrected CCG and find peak lag per group_b unit
        ccg_corr = (ccg_raw - psth_ccg_all[ii]) / np.where(theta == 0, 1.0, theta)

        for jj in range(n_b):
            peak_offsets.append(float(lags[np.argmax(np.abs(ccg_corr[jj]))]))

    return peak_offsets, n_a * n_b
