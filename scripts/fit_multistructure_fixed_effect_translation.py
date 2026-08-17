#!/usr/bin/env python3
"""Decoupled shape/translation registration across the connected V1+HVA sheet and LGd.

Shape is estimated once, from within-session-demeaned local-linear regressions of RF on
CCF position pooled across all sessions (translation-invariant by construction: a session's
own additive offset cancels exactly under within-session weighted demeaning, so no per-
session translation can leak into the fitted local slope). Path-integrating that pooled
Jacobian field along each session's own sampled anatomy gives a shape-only predicted RF field
up to one unknown per-session constant. Translation is then a single decoupled robust-
location fit of that constant, per session, combined across the cortical sheet and LGd by
reliability weight and recentered to zero mean.

V1+HVA are treated as one connected, folded map (a single shared translation, no per-area
intercept) because Allen targeted most probes to be retinotopically matched in eccentricity
across areas; a hard per-area offset would impose an artificial break at every area boundary.
LGd is a separate modality/coordinate frame and gets its own shape field and reliability
weight rather than being hand-excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from scripts.checkpoint_joint_multistructure_dispersion_likelihood import load_all
from scripts.fit_joint_multistructure_dispersion_em import recenter
from scripts.test_v1_rf_size_corroboration import robust_scale
from scripts.validate_joint_dispersion_cell_halves import stratified_halves


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
OUT = BASE / "multistructure_fixed_effect_translation"
ELIGIBLE = BASE / "joint_multistructure_dispersion_checkpoint" / "three_structure_eligible_sessions.csv"

# 2D (AP, LR) CCF, matching the already-validated check_v1_cross_animal_mean_map_support.py
# convention; depth-dependent (DV) projection is not modeled here (open decision, see plan).
CCF2 = ["anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate"]

DOMAINS = {
    "cortex": dict(groups=("V1", "HVA"), bandwidth=250., grid_step=100., grid_margin=300.,
                    min_effective_n=20., min_cell_count=3),
    "lgd": dict(groups=("LGd",), bandwidth=400., grid_step=100., grid_margin=300.,
                 min_effective_n=8., min_cell_count=3),
}
RADIUS_MULTIPLE = 3.0
MAX_EDGE_MULTIPLE = 2.0  # path-integration edges longer than this many bandwidths are refused
RIDGE = 0.05
HUBER_C = 1.5
MAX_IRLS_ITER = 25
CONDITION_LIMIT = 1e6
# A retinotopic fold/reversal is a place where the true CCF->RF map has a very steep or
# discontinuous local derivative -- exactly where a local-LINEAR fit is structurally wrong,
# regardless of how much data supports it or how well-conditioned the regression is. Found by
# inspecting the per-session registration PDF: a few grid nodes had well-conditioned, well-
# supported Jacobian fits (condition number well under CONDITION_LIMIT) but a magnitude of up
# to ~22 deg/um, which turns an ordinary ~140 um grid step into a >3000 degree jump. Cortex
# Jacobian magnitude is under ~2.8 deg/um at the 99th percentile in this cohort, so 5 deg/um
# rejects only the clear fold/singularity tail, not real steep-but-plausible gradient.
MAX_JACOBIAN_NORM_DEG_PER_UM = 5.0
SHUFFLE_REPEATS = 200
MIN_SESSION_CELLS = 8


def make_grid(points, step, margin):
    lo = points.min(axis=0) - margin
    hi = points.max(axis=0) + margin
    axis0 = np.arange(lo[0], hi[0] + step, step)
    axis1 = np.arange(lo[1], hi[1] + step, step)
    xx, yy = np.meshgrid(axis0, axis1, indexing="ij")
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return axis0, axis1, grid


def within_session_weighted_demean(x, y, sessions, weights):
    x_out = x.copy()
    y_out = y.copy()
    for sid in np.unique(sessions):
        mask = sessions == sid
        w = weights[mask]
        total = w.sum()
        if total <= 0:
            continue
        x_out[mask] = x[mask] - (w[:, None] * x[mask]).sum(0) / total
        y_out[mask] = y[mask] - (w[:, None] * y[mask]).sum(0) / total
    return x_out, y_out


def huber_irls_wls(x, y, weight, ridge, huber_c, max_iter):
    robust = np.ones(len(x))
    beta = np.zeros((x.shape[1], y.shape[1]))
    for _ in range(max_iter):
        combined = weight * robust
        gram = x.T @ (combined[:, None] * x) + ridge * np.eye(x.shape[1])
        rhs = x.T @ (combined[:, None] * y)
        updated = np.linalg.solve(gram, rhs)
        residual = y - x @ updated
        radius = np.sqrt(np.sum(residual**2, axis=1))
        scale = 1.4826 * np.median(np.abs(radius - np.median(radius))) + 1e-6
        new_robust = np.minimum(1.0, huber_c * scale / np.maximum(radius, 1e-12))
        moved = np.max(np.abs(updated - beta))
        beta = updated
        robust = new_robust
        if moved < 1e-8:
            break
    return beta, gram


def local_linear_jacobian_field(cells, grid, bandwidth, ridge=RIDGE,
                                 min_effective_n=20., min_cell_count=3,
                                 huber_c=HUBER_C, max_iter=MAX_IRLS_ITER,
                                 radius_multiple=RADIUS_MULTIPLE):
    """Pooled, within-session-demeaned local-linear RF~CCF slope at each grid node.

    Translation-invariant by construction: within_session_weighted_demean removes each
    session's own weighted-mean RF and CCF before the regression, so a session's constant
    offset contributes nothing to the fitted slope regardless of how many sessions are pooled.
    """
    points = cells[CCF2].to_numpy(float)
    rf = cells[["rf_x", "rf_y"]].to_numpy(float)
    sessions = cells["ecephys_session_id"].to_numpy(int)
    tree = cKDTree(points)
    radius = radius_multiple * bandwidth

    n_grid = len(grid)
    jac = np.full((n_grid, 2, 2), np.nan)
    eff_n = np.full(n_grid, np.nan)
    condition = np.full(n_grid, np.nan)
    session_count = np.zeros(n_grid, dtype=int)

    for i, node in enumerate(grid):
        neighbor_idx = tree.query_ball_point(node, radius)
        if len(neighbor_idx) < min_cell_count:
            continue
        neighbor_idx = np.asarray(neighbor_idx)
        delta = points[neighbor_idx] - node
        distance2 = np.sum(delta**2, axis=1)
        weight = np.exp(-0.5 * distance2 / bandwidth**2)
        local_sessions = sessions[neighbor_idx]
        n_sessions = len(np.unique(local_sessions))
        session_count[i] = n_sessions
        if n_sessions < 2:
            continue
        x_tilde, y_tilde = within_session_weighted_demean(delta, rf[neighbor_idx], local_sessions, weight)
        total = weight.sum()
        effective = total**2 / max((weight**2).sum(), 1e-12)
        eff_n[i] = effective
        if effective < min_effective_n:
            continue
        beta, gram = huber_irls_wls(x_tilde, y_tilde, weight, ridge, huber_c, max_iter)
        cond = np.linalg.cond(gram)
        condition[i] = cond
        if not np.isfinite(cond) or cond > CONDITION_LIMIT:
            continue
        if np.linalg.norm(beta) > MAX_JACOBIAN_NORM_DEG_PER_UM:
            continue
        jac[i] = beta.T  # (rf_axis, ccf_axis)
    return {"jacobian": jac, "effective_n": eff_n, "condition": condition,
            "session_count": session_count}


def jacobian_interpolators(field, axis0, axis1):
    shape = (len(axis0), len(axis1))
    components = {}
    for r in range(2):
        for c in range(2):
            values = field["jacobian"][:, r, c].reshape(shape)
            components[(r, c)] = RegularGridInterpolator((axis0, axis1), values, bounds_error=False, fill_value=np.nan)
    eff_n_interp = RegularGridInterpolator((axis0, axis1), field["effective_n"].reshape(shape), bounds_error=False, fill_value=np.nan)
    return components, eff_n_interp


def evaluate_jacobian(components, points):
    out = np.full((len(points), 2, 2), np.nan)
    for (r, c), interp in components.items():
        out[:, r, c] = interp(points)
    return out


def mst_path_integrate_session(points, components, eff_n_interp, min_effective_n, max_edge=np.inf):
    """Shape-only predicted RF for ONE session's own cells, up to one unknown constant.

    Must run per session, never pooled across sessions: an MST across cells from different
    sessions would splice physically unrelated probe tracks into one fake path. An MST edge
    longer than max_edge is refused (marks that branch, and everything downstream of it,
    unreliable) rather than integrated through: local-linear steps compound additively along a
    path, and a single long jump bridging two locally-well-supported-but-distant clusters can
    silently inject a huge, non-physical excursion even though both of its endpoints look fine
    in isolation -- this was caught by inspecting the per-session registration PDF, where the
    pooled cortex map showed RF azimuth swinging across ~270 degrees, far past the ~60 degree
    screen extent. max_edge should be a small multiple of the Jacobian fit's own bandwidth, not
    the radius used to gather neighbors for that fit.
    """
    n = len(points)
    mu = np.full((n, 2), np.nan)
    reliable = np.zeros(n, dtype=bool)
    if n < 2:
        return mu, reliable
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(delta**2, axis=2))
    tree_matrix = minimum_spanning_tree(distance).toarray()
    tree_matrix = tree_matrix + tree_matrix.T
    neighbors = [np.flatnonzero(tree_matrix[i] > 0) for i in range(n)]
    root = int(np.argmin(np.sum((points - points.mean(axis=0))**2, axis=1)))
    mu[root] = 0.0
    root_eff_n = eff_n_interp(points[[root]])[0]
    reliable[root] = np.isfinite(root_eff_n) and root_eff_n >= min_effective_n
    visited = np.zeros(n, dtype=bool)
    visited[root] = True
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in neighbors[parent]:
            if visited[child]:
                continue
            visited[child] = True
            edge_length = float(np.linalg.norm(points[child] - points[parent]))
            midpoint = 0.5 * (points[parent] + points[child])
            local_eff_n = eff_n_interp(midpoint[None, :])[0]
            jac_mid = evaluate_jacobian(components, midpoint[None, :])[0]
            step = jac_mid @ (points[child] - points[parent])
            valid_step = (np.isfinite(local_eff_n) and local_eff_n >= min_effective_n
                          and np.isfinite(step).all() and edge_length <= max_edge)
            mu[child] = mu[parent] + (step if np.isfinite(step).all() else np.nan)
            reliable[child] = reliable[parent] and valid_step
            stack.append(child)
    return mu, reliable


def mst_path_integrate(cells, components, eff_n_interp, min_effective_n, max_edge=np.inf):
    """Per-session shape-only prediction for every cell, assembled back into cells' order."""
    n = len(cells)
    mu = np.full((n, 2), np.nan)
    reliable = np.zeros(n, dtype=bool)
    for sid, local in cells.groupby("ecephys_session_id", observed=True):
        idx = local.index.to_numpy()
        points = local[CCF2].to_numpy(float)
        sub_mu, sub_reliable = mst_path_integrate_session(points, components, eff_n_interp, min_effective_n, max_edge)
        mu[idx] = sub_mu
        reliable[idx] = sub_reliable
    return mu, reliable


def grid_shape_field(field, axis0, axis1, components, eff_n_interp, min_effective_n, ridge=1e-3):
    """Population-level shape-only map (the 'common map'), display/context only.

    Reconstructed by LEAST-SQUARES over every 4-connected grid-neighbor edge simultaneously,
    not by a single spanning-tree path: a single MST path was tried first and produced a
    cortex azimuth range of roughly -90 to +180 degrees, impossible for a ~60 degree screen --
    local Jacobian noise compounds additively along one arbitrary route. Redundant grid
    connectivity lets least squares average that noise down instead of accumulating it. This
    never touches the actual per-session delta fit, which still uses mst_path_integrate on
    that session's own (much smaller) cell cluster.
    """
    n0, n1 = len(axis0), len(axis1)
    valid = (np.isfinite(field["jacobian"][:, 0, 0]) & (field["effective_n"] >= min_effective_n)).reshape(n0, n1)
    index = -np.ones((n0, n1), dtype=int)
    index[valid] = np.arange(int(valid.sum()))
    n_nodes = int(valid.sum())
    mu_flat = np.full((n0 * n1, 2), np.nan)
    reliable_flat = np.zeros(n0 * n1, dtype=bool)
    if n_nodes < 2:
        return mu_flat.reshape(n0, n1, 2), reliable_flat.reshape(n0, n1)

    edges = []
    for i in range(n0):
        for j in range(n1):
            if not valid[i, j]:
                continue
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii >= n0 or jj >= n1 or not valid[ii, jj]:
                    continue
                p0 = np.array([axis0[i], axis1[j]])
                p1 = np.array([axis0[ii], axis1[jj]])
                mid = 0.5 * (p0 + p1)
                eff = eff_n_interp(mid[None, :])[0]
                if not np.isfinite(eff) or eff < min_effective_n:
                    continue
                jac_mid = evaluate_jacobian(components, mid[None, :])[0]
                step = jac_mid @ (p1 - p0)
                if not np.isfinite(step).all():
                    continue
                edges.append((index[i, j], index[ii, jj], step[0], step[1]))
    if len(edges) < n_nodes - 1:
        return mu_flat.reshape(n0, n1, 2), reliable_flat.reshape(n0, n1)

    edges = np.asarray(edges, dtype=float)
    n_edge = len(edges)
    a_idx = edges[:, 0].astype(int)
    b_idx = edges[:, 1].astype(int)
    rows = np.concatenate([np.arange(n_edge), np.arange(n_edge), [n_edge]])
    cols = np.concatenate([a_idx, b_idx, [0]])
    data = np.concatenate([-np.ones(n_edge), np.ones(n_edge), [ridge]])
    matrix = coo_matrix((data, (rows, cols)), shape=(n_edge + 1, n_nodes)).tocsr()

    mu_nodes = np.full((n_nodes, 2), np.nan)
    for k in range(2):
        rhs = np.concatenate([edges[:, 2 + k], [0.]])
        mu_nodes[:, k] = lsqr(matrix, rhs, atol=1e-10, btol=1e-10)[0]

    node_positions = np.argwhere(valid)
    flat_idx = node_positions[:, 0] * n1 + node_positions[:, 1]
    mu_flat[flat_idx] = mu_nodes
    reliable_flat[flat_idx] = True
    return mu_flat.reshape(n0, n1, 2), reliable_flat.reshape(n0, n1)


def huber_location(values, huber_c=HUBER_C, max_iter=50):
    finite = values[np.isfinite(values).all(axis=1)]
    if len(finite) == 0:
        return np.array([np.nan, np.nan])
    location = np.median(finite, axis=0)
    for _ in range(max_iter):
        residual = finite - location
        radius = np.sqrt(np.sum(residual**2, axis=1))
        scale = 1.4826 * np.median(np.abs(radius - np.median(radius))) + 1e-6
        weight = np.minimum(1.0, huber_c * scale / np.maximum(radius, 1e-12))
        total = weight.sum()
        if total <= 0:
            break
        updated = (weight[:, None] * finite).sum(0) / total
        if np.max(np.abs(updated - location)) < 1e-6:
            location = updated
            break
        location = updated
    return location


def huber_mean_loss(values, huber_c=HUBER_C):
    radius = np.sqrt(np.sum(values**2, axis=1))
    return float(np.mean(np.where(radius <= huber_c, .5 * radius**2, huber_c * radius - .5 * huber_c**2)))


def session_shape_and_translation(cells, mu_shape, reliable):
    rows = []
    deltas = {}
    for sid, local in cells.groupby("ecephys_session_id", observed=True):
        idx = local.index.to_numpy()
        position = cells.index.get_indexer(idx)
        rf = local[["rf_x", "rf_y"]].to_numpy(float)
        predicted = mu_shape[position]
        keep = reliable[position] & np.isfinite(predicted).all(axis=1)
        if keep.sum() < MIN_SESSION_CELLS:
            rows.append({"ecephys_session_id": int(sid), "valid_cells": int(keep.sum()), "shift_az_deg": np.nan, "shift_el_deg": np.nan})
            continue
        residual = rf[keep] - predicted[keep]
        delta = huber_location(residual)
        deltas[int(sid)] = delta
        rows.append({"ecephys_session_id": int(sid), "valid_cells": int(keep.sum()), "shift_az_deg": delta[0], "shift_el_deg": delta[1]})
    return deltas, pd.DataFrame(rows)


def shuffle_reliability(cells, mu_shape, reliable, rng, repeats=SHUFFLE_REPEATS):
    rows = []
    for sid, local in cells.groupby("ecephys_session_id", observed=True):
        idx = local.index.to_numpy()
        position = cells.index.get_indexer(idx)
        rf = local[["rf_x", "rf_y"]].to_numpy(float)
        predicted = mu_shape[position]
        keep = reliable[position] & np.isfinite(predicted).all(axis=1)
        if keep.sum() < MIN_SESSION_CELLS:
            continue
        rf_keep = rf[keep]
        predicted_keep = predicted[keep]
        delta = huber_location(rf_keep - predicted_keep)
        real_loss = huber_mean_loss(rf_keep - predicted_keep - delta)
        null = np.empty(repeats)
        for r in range(repeats):
            shuffled = rng.permutation(rf_keep)
            shuffled_delta = huber_location(shuffled - predicted_keep)
            null[r] = huber_mean_loss(shuffled - predicted_keep - shuffled_delta)
        p = (1 + np.sum(null <= real_loss)) / (len(null) + 1)
        z = (np.median(null) - real_loss) / (1.4826 * np.median(np.abs(null - np.median(null))) + 1e-6)
        rows.append({"ecephys_session_id": int(sid), "valid_cells": int(keep.sum()), "real_loss": real_loss, "shuffle_p": p, "shuffle_z": z})
    return pd.DataFrame(rows)


def fit_domain(cells, domain_name, spec, rng):
    axis0, axis1, grid = make_grid(cells[CCF2].to_numpy(float), spec["grid_step"], spec["grid_margin"])
    field = local_linear_jacobian_field(cells, grid, spec["bandwidth"], min_effective_n=spec["min_effective_n"], min_cell_count=spec["min_cell_count"])
    components, eff_n_interp = jacobian_interpolators(field, axis0, axis1)
    max_edge = MAX_EDGE_MULTIPLE * spec["bandwidth"]
    mu_shape, reliable = mst_path_integrate(cells, components, eff_n_interp, spec["min_effective_n"], max_edge)
    deltas, audit = session_shape_and_translation(cells, mu_shape, reliable)
    reliability = shuffle_reliability(cells, mu_shape, reliable, rng)
    audit["domain"] = domain_name
    reliability["domain"] = domain_name
    grid_mu, grid_reliable = grid_shape_field(field, axis0, axis1, components, eff_n_interp, spec["min_effective_n"])
    field_summary = {
        "domain": domain_name,
        "grid_nodes": int(len(grid)),
        "grid_nodes_with_jacobian": int(np.isfinite(field["jacobian"][:, 0, 0]).sum()),
        "median_effective_n": float(np.nanmedian(field["effective_n"])),
        "reliable_cell_fraction": float(np.mean(reliable)),
    }
    return {"deltas": deltas, "audit": audit, "reliability": reliability, "field_summary": field_summary,
            "cells": cells, "mu_shape": mu_shape, "reliable": reliable,
            "grid_axis0": axis0, "grid_axis1": axis1, "grid_mu": grid_mu, "grid_reliable": grid_reliable}


def combine_domains(cortex_result, lgd_result, ids):
    cortex_reliability = cortex_result["reliability"].set_index("ecephys_session_id")
    lgd_reliability = lgd_result["reliability"].set_index("ecephys_session_id")
    combined = {}
    weights_rows = []
    for sid in ids:
        candidates = []
        for name, result, reliability in (("cortex", cortex_result, cortex_reliability), ("lgd", lgd_result, lgd_reliability)):
            delta = result["deltas"].get(sid)
            if delta is None:
                continue
            z = reliability.loc[sid, "shuffle_z"] if sid in reliability.index else 0.
            weight = float(np.clip(z, 0, 3) / 3)
            candidates.append((name, delta, weight))
            weights_rows.append({"ecephys_session_id": sid, "domain": name, "shift_az_deg": delta[0], "shift_el_deg": delta[1], "weight": weight})
        if not candidates:
            combined[sid] = np.array([np.nan, np.nan])
            continue
        total_weight = sum(w for _, _, w in candidates)
        if total_weight <= 0:
            combined[sid] = np.mean([d for _, d, _ in candidates], axis=0)
        else:
            combined[sid] = sum(w * d for _, d, w in candidates) / total_weight
    # Unlike the old grid-searched EM, this fit is never bounded by a search window, so the
    # recenter() clip must not reuse that script's 30 deg default -- doing so silently piled
    # several sessions onto an artificial 30 deg boundary here during development. 90 deg is a
    # generous physical cap (beyond the ~60 deg stimulus screen extent), kept only as a
    # numerical safety net, not an expected operating range.
    combined = recenter(combined, ids, bound=90.)
    return combined, pd.DataFrame(weights_rows)


def session_domain_cells(result, sid):
    cells = result["cells"]
    local = cells.loc[cells.ecephys_session_id.eq(sid)]
    if len(local) == 0:
        return None
    position = cells.index.get_indexer(local.index)
    predicted = result["mu_shape"][position]
    reliable = result["reliable"][position]
    return local, predicted, reliable


def plot_domain_row(axes_row, domain_name, result, sid, delta, shuffle_row):
    ax_az, ax_el, ax_rf = axes_row
    session_data = session_domain_cells(result, sid) if result is not None else None
    if session_data is None or delta is None or not np.all(np.isfinite(delta)):
        for ax in axes_row:
            ax.axis("off")
        ax_az.text(.5, .5, f"no usable {domain_name} data for this session", ha="center", va="center",
                   transform=ax_az.transAxes, fontsize=9)
        return
    local, predicted, reliable = session_data
    axis0, axis1 = result["grid_axis0"], result["grid_axis1"]
    grid_mu, grid_reliable = result["grid_mu"], result["grid_reliable"]
    observed_rf = local[["rf_x", "rf_y"]].to_numpy(float)

    # Window the shared population map to this session's own anatomical footprint. The map is
    # reconstructed by path-integrating a noisy local Jacobian, so far-away grid nodes can carry
    # large accumulated drift (found by inspecting an early version of this PDF, where the
    # unwindowed cortex background spanned RF azimuth from -90 to +180 degrees -- impossible for
    # a ~60 degree screen); color-scaling and cropping to what's actually relevant to this
    # session's own cells avoids both displaying and being misled by that far-field drift.
    session_ccf = local[CCF2].to_numpy(float)
    margin = 2. * DOMAINS[domain_name]["bandwidth"]
    lo = session_ccf.min(axis=0) - margin
    hi = session_ccf.max(axis=0) + margin
    window0 = (axis0 >= lo[0]) & (axis0 <= hi[0])
    window1 = (axis1 >= lo[1]) & (axis1 <= hi[1])
    for k, (ax, label) in enumerate(((ax_az, "azimuth"), (ax_el, "elevation"))):
        values = np.where(grid_reliable, grid_mu[..., k] + delta[k], np.nan)
        windowed = values[np.ix_(window0, window1)]
        finite = windowed[np.isfinite(windowed)]
        if len(finite) == 0:
            ax.axis("off")
            continue
        vmin, vmax = np.nanpercentile(finite, [2, 98])
        cs = ax.contourf(axis1, axis0, values, levels=20, cmap="coolwarm", vmin=vmin, vmax=vmax, extend="both")
        ax.scatter(local["left_right_ccf_coordinate"], local["anterior_posterior_ccf_coordinate"],
                   c=observed_rf[:, k], cmap="coolwarm", vmin=vmin, vmax=vmax, edgecolor="k", linewidth=.4, s=24)
        ax.set_xlim(lo[1], hi[1])
        ax.set_ylim(lo[0], hi[0])
        plt.colorbar(cs, ax=ax, shrink=.8, label=f"predicted RF {label} (deg)")
        ax.set(xlabel="left-right CCF (µm)", ylabel="anterior-posterior CCF (µm)",
               title=f"{domain_name}: common map + cells ({label}), windowed to this session", aspect="equal")
    keep = reliable & np.isfinite(predicted).all(axis=1)
    predicted_shifted = predicted + delta
    ax_rf.scatter(observed_rf[keep, 0], observed_rf[keep, 1], s=24, color="#4477aa", label="observed", zorder=2)
    ax_rf.scatter(predicted_shifted[keep, 0], predicted_shifted[keep, 1], s=24, marker="x", color="#ee7733",
                  label="shape + δ predicted", zorder=2)
    for o, p in zip(observed_rf[keep], predicted_shifted[keep]):
        ax_rf.plot([o[0], p[0]], [o[1], p[1]], color="0.75", lw=.5, zorder=1)
    ax_rf.axhline(0, color=".85", lw=.6)
    ax_rf.axvline(0, color=".85", lw=.6)
    z_text = f", shuffle z={shuffle_row.shuffle_z.iloc[0]:.1f}" if shuffle_row is not None and len(shuffle_row) else ""
    ax_rf.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", aspect="equal",
              title=f"{domain_name} registration (n={int(keep.sum())})\n"
                    f"δ=({delta[0]:+.1f}, {delta[1]:+.1f})°{z_text}")
    ax_rf.legend(fontsize=7, frameon=False)


def build_registration_pdf(domain_results, combined, weights, reliability, session_ids, out_path):
    weights_indexed = weights.set_index(["ecephys_session_id", "domain"])
    with PdfPages(out_path) as pdf:
        for sid in session_ids:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
            combined_delta = combined.get(sid)
            combined_text = "no combined estimate"
            if combined_delta is not None and np.all(np.isfinite(combined_delta)):
                parts = []
                for domain_name in DOMAINS:
                    if (sid, domain_name) in weights_indexed.index:
                        parts.append(f"{domain_name} weight={weights_indexed.loc[(sid, domain_name), 'weight']:.2f}")
                combined_text = (f"combined δ=({combined_delta[0]:+.1f}, {combined_delta[1]:+.1f})°; "
                                 + "; ".join(parts))
            fig.suptitle(f"Session {sid} — {combined_text}", y=.995)
            for row, domain_name in zip(axes, DOMAINS):
                result = domain_results[domain_name]
                delta = result["deltas"].get(sid)
                shuffle_row = result["reliability"].loc[result["reliability"].ecephys_session_id.eq(sid)]
                plot_domain_row(row, domain_name, result, sid, delta, shuffle_row)
            fig.tight_layout(rect=(0, 0, 1, .97))
            pdf.savefig(fig)
            plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = load_all()
    pop = pop.loc[~pop.center_bound & pop[CCF2 + ["rf_x", "rf_y"]].notna().all(axis=1)].copy()
    eligible_ids = pd.read_csv(ELIGIBLE).ecephys_session_id.astype(int).tolist()

    rng = np.random.default_rng(20260817)
    domain_results = {}
    for domain_name, spec in DOMAINS.items():
        cells = pop.loc[pop.structure_group.isin(spec["groups"])].reset_index(drop=True)
        print(f"fitting domain={domain_name} cells={len(cells)} sessions={cells.ecephys_session_id.nunique()}", flush=True)
        domain_results[domain_name] = fit_domain(cells, domain_name, spec, rng)

    all_session_ids = sorted(set(pop.ecephys_session_id.astype(int)))
    combined, weights = combine_domains(domain_results["cortex"], domain_results["lgd"], all_session_ids)

    delta_rows = [{"ecephys_session_id": sid, "shift_az_deg": combined[sid][0], "shift_el_deg": combined[sid][1]} for sid in all_session_ids]
    delta_table = pd.DataFrame(delta_rows)
    delta_table.to_csv(OUT / "session_translations.csv", index=False)
    weights.to_csv(OUT / "domain_weights.csv", index=False)

    audit = pd.concat([domain_results["cortex"]["audit"], domain_results["lgd"]["audit"]], ignore_index=True)
    audit.to_csv(OUT / "domain_translation_audit.csv", index=False)
    reliability = pd.concat([domain_results["cortex"]["reliability"], domain_results["lgd"]["reliability"]], ignore_index=True)
    reliability.to_csv(OUT / "domain_shuffle_reliability.csv", index=False)

    # split-half reproducibility of the pooled Jacobian field / final translation
    split_rows = []
    for domain_name, spec in DOMAINS.items():
        cells = pop.loc[pop.structure_group.isin(spec["groups"])].reset_index(drop=True)
        halves = stratified_halves(cells, np.random.default_rng(20260818))
        half_deltas = []
        for half_index, half_cells in enumerate(halves):
            half_cells = half_cells.reset_index(drop=True)
            axis0, axis1, grid = make_grid(half_cells[CCF2].to_numpy(float), spec["grid_step"], spec["grid_margin"])
            field = local_linear_jacobian_field(half_cells, grid, spec["bandwidth"], min_effective_n=spec["min_effective_n"], min_cell_count=spec["min_cell_count"])
            components, eff_n_interp = jacobian_interpolators(field, axis0, axis1)
            mu_shape, reliable = mst_path_integrate(half_cells, components, eff_n_interp, spec["min_effective_n"], MAX_EDGE_MULTIPLE * spec["bandwidth"])
            deltas, _ = session_shape_and_translation(half_cells, mu_shape, reliable)
            half_deltas.append(deltas)
        common = set(half_deltas[0]) & set(half_deltas[1])
        for sid in common:
            distance = float(np.linalg.norm(half_deltas[0][sid] - half_deltas[1][sid]))
            split_rows.append({"domain": domain_name, "ecephys_session_id": sid, "split_distance_deg": distance})
    split = pd.DataFrame(split_rows)
    split.to_csv(OUT / "split_half_reproducibility.csv", index=False)

    # non-fused cross-checks against existing independent estimates
    cross_rows = []
    anatomy_path = BASE / "gaze_censor_anchor_checkpoint" / "all_session_anatomy_offsets.csv"
    em_path = BASE / "joint_multistructure_dispersion_em" / "all_initialization_session_shifts.csv"
    if anatomy_path.exists():
        anatomy = pd.read_csv(anatomy_path).set_index("ecephys_session_id")
    else:
        anatomy = None
    em_summary_path = BASE / "joint_multistructure_dispersion_em" / "initialization_summary.csv"
    if em_path.exists() and em_summary_path.exists():
        best_initialization = pd.read_csv(em_summary_path).sort_values("final_objective").iloc[0].initialization
        em = pd.read_csv(em_path)
        em = em.loc[em.initialization.eq(best_initialization)].set_index("ecephys_session_id")
    else:
        em = None
    for sid in all_session_ids:
        row = {"ecephys_session_id": sid, "shift_az_deg": combined[sid][0], "shift_el_deg": combined[sid][1]}
        if anatomy is not None and sid in anatomy.index:
            offset = anatomy.loc[sid, ["offset_az_relative_deg", "offset_el_relative_deg"]].to_numpy(float)
            row["distance_from_v1_anatomy_prior_deg"] = float(np.linalg.norm(combined[sid] - offset))
        if em is not None and sid in em.index:
            em_shift = em.loc[sid, ["shift_az_deg", "shift_el_deg"]].to_numpy(float)
            row["distance_from_dispersion_em_deg"] = float(np.linalg.norm(combined[sid] - em_shift))
        cross_rows.append(row)
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(OUT / "cross_check_against_existing_estimates.csv", index=False)

    manifest = {
        "domains": {name: spec for name, spec in DOMAINS.items()},
        "ccf_columns": CCF2,
        "radius_multiple": RADIUS_MULTIPLE,
        "ridge": RIDGE,
        "huber_c": HUBER_C,
        "shuffle_repeats": SHUFFLE_REPEATS,
        "min_session_cells": MIN_SESSION_CELLS,
        "field_summary": {name: result["field_summary"] for name, result in domain_results.items()},
        "eligible_sessions_16": eligible_ids,
        "note": "V1+HVA fit as one connected domain (single shared translation, no per-area intercept); LGd fit and weighted separately per reliability, not fused into the shape fit.",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(delta_table.shift_az_deg, delta_table.shift_el_deg, alpha=.7)
    axes[0].axhline(0, color=".8"); axes[0].axvline(0, color=".8")
    axes[0].set(xlabel="az shift (deg)", ylabel="el shift (deg)", title="Final per-session translation", aspect="equal")
    if len(split):
        split.boxplot(column="split_distance_deg", by="domain", ax=axes[1])
        axes[1].set(ylabel="split-half distance (deg)", title="Shape/translation reproducibility")
        plt.suptitle("")
    for domain_name in DOMAINS:
        local = reliability.loc[reliability.domain.eq(domain_name)]
        axes[2].hist(local.shuffle_z, bins=20, alpha=.6, label=domain_name)
    axes[2].set(xlabel="shuffle z", ylabel="sessions", title="Per-session shape reliability")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_multistructure_fixed_effect_translation.png", dpi=180)
    plt.close(fig)

    pdf_path = OUT / "Figure_per_session_registration.pdf"
    build_registration_pdf(domain_results, combined, weights, reliability, all_session_ids, pdf_path)
    print(pdf_path)

    print(delta_table.to_string(index=False))
    print(split.groupby("domain").split_distance_deg.describe().to_string())


if __name__ == "__main__":
    main()
