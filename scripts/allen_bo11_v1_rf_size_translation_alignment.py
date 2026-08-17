#!/usr/bin/env python3
"""Pilot session translation matching from interior Allen V1 RF-size surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from scripts.allen_bo11_tuning_driven_limited_affine import (
    PARAMETER_NAMES,
    evaluate_model,
    load_maps,
    map_agreement,
    summarize,
    template_from_maps,
    warp_map,
)
from scripts.allen_bo11_tuning_weighted_session_surfaces import weighted_gaussian_surface
from scripts.render_allen_bo11_registration_comparison import DEFAULT_SURFACE_GRID
from scripts.render_allen_bo11_v1_rf_size_interior import (
    BO_COHORT,
    DEFAULT_INPUT,
    prepare_population,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_OUTPUT = AUDIT / "v1_rf_size_translation_alignment"
DEFAULT_COHORT = AUDIT / "ccf_retinotopy_alignment" / "selected_ccf_retinotopy_transforms.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tuning-grid", type=Path, default=DEFAULT_SURFACE_GRID)
    parser.add_argument("--session-table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-exclusion-deg", type=float, default=20.0)
    parser.add_argument("--bandwidth-deg", type=float, default=8.0)
    parser.add_argument("--minimum-effective-local-units", type=float, default=5.0)
    parser.add_argument("--translation-bound-deg", type=float, default=10.0)
    parser.add_argument("--regularization-scale-deg", type=float, default=10.0)
    parser.add_argument("--regularization-weight", type=float, default=0.08)
    parser.add_argument("--minimum-shared-grid-points", type=int, default=80)
    parser.add_argument("--skip-split-half", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_session_maps(
    population: pd.DataFrame,
    sessions: list[int],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_local_units: float,
) -> tuple[dict[tuple[int, str, str], dict[str, np.ndarray]], pd.DataFrame]:
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    targets = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps = {}
    rows = []
    for session_id in sessions:
        selected = population.loc[population["ecephys_session_id"].eq(session_id)].dropna(
            subset=["azimuth_rf", "elevation_rf", "session_standardized_log2_rf_area"]
        )
        surface = weighted_gaussian_surface(
            selected[["azimuth_rf", "elevation_rf"]].to_numpy(float),
            selected["session_standardized_log2_rf_area"].to_numpy(float),
            np.ones(len(selected)),
            targets,
            bandwidth_deg=bandwidth_deg,
            minimum_effective_local_units=minimum_effective_local_units,
        )
        value = surface["estimate_log2"].reshape(len(el_grid), len(az_grid))
        effective = surface["effective_local_units"].reshape(len(el_grid), len(az_grid))
        supported = surface["supported"].reshape(len(el_grid), len(az_grid))
        evidence = np.where(supported & np.isfinite(value), np.sqrt(np.maximum(effective, 0)), 0.0)
        finite = np.where(np.isfinite(value), value, 0.0)
        maps[(session_id, "V1", "rf_size")] = {
            "value": value,
            "evidence": evidence,
            "source_units": len(selected),
            "interpolate_evidence": RegularGridInterpolator(
                (el_grid, az_grid), evidence, bounds_error=False, fill_value=0.0
            ),
            "interpolate_numerator": RegularGridInterpolator(
                (el_grid, az_grid), finite * evidence, bounds_error=False, fill_value=0.0
            ),
        }
        rows.append(
            {
                "ecephys_session_id": session_id,
                "interior_v1_units": len(selected),
                "supported_grid_points": int(supported.sum()),
                "supported_grid_fraction": float(supported.mean()),
            }
        )
    return maps, pd.DataFrame(rows)


def packed_translation(translation: np.ndarray) -> np.ndarray:
    return np.array([translation[0], translation[1], 0.0, 0.0, 0.0, 0.0])


def alignment_loss(
    translation: np.ndarray,
    source: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    bound_deg: float,
    regularization_scale_deg: float,
    regularization_weight: float,
    minimum_points: int,
) -> tuple[float, dict[str, float]]:
    warped = warp_map(source, packed_translation(translation), az_grid, el_grid)
    agreement = map_agreement(warped, template, minimum_points)
    if not np.isfinite(agreement["correlation"]):
        return 3.0, agreement
    penalty = regularization_weight * float(
        np.mean(np.square(np.asarray(translation) / regularization_scale_deg))
    )
    loss = 1.0 - agreement["correlation"] + 0.2 * (1.0 - agreement["coverage"]) + penalty
    return float(loss), agreement


def fit_translations(
    maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    *,
    bound_deg: float,
    regularization_scale_deg: float,
    regularization_weight: float,
    minimum_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = sorted({key[0] for key in maps})
    rows = []
    profiles = []
    for index, session_id in enumerate(sessions):
        source = maps[(session_id, "V1", "rf_size")]
        template = template_from_maps(maps, "V1", "rf_size", exclude_session=session_id)
        objective = lambda shift: alignment_loss(
            shift, source, template, az_grid, el_grid,
            bound_deg=bound_deg,
            regularization_scale_deg=regularization_scale_deg,
            regularization_weight=regularization_weight,
            minimum_points=minimum_points,
        )[0]
        supported_values = source["value"][source["evidence"] > 0]
        fit_identifiable = bool(
            len(supported_values) >= 2
            and np.nanstd(supported_values) > 1e-6
            and np.nanstd(template["value"][template["evidence"] > 0]) > 1e-6
        )
        if fit_identifiable:
            result = differential_evolution(
                objective,
                [(-bound_deg, bound_deg), (-bound_deg, bound_deg)],
                seed=20260813 + index,
                maxiter=30,
                popsize=7,
                polish=True,
                workers=1,
                updating="immediate",
                tol=1e-5,
            )
            optimum = np.asarray(result.x)
        else:
            # A constant/empty surface contains no spatial information. Keep the
            # session in downstream stacks but do not assign a random optimizer shift.
            optimum = np.zeros(2)
        identity_loss, identity_agreement = alignment_loss(
            np.zeros(2), source, template, az_grid, el_grid,
            bound_deg=bound_deg,
            regularization_scale_deg=regularization_scale_deg,
            regularization_weight=regularization_weight,
            minimum_points=minimum_points,
        )
        optimum_loss, optimum_agreement = alignment_loss(
            optimum, source, template, az_grid, el_grid,
            bound_deg=bound_deg,
            regularization_scale_deg=regularization_scale_deg,
            regularization_weight=regularization_weight,
            minimum_points=minimum_points,
        )
        axis_widths = {}
        scan = np.linspace(-bound_deg, bound_deg, 81)
        for axis, label in enumerate(("azimuth", "elevation")):
            losses = []
            for value in scan:
                shift = optimum.copy()
                shift[axis] = value
                losses.append(objective(shift))
                profiles.append(
                    {
                        "ecephys_session_id": session_id,
                        "axis": label,
                        "translation_deg": value,
                        "regularized_loss": losses[-1],
                        "loss_above_optimum": losses[-1] - optimum_loss,
                    }
                )
            supported = scan[np.asarray(losses) <= optimum_loss + 0.02]
            axis_widths[label] = float(np.ptp(supported)) if len(supported) else np.nan
        rows.append(
            {
                "ecephys_session_id": session_id,
                "selected_model": "interior_v1_rf_size_translation",
                "translation_azimuth_deg": optimum[0],
                "translation_elevation_deg": optimum[1],
                "rotation_deg": 0.0,
                "log_scale_azimuth": 0.0,
                "log_scale_elevation": 0.0,
                "shear": 0.0,
                "identity_regularized_loss": identity_loss,
                "aligned_regularized_loss": optimum_loss,
                "regularized_loss_gain": identity_loss - optimum_loss,
                "identity_rf_size_correlation": identity_agreement["correlation"],
                "aligned_rf_size_correlation": optimum_agreement["correlation"],
                "azimuth_profile_width_deg": axis_widths["azimuth"],
                "elevation_profile_width_deg": axis_widths["elevation"],
                "translation_at_bound": bool(np.any(np.abs(optimum) >= 0.98 * bound_deg)),
                "fit_identifiable": fit_identifiable,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(profiles)


def render_diagnostic(
    maps: dict[tuple[int, str, str], dict[str, np.ndarray]],
    transforms: pd.DataFrame,
    profiles: pd.DataFrame,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    output_path: Path,
) -> None:
    parameters = {
        int(row.ecephys_session_id): row[list(PARAMETER_NAMES)].to_numpy(float)
        for _, row in transforms.iterrows()
    }
    raw = template_from_maps(maps, "V1", "rf_size")
    from scripts.allen_bo11_tuning_driven_limited_affine import warp_all
    aligned = template_from_maps(warp_all(maps, parameters, az_grid, el_grid), "V1", "rf_size")
    values = np.r_[raw["value"][np.isfinite(raw["value"])], aligned["value"][np.isfinite(aligned["value"])]]
    limit = max(float(np.quantile(np.abs(values), .98)), .1)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0))
    for ax, template, title in ((axes[0, 0], raw, "Raw interior V1 RF-size stack"), (axes[0, 1], aligned, "RF-size-matched stack")):
        artist = ax.pcolormesh(az_grid, el_grid, template["value"], shading="gouraud", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        ax.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title=title)
    fig.colorbar(artist, ax=axes[0, :].tolist(), fraction=.025, pad=.03, label="standardized log₂ RF area")
    axes[1, 0].quiver(
        np.zeros(len(transforms)), np.zeros(len(transforms)),
        transforms["translation_azimuth_deg"], transforms["translation_elevation_deg"],
        angles="xy", scale_units="xy", scale=1, width=.006, alpha=.7, color="#6a3d9a",
    )
    axes[1, 0].axhline(0, color="#888888", linewidth=.7)
    axes[1, 0].axvline(0, color="#888888", linewidth=.7)
    vector_limit = max(11.0, float(np.nanmax(np.abs(transforms[["translation_azimuth_deg", "translation_elevation_deg"]].to_numpy()))) + 2.0)
    axes[1, 0].set(xlim=(-vector_limit, vector_limit), ylim=(-vector_limit, vector_limit), aspect="equal", xlabel="Azimuth translation (deg)", ylabel="Elevation translation (deg)", title="RF-size-derived session offsets")
    width = transforms[["azimuth_profile_width_deg", "elevation_profile_width_deg"]].rename(columns={"azimuth_profile_width_deg": "Azimuth", "elevation_profile_width_deg": "Elevation"})
    axes[1, 1].boxplot([width["Azimuth"].dropna(), width["Elevation"].dropna()], labels=["Azimuth", "Elevation"], patch_artist=True)
    axes[1, 1].axhline(10, color="#a13d2d", linestyle="--", linewidth=1)
    axes[1, 1].set(ylabel="Profile width within Δloss ≤ 0.02 (deg)", title="Offset identifiability (smaller is better)")
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    fig.suptitle("Allen BO 1.1: can the interior V1 RF-size gradient register sessions?", fontsize=15)
    fig.subplots_adjust(left=.08, right=.94, bottom=.07, top=.91, hspace=.32, wspace=.28)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_maps, tuning_az, tuning_el = load_maps(args.tuning_grid.resolve())
    sessions = sorted({key[0] for key in tuning_maps})
    population = prepare_population(pd.read_csv(args.support.resolve(), low_memory=False))
    if args.session_table is not None:
        cohort = pd.read_csv(args.session_table.resolve())
        if "ccf_available" in cohort:
            cohort = cohort.loc[cohort["ccf_available"].fillna(False).astype(bool)]
        cohort_sessions = set(cohort["ecephys_session_id"].astype(int))
        sessions = sorted(set(sessions) & cohort_sessions)
    population = population.loc[
        population["ecephys_session_id"].isin(sessions)
        & population["distance_to_nearest_grid_edge_deg"].ge(args.edge_exclusion_deg)
    ].copy()
    sessions = [int(value) for value in sorted(population["ecephys_session_id"].astype(int).unique())]
    tuning_maps = {key: value for key, value in tuning_maps.items() if key[0] in sessions}
    az_grid = np.linspace(10 + args.edge_exclusion_deg, 90 - args.edge_exclusion_deg, 31)
    el_grid = np.linspace(-30 + args.edge_exclusion_deg, 50 - args.edge_exclusion_deg, 31)
    maps, support = build_session_maps(
        population, sessions, az_grid, el_grid,
        bandwidth_deg=args.bandwidth_deg,
        minimum_effective_local_units=args.minimum_effective_local_units,
    )
    transforms, profiles = fit_translations(
        maps, az_grid, el_grid,
        bound_deg=args.translation_bound_deg,
        regularization_scale_deg=args.regularization_scale_deg,
        regularization_weight=args.regularization_weight,
        minimum_points=args.minimum_shared_grid_points,
    )
    split_transforms = pd.DataFrame()
    split_reliability = pd.DataFrame()
    if not args.skip_split_half:
        population["split_half"] = 0
        for session_id, indices in population.groupby("ecephys_session_id", observed=True).groups.items():
            rng = np.random.default_rng(20260814 + int(session_id))
            shuffled = np.asarray(list(indices))[rng.permutation(len(indices))]
            population.loc[shuffled[len(shuffled) // 2 :], "split_half"] = 1
        split_frames = []
        for half in (0, 1):
            half_maps, _ = build_session_maps(
                population.loc[population["split_half"].eq(half)], sessions, az_grid, el_grid,
                bandwidth_deg=args.bandwidth_deg,
                minimum_effective_local_units=max(1.0, args.minimum_effective_local_units / 2),
            )
            local, _ = fit_translations(
                half_maps, az_grid, el_grid,
                bound_deg=args.translation_bound_deg,
                regularization_scale_deg=args.regularization_scale_deg,
                regularization_weight=args.regularization_weight,
                minimum_points=max(50, args.minimum_shared_grid_points // 2),
            )
            local["split_half"] = half
            split_frames.append(local)
        split_transforms = pd.concat(split_frames, ignore_index=True)
        split_wide = split_transforms.pivot(
            index="ecephys_session_id", columns="split_half",
            values=["translation_azimuth_deg", "translation_elevation_deg"],
        )
        split_rows = []
        for label in ("azimuth", "elevation"):
            first = split_wide[(f"translation_{label}_deg", 0)]
            second = split_wide[(f"translation_{label}_deg", 1)]
            split_rows.append(
                {
                    "axis": label,
                    "split_half_spearman_rho": spearmanr(first, second).statistic,
                    "median_absolute_half_difference_deg": np.median(np.abs(first - second)),
                }
            )
        split_reliability = pd.DataFrame(split_rows)
    parameters = {int(row.ecephys_session_id): row[list(PARAMETER_NAMES)].to_numpy(float) for _, row in transforms.iterrows()}
    identity = {session_id: np.zeros(6) for session_id in sessions}
    raw_metrics = evaluate_model(tuning_maps, identity, tuning_az, tuning_el, 50, "raw")
    aligned_metrics = evaluate_model(tuning_maps, parameters, tuning_az, tuning_el, 50, "interior_v1_rf_size_translation")
    tuning_summary = summarize(raw_metrics, aligned_metrics)
    support.merge(transforms, on="ecephys_session_id", validate="one_to_one").to_csv(output_dir / "selected_v1_rf_size_translations.csv", index=False, float_format="%.6g")
    profiles.to_csv(output_dir / "v1_rf_size_translation_profiles.csv", index=False, float_format="%.6g")
    if not split_transforms.empty:
        split_transforms.to_csv(output_dir / "v1_rf_size_split_half_transforms.csv", index=False, float_format="%.6g")
        split_reliability.to_csv(output_dir / "v1_rf_size_split_half_reliability.csv", index=False, float_format="%.6g")
    tuning_summary.to_csv(output_dir / "v1_rf_size_translation_tuning_summary.csv", index=False, float_format="%.6g")
    figure_path = output_dir / "Figure_allen_bo11_v1_rf_size_translation_alignment.png"
    render_diagnostic(maps, transforms, profiles, az_grid, el_grid, figure_path)
    lines = [
        "# Allen BO 1.1 interior V1 RF-size translation pilot",
        "",
        f"Each of {len(sessions)} sessions was matched to a leave-one-session-out V1 RF-size template using only RF centers at least {args.edge_exclusion_deg:g}° from every RF-grid edge.",
        "Only translation was allowed. RF size was log2 transformed and median/IQR standardized within session, so absolute between-session RF-size differences could not masquerade as position.",
        "SF, TF, and HVA data were held out and evaluated afterward.",
        "",
        f"Median RF-size correlation gain: **{(transforms.aligned_rf_size_correlation - transforms.identity_rf_size_correlation).median():+.3f}**.",
        f"Median azimuth/elevation profile widths: **{transforms.azimuth_profile_width_deg.median():.1f}° / {transforms.elevation_profile_width_deg.median():.1f}°** (smaller means better identified).",
        f"Sessions at a ±{args.translation_bound_deg:g}° bound: **{int(transforms.translation_at_bound.sum())}/{len(transforms)}**.",
        f"Spatially identifiable session surfaces: **{int(transforms.fit_identifiable.sum())}/{len(transforms)}**; non-identifiable sessions were retained with zero shift.",
        "",
    ]
    if args.skip_split_half:
        lines.append("Split-half fitting was skipped because several sessions retain too few units after this exclusion.")
    else:
        lines.extend([
            f"Independent target-unit halves reproduce azimuth offsets at **rho = {split_reliability.set_index('axis').loc['azimuth', 'split_half_spearman_rho']:+.3f}** and elevation offsets at **rho = {split_reliability.set_index('axis').loc['elevation', 'split_half_spearman_rho']:+.3f}**.",
            f"Median absolute half-to-half differences are **{split_reliability.set_index('axis').loc['azimuth', 'median_absolute_half_difference_deg']:.1f}°** azimuth and **{split_reliability.set_index('axis').loc['elevation', 'median_absolute_half_difference_deg']:.1f}°** elevation.",
        ])
    lines.extend([
        "",
        "| Group | Map | Median paired Δr versus raw |",
        "| --- | --- | ---: |",
    ])
    for row in tuning_summary.itertuples(index=False):
        lines.append(f"| {row.group} | {row.preference.upper()} | {row.median_paired_correlation_change:+.3f} |")
    (output_dir / "ALLEN_BO11_V1_RF_SIZE_TRANSLATION_ALIGNMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output_dir.iterdir() if p.is_file() and p.name != "run_manifest.json"}
    manifest = {
        "checkpoint": "06c_allen_bo11_v1_rf_size_translation_alignment",
        "status": "exploratory RF-size-only translation; tuning held out",
        "inputs": {"support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())}, "tuning_grid": {"path": str(args.tuning_grid.resolve()), "sha256": sha256(args.tuning_grid.resolve())}, "session_table": None if args.session_table is None else {"path": str(args.session_table.resolve()), "sha256": sha256(args.session_table.resolve())}},
        "parameters": {"sessions": sessions, "edge_exclusion_deg": args.edge_exclusion_deg, "bandwidth_deg": args.bandwidth_deg, "minimum_effective_local_units": args.minimum_effective_local_units, "translation_bound_deg": args.translation_bound_deg, "regularization_scale_deg": args.regularization_scale_deg, "regularization_weight": args.regularization_weight, "tuning_used_for_fit": False, "split_half_skipped": args.skip_split_half},
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen V1 RF-size translation pilot written to {output_dir}")


if __name__ == "__main__":
    main()
