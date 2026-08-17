#!/usr/bin/env python3
"""Build Allen BO 1.1 session SF/TF maps with tuning-quality unit weights.

The released unit table has no split-half tuning reliability estimate.  The
weights therefore combine lifetime sparseness (tuning strength), a saturating
stimulus firing-rate term (response evidence), and inverse Fano factor (a
trial-variability proxy).  Weights are normalized separately within each
session, map, and displayed area group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.allen_frequency_preference_surfaces import (
    DEFAULT_AUDIT,
    PREFERENCES,
    load_preference_units,
)
from scripts.render_allen_bo11_simultaneous_v1_hva_session_maps import (
    group_definitions,
    simultaneous_v1_hva_sessions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DEFAULT_AUDIT / "tuning_weighted_session_surfaces"
METRICS = {
    "sf": ("lifetime_sparseness_sg", "firing_rate_sg", "fano_sg"),
    "tf": ("lifetime_sparseness_dg", "firing_rate_dg", "fano_dg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--minimum-effective-local-units", type=float, default=10.0)
    parser.add_argument("--minimum-lifetime-sparseness", type=float, default=0.1)
    parser.add_argument("--minimum-stimulus-firing-rate", type=float, default=0.1)
    parser.add_argument("--rate-half-saturation-hz", type=float, default=1.0)
    parser.add_argument("--weight-floor", type=float, default=0.25)
    parser.add_argument("--weight-ceiling", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tuning_quality_components(
    sparseness: np.ndarray,
    firing_rate: np.ndarray,
    fano: np.ndarray,
    *,
    minimum_sparseness: float = 0.1,
    rate_half_saturation_hz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return strength, response evidence, stability proxy, and raw weight."""
    sparseness = np.asarray(sparseness, dtype=float)
    firing_rate = np.asarray(firing_rate, dtype=float)
    fano = np.asarray(fano, dtype=float)
    scale = max(1.0 - float(minimum_sparseness), 1e-6)
    strength = np.clip((sparseness - float(minimum_sparseness)) / scale, 0.0, 1.0)
    rate = np.clip(firing_rate, 0.0, None)
    response_evidence = rate / (rate + float(rate_half_saturation_hz))
    stability_proxy = 1.0 / (1.0 + np.clip(fano, 0.0, None))
    components = np.column_stack([strength, response_evidence, stability_proxy])
    components = np.clip(components, 0.05, 1.0)
    raw_weight = np.exp(np.mean(np.log(components), axis=1))
    raw_weight[~np.isfinite(raw_weight)] = np.nan
    return strength, response_evidence, stability_proxy, raw_weight


def normalize_unit_weights(
    raw_weight: np.ndarray,
    *,
    floor: float = 0.25,
    ceiling: float = 4.0,
) -> np.ndarray:
    """Normalize mean contribution to one, winsorize, and renormalize."""
    raw_weight = np.asarray(raw_weight, dtype=float)
    finite = np.isfinite(raw_weight) & (raw_weight > 0)
    result = np.full(raw_weight.shape, np.nan)
    if not finite.any():
        return result
    normalized = raw_weight[finite] / np.mean(raw_weight[finite])
    normalized = np.clip(normalized, float(floor), float(ceiling))
    normalized /= np.mean(normalized)
    result[finite] = normalized
    return result


def weighted_gaussian_surface(
    points: np.ndarray,
    values: np.ndarray,
    unit_weights: np.ndarray,
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_local_units: float,
) -> dict[str, np.ndarray]:
    """Gaussian mean with weighted Kish effective local sample size."""
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    unit_weights = np.asarray(unit_weights, dtype=float)
    grid_points = np.asarray(grid_points, dtype=float)
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(values)
        & np.isfinite(unit_weights)
        & (unit_weights > 0)
    )
    points, values, unit_weights = points[finite], values[finite], unit_weights[finite]
    if not len(values):
        empty = np.full(len(grid_points), np.nan)
        return {
            "estimate_log2": empty,
            "effective_local_units": np.zeros(len(grid_points)),
            "supported": np.zeros(len(grid_points), dtype=bool),
        }
    distance_squared = np.sum((grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-0.5 * distance_squared / float(bandwidth_deg) ** 2)
    weights = kernel * unit_weights[None, :]
    denominator = weights.sum(axis=1)
    estimate = np.divide(
        weights @ values,
        denominator,
        out=np.full(len(grid_points), np.nan),
        where=denominator > 0,
    )
    local = distance_squared <= (1.5 * float(bandwidth_deg)) ** 2
    local_weights = local * unit_weights[None, :]
    local_sum = local_weights.sum(axis=1)
    local_square_sum = np.square(local_weights).sum(axis=1)
    effective = np.divide(
        np.square(local_sum),
        local_square_sum,
        out=np.zeros(len(grid_points)),
        where=local_square_sum > 0,
    )
    supported = effective >= float(minimum_effective_local_units)
    estimate[~supported] = np.nan
    return {"estimate_log2": estimate, "effective_local_units": effective, "supported": supported}


def build_maps_and_audit(
    units: pd.DataFrame,
    sessions: list[int],
    grid_points: np.ndarray,
    *,
    bandwidth_deg: float,
    minimum_effective_local_units: float,
    minimum_sparseness: float,
    rate_half_saturation_hz: float,
    weight_floor: float,
    weight_ceiling: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for session_id in sessions:
        session = units.loc[units["ecephys_session_id"].eq(session_id)]
        specimen_id = int(session["specimen_id"].iloc[0])
        for group_label, group_mask in group_definitions(session):
            for preference, specification in PREFERENCES.items():
                group = session.loc[group_mask & session[f"tuning_eligible_{preference}"]].copy()
                sparseness, rate, fano = METRICS[preference]
                strength, response, stability, raw = tuning_quality_components(
                    group[sparseness].to_numpy(float),
                    group[rate].to_numpy(float),
                    group[fano].to_numpy(float),
                    minimum_sparseness=minimum_sparseness,
                    rate_half_saturation_hz=rate_half_saturation_hz,
                )
                weights = normalize_unit_weights(raw, floor=weight_floor, ceiling=weight_ceiling)
                group["tuning_strength"] = strength
                group["response_evidence"] = response
                group["stability_proxy"] = stability
                group["raw_tuning_quality_weight"] = raw
                group["normalized_unit_weight"] = weights
                group["map"] = preference
                group["group"] = group_label
                audits.append(
                    group[
                        [
                            "ecephys_unit_id", "ecephys_session_id", "specimen_id", "area",
                            "group", "map", sparseness, rate, fano, "tuning_strength",
                            "response_evidence", "stability_proxy", "raw_tuning_quality_weight",
                            "normalized_unit_weight",
                        ]
                    ].rename(columns={sparseness: "lifetime_sparseness", rate: "stimulus_firing_rate_hz", fano: "fano_factor"})
                )
                surface = weighted_gaussian_surface(
                    group[["azimuth_rf", "elevation_rf"]].to_numpy(float),
                    np.log2(group[specification["column"]].to_numpy(float)),
                    weights,
                    grid_points,
                    bandwidth_deg=bandwidth_deg,
                    minimum_effective_local_units=minimum_effective_local_units,
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "ecephys_session_id": session_id,
                            "specimen_id": specimen_id,
                            "group": group_label,
                            "map": preference,
                            "azimuth_deg": grid_points[:, 0],
                            "elevation_deg": grid_points[:, 1],
                            "estimate": np.exp2(surface["estimate_log2"]),
                            "estimate_log2": surface["estimate_log2"],
                            "local_units": surface["effective_local_units"],
                            "supported": surface["supported"],
                            "source_units": len(group),
                            "source_effective_units": np.square(np.nansum(weights)) / np.nansum(np.square(weights)),
                            "source_probes": group["ecephys_probe_id"].nunique(),
                            "source_areas": "+".join(sorted(group["area"].unique())),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True), pd.concat(audits, ignore_index=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    units, unit_path, audit_manifest = load_preference_units(
        args.audit_dir.resolve(),
        args.config,
        minimum_lifetime_sparseness=args.minimum_lifetime_sparseness,
        minimum_stimulus_firing_rate=args.minimum_stimulus_firing_rate,
        require_unique_preference=True,
    )
    sessions = simultaneous_v1_hva_sessions(units)
    units = units.loc[units["ecephys_session_id"].isin(sessions)].copy()
    az_grid = np.linspace(10.0, 90.0, args.grid_size)
    el_grid = np.linspace(-30.0, 50.0, args.grid_size)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    grid_points = np.column_stack([az_mesh.ravel(), el_mesh.ravel()])
    maps, audit = build_maps_and_audit(
        units,
        sessions,
        grid_points,
        bandwidth_deg=args.bandwidth_deg,
        minimum_effective_local_units=args.minimum_effective_local_units,
        minimum_sparseness=args.minimum_lifetime_sparseness,
        rate_half_saturation_hz=args.rate_half_saturation_hz,
        weight_floor=args.weight_floor,
        weight_ceiling=args.weight_ceiling,
    )
    grid_path = output_dir / "allen_bo11_tuning_weighted_surface_grid.csv"
    audit_path = output_dir / "allen_bo11_tuning_weight_audit.csv"
    summary_path = output_dir / "allen_bo11_tuning_weight_summary.csv"
    maps.to_csv(grid_path, index=False, float_format="%.6g")
    audit.to_csv(audit_path, index=False, float_format="%.6g")
    summary = (
        audit.groupby(["group", "map"], observed=True)
        .agg(
            unit_rows=("ecephys_unit_id", "size"),
            unique_units=("ecephys_unit_id", "nunique"),
            median_tuning_strength=("tuning_strength", "median"),
            median_stability_proxy=("stability_proxy", "median"),
            weight_p10=("normalized_unit_weight", lambda x: x.quantile(0.1)),
            weight_median=("normalized_unit_weight", "median"),
            weight_p90=("normalized_unit_weight", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False, float_format="%.6g")
    report = [
        "# Allen BO 1.1 tuning-quality-weighted session surfaces",
        "",
        f"Built SF/TF maps for {len(sessions)} simultaneous V1/HVA sessions. Eligible units",
        "retain the existing selectivity, response-rate, and unique-preference gates.",
        "",
        "Each unit weight is the geometric mean of lifetime-sparseness tuning strength,",
        "saturating stimulus response rate, and `1/(1 + Fano factor)`. The Fano term is",
        "only a trial-variability proxy: it is not split-half tuning reliability. Weights",
        "are clipped and renormalized to mean one within session × group × SF/TF, so this",
        "changes relative unit influence without changing a map's total nominal weight.",
        "",
        "Grid support uses weighted Kish effective local unit count within 1.5 bandwidths.",
        "The resulting grid has the same contract as the equal-unit affine pilot input.",
    ]
    (output_dir / "ALLEN_BO11_TUNING_WEIGHTED_SESSION_SURFACES.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_tuning_weighted_session_surfaces",
        "status": "tuning-strength and variability-proxy weighted session surfaces",
        "input": {"path": str(unit_path.resolve()), "sha256": sha256(unit_path.resolve())},
        "audit_manifest_sha256": hashlib.sha256(json.dumps(audit_manifest, sort_keys=True).encode()).hexdigest(),
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "sessions": sessions,
            "bandwidth_deg": args.bandwidth_deg,
            "grid_size": args.grid_size,
            "minimum_effective_local_units": args.minimum_effective_local_units,
            "minimum_lifetime_sparseness": args.minimum_lifetime_sparseness,
            "minimum_stimulus_firing_rate": args.minimum_stimulus_firing_rate,
            "rate_half_saturation_hz": args.rate_half_saturation_hz,
            "weight_floor": args.weight_floor,
            "weight_ceiling": args.weight_ceiling,
            "require_unique_preference": True,
            "reliability_metric": "inverse Fano factor; trial-variability proxy, not split-half tuning reliability",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen tuning-weighted session surfaces written to {output_dir}")


if __name__ == "__main__":
    main()
