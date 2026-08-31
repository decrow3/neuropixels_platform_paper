#!/usr/bin/env python3
"""Render a reproducible contact sheet of reliable Allen V1 receptive fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from scripts.build_allen_v1_rf_size_vs_rf_location import load_population


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FITS = (
    ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/"
    "all_session_unit_geometry_fits.csv"
)
DEFAULT_UNITS = ROOT / "data/unit_table.csv"
DEFAULT_RELIABILITY = (
    ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/"
    "allen_rf_split_half_reliability.csv.gz"
)
DEFAULT_CACHE = ROOT / "artifacts/allen_full_rf_production_v1/01_compact_cache"
DEFAULT_OUTPUT = ROOT / "artifacts/v1_rf_contact_sheet_random100_reliability_gt_0p7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-split-half-reliability", type=float, default=0.7)
    parser.add_argument("--edge-exclusion-deg", type=float, default=10.0)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--columns", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_observed_maps(
    selected: pd.DataFrame, cache_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool all Gabor orientations/repeats into mean counts at each x/y cell."""
    maps: dict[int, np.ndarray] = {}
    canonical_x: np.ndarray | None = None
    canonical_y: np.ndarray | None = None
    for session_id, session_units in selected.groupby("ecephys_session_id", sort=True):
        cache = cache_root / f"session_{int(session_id)}"
        population = pd.read_csv(
            cache / "visual_unit_population.csv",
            usecols=["ecephys_unit_id"],
            low_memory=False,
        )
        trials = pd.read_csv(
            cache / "gabor_trial_gaze_table.csv",
            usecols=["x_position", "y_position"],
            low_memory=False,
        )
        with np.load(cache / "gabor_spike_counts.npz") as payload:
            counts = payload["counts"]
            cached_ids = payload["unit_ids"].astype(int)
        population_ids = population["ecephys_unit_id"].to_numpy(int)
        if not np.array_equal(cached_ids, population_ids):
            raise ValueError(f"Session {session_id}: spike-count unit order mismatch")

        x_values = np.sort(trials["x_position"].unique().astype(float))
        y_values = np.sort(trials["y_position"].unique().astype(float))
        if canonical_x is None:
            canonical_x, canonical_y = x_values, y_values
        elif not np.array_equal(x_values, canonical_x) or not np.array_equal(y_values, canonical_y):
            raise ValueError(f"Session {session_id}: RF position grid differs from prior sessions")

        x_lookup = {value: index for index, value in enumerate(x_values)}
        y_lookup = {value: index for index, value in enumerate(y_values)}
        x_code = trials["x_position"].map(x_lookup).to_numpy(int)
        y_code = trials["y_position"].map(y_lookup).to_numpy(int)
        cell_code = y_code * len(x_values) + x_code
        trial_count = np.bincount(cell_code, minlength=len(x_values) * len(y_values))
        row_lookup = {unit_id: row for row, unit_id in enumerate(cached_ids)}
        for unit_id in session_units["ecephys_unit_id"].astype(int):
            totals = np.bincount(
                cell_code,
                weights=counts[row_lookup[unit_id]].astype(float),
                minlength=len(x_values) * len(y_values),
            )
            maps[unit_id] = (totals / trial_count).reshape(len(y_values), len(x_values))

    unit_order = selected["ecephys_unit_id"].astype(int).to_numpy()
    return np.stack([maps[unit_id] for unit_id in unit_order]), canonical_x, canonical_y


def render_contact_sheet(
    selected: pd.DataFrame,
    maps: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    output: Path,
    columns: int,
    threshold: float,
    seed: int,
) -> None:
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.55 * columns, 2.65 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()
    half_max_scale = np.sqrt(2.0 * np.log(2.0))
    extent = [
        x_values.min() - 5,
        x_values.max() + 5,
        y_values.min() - 5,
        y_values.max() + 5,
    ]
    for index, (axis, (_, unit), matrix) in enumerate(zip(axes, selected.iterrows(), maps)):
        low, high = np.quantile(matrix, [0.02, 0.98])
        if not high > low:
            high = low + 1.0
        axis.imshow(
            matrix,
            origin="lower",
            interpolation="nearest",
            extent=extent,
            cmap="magma",
            vmin=low,
            vmax=high,
            aspect="equal",
        )
        ellipse = Ellipse(
            (unit.axis_center_x_deg, unit.axis_center_y_deg),
            width=2 * half_max_scale * unit.axis_sigma_x_deg,
            height=2 * half_max_scale * unit.axis_sigma_y_deg,
            angle=0,
            fill=False,
            edgecolor="cyan",
            linewidth=1.15,
            linestyle="--",
        )
        axis.add_patch(ellipse)
        axis.plot(
            unit.axis_center_x_deg,
            unit.axis_center_y_deg,
            marker="+",
            color="cyan",
            markersize=5,
            markeredgewidth=1.0,
        )
        axis.set_title(
            f"ID {int(unit.ecephys_unit_id)}\n"
            f"r={unit.rf_split_half_spearman_brown:.2f}  A={unit.axis_area_deg2:.0f}°²",
            fontsize=6.2,
            pad=2,
        )
        axis.set_xticks([-40, 0, 40])
        axis.set_yticks([-40, 0, 40])
        axis.tick_params(labelsize=5, length=2, pad=1)
        if index // columns == rows - 1:
            axis.set_xlabel("azimuth (°)", fontsize=6)
        if index % columns == 0:
            axis.set_ylabel("elevation (°)", fontsize=6)
    for axis in axes[len(selected):]:
        axis.set_visible(False)

    fig.suptitle(
        f"Random sample of {len(selected)} Allen V1 receptive fields with split-half reliability > {threshold:g}\n"
        "Observed mean spike count on the 9×9 Gabor-position grid; cyan dashed ellipse = fitted latent half-maximum",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        f"Uniform unit-level sample without replacement; seed={seed}. "
        "Each tile is independently scaled to its 2nd–98th response percentiles.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.025, top=0.955, wspace=0.12, hspace=0.30)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fits = args.fits.resolve()
    units = args.units.resolve()
    reliability = args.reliability.resolve()
    cache_root = args.cache_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible, audit = load_population(
        fits,
        units,
        args.edge_exclusion_deg,
        reliability,
        args.minimum_split_half_reliability,
    )
    if args.sample_size > len(eligible):
        raise ValueError(f"Requested {args.sample_size} RFs from only {len(eligible)} eligible units")
    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(len(eligible), size=args.sample_size, replace=False)
    selected = eligible.iloc[selected_indices].copy().reset_index(drop=True)
    selected.insert(0, "sheet_position", np.arange(1, len(selected) + 1))

    maps, x_values, y_values = extract_observed_maps(selected, cache_root)
    display_low = np.quantile(maps, 0.02, axis=(1, 2))
    display_high = np.quantile(maps, 0.98, axis=(1, 2))
    selected["observed_mean_count_min"] = maps.min(axis=(1, 2))
    selected["observed_mean_count_max"] = maps.max(axis=(1, 2))
    selected["display_count_p02"] = display_low
    selected["display_count_p98"] = display_high

    figure_path = output_dir / "Figure_random100_v1_rf_contact_sheet_reliability_gt_0p7.png"
    render_contact_sheet(
        selected,
        maps,
        x_values,
        y_values,
        figure_path,
        args.columns,
        args.minimum_split_half_reliability,
        args.seed,
    )
    selection_path = output_dir / "selected_rf_units.csv"
    selected.to_csv(selection_path, index=False, float_format="%.8g")
    maps_path = output_dir / "selected_observed_rf_maps.npz"
    np.savez_compressed(
        maps_path,
        unit_ids=selected["ecephys_unit_id"].to_numpy(int),
        x_positions_deg=x_values,
        y_positions_deg=y_values,
        mean_spike_counts=maps,
        display_count_p02=display_low,
        display_count_p98=display_high,
    )

    manifest = {
        "status": "exploratory concrete-case audit",
        "selection": {
            "eligible_pool": int(len(eligible)),
            "sample_size": int(len(selected)),
            "sampling_unit": "ecephys unit",
            "sampling_method": "uniform without replacement",
            "seed": args.seed,
            "minimum_split_half_spearman_brown_exclusive": args.minimum_split_half_reliability,
            "edge_exclusion_deg": args.edge_exclusion_deg,
            "eligibility_audit": audit,
        },
        "observed_map": {
            "quantity": "mean spike count pooled across all Gabor orientations and repeats at each x/y cell",
            "grid_shape": list(maps.shape[1:]),
            "display_scaling": "independent per tile, 2nd to 98th percentile",
            "overlay": "axis-aligned aperture-model latent Gaussian half-maximum ellipse",
        },
        "inputs": {
            "fits": str(fits),
            "fits_sha256": sha256(fits),
            "units": str(units),
            "units_sha256": sha256(units),
            "reliability": str(reliability),
            "reliability_sha256": sha256(reliability),
            "cache_root": str(cache_root),
        },
        "outputs": {
            "figure": str(figure_path),
            "selection_table": str(selection_path),
            "observed_maps": str(maps_path),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Random 100 Allen V1 receptive fields, reliability > {args.minimum_split_half_reliability:g}

This is a concrete-case audit of the unit-level RFs used by the RF-location analysis. It is not a population summary.

- Eligible pool: **{len(eligible):,}** V1 aperture RFs after fit-bound, edge, and strict reliability filters.
- Selection: **{len(selected)} units sampled uniformly without replacement**, seed `{args.seed}`.
- Observed image: mean spike count at each of the 81 Gabor positions, pooled across orientations and repeats.
- Display: each tile is independently scaled to its 2nd–98th response percentiles.
- Overlay: cyan dashed latent half-maximum ellipse and center from the axis-aligned aperture fit.
- Full identifiers and metrics: `selected_rf_units.csv`.
- Exact observed matrices: `selected_observed_rf_maps.npz`.

## Stimulus phase provenance

All Gabors began at phase 0 cycles and traversed 15 frame phases from 0 through
14/15 cycles (0 through 336 degrees). The NWB value
`[3644.93333333, 3644.93333333]` is the final shared unwrapped phase state, not
the onset phase of each presentation. See the full derivation in
[`docs/ALLEN_RF_MAPPING_STIMULUS.md`](../../docs/ALLEN_RF_MAPPING_STIMULUS.md).

## Reproduce

```bash
python -m scripts.render_allen_v1_rf_contact_sheet --minimum-split-half-reliability {args.minimum_split_half_reliability:g} --sample-size {args.sample_size} --seed {args.seed} --output-dir {output_dir}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest["selection"], indent=2))
    print(figure_path)


if __name__ == "__main__":
    main()
