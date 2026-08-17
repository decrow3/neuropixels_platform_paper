#!/usr/bin/env python3
"""Split-half reliability of pooled pairwise residual-RF anisotropy in V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.check_v1_cross_animal_mean_map_support import (
    CCF_COLUMNS,
    DEFAULT_INPUT,
    DEFAULT_UNITS,
    RF_COLUMNS,
    load_population,
    make_block_table,
)
from scripts.check_v1_dispersion_physical_sampling import physical_blocks
from scripts.test_v1_rf_size_corroboration import nested_session_features


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_OUTPUT = CHECKPOINT / "pooled_pairwise_anisotropy_split_half"
CASES = {
    760345702: "trace point-localizing candidate",
    798911424: "trace annular-ambiguity candidate",
}
BANDS = {
    "all_0_300um": (0.0, 300.0),
    "near_0_100um": (0.0, 100.0),
    "middle_100_200um": (100.0, 200.0),
    "far_200_300um": (200.0, 300.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--physical-block-count", type=int, default=6)
    parser.add_argument("--rf-neighborhood-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--split-repeats", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pairwise_second_moment(
    residual: np.ndarray,
    ccf: np.ndarray,
    low_um: float,
    high_um: float,
) -> dict[str, float | complex]:
    """Length-weighted second angular moment of residual RF pair displacements."""
    n = len(residual)
    if n < 3:
        return {"moment": np.nan + 1j * np.nan, "pairs": 0, "cells": n}
    upper = np.triu_indices(n, 1)
    delta_residual = residual[:, None, :] - residual[None, :, :]
    delta_ccf = ccf[:, None, :] - ccf[None, :, :]
    distance_ccf = np.sqrt(np.sum(delta_ccf**2, axis=2))[upper]
    dx = delta_residual[:, :, 0][upper]
    dy = delta_residual[:, :, 1][upper]
    selected = (
        np.isfinite(distance_ccf)
        & np.isfinite(dx)
        & np.isfinite(dy)
        & (distance_ccf > low_um)
        & (distance_ccf <= high_um)
    )
    z = dx[selected] + 1j * dy[selected]
    denominator = np.sum(np.abs(z) ** 2)
    moment = np.sum(z**2) / denominator if denominator > 0 and len(z) else np.nan + 1j * np.nan
    return {"moment": moment, "pairs": int(len(z)), "cells": n}


def moment_fields(moment: complex) -> dict[str, float]:
    if not np.isfinite(moment.real) or not np.isfinite(moment.imag):
        return {
            "moment_real": np.nan,
            "moment_imag": np.nan,
            "anisotropy_magnitude": np.nan,
            "anisotropy_axis_deg": np.nan,
        }
    axis = .5 * np.degrees(np.angle(moment))
    return {
        "moment_real": float(moment.real),
        "moment_imag": float(moment.imag),
        "anisotropy_magnitude": float(abs(moment)),
        "anisotropy_axis_deg": float(axis),
    }


def axis_difference_deg(first: complex, second: complex) -> float:
    if not all(np.isfinite([first.real, first.imag, second.real, second.imag])):
        return np.nan
    return float(.5 * abs(np.degrees(np.angle(first * np.conj(second)))))


def stratified_halves(
    table: pd.DataFrame,
    block_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = physical_blocks(table["probe_vertical_position"], block_count)
    halves = [[], []]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        members = rng.permutation(members)
        offset = int(rng.integers(0, 2))
        halves[offset].extend(members[0::2])
        halves[1 - offset].extend(members[1::2])
    return np.asarray(halves[0], int), np.asarray(halves[1], int)


def randomized_orientations(residual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    length = np.sqrt(np.sum(residual**2, axis=1))
    angle = rng.uniform(-np.pi, np.pi, len(residual))
    return np.column_stack([length * np.cos(angle), length * np.sin(angle)])


def add_record(
    rows: list[dict],
    session_id: int,
    repeat: int,
    data_kind: str,
    half: str,
    band: str,
    estimate: dict,
) -> None:
    rows.append(
        {
            "ecephys_session_id": session_id,
            "repeat": repeat,
            "data_kind": data_kind,
            "half": half,
            "ccf_band": band,
            "cells": estimate["cells"],
            "pairs": estimate["pairs"],
            **moment_fields(estimate["moment"]),
        }
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = load_population(args.input.resolve(), args.unit_table.resolve())
    blocks = make_block_table(population, args.physical_block_count)
    usable = blocks.groupby("ecephys_session_id").size()
    usable = usable.index[usable >= 4]
    population = population.loc[population["ecephys_session_id"].isin(usable)].copy()
    blocks = blocks.loc[blocks["ecephys_session_id"].isin(usable)].copy()

    estimate_rows: list[dict] = []
    agreement_rows: list[dict] = []
    full_rows: list[dict] = []
    payload = {}

    for target_id, role in CASES.items():
        print(f"target {target_id}: {role}", flush=True)
        features, predictions = nested_session_features(
            target_id, population, blocks, args.ridge,
            args.rf_neighborhood_bandwidth_deg,
        )
        target = features[target_id].reset_index(drop=True)
        observed = target[list(RF_COLUMNS)].to_numpy(float)
        predicted = predictions[target_id]
        ccf = target[list(CCF_COLUMNS)].to_numpy(float)
        valid = (
            np.isfinite(observed).all(axis=1)
            & np.isfinite(predicted).all(axis=1)
            & np.isfinite(ccf).all(axis=1)
        )
        target = target.loc[valid].reset_index(drop=True)
        residual = observed[valid] - predicted[valid]
        ccf = ccf[valid]

        full_moments = {}
        for band, (low, high) in BANDS.items():
            estimate = pairwise_second_moment(residual, ccf, low, high)
            full_moments[band] = estimate["moment"]
            full_rows.append(
                {
                    "ecephys_session_id": target_id,
                    "role": role,
                    "ccf_band": band,
                    "cells": estimate["cells"],
                    "pairs": estimate["pairs"],
                    **moment_fields(estimate["moment"]),
                }
            )

        root = np.random.SeedSequence([20260816, target_id, 4])
        repeat_seeds = root.spawn(args.split_repeats)
        for repeat, seed in enumerate(repeat_seeds):
            rng = np.random.default_rng(seed)
            first, second = stratified_halves(target, args.physical_block_count, rng)
            null_residual = randomized_orientations(residual, rng)
            for band, (low, high) in BANDS.items():
                real_estimates = [
                    pairwise_second_moment(residual[index], ccf[index], low, high)
                    for index in (first, second)
                ]
                null_estimates = [
                    pairwise_second_moment(null_residual[index], ccf[index], low, high)
                    for index in (first, second)
                ]
                null_full = pairwise_second_moment(null_residual, ccf, low, high)
                for half, estimate in enumerate(real_estimates):
                    add_record(
                        estimate_rows, target_id, repeat, "observed", str(half), band, estimate
                    )
                for half, estimate in enumerate(null_estimates):
                    add_record(
                        estimate_rows, target_id, repeat, "orientation_null", str(half), band, estimate
                    )
                add_record(
                    estimate_rows, target_id, repeat, "orientation_null", "full", band, null_full
                )
                for data_kind, estimates in (
                    ("observed", real_estimates),
                    ("orientation_null", null_estimates),
                ):
                    first_moment, second_moment = estimates[0]["moment"], estimates[1]["moment"]
                    agreement_rows.append(
                        {
                            "ecephys_session_id": target_id,
                            "repeat": repeat,
                            "data_kind": data_kind,
                            "ccf_band": band,
                            "axis_difference_deg": axis_difference_deg(first_moment, second_moment),
                            "complex_moment_distance": float(abs(first_moment - second_moment)),
                            "magnitude_absolute_difference": float(
                                abs(abs(first_moment) - abs(second_moment))
                            ),
                            "cosine_double_angle": float(
                                np.cos(np.angle(first_moment * np.conj(second_moment)))
                            ),
                        }
                    )
        payload[target_id] = {
            "target": target,
            "residual": residual,
            "full_moments": full_moments,
        }

    estimates = pd.DataFrame(estimate_rows)
    agreements = pd.DataFrame(agreement_rows)
    full = pd.DataFrame(full_rows)

    metric_rows = []
    for (session_id, band), local_full in full.groupby(
        ["ecephys_session_id", "ccf_band"], observed=True
    ):
        observed_agreement = agreements.loc[
            agreements["ecephys_session_id"].eq(session_id)
            & agreements["ccf_band"].eq(band)
            & agreements["data_kind"].eq("observed")
        ]
        null_agreement = agreements.loc[
            agreements["ecephys_session_id"].eq(session_id)
            & agreements["ccf_band"].eq(band)
            & agreements["data_kind"].eq("orientation_null")
        ]
        null_half = estimates.loc[
            estimates["ecephys_session_id"].eq(session_id)
            & estimates["ccf_band"].eq(band)
            & estimates["data_kind"].eq("orientation_null")
            & estimates["half"].eq("full")
        ]
        full_magnitude = float(local_full["anisotropy_magnitude"].iloc[0])
        metric_rows.append(
            {
                "ecephys_session_id": int(session_id),
                "ccf_band": band,
                "full_anisotropy_magnitude": full_magnitude,
                "full_anisotropy_axis_deg": float(local_full["anisotropy_axis_deg"].iloc[0]),
                "full_pairs": int(local_full["pairs"].iloc[0]),
                "observed_median_axis_difference_deg": float(
                    observed_agreement["axis_difference_deg"].median()
                ),
                "observed_p90_axis_difference_deg": float(
                    observed_agreement["axis_difference_deg"].quantile(.90)
                ),
                "null_median_axis_difference_deg": float(
                    null_agreement["axis_difference_deg"].median()
                ),
                "observed_fraction_axis_within_15deg": float(
                    observed_agreement["axis_difference_deg"].le(15).mean()
                ),
                "null_fraction_axis_within_15deg": float(
                    null_agreement["axis_difference_deg"].le(15).mean()
                ),
                "observed_median_complex_distance": float(
                    observed_agreement["complex_moment_distance"].median()
                ),
                "null_median_complex_distance": float(
                    null_agreement["complex_moment_distance"].median()
                ),
                "orientation_null_magnitude_p": float(
                    (1 + np.sum(null_half["anisotropy_magnitude"].to_numpy(float) >= full_magnitude))
                    / (1 + len(null_half))
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    full.to_csv(output / "pooled_pairwise_full_data_moments.csv", index=False)
    estimates.to_csv(output / "pooled_pairwise_split_half_estimates.csv.gz", index=False, compression="gzip")
    agreements.to_csv(output / "pooled_pairwise_split_half_agreement.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output / "pooled_pairwise_split_half_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(21, 8.4))
    band = "all_0_300um"
    for row, (session_id, role) in enumerate(CASES.items()):
        local = payload[session_id]
        residual = local["residual"]
        full_moment = local["full_moments"][band]
        ax = axes[row, 0]
        ax.scatter(residual[:, 0], residual[:, 1], s=22, alpha=.65, color="#4477aa")
        scale = np.nanquantile(np.sqrt(np.sum(residual**2, axis=1)), .9)
        axis_angle = .5 * np.angle(full_moment)
        direction = np.array([np.cos(axis_angle), np.sin(axis_angle)]) * scale
        ax.plot([-direction[0], direction[0]], [-direction[1], direction[1]], color="#cc3311", lw=3)
        ax.axhline(0, color="0.7", lw=.6)
        ax.axvline(0, color="0.7", lw=.6)
        ax.set(
            xlabel="residual RF azimuth", ylabel="residual RF elevation", aspect="equal",
            title=f"{session_id}: {role}\ncell residuals; full |A2|={abs(full_moment):.2f}",
        )

        ax = axes[row, 1]
        local_est = estimates.loc[
            estimates["ecephys_session_id"].eq(session_id)
            & estimates["ccf_band"].eq(band)
            & estimates["data_kind"].eq("observed")
        ]
        for half_label, color in (("0", "#4477aa"), ("1", "#ee7733")):
            half = local_est.loc[local_est["half"].eq(half_label)]
            ax.scatter(half["moment_real"], half["moment_imag"], s=10, alpha=.18, color=color, label=f"half {half_label}")
        ax.scatter(full_moment.real, full_moment.imag, marker="*", s=130, color="black")
        circle = plt.Circle((0, 0), 1, fill=False, color="0.7", lw=.7)
        ax.add_patch(circle)
        ax.axhline(0, color="0.7", lw=.5)
        ax.axvline(0, color="0.7", lw=.5)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xlabel="Re(A2)", ylabel="Im(A2)", title="split halves in double-angle space")

        local_agree = agreements.loc[
            agreements["ecephys_session_id"].eq(session_id)
            & agreements["ccf_band"].eq(band)
        ]
        ax = axes[row, 2]
        bins = np.linspace(0, 90, 25)
        for kind, color, label in (
            ("observed", "#4477aa", "observed"),
            ("orientation_null", "0.55", "orientation null"),
        ):
            values = local_agree.loc[local_agree["data_kind"].eq(kind), "axis_difference_deg"]
            ax.hist(values, bins=bins, density=True, histtype="step", lw=2, color=color, label=label)
        ax.set(xlabel="half-to-half axis difference (deg)", ylabel="density", title="axis repeatability")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[row, 3]
        pivot = local_est.pivot(index="repeat", columns="half", values="anisotropy_magnitude")
        ax.scatter(pivot["0"], pivot["1"], s=12, alpha=.25, color="#4477aa")
        upper = max(float(pivot.max().max()), .25)
        ax.plot([0, upper], [0, upper], color="0.5", ls="--")
        ax.set(xlim=(0, upper), ylim=(0, upper), aspect="equal", xlabel="half 0 |A2|", ylabel="half 1 |A2|", title="magnitude repeatability")

        ax = axes[row, 4]
        local_metric = metrics.loc[metrics["ecephys_session_id"].eq(session_id)].set_index("ccf_band").loc[list(BANDS)]
        x = np.arange(len(local_metric))
        ax.plot(x, local_metric["observed_median_axis_difference_deg"], marker="o", color="#4477aa", label="observed")
        ax.plot(x, local_metric["null_median_axis_difference_deg"], marker="o", color="0.55", label="orientation null")
        ax.set(xticks=x, xticklabels=["0–300", "0–100", "100–200", "200–300"], xlabel="CCF pair-distance band (um)", ylabel="median half-axis difference (deg)", ylim=(0, 90), title="dependence on anatomical separation")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Pooled pairwise residual-RF anisotropy: physical-block-stratified split-half reliability\n"
        "Pairs define A2 but cells—not pairs—are randomized; CCF-to-RF mean maps held fixed",
        y=.998,
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    figure_path = output / "Figure_v1_pooled_pairwise_anisotropy_split_half.png"
    fig.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "two-case pooled-pairwise anisotropy reliability checkpoint",
        "estimand": "A2=sum((daz+i*del)^2)/sum(daz^2+del^2) for residual-RF cell-pair differences",
        "residual": "observed RF minus nested leave-one-animal-out CCF-to-RF mean-map prediction",
        "split": "random cell halves stratified within six physical probe blocks",
        "null": "independent random rotation of each cell residual vector, preserving its magnitude",
        "inference_unit": "cell/physical block; pair counts are descriptive only",
        "split_repeats": args.split_repeats,
        "ccf_distance_bands_um": BANDS,
        "outputs": [
            figure_path.name,
            "pooled_pairwise_full_data_moments.csv",
            "pooled_pairwise_split_half_estimates.csv.gz",
            "pooled_pairwise_split_half_agreement.csv.gz",
            "pooled_pairwise_split_half_metrics.csv",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(figure_path)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
