#!/usr/bin/env python3
"""Compare RF-size translation bounds using cross-half predictive performance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.allen_bo11_tuning_driven_limited_affine import map_agreement, template_from_maps, warp_map
from scripts.allen_bo11_v1_rf_size_translation_alignment import (
    build_session_maps,
    fit_translations,
    packed_translation,
)
from scripts.render_allen_bo11_registration_comparison import DEFAULT_SURFACE_GRID
from scripts.allen_bo11_tuning_driven_limited_affine import load_maps
from scripts.render_allen_bo11_v1_rf_size_interior import DEFAULT_INPUT, prepare_population


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
OUTPUT = AUDIT / "v1_rf_size_translation_bound_crossvalidation"
BOUNDS = (10.0, 15.0, 20.0, 30.0)
EDGE_EXCLUSION_DEG = 20.0
BANDWIDTH_DEG = 8.0
REGULARIZATION_SCALE_DEG = 10.0
REGULARIZATION_WEIGHT = 0.08


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_transform(
    source: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    translation: np.ndarray,
    az_grid: np.ndarray,
    el_grid: np.ndarray,
) -> dict[str, float]:
    return map_agreement(
        warp_map(source, packed_translation(translation), az_grid, el_grid),
        template,
        50,
    )


def render(scores: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    rng = np.random.default_rng(20260815)
    for index, bound in enumerate(BOUNDS):
        local = scores.loc[scores["bound_deg"].eq(bound)]
        session = local.groupby("ecephys_session_id", observed=True)["heldout_correlation_gain"].mean()
        axes[0].scatter(
            np.repeat(bound, len(session)) + rng.normal(0, .22, len(session)),
            session,
            s=22,
            alpha=.28,
            color="#4c78a8",
        )
        median = session.median()
        q25, q75 = session.quantile([.25, .75])
        axes[0].plot([bound - .7, bound + .7], [median, median], color="#a13d2d", linewidth=2.5)
        axes[0].plot([bound, bound], [q25, q75], color="#a13d2d", linewidth=2)
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Translation search bound (deg)", ylabel="Held-out-half correlation gain vs identity", title="Predictive RF-size alignment")

    axes[1].plot(summary["bound_deg"], summary["median_heldout_gain"], marker="o", linewidth=2, label="Median gain")
    axes[1].fill_between(summary["bound_deg"], summary["q25_heldout_gain"], summary["q75_heldout_gain"], alpha=.18)
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Translation search bound (deg)", ylabel="Held-out-half correlation gain", title="Same performance scale across bounds")
    twin = axes[1].twinx()
    twin.plot(summary["bound_deg"], summary["fraction_sessions_positive"], marker="s", color="#f58518", linewidth=2, label="Sessions positive")
    twin.set(ylabel="Fraction of sessions with positive gain", ylim=(0, 1))
    handles, labels = axes[1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + twin_handles, labels + twin_labels, frameon=False, loc="best")
    for ax in axes:
        ax.grid(alpha=.18)
    figure.suptitle("Allen BO 1.1 RF-size registration: fair cross-half bound comparison", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tuning_maps, _, _ = load_maps(DEFAULT_SURFACE_GRID)
    sessions = sorted({key[0] for key in tuning_maps})
    population = prepare_population(pd.read_csv(DEFAULT_INPUT, low_memory=False))
    population = population.loc[
        population["ecephys_session_id"].isin(sessions)
        & population["distance_to_nearest_grid_edge_deg"].ge(EDGE_EXCLUSION_DEG)
    ].copy()
    population["split_half"] = 0
    for session_id, indices in population.groupby("ecephys_session_id", observed=True).groups.items():
        rng = np.random.default_rng(20260814 + int(session_id))
        shuffled = np.asarray(list(indices))[rng.permutation(len(indices))]
        population.loc[shuffled[len(shuffled) // 2 :], "split_half"] = 1
    az_grid = np.linspace(30, 70, 31)
    el_grid = np.linspace(-10, 30, 31)
    half_maps = {}
    for half in (0, 1):
        half_maps[half], _ = build_session_maps(
            population.loc[population["split_half"].eq(half)],
            sessions,
            az_grid,
            el_grid,
            bandwidth_deg=BANDWIDTH_DEG,
            minimum_effective_local_units=3.0,
        )
    rows = []
    for bound in BOUNDS:
        fitted = {}
        for training_half in (0, 1):
            fitted[training_half], _ = fit_translations(
                half_maps[training_half],
                az_grid,
                el_grid,
                bound_deg=bound,
                regularization_scale_deg=REGULARIZATION_SCALE_DEG,
                regularization_weight=REGULARIZATION_WEIGHT,
                minimum_points=50,
            )
            fitted[training_half] = fitted[training_half].set_index("ecephys_session_id")
        for training_half, test_half in ((0, 1), (1, 0)):
            for session_id in sessions:
                source = half_maps[test_half][(session_id, "V1", "rf_size")]
                template = template_from_maps(
                    half_maps[test_half], "V1", "rf_size", exclude_session=session_id
                )
                identity = score_transform(source, template, np.zeros(2), az_grid, el_grid)
                translation = fitted[training_half].loc[
                    session_id, ["translation_azimuth_deg", "translation_elevation_deg"]
                ].to_numpy(float)
                aligned = score_transform(source, template, translation, az_grid, el_grid)
                rows.append(
                    {
                        "bound_deg": bound,
                        "ecephys_session_id": session_id,
                        "training_half": training_half,
                        "test_half": test_half,
                        "translation_azimuth_deg": translation[0],
                        "translation_elevation_deg": translation[1],
                        "identity_heldout_correlation": identity["correlation"],
                        "aligned_heldout_correlation": aligned["correlation"],
                        "heldout_correlation_gain": aligned["correlation"] - identity["correlation"],
                        "identity_heldout_rmse": identity["rmse"],
                        "aligned_heldout_rmse": aligned["rmse"],
                    }
                )
    scores = pd.DataFrame(rows)
    session_scores = (
        scores.groupby(["bound_deg", "ecephys_session_id"], observed=True)["heldout_correlation_gain"]
        .mean().reset_index()
    )
    summary_rows = []
    for bound, group in session_scores.groupby("bound_deg", observed=True):
        nonzero = group["heldout_correlation_gain"].dropna()
        test = wilcoxon(nonzero) if len(nonzero) and np.any(nonzero != 0) else None
        summary_rows.append(
            {
                "bound_deg": bound,
                "sessions": len(nonzero),
                "median_heldout_gain": nonzero.median(),
                "q25_heldout_gain": nonzero.quantile(.25),
                "q75_heldout_gain": nonzero.quantile(.75),
                "fraction_sessions_positive": np.mean(nonzero > 0),
                "wilcoxon_p": test.pvalue if test else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    scores.to_csv(OUTPUT / "cross_half_bound_scores.csv", index=False, float_format="%.6g")
    summary.to_csv(OUTPUT / "cross_half_bound_summary.csv", index=False, float_format="%.6g")
    figure_path = OUTPUT / "Figure_allen_bo11_v1_rf_size_cross_half_bound_comparison.png"
    render(scores, summary, figure_path)
    lines = [
        "# Allen BO 1.1 RF-size translation bound cross-validation",
        "",
        "Translations are fitted using one random unit half and scored on the other half's RF-size surface; both directions are averaged within session.",
        "All bounds use the same fixed per-degree regularization and the same held-out correlation-gain scale.",
        "This comparison is unaffected by the smaller numerical offset span available to narrow bounds.",
        "",
        "| Bound | Median held-out Δr | Positive sessions | Wilcoxon p |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| ±{row.bound_deg:g}° | {row.median_heldout_gain:+.3f} | {row.fraction_sessions_positive:.0%} | {row.wilcoxon_p:.3g} |")
    (OUTPUT / "ALLEN_BO11_V1_RF_SIZE_BOUND_CROSSVALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
