#!/usr/bin/env python3
"""Place HVA-to-HVA TTFS separation on a within- to between-area scale.

All three landmarks use the same estimand: an unbiased full-cell estimate of
the squared difference between two population means.  Cell-level sampling
variance is removed with a two-sample U statistic.  Pair distances are averaged
within session before sessions receive equal weight.

Landmarks
---------
within_v1
    Distinct MouseV2 V1 probe/location populations recorded in one session.
v1_to_hva
    Allen VISp and HVA populations recorded in one session.
hva_to_hva
    Distinct Allen HVA populations recorded in one session.

The normalized HVA position is

    (D_hva_to_hva - D_within_v1) / (D_v1_to_hva - D_within_v1).

Allen sessions must contain VISp and at least two retained HVAs, so the two
Allen landmarks use exactly the same session cohort.  Whole sessions are
resampled; MouseV2 and Allen session sets are resampled independently.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure3_robust_spread_comparison import AREA_ORDER  # noqa: E402

CORTICAL_HVA_ORDER = tuple(area for area in AREA_ORDER if area != "LP")

TTFS_BASE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance"
MOUSE_UNITS = TTFS_BASE / "figure3_response_filtered_preferred_ttfs_units.csv"
ALLEN_UNITS = TTFS_BASE / "figure3_response_filtered_ttfs_all_areas_unit_audit.csv"
CURRENT_TTFS_MEANS = TTFS_BASE / "figure3_response_filtered_ttfs_override.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/figure3/07_big_picture_concrete_first/landmark_scale/ttfs_cortical_hvas"
LANDMARK_ORDER = ["within_v1", "hva_to_hva", "v1_to_hva"]
LANDMARK_LABELS = {
    "within_v1": "V1 location ↔ V1 location",
    "hva_to_hva": "HVA ↔ HVA",
    "v1_to_hva": "V1 ↔ HVA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--min-units", type=int, default=10)
    return parser.parse_args()


def unbiased_squared_mean_distance(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Estimate (E[X] - E[Y])**2 without cell-sampling inflation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each population needs at least two finite neurons")
    naive = float((x.mean() - y.mean()) ** 2)
    correction = float(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return {
        "naive_squared_mean_difference": naive,
        "sampling_noise_correction": correction,
        "corrected_squared_mean_distance": naive - correction,
    }


def build_ttfs_units(min_units: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    mouse = pd.read_csv(MOUSE_UNITS, dtype={"session_id": str})
    mouse = mouse.loc[
        mouse["cohort"].eq("MouseV2")
        & mouse["selected"].astype(bool)
        & pd.to_numeric(mouse["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    mouse = mouse.rename(
        columns={
            "preferred_0_250_ttfs_ms": "value",
            "location": "group",
        }
    )
    mouse["unit_id"] = mouse["unit_id"].astype(str)
    mouse_counts = mouse.groupby(["session_id", "group"])["unit_id"].transform("size")
    mouse = mouse.loc[mouse_counts.ge(min_units)].copy()
    complete_mouse_sessions = mouse.groupby("session_id")["group"].nunique()
    complete_mouse_sessions = complete_mouse_sessions[complete_mouse_sessions.ge(4)].index
    mouse = mouse.loc[mouse["session_id"].isin(complete_mouse_sessions)].copy()

    allen = pd.read_csv(ALLEN_UNITS, dtype={"session_id": str})
    allen = allen.loc[
        allen["area_coarse"].isin(["V1", *CORTICAL_HVA_ORDER])
        & allen["selected_positive_responder_area"].astype(bool)
        & pd.to_numeric(allen["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    allen = allen.rename(
        columns={
            "preferred_0_250_ttfs_ms": "value",
            "area_coarse": "group",
        }
    )
    allen["unit_id"] = allen["unit_id"].astype(str)
    allen_counts = allen.groupby(["session_id", "group"])["unit_id"].transform("size")
    allen = allen.loc[allen_counts.ge(min_units)].copy()
    composition = allen.groupby("session_id")["group"].agg(set)
    eligible_allen_sessions = composition[
        composition.map(lambda groups: "V1" in groups and len(set(CORTICAL_HVA_ORDER) & groups) >= 2)
    ].index
    allen = allen.loc[allen["session_id"].isin(eligible_allen_sessions)].copy()

    columns = ["session_id", "group", "unit_id", "value"]
    for table, name in [(mouse, "MouseV2"), (allen, "Allen")]:
        if table.duplicated(["session_id", "group", "unit_id"]).any():
            raise ValueError(f"{name} contains a duplicated neuron within a population")
    return mouse[columns].copy(), allen[columns].copy()


def validate_ttfs_units(mouse: pd.DataFrame, allen: pd.DataFrame) -> pd.DataFrame:
    mouse_check = mouse.copy()
    mouse_check["dataset"] = "Within-V1"
    allen_check = allen.copy()
    allen_check["dataset"] = np.where(
        allen_check["group"].eq("V1"), "Allen-V1", "Post-V1"
    )
    allen_check["group"] = allen_check["group"].replace(
        {"V1": "Visual Coding VISp"}
    )
    reconstructed = (
        pd.concat([mouse_check, allen_check], ignore_index=True)
        .groupby(["dataset", "session_id", "group"], as_index=False)
        .agg(mean_reconstructed=("value", "mean"), n_reconstructed=("unit_id", "size"))
    )
    current = pd.read_csv(CURRENT_TTFS_MEANS, dtype={"session_id": str})
    current = current[["dataset", "session_id", "group", "mean", "n_units"]]
    audit = reconstructed.merge(
        current,
        on=["dataset", "session_id", "group"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    audit["absolute_mean_error"] = (audit["mean_reconstructed"] - audit["mean"]).abs()
    audit["unit_count_error"] = audit["n_reconstructed"] - audit["n_units"]
    if not audit["_merge"].eq("both").all():
        raise ValueError("Landmark populations are missing from the current TTFS means")
    if audit["absolute_mean_error"].max() > 1e-10 or not audit["unit_count_error"].eq(0).all():
        raise ValueError("Landmark full-cell populations do not reproduce current TTFS means")
    return audit


def _pair_row(
    session_id: str,
    landmark: str,
    group_a: str,
    values_a: np.ndarray,
    group_b: str,
    values_b: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {
        "landmark": landmark,
        "session_id": str(session_id),
        "group_a": str(group_a),
        "group_b": str(group_b),
        "n_a": int(len(values_a)),
        "n_b": int(len(values_b)),
        "cells_in_pair": int(len(values_a) + len(values_b)),
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
    }
    result.update(unbiased_squared_mean_distance(values_a, values_b))
    return result


def compute_pair_distances(mouse: pd.DataFrame, allen: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_id, session in mouse.groupby("session_id", sort=True):
        populations = {
            group: part["value"].to_numpy(float)
            for group, part in session.groupby("group", sort=True)
        }
        for group_a, group_b in combinations(sorted(populations), 2):
            rows.append(
                _pair_row(
                    session_id, "within_v1", group_a, populations[group_a],
                    group_b, populations[group_b]
                )
            )

    for session_id, session in allen.groupby("session_id", sort=True):
        populations = {
            group: part["value"].to_numpy(float)
            for group, part in session.groupby("group", sort=True)
        }
        hvas = sorted(set(populations) & set(CORTICAL_HVA_ORDER))
        for hva in hvas:
            rows.append(
                _pair_row(
                    session_id, "v1_to_hva", "V1", populations["V1"],
                    hva, populations[hva]
                )
            )
        for group_a, group_b in combinations(hvas, 2):
            rows.append(
                _pair_row(
                    session_id, "hva_to_hva", group_a, populations[group_a],
                    group_b, populations[group_b]
                )
            )
    return pd.DataFrame(rows)


def session_landmarks(pair_distances: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_distances.groupby(["landmark", "session_id"], as_index=False)
        .agg(
            distance=("corrected_squared_mean_distance", "mean"),
            naive_distance=("naive_squared_mean_difference", "mean"),
            sampling_noise_correction=("sampling_noise_correction", "mean"),
            n_pairs=("group_a", "size"),
            cells_in_pairs=("cells_in_pair", "sum"),
        )
    )


def landmark_point_summary(session_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    summary = (
        session_table.groupby("landmark", as_index=False)
        .agg(
            distance=("distance", "mean"),
            naive_distance=("naive_distance", "mean"),
            sampling_noise_correction=("sampling_noise_correction", "mean"),
            n_sessions=("session_id", "nunique"),
        )
    )
    values = summary.set_index("landmark")["distance"]
    naive_values = summary.set_index("landmark")["naive_distance"]
    denominator = float(values["v1_to_hva"] - values["within_v1"])
    score = float((values["hva_to_hva"] - values["within_v1"]) / denominator)
    derived = {
        "hva_minus_within_v1": float(values["hva_to_hva"] - values["within_v1"]),
        "between_minus_within_v1": denominator,
        "hva_minus_v1_to_hva": float(values["hva_to_hva"] - values["v1_to_hva"]),
        "hva_landmark_position": score,
        "naive_hva_landmark_position": float(
            (naive_values["hva_to_hva"] - naive_values["within_v1"])
            / (naive_values["v1_to_hva"] - naive_values["within_v1"])
        ),
    }
    return summary, derived


def bootstrap_landmarks(
    session_table: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    within = session_table.loc[session_table["landmark"].eq("within_v1")].set_index(
        "session_id"
    )[["distance", "naive_distance"]]
    allen = session_table.loc[
        session_table["landmark"].isin(["v1_to_hva", "hva_to_hva"])
    ].pivot(
        index="session_id", columns="landmark", values=["distance", "naive_distance"]
    ).dropna()
    rows = []
    for repetition in range(repetitions):
        within_indices = rng.integers(0, len(within), size=len(within))
        within_draw = within.iloc[within_indices]
        allen_indices = rng.integers(0, len(allen), size=len(allen))
        allen_draw = allen.iloc[allen_indices]
        d_within = float(within_draw["distance"].mean())
        d_between = float(allen_draw[("distance", "v1_to_hva")].mean())
        d_hva = float(allen_draw[("distance", "hva_to_hva")].mean())
        naive_within = float(within_draw["naive_distance"].mean())
        naive_between = float(allen_draw[("naive_distance", "v1_to_hva")].mean())
        naive_hva = float(allen_draw[("naive_distance", "hva_to_hva")].mean())
        denominator = d_between - d_within
        naive_denominator = naive_between - naive_within
        rows.append(
            {
                "bootstrap_repetition": repetition,
                "within_v1": d_within,
                "hva_to_hva": d_hva,
                "v1_to_hva": d_between,
                "hva_minus_within_v1": d_hva - d_within,
                "between_minus_within_v1": denominator,
                "hva_minus_v1_to_hva": d_hva - d_between,
                "hva_landmark_position": (
                    (d_hva - d_within) / denominator if denominator != 0 else np.nan
                ),
                "naive_hva_landmark_position": (
                    (naive_hva - naive_within) / naive_denominator
                    if naive_denominator != 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(
    point: pd.DataFrame,
    derived: dict[str, float],
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    point_values = point.set_index("landmark")["distance"].to_dict() | derived
    rows = []
    for estimand in [
        *LANDMARK_ORDER,
        "hva_minus_within_v1",
        "between_minus_within_v1",
        "hva_minus_v1_to_hva",
        "hva_landmark_position",
        "naive_hva_landmark_position",
    ]:
        values = bootstrap[estimand].to_numpy(float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "estimand": estimand,
                "point_estimate": point_values[estimand],
                "bootstrap_median": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "fraction_le_zero": float(np.mean(values <= 0)),
                "finite_bootstrap_draws": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_session_out(session_table: pd.DataFrame) -> pd.DataFrame:
    within = session_table.loc[session_table["landmark"].eq("within_v1")]
    allen = session_table.loc[
        session_table["landmark"].isin(["v1_to_hva", "hva_to_hva"])
    ]
    rows = []
    for source, table in [("MouseV2", within), ("Allen", allen)]:
        for session_id in sorted(table["session_id"].unique()):
            candidate = session_table.loc[
                ~(
                    session_table["session_id"].eq(session_id)
                    & session_table["landmark"].isin(table["landmark"].unique())
                )
            ]
            _, derived = landmark_point_summary(candidate)
            rows.append(
                {
                    "omitted_source": source,
                    "omitted_session_id": session_id,
                    **derived,
                }
            )
    return pd.DataFrame(rows)


def select_concrete_sessions(session_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for landmark, part in session_table.groupby("landmark", sort=False):
        part = part.sort_values("distance").copy()
        median = float(part["distance"].median())
        roles = {
            "smallest": part.iloc[0],
            "typical": part.iloc[(part["distance"] - median).abs().argmin()],
            "largest": part.iloc[-1],
        }
        for role, row in roles.items():
            record = row.to_dict()
            record["selection_role"] = role
            record["selection_criterion"] = (
                "minimum session mean" if role == "smallest" else
                "closest to session median" if role == "typical" else
                "maximum session mean"
            )
            rows.append(record)
    return pd.DataFrame(rows)


def render(
    summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    output: Path,
) -> None:
    point = summary.set_index("landmark")["distance"]
    intervals = bootstrap_summary.set_index("estimand")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax = axes[0]
    y = np.arange(3)[::-1]
    x = np.array([point[key] for key in LANDMARK_ORDER])
    low = np.array([intervals.loc[key, "ci_low"] for key in LANDMARK_ORDER])
    high = np.array([intervals.loc[key, "ci_high"] for key in LANDMARK_ORDER])
    colors = ["#7564a8", "#c47a44", "#4f5963"]
    ax.errorbar(x, y, xerr=[x - low, high - x], fmt="none", ecolor="#343a40", capsize=4)
    ax.scatter(x, y, s=80, color=colors, edgecolor="#222222", zorder=3)
    ax.axvline(0, color="#999999", linewidth=1)
    ax.set_yticks(y, [LANDMARK_LABELS[key] for key in LANDMARK_ORDER])
    ax.set_xlabel("Corrected squared population-mean distance (ms²)")
    ax.set_title("Raw TTFS landmarks", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    score = float(intervals.loc["hva_landmark_position", "point_estimate"])
    score_low = float(loo["hva_landmark_position"].min())
    score_high = float(loo["hva_landmark_position"].max())
    ax.axvspan(0, 1, color="#eef1f4", zorder=0)
    ax.axvline(0, color="#7564a8", linewidth=2)
    ax.axvline(1, color="#4f5963", linewidth=2)
    ax.errorbar(score, 0, xerr=[[score - score_low], [score_high - score]], fmt="o",
                color="#c47a44", ecolor="#c47a44", capsize=5, markersize=9)
    ax.text(0, 0.22, "Within V1", ha="center", color="#5f4f91")
    ax.text(1, 0.22, "V1 ↔ HVA", ha="center", color="#3f4850")
    ax.text(score, -0.2, f"HVA ↔ HVA\n{score:.2f}", ha="center", color="#8f542d")
    span_low = min(-0.15, score_low - 0.08)
    span_high = max(1.15, score_high + 0.08)
    ax.set_xlim(span_low, span_high)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Position from within-area (0) to between-area (1)")
    ax.set_title("Landmark position (bar: leave-one-session-out)", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)

    figure.suptitle("TTFS area-separation landmark pilot", fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_checkpoint(
    point: pd.DataFrame,
    derived: dict[str, float],
    bootstrap_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pair_distances: pd.DataFrame,
    selected_sessions: pd.DataFrame,
    loo: pd.DataFrame,
    validation: pd.DataFrame,
    path: Path,
) -> None:
    intervals = bootstrap_summary.set_index("estimand")
    lines = [
        "# TTFS within-to-between area landmark scale", "",
        "All three landmarks are full-cell, measurement-error-corrected squared",
        "population-mean distances. Pair distances are averaged within session, then",
        "sessions receive equal weight. Allen V1–HVA and HVA–HVA distances use the",
        "same sessions containing VISp and at least two retained cortical HVAs.",
        "The cortical HVA set is LM, RL, AL, PM, and AM; thalamic LP is excluded.", "",
        "## Point estimates", "",
        "| Landmark | Sessions | Distance (ms²) | 95% session-bootstrap interval |",
        "|---|---:|---:|---:|",
    ]
    for row in point.set_index("landmark").loc[LANDMARK_ORDER].itertuples():
        interval = intervals.loc[row.Index]
        lines.append(
            f"| {LANDMARK_LABELS[row.Index]} | {int(row.n_sessions)} | {row.distance:.3f} | "
            f"[{interval.ci_low:.3f}, {interval.ci_high:.3f}] |"
        )
    scale = intervals.loc["hva_landmark_position"]
    difference = intervals.loc["hva_minus_within_v1"]
    anchor_difference = intervals.loc["hva_minus_v1_to_hva"]
    positive_denominator = float(np.mean(bootstrap["between_minus_within_v1"] > 0))
    lines.extend([
        "", "## Landmark position", "",
        f"The HVA–HVA position is **{derived['hva_landmark_position']:.3f}**, with a raw",
        f"bootstrap interval of [{scale.ci_low:.3f}, {scale.ci_high:.3f}].",
        f"The HVA–HVA minus within-V1 distance is {derived['hva_minus_within_v1']:.3f} ms²",
        f"([{difference.ci_low:.3f}, {difference.ci_high:.3f}]).",
        f"HVA–HVA is {derived['hva_minus_v1_to_hva']:.3f} ms² from the V1–HVA anchor",
        f"([{anchor_difference.ci_low:.3f}, {anchor_difference.ci_high:.3f}]), so the data",
        "do not distinguish those two between-area-sized landmarks.",
        f"Without the finite-cell correction, the position is {derived['naive_hva_landmark_position']:.3f};",
        "the corrected result is not created by an inflated raw HVA distance.",
        f"Leave-one-session-out corrected positions range from {loo.hva_landmark_position.min():.3f}",
        f"to {loo.hva_landmark_position.max():.3f}.",
        f"The between-minus-within denominator is positive in {positive_denominator:.1%}",
        "of session-bootstrap draws. Normalized intervals should be treated as unstable",
        "if that fraction is not close to 100%; the three raw landmarks remain primary.",
        "", "## Independence and comparability gates", "",
        f"- All {len(validation)} retained session × population means and counts reproduce",
        f"  the current Figure 3 TTFS input (maximum mean error {validation.absolute_mean_error.max():.3g}).",
        f"- {pair_distances.loc[pair_distances.landmark.eq('within_v1'), 'session_id'].nunique()} MouseV2 sessions contribute the within-V1 landmark.",
        f"- {pair_distances.loc[pair_distances.landmark.eq('v1_to_hva'), 'session_id'].nunique()} Allen sessions contribute both Allen landmarks.",
        "- No neuron is duplicated within a population; cells contribute through the",
        "  two-sample U statistic rather than being treated as biological replicates.",
        "- Whole sessions are resampled, preserving simultaneously recorded populations.",
        "- The V1–HVA anchor is within Allen, so the known cross-dataset TTFS timing offset",
        "  does not enter that landmark.",
        "", "## Concrete-session audit", "",
        "The saved selection table includes the smallest, median-like, and largest",
        "session for every landmark; selection is algorithmic, not hand-picked.",
        "", "## Remaining limitation", "",
        "The U statistic removes finite-cell sampling variance from squared population-mean",
        "differences, but it does not RF-, layer-, cell-type-, or firing-rate-match the",
        "populations. Negative corrected distances are retained rather than clipped.",
        "", f"Selected audit rows: {len(selected_sessions)}.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mouse, allen = build_ttfs_units(args.min_units)
    validation = validate_ttfs_units(mouse, allen)
    pairs = compute_pair_distances(mouse, allen)
    sessions = session_landmarks(pairs)
    point, derived = landmark_point_summary(sessions)
    bootstrap = bootstrap_landmarks(
        sessions, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    bootstrap_summary = summarize_bootstrap(point, derived, bootstrap)
    selected = select_concrete_sessions(sessions)
    loo = leave_one_session_out(sessions)

    mouse.assign(source_dataset="MouseV2").to_csv(output / "ttfs_landmark_mouse_units.csv", index=False)
    allen.assign(source_dataset="Allen").to_csv(output / "ttfs_landmark_allen_units.csv", index=False)
    validation.to_csv(output / "ttfs_landmark_input_validation.csv", index=False)
    pairs.to_csv(output / "ttfs_landmark_pair_distances.csv", index=False)
    sessions.to_csv(output / "ttfs_landmark_session_means.csv", index=False)
    point.to_csv(output / "ttfs_landmark_point_estimates.csv", index=False)
    bootstrap.to_csv(output / "ttfs_landmark_session_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(output / "ttfs_landmark_bootstrap_summary.csv", index=False)
    selected.to_csv(output / "ttfs_landmark_selected_sessions.csv", index=False)
    loo.to_csv(output / "ttfs_landmark_leave_one_session_out.csv", index=False)
    render(point, bootstrap_summary, bootstrap, loo, output / "Figure_ttfs_landmark_scale.png")
    write_checkpoint(
        point, derived, bootstrap_summary, bootstrap, pairs, selected, loo, validation,
        output / "TTFS_LANDMARK_SCALE_CHECKPOINT.md",
    )
    print(point.to_string(index=False))
    print(bootstrap_summary.to_string(index=False))


if __name__ == "__main__":
    main()
