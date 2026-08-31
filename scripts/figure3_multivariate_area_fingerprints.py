#!/usr/bin/env python3
"""Cross-validated multivariate fingerprints for cortical HVAs and V1 sites.

The analysis uses the frozen Figure 3 session x group means.  Each observation
is represented by its session-centered TTFS, log10 F1/F0, and response
timescale.  Scaling and group centroids are learned only from other sessions.
Identity labels are shuffled within sessions for the permutation null, which
preserves the recorded group set, session offsets, and missingness pattern.

This is an exploratory, concrete-first checkpoint.  It tests stable functional
identity and keeps that estimand separate from alignment to the published
anatomical hierarchy score.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
DEFAULT_OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/"
    "multivariate_area_fingerprints"
)
METRICS = ["TTFS (ms)", "log10 F1/F0", "Response timescale (ms)"]
DATASET_GROUPS = {
    "Post-V1": ["LM", "RL", "AL", "PM", "AM"],
    "Within-V1": ["A", "B", "C", "E"],
}
DATASET_LABELS = {
    "Post-V1": "Cortical HVAs",
    "Within-V1": "V1 probe locations",
}
HIERARCHY_SCORES = {
    "LM": -0.093,
    "RL": -0.059,
    "AL": 0.152,
    "PM": 0.327,
    "AM": 0.441,
}
SEED = 20260827


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_source(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"session_id": str})
    required = {
        "dataset", "metric", "session_id", "group", "mean", "centered_mean",
        "session_mean", "n_groups_in_session", "n_units",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Source is missing columns: {sorted(missing)}")
    if table.duplicated(["dataset", "metric", "session_id", "group"]).any():
        raise ValueError("Duplicate dataset x metric x session x group rows")
    reconstructed = table["mean"] - table["session_mean"]
    maximum_error = float((reconstructed - table["centered_mean"]).abs().max())
    if maximum_error > 1e-10:
        raise ValueError(f"Session-centering reconstruction error: {maximum_error}")
    return table


def build_fingerprint_table(
    source: pd.DataFrame,
    dataset: str,
    *,
    minimum_metrics: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = DATASET_GROUPS[dataset]
    local = source.loc[
        source["dataset"].eq(dataset)
        & source["group"].isin(groups)
        & source["metric"].isin(METRICS)
    ].copy()
    # Recenter after the cortical-HVA restriction.  The frozen source also
    # contains thalamic LP, so its stored post-V1 session center is not the
    # correct nuisance intercept for a cortical-HVA-only analysis.
    local["analysis_groups_in_session"] = local.groupby(
        ["metric", "session_id"], observed=True
    )["group"].transform("nunique")
    local = local.loc[local["analysis_groups_in_session"].ge(2)].copy()
    local["analysis_session_mean"] = local.groupby(
        ["metric", "session_id"], observed=True
    )["mean"].transform("mean")
    local["analysis_centered_mean"] = (
        local["mean"] - local["analysis_session_mean"]
    )
    values = local.pivot(
        index=["session_id", "group"], columns="metric", values="analysis_centered_mean"
    ).reindex(columns=METRICS)
    counts = local.pivot(
        index=["session_id", "group"], columns="metric", values="n_units"
    ).reindex(columns=METRICS)
    values.columns = [f"value::{metric}" for metric in values.columns]
    counts.columns = [f"n_units::{metric}" for metric in counts.columns]
    table = values.join(counts).reset_index()
    value_columns = [f"value::{metric}" for metric in METRICS]
    table["available_metrics"] = table[value_columns].notna().sum(axis=1)
    table = table.loc[table["available_metrics"].ge(minimum_metrics)].copy()
    eligible_per_session = table.groupby("session_id")["group"].transform("size")
    table = table.loc[eligible_per_session.ge(2)].copy()
    table["dataset"] = dataset
    table = table.sort_values(["session_id", "group"]).reset_index(drop=True)

    coverage = []
    for group in groups:
        part = table.loc[table["group"].eq(group)]
        row: dict[str, object] = {
            "dataset": dataset,
            "group": group,
            "eligible_rows": len(part),
            "sessions": part["session_id"].nunique(),
            "complete_three_metric_rows": int(
                part[value_columns].notna().all(axis=1).sum()
            ),
        }
        for metric in METRICS:
            row[f"rows::{metric}"] = int(part[f"value::{metric}"].notna().sum())
        coverage.append(row)
    return table, pd.DataFrame(coverage)


def balanced_accuracy(table: pd.DataFrame) -> float:
    eligible = table.loc[table["eligible"].astype(bool)]
    if eligible.empty:
        return np.nan
    recalls = eligible.groupby("true_group", observed=True)["correct"].mean()
    return float(recalls.mean())


def macro_f1(table: pd.DataFrame, groups: list[str]) -> float:
    eligible = table.loc[table["eligible"].astype(bool)]
    scores = []
    for group in groups:
        true = eligible["true_group"].eq(group)
        predicted = eligible["predicted_group"].eq(group)
        true_positive = int((true & predicted).sum())
        false_positive = int((~true & predicted).sum())
        false_negative = int((true & ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def run_loso_centroid(
    table: pd.DataFrame,
    dataset: str,
    *,
    metrics: list[str] | None = None,
    minimum_features: int = 2,
    label_column: str = "group",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    metrics = METRICS if metrics is None else metrics
    groups = DATASET_GROUPS[dataset]
    prediction_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    squared_error = 0.0
    null_squared_error = 0.0
    evaluated_components = 0

    for session_id in sorted(table["session_id"].unique()):
        train = table.loc[~table["session_id"].eq(session_id)].copy()
        test = table.loc[table["session_id"].eq(session_id)].copy()
        scales: dict[str, float] = {}
        centroids: dict[str, dict[str, float]] = {group: {} for group in groups}
        centroid_counts: dict[str, dict[str, int]] = {group: {} for group in groups}
        for metric in metrics:
            column = f"value::{metric}"
            scale = float(train[column].std(ddof=1))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            scales[metric] = scale
            for group in groups:
                values = train.loc[train[label_column].eq(group), column].dropna()
                centroids[group][metric] = (
                    float(values.mean() / scale) if len(values) else np.nan
                )
                centroid_counts[group][metric] = int(len(values))

        for record in test.to_dict("records"):
            true_group = str(record[label_column])
            z_values = {
                metric: (
                    float(record[f"value::{metric}"] / scales[metric])
                    if pd.notna(record[f"value::{metric}"]) else np.nan
                )
                for metric in metrics
            }
            candidates = []
            for candidate in groups:
                used = [
                    metric for metric in metrics
                    if np.isfinite(z_values[metric])
                    and np.isfinite(centroids[candidate][metric])
                    and centroid_counts[candidate][metric] >= 2
                ]
                distance = (
                    float(np.mean([
                        (z_values[metric] - centroids[candidate][metric]) ** 2
                        for metric in used
                    ]))
                    if len(used) >= minimum_features else np.nan
                )
                distance_rows.append(
                    {
                        "dataset": dataset,
                        "session_id": session_id,
                        "true_group": true_group,
                        "candidate_group": candidate,
                        "distance": distance,
                        "features_used": ";".join(used),
                        "n_features_used": len(used),
                    }
                )
                if np.isfinite(distance):
                    candidates.append((distance, candidate, used))

            own = [item for item in candidates if item[1] == true_group]
            eligible = bool(candidates and own)
            if eligible:
                predicted_distance, predicted_group, predicted_used = min(candidates)
                own_distance, _, own_used = own[0]
                other = min(item for item in candidates if item[1] != true_group)
                margin = float(other[0] - own_distance)
                for metric in own_used:
                    squared_error += (
                        z_values[metric] - centroids[true_group][metric]
                    ) ** 2
                    null_squared_error += z_values[metric] ** 2
                    evaluated_components += 1
            else:
                predicted_distance = np.nan
                predicted_group = ""
                predicted_used = []
                own_distance = np.nan
                margin = np.nan
                own_used = []
            row: dict[str, object] = {
                "dataset": dataset,
                "session_id": session_id,
                "true_group": true_group,
                "predicted_group": predicted_group,
                "correct": bool(eligible and predicted_group == true_group),
                "eligible": eligible,
                "own_distance": own_distance,
                "predicted_distance": predicted_distance,
                "classification_margin": margin,
                "features_used": ";".join(predicted_used),
                "n_features_used": len(predicted_used),
            }
            for metric in metrics:
                safe = metric.replace(" ", "_")
                row[f"z::{safe}"] = z_values[metric]
                row[f"own_centroid::{safe}"] = centroids.get(true_group, {}).get(metric, np.nan)
                row[f"predicted_centroid::{safe}"] = centroids.get(
                    predicted_group, {}
                ).get(metric, np.nan)
            prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    distances = pd.DataFrame(distance_rows)
    identity_r2 = (
        1.0 - squared_error / null_squared_error
        if null_squared_error > 0 else np.nan
    )
    summary = {
        "balanced_accuracy": balanced_accuracy(predictions),
        "macro_f1": macro_f1(predictions, groups),
        "identity_r2": float(identity_r2),
        "evaluated_rows": int(predictions["eligible"].sum()),
        "evaluated_components": int(evaluated_components),
    }
    return predictions, distances, summary


def permute_labels_within_sessions(
    table: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    result = table.copy()
    labels = result["group"].astype(str).to_numpy(copy=True)
    for indices in result.groupby("session_id", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        labels[indices] = rng.permutation(labels[indices])
    result["permuted_group"] = labels
    return result


def fast_loso_summary(
    table: pd.DataFrame,
    dataset: str,
    *,
    labels: np.ndarray | None = None,
    metrics: list[str] | None = None,
    minimum_features: int = 2,
) -> dict[str, float]:
    """Numerical-only LOSO path used for resampling.

    This reproduces ``run_loso_centroid`` without constructing prediction and
    candidate-distance DataFrames on every permutation.
    """
    metrics = METRICS if metrics is None else metrics
    groups = DATASET_GROUPS[dataset]
    group_index = {group: index for index, group in enumerate(groups)}
    observed_labels = table["group"].astype(str).to_numpy()
    labels = observed_labels if labels is None else np.asarray(labels, dtype=str)
    label_codes = np.asarray([group_index[value] for value in labels], dtype=int)
    true_codes = label_codes
    session_values = table["session_id"].astype(str).to_numpy()
    sessions = np.asarray(sorted(np.unique(session_values)), dtype=object)
    values = table[[f"value::{metric}" for metric in metrics]].to_numpy(float)
    predicted_codes = np.full(len(table), -1, dtype=int)
    eligible = np.zeros(len(table), dtype=bool)
    squared_error = 0.0
    null_squared_error = 0.0
    evaluated_components = 0

    for session_id in sessions:
        test_mask = session_values == session_id
        train_mask = ~test_mask
        train_values = values[train_mask]
        scales = np.nanstd(train_values, axis=0, ddof=1)
        scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
        train_z = train_values / scales
        test_z = values[test_mask] / scales
        train_codes = label_codes[train_mask]
        centroids = np.full((len(groups), len(metrics)), np.nan)
        centroid_counts = np.zeros((len(groups), len(metrics)), dtype=int)
        for group_code in range(len(groups)):
            local = train_z[train_codes == group_code]
            if not len(local):
                continue
            counts = np.isfinite(local).sum(axis=0)
            centroid_counts[group_code] = counts
            valid = counts > 0
            centroids[group_code, valid] = np.nansum(local[:, valid], axis=0) / counts[valid]

        test_indices = np.flatnonzero(test_mask)
        for local_index, global_index in enumerate(test_indices):
            row = test_z[local_index]
            distances = np.full(len(groups), np.nan)
            feature_masks = []
            for group_code in range(len(groups)):
                used = (
                    np.isfinite(row)
                    & np.isfinite(centroids[group_code])
                    & (centroid_counts[group_code] >= 2)
                )
                feature_masks.append(used)
                if int(used.sum()) >= minimum_features:
                    distances[group_code] = float(
                        np.mean((row[used] - centroids[group_code, used]) ** 2)
                    )
            own_code = true_codes[global_index]
            if not np.isfinite(distances).any() or not np.isfinite(distances[own_code]):
                continue
            predicted_codes[global_index] = int(np.nanargmin(distances))
            eligible[global_index] = True
            own_used = feature_masks[own_code]
            squared_error += float(
                np.sum((row[own_used] - centroids[own_code, own_used]) ** 2)
            )
            null_squared_error += float(np.sum(row[own_used] ** 2))
            evaluated_components += int(own_used.sum())

    recalls = []
    f1_scores = []
    for group_code in range(len(groups)):
        truth = true_codes == group_code
        predicted = predicted_codes == group_code
        group_eligible = eligible & truth
        if group_eligible.any():
            recalls.append(float((predicted[group_eligible]).mean()))
        true_positive = int((eligible & truth & predicted).sum())
        false_positive = int((eligible & ~truth & predicted).sum())
        false_negative = int((eligible & truth & ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    identity_r2 = (
        1.0 - squared_error / null_squared_error
        if null_squared_error > 0 else np.nan
    )
    return {
        "balanced_accuracy": float(np.mean(recalls)) if recalls else np.nan,
        "macro_f1": float(np.mean(f1_scores)),
        "identity_r2": float(identity_r2),
        "evaluated_rows": int(eligible.sum()),
        "evaluated_components": int(evaluated_components),
    }


def permutation_null(
    table: pd.DataFrame,
    dataset: str,
    *,
    repetitions: int,
    rng: np.random.Generator,
    metrics: list[str] | None = None,
    minimum_features: int = 2,
) -> pd.DataFrame:
    rows = []
    base_labels = table["group"].astype(str).to_numpy()
    session_indices = [
        np.asarray(indices, dtype=int)
        for indices in table.groupby("session_id", sort=False).indices.values()
    ]
    for repetition in range(repetitions):
        labels = base_labels.copy()
        for indices in session_indices:
            labels[indices] = rng.permutation(labels[indices])
        summary = fast_loso_summary(
            table,
            dataset,
            labels=labels,
            metrics=metrics,
            minimum_features=minimum_features,
        )
        rows.append({"repetition": repetition, **summary})
    return pd.DataFrame(rows)


def bootstrap_sessions(
    table: pd.DataFrame,
    dataset: str,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sessions = np.asarray(sorted(table["session_id"].unique()), dtype=object)
    rows = []
    for repetition in range(repetitions):
        selected = rng.choice(sessions, size=len(sessions), replace=True)
        blocks = []
        for block_index, session_id in enumerate(selected):
            block = table.loc[table["session_id"].eq(session_id)].copy()
            block["session_id"] = f"bootstrap_{block_index}:{session_id}"
            blocks.append(block)
        sample = pd.concat(blocks, ignore_index=True)
        summary = fast_loso_summary(sample, dataset)
        rows.append({"repetition": repetition, **summary})
    return pd.DataFrame(rows)


def hierarchy_loso_r2(
    table: pd.DataFrame,
    score_map: dict[str, float],
) -> tuple[float, pd.DataFrame]:
    squared_error = 0.0
    null_squared_error = 0.0
    rows = []
    for session_id in sorted(table["session_id"].unique()):
        train = table.loc[~table["session_id"].eq(session_id)].copy()
        test = table.loc[table["session_id"].eq(session_id)].copy()
        for metric in METRICS:
            column = f"value::{metric}"
            train_metric = train.loc[train[column].notna()].copy()
            test_metric = test.loc[test[column].notna()].copy()
            if len(train_metric) < 5 or test_metric.empty:
                continue
            scale = float(train_metric[column].std(ddof=1))
            if not np.isfinite(scale) or scale <= 0:
                continue
            x_train = train_metric["group"].map(score_map).to_numpy(float)
            y_train = train_metric[column].to_numpy(float) / scale
            design = np.column_stack([np.ones(len(x_train)), x_train])
            beta = np.linalg.lstsq(design, y_train, rcond=None)[0]
            x_test = test_metric["group"].map(score_map).to_numpy(float)
            observed = test_metric[column].to_numpy(float) / scale
            predicted = beta[0] + beta[1] * x_test
            squared_error += float(np.sum((observed - predicted) ** 2))
            null_squared_error += float(np.sum(observed**2))
            for record, obs, pred in zip(
                test_metric.to_dict("records"), observed, predicted
            ):
                rows.append(
                    {
                        "session_id": session_id,
                        "group": record["group"],
                        "metric": metric,
                        "observed_z": obs,
                        "predicted_z": pred,
                        "hierarchy_score": score_map[str(record["group"])],
                    }
                )
    r2 = 1.0 - squared_error / null_squared_error
    return float(r2), pd.DataFrame(rows)


def exact_hierarchy_null(table: pd.DataFrame) -> pd.DataFrame:
    groups = DATASET_GROUPS["Post-V1"]
    scores = [HIERARCHY_SCORES[group] for group in groups]
    rows = []
    for permutation_index, ordered_scores in enumerate(permutations(scores)):
        score_map = dict(zip(groups, ordered_scores))
        r2, _ = hierarchy_loso_r2(table, score_map)
        rows.append(
            {
                "permutation": permutation_index,
                "hierarchy_r2": r2,
                "mapping": ";".join(
                    f"{group}:{score_map[group]:.3f}" for group in groups
                ),
                "is_published_mapping": all(
                    score_map[group] == HIERARCHY_SCORES[group] for group in groups
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_null(
    dataset: str,
    observed: dict[str, float],
    null: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        **observed,
        "permutations": len(null),
        "bootstraps": len(bootstrap),
    }
    for metric in ["balanced_accuracy", "macro_f1", "identity_r2"]:
        values = null[metric].dropna().to_numpy(float)
        boot = bootstrap[metric].dropna().to_numpy(float)
        row[f"null_mean::{metric}"] = float(np.mean(values))
        row[f"null_q025::{metric}"] = float(np.quantile(values, 0.025))
        row[f"null_q975::{metric}"] = float(np.quantile(values, 0.975))
        row[f"permutation_p::{metric}"] = float(
            (1 + np.sum(values >= observed[metric])) / (1 + len(values))
        )
        row[f"bootstrap_q025::{metric}"] = float(np.quantile(boot, 0.025))
        row[f"bootstrap_q975::{metric}"] = float(np.quantile(boot, 0.975))
    return row


def feature_ablation(
    table: pd.DataFrame,
    dataset: str,
    *,
    permutations_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    subsets = [("all three, missingness-aware", METRICS, 2)] + [
        (metric, [metric], 1) for metric in METRICS
    ]
    for label, metrics, minimum_features in subsets:
        _, _, observed = run_loso_centroid(
            table, dataset, metrics=metrics, minimum_features=minimum_features
        )
        null = permutation_null(
            table,
            dataset,
            repetitions=permutations_count,
            rng=rng,
            metrics=metrics,
            minimum_features=minimum_features,
        )
        rows.append(
            {
                "dataset": dataset,
                "feature_set": label,
                **observed,
                "null_accuracy_mean": float(null["balanced_accuracy"].mean()),
                "permutation_p_accuracy": float(
                    (1 + (null["balanced_accuracy"] >= observed["balanced_accuracy"]).sum())
                    / (1 + len(null))
                ),
                "null_r2_mean": float(null["identity_r2"].mean()),
                "permutation_p_r2": float(
                    (1 + (null["identity_r2"] >= observed["identity_r2"]).sum())
                    / (1 + len(null))
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for dataset, part in predictions.groupby("dataset", sort=False):
        eligible = part.loc[part["eligible"]].copy()
        correct = eligible.loc[eligible["correct"]].copy()
        failures = eligible.loc[~eligible["correct"]].copy()
        roles: list[tuple[str, pd.Series]] = []
        if not correct.empty:
            roles.append(("strongest correct", correct.loc[correct["classification_margin"].idxmax()]))
            median_margin = float(correct["classification_margin"].median())
            typical_index = (correct["classification_margin"] - median_margin).abs().idxmin()
            roles.append(("typical correct", correct.loc[typical_index]))
        if not failures.empty:
            roles.append(("strongest failure", failures.loc[failures["classification_margin"].idxmin()]))
        ambiguous_index = eligible["classification_margin"].abs().idxmin()
        roles.append(("boundary case", eligible.loc[ambiguous_index]))
        missingness = eligible.loc[eligible["n_features_used"].eq(eligible["n_features_used"].min())]
        missingness_index = missingness["classification_margin"].abs().idxmin()
        roles.append(("missingness boundary", eligible.loc[missingness_index]))
        used_keys = set()
        for role, row in roles:
            key = (row["session_id"], row["true_group"])
            if key in used_keys:
                continue
            used_keys.add(key)
            local = row.copy()
            local["selection_role"] = role
            local["selection_criterion"] = {
                "strongest correct": "maximum positive classification margin",
                "typical correct": "closest to median positive margin",
                "strongest failure": "most negative classification margin",
                "boundary case": "minimum absolute classification margin",
                "missingness boundary": "fewest features, then minimum absolute margin",
            }[role]
            selected.append(local)
    columns = ["dataset", "selection_role", "selection_criterion"] + [
        column for column in predictions.columns
        if column not in {"dataset"}
    ]
    return pd.DataFrame(selected)[columns]


def confusion_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in predictions.loc[predictions["eligible"]].groupby("dataset"):
        groups = DATASET_GROUPS[dataset]
        matrix = pd.crosstab(part["true_group"], part["predicted_group"]).reindex(
            index=groups, columns=groups, fill_value=0
        )
        normalized = matrix.div(matrix.sum(axis=1), axis=0)
        for true_group in groups:
            for predicted_group in groups:
                rows.append(
                    {
                        "dataset": dataset,
                        "true_group": true_group,
                        "predicted_group": predicted_group,
                        "count": int(matrix.loc[true_group, predicted_group]),
                        "row_fraction": float(normalized.loc[true_group, predicted_group]),
                    }
                )
    return pd.DataFrame(rows)


def render_summary(
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    hierarchy_r2: float,
    hierarchy_null: pd.DataFrame,
    output: Path,
) -> None:
    figure = plt.figure(figsize=(13.2, 8.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[0.9, 1.15])
    ax_accuracy = figure.add_subplot(grid[0, 0])
    ax_r2 = figure.add_subplot(grid[0, 1])
    ax_hierarchy = figure.add_subplot(grid[0, 2])
    colors = {"Post-V1": "#315f86", "Within-V1": "#c27a3f"}
    positions = np.arange(len(summary))
    for ax, metric, label in [
        (ax_accuracy, "balanced_accuracy", "Balanced identity accuracy"),
        (ax_r2, "identity_r2", "Cross-validated identity R²"),
    ]:
        observed = summary[metric].to_numpy(float)
        low = summary[f"bootstrap_q025::{metric}"].to_numpy(float)
        high = summary[f"bootstrap_q975::{metric}"].to_numpy(float)
        null_mean = summary[f"null_mean::{metric}"].to_numpy(float)
        ax.vlines(positions, low, high, color="#252c32", linewidth=1.4)
        ax.scatter(positions, low, marker="_", s=70, color="#252c32")
        ax.scatter(positions, high, marker="_", s=70, color="#252c32")
        for index, row in summary.reset_index(drop=True).iterrows():
            ax.scatter(index, row[metric], s=85, color=colors[row["dataset"]], zorder=3)
        ax.scatter(positions, null_mean, marker="_", s=480, linewidth=2.2,
                   color="#6f7880", label="permutation mean")
        ax.set_xticks(positions, [DATASET_LABELS[value] for value in summary["dataset"]], rotation=16)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.axhline(0, color="#a4abb1", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax_accuracy.set_ylim(0, 1)
    ax_accuracy.legend(frameon=False, fontsize=8)

    null_values = hierarchy_null["hierarchy_r2"].to_numpy(float)
    ax_hierarchy.hist(null_values, bins=18, color="#d5dce2", edgecolor="#67727c")
    ax_hierarchy.axvline(hierarchy_r2, color="#315f86", linewidth=2.2,
                        label=f"published order: {hierarchy_r2:.3f}")
    ax_hierarchy.set_xlabel("Cross-validated hierarchy R²")
    ax_hierarchy.set_ylabel("Exact score permutations")
    ax_hierarchy.set_title("Hierarchy ordering", loc="left", fontweight="bold")
    ax_hierarchy.legend(frameon=False, fontsize=8)
    ax_hierarchy.spines[["top", "right"]].set_visible(False)

    for column, dataset in enumerate(["Post-V1", "Within-V1"]):
        ax = figure.add_subplot(grid[1, column])
        groups = DATASET_GROUPS[dataset]
        local = confusion.loc[confusion["dataset"].eq(dataset)]
        matrix = local.pivot(index="true_group", columns="predicted_group", values="row_fraction").reindex(
            index=groups, columns=groups
        )
        image = ax.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="Blues")
        for row in range(len(groups)):
            for col in range(len(groups)):
                value = matrix.iloc[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if value > 0.55 else "#26343f")
        ax.set_xticks(range(len(groups)), groups)
        ax.set_yticks(range(len(groups)), groups)
        ax.set_xlabel("Predicted identity")
        ax.set_ylabel("Observed identity")
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row fraction")

    ax_note = figure.add_subplot(grid[1, 2])
    ax_note.axis("off")
    hva = summary.set_index("dataset").loc["Post-V1"]
    v1 = summary.set_index("dataset").loc["Within-V1"]
    text = (
        "Reading rule\n\n"
        f"HVA accuracy: {hva['balanced_accuracy']:.3f}\n"
        f"HVA identity R²: {hva['identity_r2']:.3f}\n\n"
        f"V1 accuracy: {v1['balanced_accuracy']:.3f}\n"
        f"V1 identity R²: {v1['identity_r2']:.3f}\n\n"
        "Accuracy tests label recovery. R² tests whether the held-out\n"
        "identity centroid predicts values better than a zero, session-\n"
        "centered baseline. Both are session-blocked."
    )
    ax_note.text(0.03, 0.97, text, va="top", ha="left", fontsize=10,
                 family="monospace", linespacing=1.35)
    figure.suptitle(
        "Cross-validated multivariate identity: cortical HVAs versus V1 locations",
        fontsize=14, fontweight="bold",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_cases(cases: pd.DataFrame, output: Path) -> None:
    number = len(cases)
    columns = 2
    rows = int(np.ceil(number / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.4 * rows), squeeze=False)
    x = np.arange(len(METRICS))
    labels = ["TTFS", "log₁₀ F1/F0", "Timescale"]
    for ax, (_, case) in zip(axes.flat, cases.iterrows()):
        observed, own, predicted = [], [], []
        for metric in METRICS:
            safe = metric.replace(" ", "_")
            observed.append(case.get(f"z::{safe}", np.nan))
            own.append(case.get(f"own_centroid::{safe}", np.nan))
            predicted.append(case.get(f"predicted_centroid::{safe}", np.nan))
        observed_array = np.asarray(observed, dtype=float)
        own = np.where(np.isfinite(observed_array), np.asarray(own, dtype=float), np.nan)
        predicted = np.where(
            np.isfinite(observed_array), np.asarray(predicted, dtype=float), np.nan
        )
        ax.plot(x, own, marker="o", color="#315f86", linewidth=1.5,
                label=f"held-out centroid: {case['true_group']}")
        if case["predicted_group"] != case["true_group"]:
            ax.plot(x, predicted, marker="s", linestyle="--", color="#c27a3f",
                    linewidth=1.4, label=f"nearest centroid: {case['predicted_group']}")
        ax.scatter(x, observed, s=75, color="#20272d", zorder=4, label="observed")
        ax.axhline(0, color="#aab0b5", linewidth=0.8, linestyle=":")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Training-scaled, session-centered value")
        ax.set_title(
            f"{DATASET_LABELS[case['dataset']]} · {case['selection_role']}\n"
            f"session {case['session_id']}, {case['true_group']} → {case['predicted_group']} "
            f"(margin {case['classification_margin']:+.3f})",
            loc="left", fontsize=10, fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=7, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[number:]:
        ax.axis("off")
    figure.suptitle(
        "Auditable fingerprint cases: successes, failures, and boundaries",
        fontsize=14, fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_checkpoint(
    output: Path,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    hierarchy_r2: float,
    hierarchy_p: float,
    cases: pd.DataFrame,
    ablation: pd.DataFrame,
    contrast: pd.DataFrame,
) -> None:
    indexed = summary.set_index("dataset")
    hva_ablation = ablation.loc[ablation["dataset"].eq("Post-V1")].set_index(
        "feature_set"
    )
    contrast_summary = {}
    for metric in ["balanced_accuracy", "identity_r2"]:
        values = contrast[f"difference::{metric}"].dropna().to_numpy(float)
        contrast_summary[metric] = {
            "median": float(np.median(values)),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
            "probability_le_zero": float(np.mean(values <= 0)),
        }
    lines = [
        "# Multivariate HVA identity checkpoint",
        "",
        "## Technical summary",
        "",
    ]
    for dataset in ["Post-V1", "Within-V1"]:
        row = indexed.loc[dataset]
        lines.append(
            f"- **{DATASET_LABELS[dataset]}:** balanced leave-one-session-out "
            f"identity accuracy {row['balanced_accuracy']:.3f} "
            f"(session-bootstrap 95% interval "
            f"[{row['bootstrap_q025::balanced_accuracy']:.3f}, "
            f"{row['bootstrap_q975::balanced_accuracy']:.3f}]); within-session "
            f"permutation p={row['permutation_p::balanced_accuracy']:.4f}. "
            f"Cross-validated identity R²={row['identity_r2']:.3f} "
            f"(p={row['permutation_p::identity_r2']:.4f})."
        )
    lines.extend(
        [
            f"- **Published HVA hierarchy order:** cross-validated R²={hierarchy_r2:.3f}; "
            f"exact five-area score-permutation p={hierarchy_p:.4f}.",
            f"- **The HVA signal is primarily timescale-driven:** timescale-only "
            f"identity R²={hva_ablation.loc['Response timescale (ms)', 'identity_r2']:.3f} "
            f"(p={hva_ablation.loc['Response timescale (ms)', 'permutation_p_r2']:.4f}), "
            f"whereas F1/F0 R²={hva_ablation.loc['log10 F1/F0', 'identity_r2']:.3f} "
            f"and TTFS R²={hva_ablation.loc['TTFS (ms)', 'identity_r2']:.3f}.",
            "",
            "Identity accuracy asks whether the three functional measures recover the",
            "named group. Identity R² asks whether the held-out group centroid predicts",
            "the observed session-centered values better than a zero-effect baseline.",
            "These are evidence for reproducible functional identity, not proof of",
            "anatomical boundaries or causal hierarchy.",
            "The direct HVA-minus-V1 bootstrap comparison remains inconclusive: the",
            f"identity-R² difference has a 95% interval "
            f"[{contrast_summary['identity_r2']['low']:.3f}, "
            f"{contrast_summary['identity_r2']['high']:.3f}], reflecting the eight-session",
            "V1 control. The datasets should therefore be described side by side rather",
            "than as a definitive classifier-performance difference.",
            "",
            "## Scope and data sufficiency",
            "",
            "Rows require at least two available metrics and at least two eligible groups",
            "in the same session. The post-V1 set is restricted to cortical HVAs LM, RL,",
            "AL, PM, and AM; thalamic LP is excluded.",
            "",
            "| Dataset | Group | Eligible rows | Complete three-metric rows |",
            "|---|---|---:|---:|",
        ]
    )
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.group} | {row.eligible_rows} | "
            f"{row.complete_three_metric_rows} |"
        )
    lines.extend(
        [
            "",
            "## Model and null",
            "",
            "For every held-out session, metric scales and group centroids are estimated",
            "only from other sessions. A row is assigned to the nearest group centroid",
            "using its available metrics. The permutation null shuffles group identities",
            "within sessions, preserving the sampled group set, session structure, and",
            "missingness. Session bootstrap intervals resample entire sessions.",
            "",
            "Hierarchy alignment is tested separately with a leave-one-session-out linear",
            "prediction from the published area score. Its exact null enumerates all 120",
            "assignments of the five published scores to the five named HVAs.",
            "",
            "## Selected success and failure cases",
            "",
            "| Dataset | Role | Session | Observed | Predicted | Margin | Features |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in cases.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.selection_role} | {row.session_id} | "
            f"{row.true_group} | {row.predicted_group} | "
            f"{row.classification_margin:+.3f} | {row.n_features_used} |"
        )
    lines.extend(
        [
            "",
            "## Limitations and claim boundary",
            "",
            "- This is exploratory because the multivariate test was defined after the",
            "  univariate variance analysis was inspected.",
            "- HVA sampling is unbalanced, and complete three-metric coverage is sparse",
            "  for PM (one complete row). A five-area complete-case LOSO analysis is",
            "  therefore not identified. The primary analysis uses any two or",
            "  three features and never uses missingness itself as a predictor.",
            "- The positive multivariate identity result is mostly carried by response",
            "  timescale. It should not be called a broad three-metric fingerprint.",
            "- Session centering tests within-session relative identity. It intentionally",
            "  removes global session offsets and cannot establish absolute area means.",
            "- A significant categorical identity result would not automatically imply a",
            "  monotonic hierarchy; the separate hierarchy test must support that claim.",
            "",
            "## Audit artifacts",
            "",
            "- `fingerprint_input.csv`: exact eligible session × group features.",
            "- `fingerprint_predictions.csv`: held-out predictions and margins.",
            "- `fingerprint_candidate_distances.csv`: every candidate distance.",
            "- `fingerprint_permutation_null.csv`: session-preserving null draws.",
            "- `fingerprint_session_bootstrap.csv`: whole-session bootstrap draws.",
            "- `feature_ablation.csv`: one-metric and multivariate sensitivities.",
            "- `hierarchy_exact_null.csv`: all 120 score assignments.",
            "- `selected_fingerprint_cases.csv`: auditable selection roles.",
            "- `Figure_multivariate_identity_summary.png` and",
            "  `Figure_multivariate_identity_cases.png`: summary and concrete evidence.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_source(args.source)
    seed_sequence = np.random.SeedSequence(args.seed)
    children = iter(seed_sequence.spawn(12))
    tables = {}
    coverage_parts = []
    predictions_parts = []
    distances_parts = []
    null_parts = []
    bootstrap_parts = []
    summary_rows = []
    ablations = []

    for dataset in ["Post-V1", "Within-V1"]:
        table, coverage = build_fingerprint_table(source, dataset)
        tables[dataset] = table
        coverage_parts.append(coverage)
        predictions, distances, observed = run_loso_centroid(table, dataset)
        predictions_parts.append(predictions)
        distances_parts.append(distances)
        null = permutation_null(
            table,
            dataset,
            repetitions=args.permutations,
            rng=np.random.default_rng(next(children)),
        )
        null["dataset"] = dataset
        null_parts.append(null)
        bootstrap = bootstrap_sessions(
            table,
            dataset,
            repetitions=args.bootstraps,
            rng=np.random.default_rng(next(children)),
        )
        bootstrap["dataset"] = dataset
        bootstrap_parts.append(bootstrap)
        summary_rows.append(summarize_null(dataset, observed, null, bootstrap))
        ablations.append(
            feature_ablation(
                table,
                dataset,
                permutations_count=max(500, args.permutations // 5),
                rng=np.random.default_rng(next(children)),
            )
        )

    fingerprint_input = pd.concat(tables.values(), ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    predictions = pd.concat(predictions_parts, ignore_index=True)
    distances = pd.concat(distances_parts, ignore_index=True)
    null = pd.concat(null_parts, ignore_index=True)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    ablation = pd.concat(ablations, ignore_index=True)
    cases = choose_cases(predictions)
    confusion = confusion_table(predictions)

    hierarchy_r2, hierarchy_predictions = hierarchy_loso_r2(
        tables["Post-V1"], HIERARCHY_SCORES
    )
    hierarchy_null = exact_hierarchy_null(tables["Post-V1"])
    hierarchy_p = float(
        (hierarchy_null["hierarchy_r2"] >= hierarchy_r2 - 1e-12).mean()
    )
    hva_bootstrap = bootstrap.loc[bootstrap["dataset"].eq("Post-V1")].sort_values(
        "repetition"
    )
    v1_bootstrap = bootstrap.loc[bootstrap["dataset"].eq("Within-V1")].sort_values(
        "repetition"
    )
    contrast = pd.DataFrame({
        "repetition": hva_bootstrap["repetition"].to_numpy(int),
        "difference::balanced_accuracy": (
            hva_bootstrap["balanced_accuracy"].to_numpy(float)
            - v1_bootstrap["balanced_accuracy"].to_numpy(float)
        ),
        "difference::identity_r2": (
            hva_bootstrap["identity_r2"].to_numpy(float)
            - v1_bootstrap["identity_r2"].to_numpy(float)
        ),
    })

    fingerprint_input.to_csv(args.output_dir / "fingerprint_input.csv", index=False)
    coverage.to_csv(args.output_dir / "fingerprint_coverage.csv", index=False)
    predictions.to_csv(args.output_dir / "fingerprint_predictions.csv", index=False)
    distances.to_csv(args.output_dir / "fingerprint_candidate_distances.csv", index=False)
    null.to_csv(args.output_dir / "fingerprint_permutation_null.csv", index=False)
    bootstrap.to_csv(args.output_dir / "fingerprint_session_bootstrap.csv", index=False)
    summary.to_csv(args.output_dir / "fingerprint_summary.csv", index=False)
    ablation.to_csv(args.output_dir / "feature_ablation.csv", index=False)
    cases.to_csv(args.output_dir / "selected_fingerprint_cases.csv", index=False)
    confusion.to_csv(args.output_dir / "fingerprint_confusion.csv", index=False)
    hierarchy_predictions.to_csv(args.output_dir / "hierarchy_predictions.csv", index=False)
    hierarchy_null.to_csv(args.output_dir / "hierarchy_exact_null.csv", index=False)
    contrast.to_csv(args.output_dir / "dataset_contrast_bootstrap.csv", index=False)

    render_summary(
        summary,
        confusion,
        hierarchy_r2,
        hierarchy_null,
        args.output_dir / "Figure_multivariate_identity_summary.png",
    )
    render_cases(
        cases,
        args.output_dir / "Figure_multivariate_identity_cases.png",
    )
    write_checkpoint(
        args.output_dir / "MULTIVARIATE_AREA_FINGERPRINTS.md",
        summary,
        coverage,
        hierarchy_r2,
        hierarchy_p,
        cases,
        ablation,
        contrast,
    )
    print(summary.to_string(index=False))
    print(f"hierarchy_r2={hierarchy_r2:.6g}; exact_p={hierarchy_p:.6g}")


if __name__ == "__main__":
    main()
