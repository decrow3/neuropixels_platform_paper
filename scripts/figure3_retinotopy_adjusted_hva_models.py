#!/usr/bin/env python3
"""Compare pooled-V2, hierarchy, and categorical-HVA models after RF adjustment.

The analysis rebuilds the exact Figure 3 Allen outcome populations at unit
level, intersects each with the published-quality receptive-field population,
and restricts the primary analysis to a robust five-area RF overlap box.  The
inferential observations are session x area means, centered within session.
Models are evaluated by leave-one-session-out prediction with ridge strength
chosen by an inner leave-one-session-out loop.

This is an exploratory model-comparison checkpoint.  It distinguishes evidence
for stable named-area effects from response variation explainable by achieved
retinotopy.  It does not test whether the anatomical areas exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.population_masks import common_qc_mask  # noqa: E402


HVA_ORDER = ("LM", "RL", "AL", "PM", "AM")
HIERARCHY_SCORE = {
    "LM": -0.093,
    "RL": -0.059,
    "AL": 0.152,
    "PM": 0.327,
    "AM": 0.441,
}
METRIC_ORDER = ("TTFS (ms)", "log10 F1/F0", "Response timescale (ms)")
MODEL_ORDER = ("pooled_v2", "retinotopy", "hierarchy", "categorical_hva")
MODEL_LABELS = {
    "pooled_v2": "Pooled V2",
    "retinotopy": "Retinotopy only",
    "hierarchy": "Retinotopy + hierarchy",
    "categorical_hva": "Retinotopy + named HVA",
}
AREA_COLORS = {
    "LM": "#1B9E77",
    "RL": "#D95F02",
    "AL": "#7570B3",
    "PM": "#E7298A",
    "AM": "#66A61E",
}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
ORIGINAL_MIN_UNITS = {
    "TTFS (ms)": 10,
    "log10 F1/F0": 5,
    "Response timescale (ms)": 5,
}
SEED = 20260828

SUPPORT_PATH = ROOT / "artifacts/figure3/06c_allen_rf_matching/rf_unit_common_support.csv"
TTFS_PATH = ROOT / (
    "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance/"
    "figure3_response_filtered_ttfs_all_areas_unit_audit.csv"
)
F1_PATH = ROOT / (
    "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/"
    "07_figure3_harmonized_f1_f0/allen_bo_area_unit_f1_f0_harmonized.csv"
)
UNIT_PATH = ROOT / "data/unit_table.csv"
FROZEN_MEANS = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
DEFAULT_OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/"
    "retinotopy_adjusted_hva_models"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--support-quantile", type=float, default=0.025)
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--split-halves", type=int, default=2000)
    parser.add_argument(
        "--equivalence-margin-r2", type=float, default=0.05,
        help="Exploratory upper bound for a meaningful categorical gain over hierarchy.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exact_outcomes() -> dict[str, pd.DataFrame]:
    """Load the exact unit-level populations behind the frozen Allen rows."""
    ttfs = pd.read_csv(TTFS_PATH, dtype={"session_id": str})
    ttfs = ttfs.loc[
        ttfs["area_coarse"].isin(HVA_ORDER)
        & ttfs["selected_positive_responder_area"].astype(bool)
        & pd.to_numeric(ttfs["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    ttfs = ttfs.rename(
        columns={
            "unit_id": "ecephys_unit_id",
            "session_id": "ecephys_session_id",
            "area_coarse": "area",
            "preferred_0_250_ttfs_ms": "outcome",
        }
    )

    f1 = pd.read_csv(F1_PATH, dtype={"ecephys_session_id": str})
    if not f1["population_profile"].eq("common_qc").all():
        raise ValueError("Harmonized Allen F1/F0 source is not uniformly common_qc")
    f1 = f1.loc[
        f1["area_coarse"].isin(HVA_ORDER)
        & pd.to_numeric(f1["f1_f0_dg_harmonized"], errors="coerce").gt(0)
    ].copy()
    f1 = f1.rename(columns={"area_coarse": "area"})
    f1["outcome"] = np.log10(pd.to_numeric(f1["f1_f0_dg_harmonized"]))

    usecols = [
        "ecephys_unit_id", "ecephys_session_id", "ecephys_structure_acronym",
        "amplitude_cutoff", "presence_ratio", "isi_violations", "timescale_ac",
        "spike_count_ac", "err_ac",
    ]
    timescale = pd.read_csv(UNIT_PATH, usecols=usecols, low_memory=False)
    timescale["ecephys_session_id"] = timescale["ecephys_session_id"].astype(str)
    timescale["area"] = timescale["ecephys_structure_acronym"].map(
        {"VISl": "LM", "VISrl": "RL", "VISal": "AL", "VISpm": "PM", "VISam": "AM"}
    )
    valid = common_qc_mask(timescale, dataset="allen")
    valid &= timescale["area"].isin(HVA_ORDER)
    valid &= pd.to_numeric(timescale["timescale_ac"], errors="coerce").between(1, 300)
    valid &= pd.to_numeric(timescale["spike_count_ac"], errors="coerce").gt(50)
    valid &= pd.to_numeric(timescale["err_ac"], errors="coerce").lt(20)
    timescale = timescale.loc[valid].copy()
    timescale["outcome"] = pd.to_numeric(timescale["timescale_ac"])

    columns = ["ecephys_unit_id", "ecephys_session_id", "area", "outcome"]
    result = {
        "TTFS (ms)": ttfs[columns].copy(),
        "log10 F1/F0": f1[columns].copy(),
        "Response timescale (ms)": timescale[columns].copy(),
    }
    for metric, table in result.items():
        if table["ecephys_unit_id"].duplicated().any():
            raise ValueError(f"Duplicate unit IDs in {metric} source")
        if not np.isfinite(table["outcome"]).all():
            raise ValueError(f"Non-finite exact outcomes remain for {metric}")
    return result


def validate_frozen_reconstruction(outcomes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reproduce raw session x HVA means before adding the RF requirement."""
    frozen = pd.read_csv(FROZEN_MEANS, dtype={"session_id": str})
    frozen = frozen.loc[
        frozen["dataset"].eq("Post-V1") & frozen["group"].isin(HVA_ORDER)
    ].copy()
    rows: list[dict[str, object]] = []
    minimum = {"TTFS (ms)": 10, "log10 F1/F0": 5, "Response timescale (ms)": 5}
    for metric, units in outcomes.items():
        rebuilt = (
            units.groupby(["ecephys_session_id", "area"], sort=True)
            .agg(rebuilt_mean=("outcome", "mean"), rebuilt_n=("ecephys_unit_id", "size"))
            .reset_index()
        )
        rebuilt = rebuilt.loc[rebuilt["rebuilt_n"].ge(minimum[metric])]
        target = frozen.loc[frozen["metric"].eq(metric)].copy()
        joined = target.merge(
            rebuilt,
            left_on=["session_id", "group"],
            right_on=["ecephys_session_id", "area"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        mean_error = (joined["mean"] - joined["rebuilt_mean"]).abs()
        n_error = (joined["n_units"] - joined["rebuilt_n"]).abs()
        rows.append(
            {
                "metric": metric,
                "frozen_cells": len(target),
                "rebuilt_cells": len(rebuilt),
                "matched_cells": int(joined["_merge"].eq("both").sum()),
                "unmatched_cells": int(joined["_merge"].ne("both").sum()),
                "max_abs_mean_error": float(mean_error.max()) if mean_error.notna().any() else np.nan,
                "max_abs_n_error": float(n_error.max()) if n_error.notna().any() else np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    if (audit["unmatched_cells"] > 0).any() or (audit["max_abs_mean_error"] > 1e-10).any() or (audit["max_abs_n_error"] > 0).any():
        raise ValueError(f"Exact Figure 3 reconstruction failed:\n{audit.to_string(index=False)}")
    return audit


def load_rf_support() -> pd.DataFrame:
    support = pd.read_csv(SUPPORT_PATH, dtype={"ecephys_session_id": str})
    columns = [
        "ecephys_unit_id", "ecephys_session_id", "area", "relative_azimuth_deg",
        "relative_elevation_deg", "inside_v1_robust_box", "inside_v1_convex_hull",
        "area_rf", "p_value_rf",
    ]
    missing = set(columns).difference(support.columns)
    if missing:
        raise ValueError(f"RF support source lacks {sorted(missing)}")
    support = support.loc[support["area"].isin(HVA_ORDER), columns].copy()
    if support["ecephys_unit_id"].duplicated().any():
        raise ValueError("RF support contains duplicate unit IDs")
    return support


def robust_common_box(table: pd.DataFrame, quantile: float) -> dict[str, float]:
    if not 0 <= quantile < 0.5:
        raise ValueError("support quantile must lie in [0, 0.5)")
    bounds = []
    for area in HVA_ORDER:
        local = table.loc[table["area"].eq(area)]
        if local.empty:
            raise ValueError(f"No RF observations for {area}")
        bounds.append(
            {
                "az_low": float(local["relative_azimuth_deg"].quantile(quantile)),
                "az_high": float(local["relative_azimuth_deg"].quantile(1 - quantile)),
                "el_low": float(local["relative_elevation_deg"].quantile(quantile)),
                "el_high": float(local["relative_elevation_deg"].quantile(1 - quantile)),
            }
        )
    result = {
        "azimuth_low_deg": max(row["az_low"] for row in bounds),
        "azimuth_high_deg": min(row["az_high"] for row in bounds),
        "elevation_low_deg": max(row["el_low"] for row in bounds),
        "elevation_high_deg": min(row["el_high"] for row in bounds),
    }
    if result["azimuth_low_deg"] >= result["azimuth_high_deg"] or result["elevation_low_deg"] >= result["elevation_high_deg"]:
        raise ValueError(f"Five-area RF overlap box is empty: {result}")
    return result


def add_common_box_flag(table: pd.DataFrame, box: dict[str, float]) -> pd.DataFrame:
    result = table.copy()
    result["inside_hva_common_box"] = (
        result["relative_azimuth_deg"].between(box["azimuth_low_deg"], box["azimuth_high_deg"])
        & result["relative_elevation_deg"].between(box["elevation_low_deg"], box["elevation_high_deg"])
    )
    return result


def build_unit_and_cell_tables(
    outcomes: dict[str, pd.DataFrame],
    support: pd.DataFrame,
    *,
    quantile: float,
    min_units: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_frames = []
    flow_rows = []
    box_rows = []
    for metric, exact in outcomes.items():
        joined = exact.merge(
            support,
            on=["ecephys_unit_id", "ecephys_session_id", "area"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        rf = joined.loc[joined["_merge"].eq("both")].drop(columns="_merge").copy()
        box = robust_common_box(rf, quantile)
        rf = add_common_box_flag(rf, box)
        rf["metric"] = metric
        unit_frames.append(rf)
        box_rows.append({"metric": metric, "support_quantile": quantile, **box})
        for stage, frame in (
            ("exact_metric_population", exact),
            ("published_quality_rf_intersection", rf),
            ("five_area_common_box", rf.loc[rf["inside_hva_common_box"]]),
        ):
            flow_rows.append(
                {
                    "metric": metric,
                    "stage": stage,
                    "units": len(frame),
                    "sessions": frame["ecephys_session_id"].nunique(),
                    "session_area_cells": frame.groupby(["ecephys_session_id", "area"]).ngroups,
                }
            )

    units = pd.concat(unit_frames, ignore_index=True)
    cell_frames = []
    cell_flow = []
    for metric in METRIC_ORDER:
        local = units.loc[units["metric"].eq(metric)].copy()
        for population, subset in (
            ("rf_quality", local),
            ("common_box", local.loc[local["inside_hva_common_box"]]),
        ):
            cells = (
                subset.groupby(["ecephys_session_id", "area"], sort=True)
                .agg(
                    outcome=("outcome", "mean"),
                    n_units=("ecephys_unit_id", "size"),
                    rf_azimuth_deg=("relative_azimuth_deg", "median"),
                    rf_elevation_deg=("relative_elevation_deg", "median"),
                    rf_azimuth_iqr_deg=("relative_azimuth_deg", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                    rf_elevation_iqr_deg=("relative_elevation_deg", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                )
                .reset_index()
            )
            before = len(cells)
            required_min_units = max(min_units, ORIGINAL_MIN_UNITS[metric])
            cells = cells.loc[cells["n_units"].ge(required_min_units)].copy()
            eligible_sessions = cells.groupby("ecephys_session_id")["area"].transform("nunique").ge(2)
            cells = cells.loc[eligible_sessions].copy()
            cells["session_mean"] = cells.groupby("ecephys_session_id")["outcome"].transform("mean")
            cells["centered_outcome"] = cells["outcome"] - cells["session_mean"]
            cells["hierarchy_score"] = cells["area"].map(HIERARCHY_SCORE)
            cells["metric"] = metric
            cells["population"] = population
            cells["required_min_units"] = required_min_units
            cell_frames.append(cells)
            cell_flow.append(
                {
                    "metric": metric,
                    "population": population,
                    "raw_cells": before,
                    "eligible_cells": len(cells),
                    "eligible_sessions": cells["ecephys_session_id"].nunique(),
                    "median_areas_per_session": float(cells.groupby("ecephys_session_id")["area"].nunique().median()) if len(cells) else np.nan,
                    "required_min_units": required_min_units,
                }
            )
    return (
        units,
        pd.concat(cell_frames, ignore_index=True),
        pd.DataFrame(flow_rows),
        pd.DataFrame(box_rows),
        pd.DataFrame(cell_flow),
    )


def raw_features(table: pd.DataFrame, model: str, *, label_column: str = "area") -> tuple[np.ndarray, list[bool]]:
    arrays = []
    continuous = []
    if model in {"retinotopy", "hierarchy", "categorical_hva"}:
        rf = table[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
        arrays.append(PolynomialFeatures(degree=2, include_bias=False).fit_transform(rf))
        continuous.extend([True] * 5)
    if model == "hierarchy":
        score = table[label_column].map(HIERARCHY_SCORE).to_numpy(float)[:, None]
        arrays.append(score)
        continuous.append(True)
    if model == "categorical_hva":
        labels = table[label_column].astype(str)
        one_hot = np.column_stack([labels.eq(area).to_numpy(float) for area in HVA_ORDER[1:]])
        arrays.append(one_hot)
        continuous.extend([False] * len(HVA_ORDER[1:]))
    if not arrays:
        return np.empty((len(table), 0)), []
    return np.column_stack(arrays), continuous


def transform_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: str,
    *,
    label_column: str = "area",
) -> tuple[np.ndarray, np.ndarray]:
    x_train, continuous = raw_features(train, model, label_column=label_column)
    x_test, _ = raw_features(test, model, label_column=label_column)
    if any(continuous):
        index = np.asarray(continuous, dtype=bool)
        scaler = StandardScaler().fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])
    return x_train, x_test


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: str,
    alpha: float,
    *,
    label_column: str = "area",
) -> tuple[np.ndarray, float]:
    y_train = train["centered_outcome"].to_numpy(float)
    if model == "pooled_v2":
        prediction = np.zeros(len(test), dtype=float)
        sigma = float(np.sqrt(np.mean(y_train**2)))
        return prediction, max(sigma, 1e-8)
    x_train, x_test = transform_features(train, test, model, label_column=label_column)
    fit = Ridge(alpha=alpha, fit_intercept=True).fit(x_train, y_train)
    residual = y_train - fit.predict(x_train)
    sigma = float(np.sqrt(np.mean(residual**2)))
    return fit.predict(x_test), max(sigma, 1e-8)


def choose_alpha(train: pd.DataFrame, model: str, *, label_column: str = "area") -> float:
    sessions = sorted(train["ecephys_session_id"].unique())
    if len(sessions) < 5:
        return 1.0
    errors = {alpha: 0.0 for alpha in ALPHAS}
    counts = {alpha: 0 for alpha in ALPHAS}
    splitter = GroupKFold(n_splits=min(5, len(sessions)))
    groups = train["ecephys_session_id"].to_numpy()
    for train_indices, test_indices in splitter.split(train, groups=groups):
        inner_train = train.iloc[train_indices]
        inner_test = train.iloc[test_indices]
        for alpha in ALPHAS:
            prediction, _ = fit_predict(
                inner_train, inner_test, model, alpha, label_column=label_column
            )
            errors[alpha] += float(np.sum((inner_test["centered_outcome"].to_numpy() - prediction) ** 2))
            counts[alpha] += len(inner_test)
    return min(ALPHAS, key=lambda alpha: errors[alpha] / counts[alpha])


def loso_predictions(
    table: pd.DataFrame,
    *,
    models: tuple[str, ...] = MODEL_ORDER,
    label_column: str = "area",
    fixed_alphas: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for session in sorted(table["ecephys_session_id"].unique()):
        train = table.loc[~table["ecephys_session_id"].eq(session)].copy()
        test = table.loc[table["ecephys_session_id"].eq(session)].copy()
        train_scale = float(np.sqrt(np.mean(train["centered_outcome"].to_numpy(float) ** 2)))
        train_scale = max(train_scale, 1e-8)
        for model in models:
            alpha = 0.0 if model == "pooled_v2" else (
                fixed_alphas[model] if fixed_alphas is not None else choose_alpha(train, model, label_column=label_column)
            )
            prediction, sigma = fit_predict(train, test, model, alpha, label_column=label_column)
            observed = test["centered_outcome"].to_numpy(float)
            for record, predicted, value in zip(test.to_dict("records"), prediction, observed):
                rows.append(
                    {
                        "metric": record["metric"],
                        "population": record["population"],
                        "session_id": record["ecephys_session_id"],
                        "area": record["area"],
                        "model": model,
                        "observed": value,
                        "predicted": float(predicted),
                        "squared_error": float((value - predicted) ** 2),
                        "scaled_squared_error": float(((value - predicted) / train_scale) ** 2),
                        "log_predictive_density": float(norm.logpdf(value, loc=predicted, scale=sigma)),
                        "selected_alpha": alpha,
                        "n_units": record["n_units"],
                        "rf_azimuth_deg": record["rf_azimuth_deg"],
                        "rf_elevation_deg": record["rf_elevation_deg"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["metric", "population"]
    for (metric, population), local in predictions.groupby(keys, sort=False):
        pooled_sse = local.loc[local["model"].eq("pooled_v2"), "squared_error"].sum()
        rf_sse = local.loc[local["model"].eq("retinotopy"), "squared_error"].sum()
        for model in MODEL_ORDER:
            part = local.loc[local["model"].eq(model)]
            sse = float(part["squared_error"].sum())
            rows.append(
                {
                    "metric": metric,
                    "population": population,
                    "model": model,
                    "cv_r2_vs_pooled": 1 - sse / pooled_sse if pooled_sse else np.nan,
                    "incremental_r2_vs_retinotopy": (rf_sse - sse) / pooled_sse if pooled_sse else np.nan,
                    "mean_log_predictive_density": part["log_predictive_density"].mean(),
                    "cells": len(part),
                    "sessions": part["session_id"].nunique(),
                    "median_selected_alpha": part["selected_alpha"].median(),
                }
            )
    return pd.DataFrame(rows)


def session_bootstrap(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for (metric, population), local in predictions.groupby(["metric", "population"], sort=False):
        pivot = local.pivot_table(
            index="session_id", columns="model", values="squared_error", aggfunc="sum"
        ).dropna(subset=list(MODEL_ORDER))
        sessions = pivot.index.to_numpy()
        for draw in range(n_bootstrap):
            sampled = rng.choice(sessions, len(sessions), replace=True)
            sums = pivot.loc[sampled].sum()
            pooled = sums["pooled_v2"]
            rf = sums["retinotopy"]
            for model in MODEL_ORDER:
                rows.append(
                    {
                        "metric": metric,
                        "population": population,
                        "draw": draw,
                        "model": model,
                        "cv_r2_vs_pooled": 1 - sums[model] / pooled,
                        "incremental_r2_vs_retinotopy": (rf - sums[model]) / pooled,
                    }
                )
    return pd.DataFrame(rows)


def permute_labels_within_sessions(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = table.copy()
    labels = result["area"].astype(str).to_numpy(copy=True)
    for indices in result.groupby("ecephys_session_id", sort=False).indices.values():
        idx = np.asarray(indices, dtype=int)
        labels[idx] = rng.permutation(labels[idx])
    result["permuted_area"] = labels
    return result


def area_label_permutation_null(
    cells: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Fast within-session label null with observed-model ridge strengths.

    RF polynomial features and fold-specific scaling are invariant to the label
    shuffle, so they are cached once.  The direct ridge solve is algebraically
    equivalent to an intercept-bearing least-squares ridge fit with unpenalized
    intercept and makes the session-preserving null computationally practical.
    """
    def ridge_prediction(x_train, y_train, x_test, alpha):
        x_mean = x_train.mean(axis=0)
        y_mean = y_train.mean()
        x_centered = x_train - x_mean
        y_centered = y_train - y_mean
        gram = x_centered.T @ x_centered + alpha * np.eye(x_train.shape[1])
        coefficient = np.linalg.solve(gram, x_centered.T @ y_centered)
        return (x_test - x_mean) @ coefficient + y_mean

    rows = []
    for metric in METRIC_ORDER:
        local = cells.loc[cells["metric"].eq(metric)].copy().reset_index(drop=True)
        if local["ecephys_session_id"].nunique() < 10:
            continue
        summary = observed.loc[observed["metric"].eq(metric)]
        fixed = {
            model: float(summary.loc[summary["model"].eq(model), "median_selected_alpha"].iloc[0])
            for model in ("retinotopy", "hierarchy", "categorical_hva")
        }
        observed_metric = observed.loc[observed["metric"].eq(metric)].set_index("model")
        # Recover the common SSE scale from R2 relations.  The absolute scale
        # is computed directly for clarity and the RF SSE follows from it.
        pooled_sse = float(np.sum(local["centered_outcome"].to_numpy(float) ** 2))
        rf_sse = pooled_sse * (1 - observed_metric.loc["retinotopy", "cv_r2_vs_pooled"])
        rf_raw = PolynomialFeatures(degree=2, include_bias=False).fit_transform(
            local[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
        )
        y = local["centered_outcome"].to_numpy(float)
        session_indices = [
            np.asarray(indices, dtype=int)
            for indices in local.groupby("ecephys_session_id", sort=False).indices.values()
        ]
        folds = []
        for test_indices in session_indices:
            train_mask = np.ones(len(local), dtype=bool)
            train_mask[test_indices] = False
            train_indices = np.flatnonzero(train_mask)
            scaler = StandardScaler().fit(rf_raw[train_indices])
            folds.append(
                (
                    train_indices,
                    test_indices,
                    scaler.transform(rf_raw[train_indices]),
                    scaler.transform(rf_raw[test_indices]),
                )
            )
        for permutation in range(n_permutations):
            labels = local["area"].astype(str).to_numpy(copy=True)
            for indices in session_indices:
                labels[indices] = rng.permutation(labels[indices])
            for model in ("hierarchy", "categorical_hva"):
                sse = 0.0
                for train_indices, test_indices, rf_train, rf_test in folds:
                    if model == "hierarchy":
                        extra_train = np.array([HIERARCHY_SCORE[label] for label in labels[train_indices]], dtype=float)[:, None]
                        extra_test = np.array([HIERARCHY_SCORE[label] for label in labels[test_indices]], dtype=float)[:, None]
                        score_scaler = StandardScaler().fit(extra_train)
                        extra_train = score_scaler.transform(extra_train)
                        extra_test = score_scaler.transform(extra_test)
                    else:
                        extra_train = np.column_stack([labels[train_indices] == area for area in HVA_ORDER[1:]]).astype(float)
                        extra_test = np.column_stack([labels[test_indices] == area for area in HVA_ORDER[1:]]).astype(float)
                    prediction = ridge_prediction(
                        np.column_stack([rf_train, extra_train]),
                        y[train_indices],
                        np.column_stack([rf_test, extra_test]),
                        fixed[model],
                    )
                    sse += float(np.sum((y[test_indices] - prediction) ** 2))
                rows.append(
                    {
                        "metric": metric,
                        "permutation": permutation,
                        "model": model,
                        "incremental_r2_vs_retinotopy": (rf_sse - sse) / pooled_sse,
                    }
                )
    return pd.DataFrame(rows)


def split_half_reproducibility(
    predictions: pd.DataFrame,
    *,
    n_splits: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rf = predictions.loc[predictions["model"].eq("retinotopy")].copy()
    rf["rf_residual"] = rf["observed"] - rf["predicted"]
    rows = []
    for metric in METRIC_ORDER:
        local = rf.loc[rf["metric"].eq(metric)]
        if local.empty:
            continue
        sessions = np.array(sorted(local["session_id"].unique()))
        for split in range(n_splits):
            shuffled = rng.permutation(sessions)
            first = set(shuffled[: len(shuffled) // 2])
            a = local.loc[local["session_id"].isin(first)].groupby("area")["rf_residual"].mean().reindex(HVA_ORDER)
            b = local.loc[~local["session_id"].isin(first)].groupby("area")["rf_residual"].mean().reindex(HVA_ORDER)
            valid = a.notna() & b.notna()
            correlation = pearsonr(a[valid], b[valid]).statistic if valid.sum() >= 3 else np.nan
            rows.append({"metric": metric, "split": split, "correlation": correlation, "areas": int(valid.sum())})
    return pd.DataFrame(rows)


def residual_area_summary(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rf = predictions.loc[
        predictions["population"].eq("common_box") & predictions["model"].eq("retinotopy")
    ].copy()
    rf["rf_residual"] = rf["observed"] - rf["predicted"]
    rows = []
    for metric, local in rf.groupby("metric"):
        sessions = np.array(sorted(local["session_id"].unique()))
        draws = {area: [] for area in HVA_ORDER}
        groups = {session: local.loc[local["session_id"].eq(session)] for session in sessions}
        for _ in range(n_bootstrap):
            sampled = rng.choice(sessions, len(sessions), replace=True)
            draw = pd.concat([groups[session] for session in sampled], ignore_index=True)
            means = draw.groupby("area")["rf_residual"].mean()
            for area in HVA_ORDER:
                if area in means:
                    draws[area].append(float(means[area]))
        observed = local.groupby("area")["rf_residual"].agg(["mean", "median", "size"])
        for area in HVA_ORDER:
            if area not in observed.index:
                continue
            low, high = np.percentile(draws[area], [2.5, 97.5])
            rows.append(
                {
                    "metric": metric,
                    "area": area,
                    "hierarchy_score": HIERARCHY_SCORE[area],
                    "mean_rf_residual": observed.loc[area, "mean"],
                    "median_rf_residual": observed.loc[area, "median"],
                    "cells": int(observed.loc[area, "size"]),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def add_inference(summary: pd.DataFrame, boot: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["bootstrap_ci_low"] = np.nan
    result["bootstrap_ci_high"] = np.nan
    result["incremental_ci_low"] = np.nan
    result["incremental_ci_high"] = np.nan
    result["permutation_p"] = np.nan
    for index, row in result.iterrows():
        b = boot.loc[
            boot["metric"].eq(row.metric)
            & boot["population"].eq(row.population)
            & boot["model"].eq(row.model)
        ]
        if not b.empty:
            result.loc[index, ["bootstrap_ci_low", "bootstrap_ci_high"]] = np.percentile(b["cv_r2_vs_pooled"], [2.5, 97.5])
            result.loc[index, ["incremental_ci_low", "incremental_ci_high"]] = np.percentile(b["incremental_r2_vs_retinotopy"], [2.5, 97.5])
        n = null.loc[null["metric"].eq(row.metric) & null["model"].eq(row.model)] if row.population == "common_box" else pd.DataFrame()
        if not n.empty:
            observed = row.incremental_r2_vs_retinotopy
            result.loc[index, "permutation_p"] = (1 + (n["incremental_r2_vs_retinotopy"] >= observed).sum()) / (len(n) + 1)
    return result


def add_categorical_vs_hierarchy(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    equivalence_margin: float,
) -> pd.DataFrame:
    result = summary.copy()
    for column in (
        "categorical_r2_minus_hierarchy",
        "categorical_minus_hierarchy_ci_low",
        "categorical_minus_hierarchy_ci_high",
        "categorical_gain_below_margin",
    ):
        result[column] = np.nan
    for (metric, population), indices in result.groupby(["metric", "population"]).groups.items():
        local = result.loc[indices].set_index("model")
        if not {"hierarchy", "categorical_hva"}.issubset(local.index):
            continue
        point = local.loc["categorical_hva", "cv_r2_vs_pooled"] - local.loc["hierarchy", "cv_r2_vs_pooled"]
        draws = bootstrap.loc[
            bootstrap["metric"].eq(metric) & bootstrap["population"].eq(population)
        ].pivot(index="draw", columns="model", values="cv_r2_vs_pooled")
        difference = draws["categorical_hva"] - draws["hierarchy"]
        low, high = np.percentile(difference, [2.5, 97.5])
        target = result.index[
            result["metric"].eq(metric)
            & result["population"].eq(population)
            & result["model"].eq("categorical_hva")
        ]
        result.loc[target, "categorical_r2_minus_hierarchy"] = point
        result.loc[target, "categorical_minus_hierarchy_ci_low"] = low
        result.loc[target, "categorical_minus_hierarchy_ci_high"] = high
        result.loc[target, "categorical_gain_below_margin"] = float(high < equivalence_margin)
    return result


def select_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    local = predictions.loc[predictions["population"].eq("common_box")].copy()
    wide = local.pivot_table(
        index=["metric", "session_id", "area", "observed", "n_units", "rf_azimuth_deg", "rf_elevation_deg"],
        columns="model", values=["predicted", "squared_error"]
    ).reset_index()
    wide.columns = ["::".join([str(x) for x in column if str(x)]) if isinstance(column, tuple) else column for column in wide.columns]
    wide["categorical_gain"] = wide["squared_error::retinotopy"] - wide["squared_error::categorical_hva"]
    selected = []
    for metric, part in wide.groupby("metric"):
        ordered = part.sort_values("categorical_gain")
        roles = [
            ("strongest categorical failure", ordered.iloc[0]),
            ("boundary case", ordered.iloc[(ordered["categorical_gain"].abs()).argmin()]),
            ("strongest categorical improvement", ordered.iloc[-1]),
        ]
        for role, row in roles:
            record = row.to_dict()
            record["selection_role"] = role
            selected.append(record)
    return pd.DataFrame(selected)


def render_support(units: pd.DataFrame, boxes: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    rng = np.random.default_rng(SEED)
    for axis, metric in zip(axes, METRIC_ORDER):
        local = units.loc[units["metric"].eq(metric)]
        for area in HVA_ORDER:
            part = local.loc[local["area"].eq(area)]
            if len(part) > 600:
                part = part.iloc[rng.choice(len(part), 600, replace=False)]
            axis.scatter(part["relative_azimuth_deg"], part["relative_elevation_deg"], s=7, alpha=0.2, color=AREA_COLORS[area], label=area)
        box = boxes.loc[boxes["metric"].eq(metric)].iloc[0]
        axis.add_patch(
            plt.Rectangle(
                (box.azimuth_low_deg, box.elevation_low_deg),
                box.azimuth_high_deg - box.azimuth_low_deg,
                box.elevation_high_deg - box.elevation_low_deg,
                fill=False, color="black", linewidth=1.5, linestyle="--",
            )
        )
        axis.axhline(0, color="#999999", linewidth=0.5)
        axis.axvline(0, color="#999999", linewidth=0.5)
        axis.set(title=metric, xlabel="RF azimuth relative to V1 (deg)", ylabel="RF elevation relative to V1 (deg)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("Metric-specific HVA receptive-field support", fontsize=14, fontweight="bold", y=1.08)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_models(summary: pd.DataFrame, output: Path) -> None:
    primary = summary.loc[summary["population"].eq("common_box") & ~summary["model"].eq("pooled_v2")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False, constrained_layout=True)
    colors = {"retinotopy": "#777777", "hierarchy": "#4C78A8", "categorical_hva": "#E07B39"}
    for axis, metric in zip(axes, METRIC_ORDER):
        part = primary.loc[primary["metric"].eq(metric)].set_index("model").reindex(["retinotopy", "hierarchy", "categorical_hva"]).reset_index()
        if part["cv_r2_vs_pooled"].isna().all():
            axis.text(0.5, 0.5, "Insufficient cells at\noriginal eligibility threshold", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(metric)
            axis.set_axis_off()
            continue
        x = np.arange(len(part))
        y = part["cv_r2_vs_pooled"].to_numpy(float)
        low = part["bootstrap_ci_low"].to_numpy(float)
        high = part["bootstrap_ci_high"].to_numpy(float)
        axis.bar(x, y, color=[colors[m] for m in part.model], alpha=0.85)
        axis.errorbar(x, y, yerr=np.vstack([y-low, high-y]), fmt="none", color="black", capsize=3, linewidth=1)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["RF", "RF +\nhierarchy", "RF +\nHVA"], rotation=0)
        axis.set_title(metric)
        axis.set_ylabel("Held-out-session R² vs pooled V2")
        span = max(float(np.nanmax(high) - np.nanmin(low)), 0.08)
        axis.set_ylim(float(np.nanmin(low) - 0.08 * span), float(np.nanmax(high) + 0.22 * span))
        for xi, row in part.iterrows():
            if row.model in {"hierarchy", "categorical_hva"} and np.isfinite(row.permutation_p):
                axis.text(xi, high[xi] + 0.015, f"p={row.permutation_p:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Does hierarchy or named HVA improve prediction after retinotopy?", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_separateness(
    residuals: pd.DataFrame,
    summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    timescale = residuals.loc[residuals["metric"].eq("Response timescale (ms)")].sort_values("hierarchy_score")
    axis = axes[0]
    for row in timescale.itertuples():
        axis.errorbar(
            row.hierarchy_score, row.mean_rf_residual,
            yerr=[[row.mean_rf_residual - row.bootstrap_ci_low], [row.bootstrap_ci_high - row.mean_rf_residual]],
            fmt="o", color=AREA_COLORS[row.area], capsize=3, markersize=7,
        )
        axis.text(row.hierarchy_score, row.bootstrap_ci_high + 0.35, row.area, ha="center", fontsize=9)
    if len(timescale) >= 2:
        coefficient = np.polyfit(timescale["hierarchy_score"], timescale["mean_rf_residual"], 1)
        grid = np.linspace(timescale["hierarchy_score"].min(), timescale["hierarchy_score"].max(), 100)
        axis.plot(grid, np.polyval(coefficient, grid), color="#4C78A8", linewidth=1.2)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(title="RF-adjusted timescale pattern", xlabel="Published hierarchy score", ylabel="Mean held-out RF residual (ms)")

    axis = axes[1]
    direct = summary.loc[
        summary["population"].eq("common_box") & summary["model"].eq("categorical_hva")
    ].copy()
    y = np.arange(len(direct))
    point = direct["categorical_r2_minus_hierarchy"].to_numpy(float)
    low = direct["categorical_minus_hierarchy_ci_low"].to_numpy(float)
    high = direct["categorical_minus_hierarchy_ci_high"].to_numpy(float)
    axis.errorbar(point, y, xerr=np.vstack([point-low, high-point]), fmt="o", color="#E07B39", capsize=3)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvline(0.05, color="#777777", linewidth=1.0, linestyle="--")
    axis.set_yticks(y, direct["metric"])
    axis.set(title="Five categories minus continuum", xlabel="Categorical − hierarchy held-out R²")
    axis.text(0.05, 0.98, "exploratory 5-point margin", rotation=90, va="top", ha="right", color="#666666", fontsize=8, transform=axis.get_xaxis_transform())

    axis = axes[2]
    rep_rows = []
    for metric, part in reproducibility.groupby("metric"):
        valid = part["correlation"].dropna()
        rep_rows.append((metric, valid.median(), valid.quantile(0.025), valid.quantile(0.975)))
    rep = pd.DataFrame(rep_rows, columns=["metric", "median", "low", "high"])
    y = np.arange(len(rep))
    axis.errorbar(rep["median"], y, xerr=np.vstack([rep["median"]-rep["low"], rep["high"]-rep["median"]]), fmt="o", color="#4C78A8", capsize=3)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, rep["metric"])
    axis.set_xlim(-1.02, 1.02)
    axis.set(title="Area-pattern split-half stability", xlabel="Correlation across session halves")

    fig.suptitle("A timescale continuum is more supported than five separate functional categories", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    *,
    reconstruction: pd.DataFrame,
    flow: pd.DataFrame,
    cell_flow: pd.DataFrame,
    summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    cases: pd.DataFrame,
    residuals: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    primary = summary.loc[summary["population"].eq("common_box")]
    lines = [
        "# Retinotopy-adjusted HVA model comparison", "",
        "## Technical summary", "",
        "This exploratory checkpoint compares a retinotopy-only heterogeneous-V2 model with",
        "models that additionally use either the published hierarchy score or categorical HVA",
        "identity. All predictions hold out complete sessions. The analysis concerns whether",
        "the present functional measurements identify stable area effects; it does not test",
        "whether the anatomical areas exist.", "",
        "## Population contract", "",
        "Each outcome is rebuilt from its exact Figure 3 unit source and then intersected with",
        "the published-quality RF population. The primary population is restricted to the",
        f"{100*args.support_quantile:.1f}–{100*(1-args.support_quantile):.1f}% five-area RF overlap box,",
        "preserves the original per-metric unit threshold (10 for TTFS; 5 for F1/F0",
        "and timescale), and requires at least",
        "two eligible HVAs in a session. Outcomes are centered within session.", "",
        "### Exact-source reconstruction", "",
        reconstruction.to_markdown(index=False, floatfmt=".3g"), "",
        "### Eligible cell coverage", "",
        cell_flow.to_markdown(index=False, floatfmt=".3g"), "",
        "Strict common RF support leaves only four TTFS sessions at the original",
        "10-unit threshold, so no primary TTFS model is fit. This is an identification",
        "limit rather than a null TTFS result; the broader RF-quality population remains",
        "a sensitivity analysis but does not enforce five-area overlap.", "",
        "## Held-out-session model comparison", "",
        "R² is relative to the pooled-V2 prediction of zero after within-session centering.",
        "Incremental R² compares hierarchy or categorical identity with the retinotopy-only",
        "model on the same cells. Confidence intervals resample complete held-out sessions;",
        "permutation p-values shuffle area labels within sessions while retaining outcomes and RF coordinates.", "",
        primary[["metric", "model", "cv_r2_vs_pooled", "bootstrap_ci_low", "bootstrap_ci_high", "incremental_r2_vs_retinotopy", "incremental_ci_low", "incremental_ci_high", "permutation_p", "cells", "sessions"]].to_markdown(index=False, floatfmt=".4f"), "",
        "### Does a five-category model add anything beyond the hierarchy continuum?", "",
        "The following paired comparison subtracts hierarchy-model R² from categorical-HVA",
        "R². The 5-percentage-point margin is an exploratory smallest effect of interest,",
        "not a preregistered confirmatory threshold.", "",
        primary.loc[primary["model"].eq("categorical_hva"), ["metric", "categorical_r2_minus_hierarchy", "categorical_minus_hierarchy_ci_low", "categorical_minus_hierarchy_ci_high", "categorical_gain_below_margin"]].to_markdown(index=False, floatfmt=".4f"), "",
        "## Split-half reproducibility", "",
    ]
    rep_rows = []
    for metric, part in reproducibility.groupby("metric"):
        valid = part["correlation"].dropna()
        rep_rows.append({
            "metric": metric,
            "median_area_pattern_r": valid.median(),
            "ci_low": valid.quantile(0.025),
            "ci_high": valid.quantile(0.975),
            "positive_share": (valid > 0).mean(),
        })
    lines.extend([pd.DataFrame(rep_rows).to_markdown(index=False, floatfmt=".3f"), ""])
    lines.extend([
        "The split-half statistic correlates the five named-area means of RF-only held-out",
        "residuals across random session halves. A stable categorical fingerprint should be",
        "positive and reproducible across splits.", "",
        "### RF-adjusted area residuals", "",
        residuals.to_markdown(index=False, floatfmt=".3f"), "",
        "## Auditable cases", "",
        cases[["metric", "session_id", "area", "selection_role", "categorical_gain", "observed", "predicted::retinotopy", "predicted::categorical_hva", "n_units", "rf_azimuth_deg", "rf_elevation_deg"]].to_markdown(index=False, floatfmt=".3f"), "",
        "## Interpretation boundary", "",
        "A failure of hierarchy or categorical identity to improve held-out prediction is",
        "evidence about these response metrics under achieved RF sampling, not proof that the",
        "named anatomical fields are absent. Global overlap-box construction is outcome-blind",
        "but exploratory; the all-RF-quality population is retained as a sensitivity analysis.", "",
        "## Outputs", "",
        "- `metric_specific_unit_population.csv`: exact outcomes joined to RF coordinates.",
        "- `metric_specific_cell_population.csv`: session × area model observations.",
        "- `population_flow.csv` and `cell_coverage.csv`: attrition and eligibility audit.",
        "- `loso_predictions.csv`: every held-out prediction and error.",
        "- `model_comparison_summary.csv`: predictive scores, uncertainty, and permutation tests.",
        "- `split_half_reproducibility.csv`: area-pattern replication across session halves.",
        "- `selected_cases.csv`: transparently selected successes, failures, and boundaries.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    outcomes = load_exact_outcomes()
    reconstruction = validate_frozen_reconstruction(outcomes)
    support = load_rf_support()
    units, cells, flow, boxes, cell_flow = build_unit_and_cell_tables(
        outcomes, support, quantile=args.support_quantile, min_units=args.min_units
    )

    predictions = []
    for population in ("common_box", "rf_quality"):
        for metric in METRIC_ORDER:
            local = cells.loc[cells["population"].eq(population) & cells["metric"].eq(metric)]
            if local["ecephys_session_id"].nunique() < 10:
                continue
            predictions.append(loso_predictions(local))
    predictions = pd.concat(predictions, ignore_index=True)
    basic_summary = summarize_predictions(predictions)
    bootstrap = session_bootstrap(
        predictions, n_bootstrap=args.bootstraps, rng=np.random.default_rng(args.seed + 1)
    )
    primary_cells = cells.loc[cells["population"].eq("common_box")].copy()
    null = area_label_permutation_null(
        primary_cells,
        basic_summary.loc[basic_summary["population"].eq("common_box")],
        n_permutations=args.permutations,
        rng=np.random.default_rng(args.seed + 2),
    )
    summary = add_inference(basic_summary, bootstrap, null)
    summary = add_categorical_vs_hierarchy(
        summary, bootstrap, equivalence_margin=args.equivalence_margin_r2
    )
    reproducibility = split_half_reproducibility(
        predictions.loc[predictions["population"].eq("common_box")],
        n_splits=args.split_halves,
        rng=np.random.default_rng(args.seed + 3),
    )
    residuals = residual_area_summary(
        predictions,
        n_bootstrap=args.bootstraps,
        rng=np.random.default_rng(args.seed + 4),
    )
    cases = select_cases(predictions)

    reconstruction.to_csv(output / "frozen_reconstruction_audit.csv", index=False)
    units.to_csv(output / "metric_specific_unit_population.csv", index=False)
    cells.to_csv(output / "metric_specific_cell_population.csv", index=False)
    flow.to_csv(output / "population_flow.csv", index=False)
    boxes.to_csv(output / "common_support_boxes.csv", index=False)
    cell_flow.to_csv(output / "cell_coverage.csv", index=False)
    predictions.to_csv(output / "loso_predictions.csv", index=False)
    bootstrap.to_csv(output / "session_bootstrap.csv", index=False)
    null.to_csv(output / "area_label_permutation_null.csv", index=False)
    summary.to_csv(output / "model_comparison_summary.csv", index=False)
    reproducibility.to_csv(output / "split_half_reproducibility.csv", index=False)
    residuals.to_csv(output / "rf_residual_area_summary.csv", index=False)
    cases.to_csv(output / "selected_cases.csv", index=False)
    render_support(units, boxes, output / "Figure_metric_specific_rf_support.png")
    render_models(summary, output / "Figure_retinotopy_adjusted_model_comparison.png")
    render_separateness(
        residuals,
        summary,
        reproducibility,
        output / "Figure_hierarchy_continuum_vs_separate_hvas.png",
    )
    write_report(
        output / "RETINOTOPY_ADJUSTED_HVA_MODELS.md",
        reconstruction=reconstruction,
        flow=flow,
        cell_flow=cell_flow,
        summary=summary,
        reproducibility=reproducibility,
        cases=cases,
        residuals=residuals,
        args=args,
    )
    manifest = {
        "analysis": "retinotopy_adjusted_hva_models",
        "parameters": vars(args) | {"output_dir": str(output)},
        "sources": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (SUPPORT_PATH, TTFS_PATH, F1_PATH, UNIT_PATH, FROZEN_MEANS)
        },
        "models": MODEL_LABELS,
        "primary_population": "common_box",
        "interpretation_boundary": "Functional identification under measured RF sampling; not anatomical existence.",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(summary.loc[summary["population"].eq("common_box")].to_string(index=False))


if __name__ == "__main__":
    main()
