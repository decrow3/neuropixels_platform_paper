#!/usr/bin/env python3
"""Fit session-aware Allen response models adjusted for achieved RF position.

The script consumes the Iteration 6C targeting-audit contract, fits within-
session area contrasts with session-clustered uncertainty, and performs a
same-session nearest-neighbor V1 matching sensitivity on two-dimensional RF
centers.  It does not modify the released Allen unit table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_config  # noqa: E402


DEFAULT_AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_OUTPUT = DEFAULT_AUDIT / "response_adjustment"
AREA_ORDER = ("V1", "LM", "RL", "AL", "PM", "AM")
HVA_ORDER = AREA_ORDER[1:]
AREA_COLORS = {
    "V1": "#4D4D4D",
    "LM": "#1B9E77",
    "RL": "#D95F02",
    "AL": "#7570B3",
    "PM": "#E7298A",
    "AM": "#66A61E",
}
HIERARCHY_SCORE = {
    "V1": -0.357,
    "LM": -0.093,
    "RL": -0.059,
    "AL": 0.152,
    "PM": 0.327,
    "AM": 0.441,
}
OUTCOME_LABELS = {
    "ttfs_ms": "TTFS (ms)",
    "log10_mod_idx": "log10 modulation index",
    "log10_f1_f0": "log10 F1/F0",
    "timescale_ms": "response timescale (ms)",
    "rf_area_deg2": "RF area (deg²)",
}
OUTCOME_ORDER = tuple(OUTCOME_LABELS)
RESPONSE_COLUMNS = {
    "time_to_first_spike_fl",
    "mod_idx_dg",
    "f1_f0_dg",
    "timescale_ac",
    "err_ac",
    "spike_count_ac",
    "cortical_depth",
    "cortical_layer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=611)
    parser.add_argument(
        "--match-caliper-deg",
        type=float,
        default=10.0,
        help="Maximum same-session HVA-to-V1 RF-center match distance.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_outcomes(table: pd.DataFrame) -> pd.DataFrame:
    """Add paper-compatible metric outcomes with invalid values set missing."""
    result = table.copy()
    ttfs = pd.to_numeric(result["time_to_first_spike_fl"], errors="coerce")
    modulation = pd.to_numeric(result["mod_idx_dg"], errors="coerce")
    f1_f0 = pd.to_numeric(result["f1_f0_dg"], errors="coerce")
    timescale = pd.to_numeric(result["timescale_ac"], errors="coerce")
    fit_error = pd.to_numeric(result["err_ac"], errors="coerce")
    spike_count = pd.to_numeric(result["spike_count_ac"], errors="coerce")
    result["ttfs_ms"] = (ttfs * 1000).where(ttfs.lt(0.1))
    result["log10_mod_idx"] = np.log10(modulation.where(modulation.gt(0)))
    result["log10_f1_f0"] = np.log10(f1_f0.where(f1_f0.gt(0)))
    result["timescale_ms"] = timescale.where(
        timescale.between(1, 300) & spike_count.gt(50) & fit_error.lt(20)
    )
    result["rf_area_deg2"] = pd.to_numeric(result["area_rf"], errors="coerce")
    result["rf_area_z"] = (
        result["rf_area_deg2"] - result["rf_area_deg2"].mean()
    ) / result["rf_area_deg2"].std(ddof=0)
    depth = pd.to_numeric(result["cortical_depth"], errors="coerce")
    result["cortical_depth_z"] = (depth - depth.mean()) / depth.std(ddof=0)
    result["session_id"] = result["ecephys_session_id"].astype(str)
    result["area"] = pd.Categorical(result["area"], categories=AREA_ORDER)
    return result


def load_model_table(
    *, audit_dir: Path, config_path: Path | None
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    support_path = audit_dir / "rf_unit_common_support.csv"
    manifest_path = audit_dir / "run_manifest.json"
    if not support_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing Iteration 6C targeting-audit contract in {audit_dir}"
        )
    audit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit_manifest["parameters"]["population"] != "published_like":
        raise ValueError("RF-adjusted checkpoint currently requires published_like audit support")
    support = pd.read_csv(support_path)
    config = load_config(config_path)
    unit_path = Path(config["allen_unit_table"])
    if not unit_path.is_absolute():
        unit_path = ROOT / unit_path
    usecols = sorted(
        {"ecephys_unit_id", "area_rf"}.union(RESPONSE_COLUMNS)
    )
    responses = pd.read_csv(unit_path, usecols=usecols, low_memory=False)
    if responses["ecephys_unit_id"].duplicated().any():
        raise ValueError("Released Allen response table contains duplicate unit IDs")
    merged = support.merge(
        responses, on=["ecephys_unit_id", "area_rf"], how="left", validate="one_to_one"
    )
    if merged["time_to_first_spike_fl"].isna().all():
        raise ValueError("RF-support contract did not map to Allen response metrics")
    return prepare_outcomes(merged), audit_manifest, unit_path


def model_formulas(outcome: str) -> dict[str, str]:
    area = "C(area, Treatment(reference='V1'))"
    session = "C(session_id)"
    base = f"{outcome} ~ {area} + {session}"
    spatial = (
        "bs(relative_azimuth_deg, df=4, degree=2, include_intercept=False) + "
        "bs(relative_elevation_deg, df=4, degree=2, include_intercept=False)"
    )
    area_term = "" if outcome == "rf_area_deg2" else " + rf_area_z"
    return {
        "unadjusted_full": base,
        "unadjusted_common_support": base,
        "rf_adjusted_common_support": f"{base} + {spatial}{area_term}",
        "rf_depth_adjusted_common_support": (
            f"{base} + {spatial}{area_term} + cortical_depth_z + C(cortical_layer)"
        ),
        "rf_interaction_common_support": (
            f"{base} + relative_azimuth_deg + relative_elevation_deg{area_term} + "
            f"{area}:relative_azimuth_deg + {area}:relative_elevation_deg"
        ),
    }


def _area_from_term(term: str) -> str | None:
    if ":" in term:
        return None
    match = re.search(r"\[T\.([^]]+)\]", term)
    return match.group(1) if match else None


def fit_models(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    area_rows: list[dict[str, object]] = []
    for outcome in OUTCOME_ORDER:
        valid = table.loc[np.isfinite(table[outcome])].copy()
        formulas = model_formulas(outcome)
        for model_name, formula in formulas.items():
            model_data = valid.copy()
            if model_name != "unadjusted_full":
                model_data = model_data.loc[model_data["inside_v1_robust_box"]].copy()
            required = ["relative_azimuth_deg", "relative_elevation_deg"]
            if "depth" in model_name:
                required.extend(["cortical_depth_z", "cortical_layer"])
            model_data = model_data.dropna(subset=required)
            if model_data["session_id"].nunique() < 10:
                raise ValueError(f"Too few sessions for {outcome}/{model_name}")
            fit = smf.ols(formula, data=model_data).fit(
                cov_type="cluster",
                cov_kwds={"groups": model_data["session_id"], "use_correction": True},
            )
            interaction_terms = [
                index
                for index, term in enumerate(fit.params.index)
                if ":relative_azimuth_deg" in term or ":relative_elevation_deg" in term
            ]
            interaction_p = np.nan
            if interaction_terms:
                restrictions = np.eye(len(fit.params))[interaction_terms]
                interaction_p = float(fit.wald_test(restrictions, scalar=True).pvalue)
            fit_rows.append(
                {
                    "outcome": outcome,
                    "model": model_name,
                    "n_units": int(fit.nobs),
                    "n_sessions": model_data["session_id"].nunique(),
                    "r_squared": fit.rsquared,
                    "aic_nonrobust_likelihood": fit.aic,
                    "joint_area_by_rf_interaction_p": interaction_p,
                    "formula": formula,
                }
            )
            confidence = fit.conf_int()
            for term in fit.params.index:
                row = {
                    "outcome": outcome,
                    "model": model_name,
                    "term": term,
                    "estimate": fit.params[term],
                    "cluster_se": fit.bse[term],
                    "p_value": fit.pvalues[term],
                    "ci_low": confidence.loc[term, 0],
                    "ci_high": confidence.loc[term, 1],
                    "n_units": int(fit.nobs),
                    "n_sessions": model_data["session_id"].nunique(),
                }
                coefficient_rows.append(row)
                area_name = _area_from_term(term)
                if area_name in HVA_ORDER:
                    area_rows.append({**row, "area": area_name})
    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(fit_rows),
        pd.DataFrame(area_rows),
    )


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def nearest_v1_matches(
    hva: pd.DataFrame, v1: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Match every HVA unit to its nearest same-session V1 RF with replacement."""
    coordinates = ["relative_azimuth_deg", "relative_elevation_deg"]
    if hva.empty or v1.empty:
        return v1.iloc[0:0].copy(), np.array([], dtype=float)
    tree = cKDTree(v1[coordinates].to_numpy(dtype=float))
    distances, indices = tree.query(hva[coordinates].to_numpy(dtype=float), k=1)
    matched = v1.iloc[np.asarray(indices, dtype=int)].copy().reset_index(drop=True)
    return matched, np.asarray(distances, dtype=float)


def matched_session_contrasts(
    table: pd.DataFrame, *, match_caliper_deg: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrast_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    support = table.loc[table["inside_v1_robust_box"]].copy()
    for outcome in OUTCOME_ORDER:
        valid = support.loc[np.isfinite(support[outcome])].copy()
        for session_id, session in valid.groupby("session_id", observed=True):
            v1 = session.loc[session["area"].eq("V1")].copy()
            if len(v1) < 5:
                continue
            for area in HVA_ORDER:
                hva = session.loc[session["area"].eq(area)].copy()
                if len(hva) < 5:
                    continue
                matched, distances = nearest_v1_matches(hva, v1)
                if matched.empty:
                    continue
                keep = distances <= match_caliper_deg
                matched_hva = hva.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True)
                matched = matched.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True)
                distances = distances[keep]
                if len(matched_hva) < 5:
                    continue
                contrast_rows.append(
                    {
                        "outcome": outcome,
                        "area": area,
                        "session_id": session_id,
                        "cohort": session["cohort"].iloc[0],
                        "n_hva_available": len(hva),
                        "n_hva_matched": len(matched_hva),
                        "hva_discarded_fraction": 1 - len(matched_hva) / len(hva),
                        "n_v1_candidates": len(v1),
                        "n_unique_v1_matched": matched["ecephys_unit_id"].nunique(),
                        "hva_mean": matched_hva[outcome].mean(),
                        "matched_v1_mean": matched[outcome].mean(),
                        "matched_difference_hva_minus_v1": matched_hva[outcome].mean()
                        - matched[outcome].mean(),
                        "mean_match_distance_deg": float(np.mean(distances)),
                        "median_match_distance_deg": float(np.median(distances)),
                    }
                )
                row = {
                    "outcome": outcome,
                    "area": area,
                    "session_id": session_id,
                    "cohort": session["cohort"].iloc[0],
                    "n_hva_available": len(hva),
                    "n_hva_matched": len(matched_hva),
                    "hva_discarded_fraction": 1 - len(matched_hva) / len(hva),
                    "n_v1_candidates": len(v1),
                    "n_unique_v1_matched": matched["ecephys_unit_id"].nunique(),
                    "mean_match_distance_deg": float(np.mean(distances)),
                }
                for coordinate in ("relative_azimuth_deg", "relative_elevation_deg"):
                    suffix = "azimuth" if "azimuth" in coordinate else "elevation"
                    row[f"smd_{suffix}_before"] = standardized_mean_difference(
                        hva[coordinate], v1[coordinate]
                    )
                    row[f"smd_{suffix}_after"] = standardized_mean_difference(
                        matched_hva[coordinate], matched[coordinate]
                    )
                balance_rows.append(row)
    return pd.DataFrame(contrast_rows), pd.DataFrame(balance_rows)


def bootstrap_matched_summary(
    contrasts: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (outcome, area), group in contrasts.groupby(["outcome", "area"], observed=True):
        values = group["matched_difference_hva_minus_v1"].to_numpy(dtype=float)
        boot = np.empty(n_bootstrap)
        for index in range(n_bootstrap):
            boot[index] = np.mean(rng.choice(values, size=len(values), replace=True))
        rows.append(
            {
                "outcome": outcome,
                "area": area,
                "session_pairs": len(values),
                "equal_session_mean_difference": float(np.mean(values)),
                "median_session_difference": float(np.median(values)),
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "bootstrap_p_mean_le_zero": float(np.mean(boot <= 0)),
            }
        )
    return pd.DataFrame(rows)


def matching_caliper_sensitivity(
    table: pd.DataFrame, *, calipers: tuple[float, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for caliper in calipers:
        contrasts, balance = matched_session_contrasts(
            table, match_caliper_deg=caliper
        )
        for (outcome, area), group in balance.groupby(
            ["outcome", "area"], observed=True
        ):
            effects = contrasts.loc[
                contrasts["outcome"].eq(outcome) & contrasts["area"].eq(area),
                "matched_difference_hva_minus_v1",
            ]
            before = np.abs(
                group[["smd_azimuth_before", "smd_elevation_before"]].to_numpy(dtype=float)
            )
            after = np.abs(
                group[["smd_azimuth_after", "smd_elevation_after"]].to_numpy(dtype=float)
            )
            rows.append(
                {
                    "caliper_deg": caliper,
                    "outcome": outcome,
                    "area": area,
                    "session_pairs": len(group),
                    "equal_session_mean_difference": effects.mean(),
                    "mean_absolute_smd_before": np.nanmean(before),
                    "mean_absolute_smd_after": np.nanmean(after),
                    "mean_hva_discarded_fraction": group["hva_discarded_fraction"].mean(),
                    "mean_match_distance_deg": group["mean_match_distance_deg"].mean(),
                }
            )
    return pd.DataFrame(rows)


def hierarchy_associations(
    area_effects: pd.DataFrame, matched_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_models = ("unadjusted_full", "unadjusted_common_support", "rf_adjusted_common_support")
    for (outcome, model), group in area_effects.loc[
        area_effects["model"].isin(selected_models)
    ].groupby(["outcome", "model"], observed=True):
        scores = group["area"].map(HIERARCHY_SCORE).to_numpy(dtype=float)
        effects = group["estimate"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(scores, effects, 1)
        rows.append(
            {
                "outcome": outcome,
                "view": model,
                "n_areas": len(group),
                "hierarchy_slope": slope,
                "hierarchy_intercept": intercept,
                "pearson_r": pearsonr(scores, effects).statistic,
                "note": "descriptive across five adjusted area coefficients",
            }
        )
    for outcome, group in matched_summary.groupby("outcome", observed=True):
        scores = group["area"].map(HIERARCHY_SCORE).to_numpy(dtype=float)
        effects = group["equal_session_mean_difference"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(scores, effects, 1)
        rows.append(
            {
                "outcome": outcome,
                "view": "same_session_nearest_v1_matching",
                "n_areas": len(group),
                "hierarchy_slope": slope,
                "hierarchy_intercept": intercept,
                "pearson_r": pearsonr(scores, effects).statistic,
                "note": "descriptive across five matched area contrasts",
            }
        )
    return pd.DataFrame(rows)


def render_figure(
    area_effects: pd.DataFrame,
    matched_summary: pd.DataFrame,
    balance: pd.DataFrame,
    output_path: Path,
) -> None:
    outcomes = ("ttfs_ms", "log10_mod_idx", "timescale_ms")
    fig, axes = plt.subplots(len(outcomes), 3, figsize=(14, 11))
    models = ("unadjusted_full", "rf_adjusted_common_support")
    model_labels = {"unadjusted_full": "unadjusted", "rf_adjusted_common_support": "RF-adjusted"}
    offsets = {-1: -0.09, 1: 0.09}
    for row_index, outcome in enumerate(outcomes):
        ax = axes[row_index, 0]
        x = np.arange(len(HVA_ORDER))
        for direction, model in zip((-1, 1), models):
            group = area_effects.loc[
                area_effects["outcome"].eq(outcome) & area_effects["model"].eq(model)
            ].set_index("area").reindex(HVA_ORDER)
            ax.errorbar(
                x + offsets[direction],
                group["estimate"],
                yerr=np.vstack([group["estimate"] - group["ci_low"], group["ci_high"] - group["estimate"]]),
                fmt="o",
                capsize=3,
                label=model_labels[model],
            )
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_xticks(x, HVA_ORDER)
        ax.set_ylabel(f"HVA − V1: {OUTCOME_LABELS[outcome]}")
        ax.set_title("Session-FE area coefficients")
        if row_index == 0:
            ax.legend(frameon=False)

        ax = axes[row_index, 1]
        group = matched_summary.loc[matched_summary["outcome"].eq(outcome)].set_index("area").reindex(HVA_ORDER)
        ax.errorbar(
            x,
            group["equal_session_mean_difference"],
            yerr=np.vstack([
                group["equal_session_mean_difference"] - group["bootstrap_ci_low"],
                group["bootstrap_ci_high"] - group["equal_session_mean_difference"],
            ]),
            fmt="o",
            color="#333333",
            capsize=3,
        )
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_xticks(x, HVA_ORDER)
        ax.set_ylabel(f"matched HVA − V1: {OUTCOME_LABELS[outcome]}")
        ax.set_title("Same-session nearest V1")

        ax = axes[row_index, 2]
        group = balance.loc[balance["outcome"].eq(outcome)].copy()
        before = np.nanmean(np.abs(group[["smd_azimuth_before", "smd_elevation_before"]]), axis=1)
        after = np.nanmean(np.abs(group[["smd_azimuth_after", "smd_elevation_after"]]), axis=1)
        ax.scatter(before, after, s=18, alpha=0.55, color="#2C7FB8")
        limit = max(0.5, np.nanquantile(np.r_[before, after], 0.98))
        ax.plot([0, limit], [0, limit], color="#888888", ls="--", lw=0.8)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("mean |SMD| before")
        ax.set_ylabel("mean |SMD| after")
        ax.set_title("RF-center balance")

    for ax in axes.flat:
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Allen response differences after achieved-RF adjustment", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    area_effects: pd.DataFrame,
    matched_summary: pd.DataFrame,
    fit_summary: pd.DataFrame,
    balance: pd.DataFrame,
    caliper_sensitivity: pd.DataFrame,
    match_caliper_deg: float,
    output_path: Path,
) -> None:
    lines = [
        "# Iteration 6C — RF-adjusted Allen response analysis",
        "",
        "## Status: first response-adjustment checkpoint implemented",
        "",
        "Area contrasts are identified within Allen sessions using session fixed effects",
        "and session-clustered uncertainty. The primary RF-adjusted view is restricted to",
        "the conservative session-centered V1 support box and uses flexible azimuth and",
        "elevation terms. A separate same-session nearest-neighbor sensitivity matches each",
        "HVA unit to a V1 unit in two-dimensional RF-center space with replacement.",
        "",
        "## Area coefficients before and after RF adjustment",
        "",
        "| Outcome | Area | Unadjusted | RF-adjusted |",
        "| --- | --- | ---: | ---: |",
    ]
    for outcome in OUTCOME_ORDER:
        for area in HVA_ORDER:
            subset = area_effects.loc[area_effects["outcome"].eq(outcome) & area_effects["area"].eq(area)]
            raw = subset.loc[subset["model"].eq("unadjusted_full"), "estimate"]
            adjusted = subset.loc[subset["model"].eq("rf_adjusted_common_support"), "estimate"]
            if len(raw) and len(adjusted):
                lines.append(
                    f"| {OUTCOME_LABELS[outcome]} | {area} | {raw.iloc[0]:+.3f} | {adjusted.iloc[0]:+.3f} |"
                )
    lines.extend(
        [
            "",
            "The coefficient change combines explicit RF adjustment with restriction to V1",
            "common support. `model_area_effects.csv` also contains an unadjusted",
            "common-support model, which separates those two changes.",
            "",
            "## Matched sensitivity",
            "",
            "| Outcome | Area | Sessions | Matched HVA−V1 | 95% bootstrap CI |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in matched_summary.iterrows():
        lines.append(
            f"| {OUTCOME_LABELS[row.outcome]} | {row.area} | {int(row.session_pairs)} | "
            f"{row.equal_session_mean_difference:+.3f} | {row.bootstrap_ci_low:+.3f} to {row.bootstrap_ci_high:+.3f} |"
        )
    mean_before = float(np.nanmean(np.abs(balance[["smd_azimuth_before", "smd_elevation_before"]])))
    mean_after = float(np.nanmean(np.abs(balance[["smd_azimuth_after", "smd_elevation_after"]])))
    mean_discarded = float(balance["hva_discarded_fraction"].mean())
    caliper_tradeoff = (
        caliper_sensitivity.groupby("caliper_deg", observed=True)
        .agg(
            mean_absolute_smd_after=("mean_absolute_smd_after", "mean"),
            mean_hva_discarded_fraction=("mean_hva_discarded_fraction", "mean"),
            minimum_session_pairs=("session_pairs", "min"),
        )
        .reset_index()
    )
    interaction_count = int(
        fit_summary.loc[
            fit_summary["model"].eq("rf_interaction_common_support"),
            "joint_area_by_rf_interaction_p",
        ].lt(0.05).sum()
    )
    lines.extend(
        [
            "",
            f"Mean absolute RF-coordinate SMD changes from {mean_before:.3f} before matching to {mean_after:.3f} after matching.",
            f"The {match_caliper_deg:g}° caliper discards a mean {100 * mean_discarded:.1f}% of otherwise eligible HVA units across session × area × outcome cells.",
            "",
            "### Caliper trade-off",
            "",
            "| Caliper (deg) | Mean |SMD| after | Mean HVA discarded | Minimum session pairs |",
            "| ---: | ---: | ---: | ---: |",
            *[
                f"| {row.caliper_deg:g} | {row.mean_absolute_smd_after:.3f} | {100 * row.mean_hva_discarded_fraction:.1f}% | {int(row.minimum_session_pairs)} |"
                for row in caliper_tradeoff.itertuples(index=False)
            ],
            f"Area × RF-position interactions have joint p < 0.05 for {interaction_count}/{len(OUTCOME_ORDER)} outcomes; these are model checks, not a second primary test.",
            "",
            "## Interpretation boundary",
            "",
            "This checkpoint addresses achieved RF-center sampling within the Allen dataset.",
            "It does not calibrate MouseV2 and Allen response levels, validate MouseV2 RF",
            "area/significance, or make the cross-dataset claim pass. Matching is with",
            "replacement and balance/discarded support must accompany every reported result.",
            "",
            "## Outputs",
            "",
            "- `model_coefficients.csv`: every fitted coefficient and clustered interval.",
            "- `model_area_effects.csv`: HVA−V1 coefficients for each model.",
            "- `model_fit_summary.csv`: formulas, sample sizes, fit summaries, and interaction tests.",
            "- `matched_session_contrasts.csv`: paired session-level matched contrasts.",
            "- `matched_balance.csv`: RF balance and match-distance diagnostics.",
            "- `matched_area_summary.csv`: equal-session contrasts and bootstrap intervals.",
            "- `matching_caliper_sensitivity.csv`: balance, attrition, and effect sensitivity across calipers.",
            "- `hierarchy_associations.csv`: descriptive hierarchy slopes across five areas.",
            "- `Figure_allen_rf_adjusted_response.png`: primary diagnostic figure.",
            "- `run_manifest.json`: input/code/output checksums and parameters.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty checkpoint {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    table, audit_manifest, unit_path = load_model_table(
        audit_dir=audit_dir, config_path=args.config
    )
    coefficients, fit_summary, area_effects = fit_models(table)
    matched_contrasts, balance = matched_session_contrasts(
        table, match_caliper_deg=args.match_caliper_deg
    )
    matched_summary = bootstrap_matched_summary(
        matched_contrasts, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    caliper_sensitivity = matching_caliper_sensitivity(
        table, calipers=(5.0, 7.5, 10.0, 15.0)
    )
    associations = hierarchy_associations(area_effects, matched_summary)

    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)
    fit_summary.to_csv(output_dir / "model_fit_summary.csv", index=False)
    area_effects.to_csv(output_dir / "model_area_effects.csv", index=False)
    matched_contrasts.to_csv(output_dir / "matched_session_contrasts.csv", index=False)
    balance.to_csv(output_dir / "matched_balance.csv", index=False)
    matched_summary.to_csv(output_dir / "matched_area_summary.csv", index=False)
    caliper_sensitivity.to_csv(
        output_dir / "matching_caliper_sensitivity.csv", index=False
    )
    associations.to_csv(output_dir / "hierarchy_associations.csv", index=False)
    render_figure(
        area_effects,
        matched_summary,
        balance,
        output_dir / "Figure_allen_rf_adjusted_response.png",
    )
    write_report(
        area_effects=area_effects,
        matched_summary=matched_summary,
        fit_summary=fit_summary,
        balance=balance,
        caliper_sensitivity=caliper_sensitivity,
        match_caliper_deg=args.match_caliper_deg,
        output_path=output_dir / "ALLEN_RF_ADJUSTED_RESPONSE.md",
    )

    output_records = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            output_records[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "checkpoint": "06c_allen_rf_adjusted_response",
        "status": "first RF-adjusted response checkpoint implemented",
        "inputs": {
            "allen_unit_table": {"path": str(unit_path), "sha256": sha256(unit_path)},
            "rf_audit_manifest": {
                "path": str(audit_dir / "run_manifest.json"),
                "sha256": sha256(audit_dir / "run_manifest.json"),
                "population": audit_manifest["parameters"]["population"],
            },
            "rf_support": {
                "path": str(audit_dir / "rf_unit_common_support.csv"),
                "sha256": sha256(audit_dir / "rf_unit_common_support.csv"),
            },
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "common_support": "inside_v1_robust_box",
            "session_adjustment": "session fixed effects",
            "uncertainty": "session-clustered covariance",
            "matching": "same-session 1-nearest V1 RF center with replacement",
            "match_caliper_deg": args.match_caliper_deg,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "outputs": output_records,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Allen RF-adjusted response checkpoint written to {output_dir}")


if __name__ == "__main__":
    main()
