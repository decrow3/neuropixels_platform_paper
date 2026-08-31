"""Quantify HVA-area variance relative to V1-location variance.

For each session, compute the sample variance across retained group means. The
primary estimand is the ratio of mean session variances:

    E_session[Var(area means)] / E_session[Var(probe means)].

Sessions, not neurons, are the resampling unit. The primary comparison fixes
both datasets at exactly four observed groups. A sensitivity retains the
Figure 3 rule of all four V1 probes and at least three HVA areas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
OUTPUT = ROOT / "artifacts/figure3/07_big_picture_concrete_first"
METRICS = ["TTFS (ms)", "log10 F1/F0", "Response timescale (ms)"]
N_BOOTSTRAP = 50_000
SEED = 20260826


def session_variances(group_means: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric, dataset, session_id), part in group_means.groupby(
        ["metric", "dataset", "session_id"], sort=False
    ):
        n_groups = int(part["group"].nunique())
        if n_groups < 2:
            continue
        rows.append(
            {
                "metric": metric,
                "dataset": dataset,
                "session_id": str(session_id),
                "n_groups": n_groups,
                "variance_across_group_means": float(part["mean"].var(ddof=1)),
                "sd_across_group_means": float(part["mean"].std(ddof=1)),
                "groups_present": ";".join(sorted(part["group"].astype(str))),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    sessions: pd.DataFrame,
    *,
    comparison: str,
    min_hva_groups: int,
    exact_hva_groups: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        metric_sessions = sessions.loc[sessions["metric"].eq(metric)]
        v1 = metric_sessions.loc[
            metric_sessions["dataset"].eq("Within-V1")
            & metric_sessions["n_groups"].eq(4),
            "variance_across_group_means",
        ].to_numpy(float)
        hva_keep = metric_sessions["dataset"].eq("Post-V1") & metric_sessions[
            "n_groups"
        ].ge(min_hva_groups)
        if exact_hva_groups is not None:
            hva_keep &= metric_sessions["n_groups"].eq(exact_hva_groups)
        hva = metric_sessions.loc[hva_keep, "variance_across_group_means"].to_numpy(float)

        v1_draws = rng.choice(v1, size=(N_BOOTSTRAP, len(v1)), replace=True).mean(axis=1)
        hva_draws = rng.choice(hva, size=(N_BOOTSTRAP, len(hva)), replace=True).mean(axis=1)
        ratio_draws = hva_draws / v1_draws
        difference_draws = hva_draws - v1_draws
        observed_ratio = float(hva.mean() / v1.mean())
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "v1_sessions": len(v1),
                "hva_sessions": len(hva),
                "mean_v1_session_variance": float(v1.mean()),
                "mean_hva_session_variance": float(hva.mean()),
                "hva_to_v1_variance_ratio": observed_ratio,
                "variance_ratio_ci_low": float(np.quantile(ratio_draws, 0.025)),
                "variance_ratio_ci_high": float(np.quantile(ratio_draws, 0.975)),
                "bootstrap_probability_ratio_le_1": float(np.mean(ratio_draws <= 1)),
                "hva_minus_v1_variance": float(hva.mean() - v1.mean()),
                "variance_difference_ci_low": float(np.quantile(difference_draws, 0.025)),
                "variance_difference_ci_high": float(np.quantile(difference_draws, 0.975)),
                "equivalent_hva_to_v1_sd_ratio": float(np.sqrt(observed_ratio)),
                "ratio_of_median_session_variances": float(np.median(hva) / np.median(v1)),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, path: Path) -> None:
    primary = summary.loc[summary["comparison"].eq("exactly_four_groups")]
    sensitivity = summary.loc[summary["comparison"].eq("hva_at_least_three_groups")]
    lines = [
        "# Figure 3 session-variance ratio",
        "",
        "## Estimand",
        "",
        "For each recording session, calculate the sample variance across the",
        "available probe or area means. Compare datasets using the ratio of the",
        "mean HVA session variance to the mean V1 session variance. Whole sessions",
        "are independently resampled within dataset for the percentile interval.",
        "",
        "The primary analysis fixes both datasets at exactly four observed groups.",
        "This is an observed-mean variance ratio; it has not yet been corrected for",
        "finite-neuron measurement error in each session × group mean.",
        "",
        "## Primary result: exactly four groups",
        "",
        "| Metric | V1 sessions | HVA sessions | Mean V1 variance | Mean HVA variance | HVA/V1 variance ratio | 95% CI | Equivalent SD ratio | P(ratio <= 1) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.v1_sessions} | {row.hva_sessions} | "
            f"{row.mean_v1_session_variance:.5g} | {row.mean_hva_session_variance:.5g} | "
            f"{row.hva_to_v1_variance_ratio:.2f}× | "
            f"[{row.variance_ratio_ci_low:.2f}, {row.variance_ratio_ci_high:.2f}] | "
            f"{row.equivalent_hva_to_v1_sd_ratio:.2f}× | "
            f"{row.bootstrap_probability_ratio_le_1:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity: all four V1 probes versus at least three HVA areas",
            "",
            "| Metric | HVA sessions | HVA/V1 variance ratio | 95% CI | P(ratio <= 1) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.hva_sessions} | "
            f"{row.hva_to_v1_variance_ratio:.2f}× | "
            f"[{row.variance_ratio_ci_low:.2f}, {row.variance_ratio_ci_high:.2f}] | "
            f"{row.bootstrap_probability_ratio_le_1:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This directly answers whether observed across-area variation is similar",
            "to observed across-location variation. Because Allen HVA cells generally",
            "contain fewer neurons and have larger uncertainty than MouseV2 V1 cells,",
            "the next inferential step is a measurement-error-corrected variance ratio.",
            "The observed ratio should not be presented as de-noised biological variance",
            "until that correction is carried through the current metric definitions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group_means = pd.read_csv(SOURCE)
    sessions = session_variances(group_means)
    rng = np.random.default_rng(SEED)
    primary = summarize(
        sessions,
        comparison="exactly_four_groups",
        min_hva_groups=4,
        exact_hva_groups=4,
        rng=rng,
    )
    sensitivity = summarize(
        sessions,
        comparison="hva_at_least_three_groups",
        min_hva_groups=3,
        exact_hva_groups=None,
        rng=rng,
    )
    summary = pd.concat([primary, sensitivity], ignore_index=True)
    sessions.to_csv(OUTPUT / "session_group_variances.csv", index=False)
    summary.to_csv(OUTPUT / "session_variance_ratio_summary.csv", index=False)
    write_report(summary, OUTPUT / "SESSION_VARIANCE_RATIO.md")


if __name__ == "__main__":
    main()
