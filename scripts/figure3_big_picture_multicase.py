"""Select and display concrete session cases for the Figure 3 big-picture question.

This is an exploratory concrete-first checkpoint, not a population-level test.
It reads the frozen session-by-group means from the robust Figure 3 analysis and
holds the observed group count fixed at four in both datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
OUTPUT = ROOT / "artifacts/figure3/07_big_picture_concrete_first"
PROBE_ORDER = ["B", "C", "A", "E"]
HIERARCHY_SCORES = {
    "LM": -0.093,
    "RL": -0.059,
    "LP": 0.105,
    "AL": 0.152,
    "PM": 0.327,
    "AM": 0.441,
}
METRICS = ["TTFS (ms)", "log10 F1/F0", "Response timescale (ms)"]


def session_summaries(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric, dataset, session_id), part in table.groupby(
        ["metric", "dataset", "session_id"], sort=False
    ):
        if part["group"].nunique() != 4:
            continue
        values = part["mean"].to_numpy(float)
        row: dict[str, object] = {
            "metric": metric,
            "dataset": dataset,
            "session_id": str(session_id),
            "n_groups": 4,
            "spread_sd": float(np.std(values, ddof=1)),
            "spread_range": float(np.ptp(values)),
            "groups_present": ";".join(sorted(part["group"].astype(str))),
        }
        if dataset == "Post-V1":
            ordered = part.assign(
                hierarchy_score=part["group"].map(HIERARCHY_SCORES)
            ).sort_values("hierarchy_score")
            x = ordered["hierarchy_score"].to_numpy(float)
            y = ordered["mean"].to_numpy(float)
            row["within_session_hierarchy_slope"] = float(np.polyfit(x, y, 1)[0])
            row["within_session_hierarchy_r"] = float(np.corrcoef(x, y)[0, 1])
        else:
            row["within_session_hierarchy_slope"] = np.nan
            row["within_session_hierarchy_r"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def choose_roles(summaries: pd.DataFrame) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    for metric in METRICS:
        for dataset in ["Within-V1", "Post-V1"]:
            part = summaries.loc[
                summaries["metric"].eq(metric) & summaries["dataset"].eq(dataset)
            ].copy()
            median_sd = float(part["spread_sd"].median())
            part["distance_from_median_sd"] = (part["spread_sd"] - median_sd).abs()
            part["reference_median_sd"] = median_sd
            roles = [
                ("typical spread", "distance_from_median_sd", True),
                ("lowest spread", "spread_sd", True),
                ("largest spread", "spread_sd", False),
            ]
            if dataset == "Post-V1":
                roles.extend(
                    [
                        ("strongest hierarchy-concordant", "within_session_hierarchy_r", False),
                        ("strongest hierarchy-discordant", "within_session_hierarchy_r", True),
                    ]
                )
            for role, column, ascending in roles:
                selected = part.sort_values(
                    [column, "session_id"], ascending=[ascending, True]
                ).iloc[0].copy()
                selected["selection_role"] = role
                selected["selection_variable"] = column
                selected["selection_value"] = selected[column]
                chosen.append(selected)
    columns = [
        "metric",
        "dataset",
        "session_id",
        "selection_role",
        "selection_variable",
        "selection_value",
        "n_groups",
        "spread_sd",
        "spread_range",
        "reference_median_sd",
        "distance_from_median_sd",
        "within_session_hierarchy_slope",
        "within_session_hierarchy_r",
        "groups_present",
    ]
    return pd.DataFrame(chosen)[columns]


def selected_values(table: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    keys = selection[["metric", "dataset", "session_id", "selection_role"]].copy()
    source = table.copy()
    source["session_id"] = source["session_id"].astype(str)
    values = keys.merge(source, on=["metric", "dataset", "session_id"], how="left")
    values["centered_for_display"] = values["mean"] - values.groupby(
        ["metric", "dataset", "session_id", "selection_role"]
    )["mean"].transform("mean")
    values["hierarchy_score"] = values["group"].map(HIERARCHY_SCORES)
    return values


def render(values: pd.DataFrame, output: Path) -> None:
    role_colors = {
        "typical spread": "#222222",
        "lowest spread": "#4daf4a",
        "largest spread": "#984ea3",
        "strongest hierarchy-concordant": "#377eb8",
        "strongest hierarchy-discordant": "#e41a1c",
    }
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for row, metric in enumerate(METRICS):
        for column, dataset in enumerate(["Within-V1", "Post-V1"]):
            ax = axes[row, column]
            part = values.loc[
                values["metric"].eq(metric) & values["dataset"].eq(dataset)
            ]
            for (role, session_id), case in part.groupby(
                ["selection_role", "session_id"], sort=False
            ):
                if dataset == "Within-V1":
                    case = case.assign(
                        plot_x=case["group"].map({g: i for i, g in enumerate(PROBE_ORDER)})
                    ).sort_values("plot_x")
                    x = case["plot_x"]
                else:
                    case = case.sort_values("hierarchy_score")
                    x = case["hierarchy_score"]
                ax.plot(
                    x,
                    case["centered_for_display"],
                    marker="o",
                    linewidth=1.5,
                    alpha=0.85,
                    color=role_colors[role],
                    label=f"{role}: {session_id}",
                )
                if dataset == "Post-V1":
                    for _, point in case.iterrows():
                        ax.annotate(
                            point["group"],
                            (point["hierarchy_score"], point["centered_for_display"]),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7,
                            color=role_colors[role],
                        )
            ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
            if dataset == "Within-V1":
                ax.set_xticks(range(len(PROBE_ORDER)), PROBE_ORDER)
                ax.set_xlabel("Probe identity (display order only)")
            else:
                ax.set_xlabel("Published hierarchy score")
            ax.set_ylabel(f"Session-centered {metric}")
            ax.set_title(f"{metric}: {'V1 controls' if dataset == 'Within-V1' else 'four-area cases'}")
            ax.legend(fontsize=7, frameon=False, loc="best")
            ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Concrete session cases: separation versus hierarchy-aligned ordering",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_report(selection: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Big-picture Figure 3 analysis: multi-case checkpoint",
        "",
        "All selections use sessions with exactly four observed groups. Roles are",
        "chosen independently within each metric and dataset. The V1 cases provide",
        "low, typical, and high within-session positional spread controls. The HVA",
        "cases additionally expose the strongest positive and negative correlation",
        "between the four observed area means and published hierarchy score.",
        "",
        "This is an exploratory case-comparison checkpoint, not a prevalence estimate.",
        "",
        "## Selected HVA cases",
        "",
        "| Metric | Role | Session | SD | Hierarchy slope | r | Areas |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    hva = selection.loc[selection["dataset"].eq("Post-V1")]
    for row in hva.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.selection_role} | {row.session_id} | "
            f"{row.spread_sd:.4g} | {row.within_session_hierarchy_slope:+.4g} | "
            f"{row.within_session_hierarchy_r:+.3f} | {row.groups_present} |"
        )
    lines.extend(
        [
            "",
            "## Reading rule",
            "",
            "Larger SD establishes within-session heterogeneity. Positive hierarchy",
            "correlation establishes hierarchy-aligned ordering only for the displayed",
            "session. A stable hierarchy requires that ordering to recur across sessions",
            "and survive a session-controlled population model.",
            "",
            "## Audit artifacts",
            "",
            "- `multi_case_selection.csv`: predeclared roles, criteria, and selected IDs.",
            "- `multi_case_values.csv`: source group means and session-centered display values.",
            "- `Figure_multicase_comparison.png`/`.pdf`: direct selected-case comparison.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(SOURCE)
    table["session_id"] = table["session_id"].astype(str)
    summaries = session_summaries(table)
    selection = choose_roles(summaries)
    values = selected_values(table, selection)
    selection.to_csv(OUTPUT / "multi_case_selection.csv", index=False)
    values.to_csv(OUTPUT / "multi_case_values.csv", index=False)
    summaries.to_csv(OUTPUT / "four_group_session_summaries.csv", index=False)
    render(values, OUTPUT / "Figure_multicase_comparison.png")
    write_report(selection, OUTPUT / "MULTI_CASE_CHECKPOINT.md")


if __name__ == "__main__":
    main()
