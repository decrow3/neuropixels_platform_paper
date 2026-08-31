#!/usr/bin/env python3
"""Compare conventional F1/F0 across MouseV2 and Allen V1 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units  # noqa: E402


OUTPUT = ROOT / "artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/03_v1_f1_f0_dataset_comparison"
MOUSE_METRICS = ROOT / "data/imports/mousev2_grating_metrics_v1"
MOUSE_COMMON = ROOT / "data/imports/mousev2_grating_common_support_v1/unit_metric_comparison.csv"
ALLEN_BRIDGE = ROOT / "data/imports/allen_v1_raw_bridge_v2/session_summary.csv"
THRESHOLDS = (0.0, 0.1, 1.0, 2.0, 5.0, 10.0)
COLORS = {"Allen BO": "#6F63A6", "Allen FC": "#B07AA1", "MouseV2": "#D95F02"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_session_difference(
    mouse: np.ndarray, allen: np.ndarray, *, draws: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    for draw in range(draws):
        differences[draw] = (
            rng.choice(mouse, len(mouse), replace=True).mean()
            - rng.choice(allen, len(allen), replace=True).mean()
        )
    return (
        float(mouse.mean() - allen.mean()),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


def native_tables(draws: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen.area_coarse.eq("V1")].copy()
    allen["cohort"] = allen.session_type.map(
        {"brain_observatory_1.1": "Allen BO", "functional_connectivity": "Allen FC"}
    )
    allen["session_id"] = allen.ecephys_session_id.astype(str)
    allen["unit_id"] = allen.ecephys_unit_id.astype(str)

    mouse = load_mousev2_units(
        apply_qc=False,
        population_profile="common_qc",
        grating_metrics_dir=MOUSE_METRICS,
    )
    mouse["cohort"] = "MouseV2"
    mouse["session_id"] = mouse.session_num.astype(str)
    mouse["unit_id"] = mouse.unit_id.astype(str)

    columns = ["cohort", "session_id", "unit_id", "f1_f0_dg", "firing_rate_dg"]
    units = pd.concat([allen[columns], mouse[columns]], ignore_index=True)
    units["f1_f0_dg"] = pd.to_numeric(units.f1_f0_dg, errors="coerce")
    units["firing_rate_dg"] = pd.to_numeric(units.firing_rate_dg, errors="coerce")
    units["log10_f1_f0"] = np.log10(units.f1_f0_dg.where(units.f1_f0_dg > 0))

    rows = []
    for threshold in THRESHOLDS:
        selected = units.loc[units.firing_rate_dg.ge(threshold)]
        for (cohort, session_id), group in selected.groupby(["cohort", "session_id"]):
            values = group.log10_f1_f0.dropna()
            if len(values) < 5:
                continue
            rows.append(
                {
                    "view": "native_full_preference",
                    "firing_rate_threshold_hz": threshold,
                    "cohort": cohort,
                    "session_id": session_id,
                    "valid_units": len(values),
                    "mean_log10_f1_f0": values.mean(),
                    "median_f1_f0": group.loc[values.index, "f1_f0_dg"].median(),
                }
            )
    sessions = pd.DataFrame(rows)

    contrasts = []
    for threshold in THRESHOLDS:
        subset = sessions.loc[sessions.firing_rate_threshold_hz.eq(threshold)]
        mouse_values = subset.loc[subset.cohort.eq("MouseV2"), "mean_log10_f1_f0"].to_numpy()
        for index, cohort in enumerate(("Allen BO", "Allen FC")):
            allen_values = subset.loc[subset.cohort.eq(cohort), "mean_log10_f1_f0"].to_numpy()
            difference, low, high = bootstrap_session_difference(
                mouse_values, allen_values, draws=draws, seed=20260825 + int(threshold * 10) + index
            )
            contrasts.append(
                {
                    "view": "native_full_preference",
                    "firing_rate_threshold_hz": threshold,
                    "contrast": f"MouseV2 minus {cohort}",
                    "mouse_sessions": len(mouse_values),
                    "allen_sessions": len(allen_values),
                    "difference_log10_f1_f0": difference,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "geometric_mean_ratio": 10**difference,
                }
            )
    return units, sessions, pd.DataFrame(contrasts)


def harmonized_summary() -> pd.DataFrame:
    mouse = pd.read_csv(MOUSE_COMMON)
    mouse = mouse.loc[mouse.default_qc & mouse.f1_f0_dg_common_support.gt(0)].copy()
    mouse["log10_f1_f0"] = np.log10(mouse.f1_f0_dg_common_support)
    mouse_sessions = mouse.groupby("site_number").log10_f1_f0.mean()
    mouse_center = float(mouse_sessions.mean())

    bridge = pd.read_csv(ALLEN_BRIDGE)
    bridge = bridge.loc[
        bridge.metric.eq("f1_f0_dg")
        & bridge.selection_role.eq("representative")
    ].copy()
    rows = [
        {
            "cohort": "MouseV2",
            "view": "shared_conditions_1s_15trials",
            "sessions": len(mouse_sessions),
            "center_mean_log10_f1_f0": mouse_center,
            "center_geometric_f1_f0": 10**mouse_center,
            "mouse_minus_allen": np.nan,
            "subsample_2.5pct": np.nan,
            "subsample_97.5pct": np.nan,
            "scope": "all MouseV2 sessions",
        }
    ]
    names = {
        "Allen Brain Observatory 1.1": "Allen BO",
        "Allen Functional Connectivity": "Allen FC",
    }
    for source, cohort in names.items():
        native = bridge.loc[bridge.cohort.eq(source) & bridge.view.eq("released"), "mean_log10"]
        common = bridge.loc[
            bridge.cohort.eq(source) & bridge.view.eq("common_1s_15trials"), "mean_log10"
        ]
        differences = mouse_center - common.to_numpy()
        rows.append(
            {
                "cohort": cohort,
                "view": "representative_native_2s" if len(native) else "",
                "sessions": 1,
                "center_mean_log10_f1_f0": float(native.iloc[0]),
                "center_geometric_f1_f0": 10 ** float(native.iloc[0]),
                "mouse_minus_allen": np.nan,
                "subsample_2.5pct": np.nan,
                "subsample_97.5pct": np.nan,
                "scope": "one prespecified representative raw session",
            }
        )
        rows.append(
            {
                "cohort": cohort,
                "view": "shared_conditions_1s_15trials",
                "sessions": 1,
                "center_mean_log10_f1_f0": float(common.median()),
                "center_geometric_f1_f0": 10 ** float(common.median()),
                "mouse_minus_allen": float(np.median(differences)),
                "subsample_2.5pct": float(np.quantile(differences, 0.025)),
                "subsample_97.5pct": float(np.quantile(differences, 0.975)),
                "scope": "one prespecified representative raw session",
            }
        )
    return pd.DataFrame(rows)


def select_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    base = sessions.loc[sessions.firing_rate_threshold_hz.eq(0)].copy()
    rows = []
    for cohort, group in base.groupby("cohort"):
        center = group.mean_log10_f1_f0.median()
        choices = {
            "lowest_session": group.mean_log10_f1_f0.idxmin(),
            "typical_session": (group.mean_log10_f1_f0 - center).abs().idxmin(),
            "highest_session": group.mean_log10_f1_f0.idxmax(),
        }
        for role, index in choices.items():
            row = group.loc[index].to_dict()
            row["selection_role"] = role
            row["selection_rule"] = "minimum, nearest cohort median, or maximum session mean"
            rows.append(row)
    return pd.DataFrame(rows)


def render(sessions: pd.DataFrame, contrasts: pd.DataFrame, harmonized: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    order = ("Allen BO", "Allen FC", "MouseV2")
    base = sessions.loc[sessions.firing_rate_threshold_hz.eq(0)]
    rng = np.random.default_rng(20260825)
    for position, cohort in enumerate(order):
        values = base.loc[base.cohort.eq(cohort), "mean_log10_f1_f0"].to_numpy()
        axes[0].scatter(position + rng.uniform(-.11, .11, len(values)), values, s=28, alpha=.65, color=COLORS[cohort])
        axes[0].plot([position-.18, position+.18], [values.mean(), values.mean()], color="black", lw=2)
    axes[0].set_xticks(range(3), order)
    axes[0].set(ylabel="session mean log10 F1/F0", title="Native-window V1 comparison")
    axes[0].axhline(0, color="0.75", lw=.8)

    x = np.arange(len(THRESHOLDS))
    for cohort in ("Allen BO", "Allen FC"):
        group = contrasts.loc[contrasts.contrast.eq(f"MouseV2 minus {cohort}")].sort_values("firing_rate_threshold_hz")
        axes[1].plot(x, group.difference_log10_f1_f0, "o-", label=cohort, color=COLORS[cohort])
        axes[1].fill_between(x, group.bootstrap_95ci_low, group.bootstrap_95ci_high, color=COLORS[cohort], alpha=.15)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_xticks(x, [f"{v:g}" for v in THRESHOLDS])
    axes[1].set(xlabel="minimum preferred firing rate (Hz)", ylabel="MouseV2 minus Allen log10 F1/F0", title="Response-support sensitivity")
    axes[1].legend(frameon=False)
    fig.suptitle("V1 conventional trialwise F1/F0")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    units, sessions, contrasts = native_tables(args.bootstrap_draws)
    harmonized = harmonized_summary()
    cases = select_sessions(sessions)
    sessions.to_csv(output / "session_f1_f0_summary.csv", index=False)
    contrasts.to_csv(output / "native_dataset_contrasts.csv", index=False)
    harmonized.to_csv(output / "harmonized_bridge_summary.csv", index=False)
    cases.to_csv(output / "session_case_selection.csv", index=False)
    render(sessions, contrasts, harmonized, output / "v1_f1_f0_dataset_comparison.png")
    sources = [ROOT / "data/unit_table.csv", MOUSE_METRICS / "unit_metric_comparison.csv", MOUSE_COMMON, ALLEN_BRIDGE, Path(__file__).resolve()]
    manifest = {
        "status": "descriptive_cross_dataset_checkpoint",
        "metric": "conventional trialwise F1/F0",
        "population": "VISp/V1 common_qc",
        "bootstrap_unit": "session",
        "bootstrap_draws": args.bootstrap_draws,
        "sources": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sources],
        "outputs": [],
    }
    for name in (
        "session_f1_f0_summary.csv",
        "native_dataset_contrasts.csv",
        "harmonized_bridge_summary.csv",
        "session_case_selection.csv",
        "v1_f1_f0_dataset_comparison.png",
        "V1_F1_F0_COMPARISON_CHECKPOINT.md",
    ):
        path = output / name
        if path.is_file():
            manifest["outputs"].append(
                {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote V1 F1/F0 comparison to {output}")


if __name__ == "__main__":
    main()
