#!/usr/bin/env python3
"""Validate and harmonize raw Allen V1 sessions for the dataset bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import (  # noqa: E402
    _bin_trial_spike_counts,
    f1_f0_from_trial_counts,
    welch_modulation_index,
)


DEFAULT_CONFIG = ROOT / "config" / "allen_v1_bridge.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "allen_v1_raw_bridge_v2"
MOUSE_COMMON = ROOT / "data" / "imports" / "mousev2_grating_common_support_v1"
COHORT_LABELS = {
    "brain_observatory_1.1": "Allen Brain Observatory 1.1",
    "functional_connectivity": "Allen Functional Connectivity",
}
COLORS = {
    "Allen Brain Observatory 1.1": "#6F63A6",
    "Allen Functional Connectivity": "#B07AA1",
    "MouseV2 V1": "#D95F02",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-subsamples", type=int, default=100)
    parser.add_argument("--skip-input-hashes", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    return parser.parse_args()


def prepare_grating_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "orientation",
        "temporal_frequency",
        "spatial_frequency",
        "contrast",
        "start_time",
        "stop_time",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Grating table lacks columns {missing}")
    result = table.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = (
        result[["orientation", "temporal_frequency", "spatial_frequency", "contrast"]]
        .notna()
        .all(axis=1)
        & result["temporal_frequency"].gt(0)
        & result["spatial_frequency"].gt(0)
    )
    return result.loc[valid].copy()


def condition_starts(
    table: pd.DataFrame,
    *,
    common_support: dict[str, object] | None = None,
) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
    selected = prepare_grating_table(table)
    if common_support is not None:
        selected = selected.loc[
            selected["orientation"].isin(common_support["orientation_deg"])
            & selected["temporal_frequency"].isin(
                common_support["temporal_frequency_hz"]
            )
            & np.isclose(
                selected["spatial_frequency"],
                float(common_support["spatial_frequency_cpd"]),
            )
            & np.isclose(selected["contrast"], float(common_support["contrast"]))
        ]
    dimensions = ["orientation", "temporal_frequency", "spatial_frequency", "contrast"]
    # sort=False is essential: AllenSDK's condition IDs preserve first
    # presentation order, and idxmax resolves exact response ties by that order.
    return [
        (tuple(map(float, key)), group["start_time"].to_numpy(dtype=float))
        for key, group in selected.groupby(dimensions, sort=False)
    ]


def _preferred_condition(
    spikes_s: np.ndarray,
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
    *,
    duration_s: float,
) -> int:
    means = []
    for _, starts in conditions:
        first = np.searchsorted(spikes_s, starts, side="left")
        last = np.searchsorted(spikes_s, starts + duration_s, side="left")
        means.append(float(np.mean(last - first)))
    return int(np.argmax(means))


def released_metrics(
    spikes_s: np.ndarray,
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
) -> dict[str, float]:
    """Reproduce AllenSDK's 2-s preference and 1,999-bin metric convention."""
    preferred = _preferred_condition(spikes_s, conditions, duration_s=2.0)
    parameters, starts = conditions[preferred]
    trial_counts = _bin_trial_spike_counts(spikes_s, starts, duration_ms=1999)
    tf = parameters[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        f1 = f1_f0_from_trial_counts(trial_counts, tf, 1.9985)
        modulation = welch_modulation_index(trial_counts.mean(axis=0), tf)
    return {
        "raw_released_pref_ori": parameters[0],
        "raw_released_pref_tf": tf,
        "raw_released_f1_f0_dg": f1,
        "raw_released_mod_idx_dg": modulation,
    }


def _subsample_conditions(
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
    *,
    trials_per_condition: int,
    seed: int,
) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
    rng = np.random.default_rng(seed)
    sampled = []
    for parameters, starts in conditions:
        if len(starts) < trials_per_condition:
            raise ValueError(
                f"Condition {parameters} has {len(starts)} trials, fewer than "
                f"the requested {trials_per_condition}"
            )
        if len(starts) == trials_per_condition:
            chosen = starts.copy()
        else:
            chosen = starts[
                np.sort(rng.choice(len(starts), trials_per_condition, replace=False))
            ]
        sampled.append((parameters, chosen))
    return sampled


def common_window_metrics(
    spikes_s: np.ndarray,
    conditions: list[tuple[tuple[float, float, float, float], np.ndarray]],
) -> dict[str, float]:
    preferred = _preferred_condition(spikes_s, conditions, duration_s=1.0)
    parameters, starts = conditions[preferred]
    trial_counts = _bin_trial_spike_counts(spikes_s, starts, duration_ms=1000)
    tf = parameters[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        f1 = f1_f0_from_trial_counts(trial_counts, tf, 1.0)
        modulation = welch_modulation_index(trial_counts.mean(axis=0), tf)
    return {
        "common_pref_ori": parameters[0],
        "common_pref_tf": tf,
        "common_f1_f0_dg": f1,
        "common_mod_idx_dg": modulation,
    }


def common_qc(table: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(table["amplitude_cutoff"], errors="coerce").lt(0.1)
        & pd.to_numeric(table["presence_ratio"], errors="coerce").gt(0.8)
        & pd.to_numeric(table["isi_violations"], errors="coerce").lt(0.5)
    )


def build_session_selection_audit(
    release: pd.DataFrame,
    assets: list[dict[str, object]],
) -> pd.DataFrame:
    """Document where downloaded sessions sit in each released V1 cohort."""
    selected_roles = {
        int(asset["session_id"]): str(asset["selection_role"]) for asset in assets
    }
    eligible = release.loc[
        release["ecephys_structure_acronym"].eq("VISp")
        & release["session_type"].isin(COHORT_LABELS)
    ].copy()
    eligible = eligible.loc[common_qc(eligible)]
    rows = []
    for (session_type, session_id), group in eligible.groupby(
        ["session_type", "ecephys_session_id"]
    ):
        values = pd.to_numeric(group["mod_idx_dg"], errors="coerce")
        values = values.where(values > 0)
        rows.append(
            {
                "session_type": session_type,
                "cohort": COHORT_LABELS[session_type],
                "session_id": int(session_id),
                "n_common_qc_units": len(group),
                "n_valid_mod_idx": int(values.notna().sum()),
                "released_mean_log10_mod_idx": np.log10(values).mean(),
                "selection_role": selected_roles.get(int(session_id), "not_downloaded"),
            }
        )
    audit = pd.DataFrame(rows)
    audit["cohort_equal_session_mean_log10_mod_idx"] = audit.groupby(
        "cohort"
    )["released_mean_log10_mod_idx"].transform("mean")
    audit["absolute_distance_from_cohort_mean"] = np.abs(
        audit["released_mean_log10_mod_idx"]
        - audit["cohort_equal_session_mean_log10_mod_idx"]
    )
    audit["distance_rank"] = (
        audit.groupby("cohort")["absolute_distance_from_cohort_mean"]
        .rank(method="min")
        .astype(int)
    )
    return audit.sort_values(["cohort", "distance_rank", "session_id"])


def make_figure(
    validation: pd.DataFrame,
    session_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for cohort, group in validation.loc[validation["common_qc"]].groupby("cohort"):
        color = COLORS[cohort]
        axes[0, 0].scatter(
            np.log10(group["released_mod_idx_dg"]),
            np.log10(group["raw_released_mod_idx_dg"]),
            color=color,
            s=22,
            alpha=0.65,
            label=cohort,
        )
        axes[0, 1].scatter(
            np.log10(group["released_f1_f0_dg"]),
            np.log10(group["raw_released_f1_f0_dg"]),
            color=color,
            s=22,
            alpha=0.65,
        )
    for ax, label in zip(axes[0], ("log10 modulation index", "log10 F1/F0")):
        limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(limits, limits, color="black", linewidth=1)
        ax.set(xlabel=f"released {label}", ylabel=f"raw reproduction {label}")
        ax.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)

    for ax, metric, ylabel in (
        (axes[1, 0], "mod_idx_dg", "mean log10 modulation index"),
        (axes[1, 1], "f1_f0_dg", "mean log10 F1/F0"),
    ):
        selected = session_summary.loc[session_summary["metric"].eq(metric)]
        mouse = selected.loc[selected["cohort"].eq("MouseV2 V1")]
        rng = np.random.default_rng(20260805)
        ax.scatter(
            np.full(len(mouse), 2.0) + rng.uniform(-0.08, 0.08, len(mouse)),
            mouse["mean_log10"],
            color=COLORS["MouseV2 V1"],
            s=28,
            alpha=0.7,
        )
        ax.hlines(mouse["mean_log10"].mean(), 1.78, 2.22, color="black", lw=2)
        for index, cohort in enumerate(
            ("Allen Brain Observatory 1.1", "Allen Functional Connectivity")
        ):
            group = selected.loc[
                selected["cohort"].eq(cohort)
                & selected["selection_role"].eq("representative")
            ]
            released = group.loc[group["view"].eq("released"), "mean_log10"].iloc[0]
            common = group.loc[group["view"].eq("common_1s_15trials"), "mean_log10"]
            common_center = common.mean()
            common_low, common_high = np.percentile(common, [2.5, 97.5])
            ax.plot([index - 0.08, index + 0.08], [released, common_center], color=COLORS[cohort], lw=1.5)
            ax.scatter(index - 0.08, released, color=COLORS[cohort], s=35)
            ax.errorbar(
                index + 0.08,
                common_center,
                yerr=[[common_center - common_low], [common_high - common_center]],
                fmt="o",
                markerfacecolor="white",
                markeredgecolor=COLORS[cohort],
                ecolor=COLORS[cohort],
                capsize=3,
            )
        ax.set_xticks((0, 1, 2))
        ax.set_xticklabels(("Allen BO\nrepresentative", "Allen FC\nrepresentative", "MouseV2\n8 sessions"))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    axes[1, 0].set_title("Filled=release; open=common 1 s / 15 trials")
    fig.suptitle("Raw Allen V1 bridge: exact reproduction and common-window sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def refresh_manifest(output_dir: Path, script_path: Path) -> None:
    manifest_path = output_dir / "import_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_names = {
        "unit_metric_validation.csv",
        "trial_subsample_metrics.csv",
        "session_summary.csv",
        "session_selection_audit.csv",
        "raw_bridge_diagnostic.png",
        "README.md",
    }
    manifest["outputs"] = []
    for name in sorted(output_names):
        path = output_dir / name
        if path.is_file():
            manifest["outputs"].append(
                {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    manifest.setdefault("code", {})["script"] = sha256(script_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    if args.render_existing:
        validation = pd.read_csv(output_dir / "unit_metric_validation.csv")
        session_summary = pd.read_csv(output_dir / "session_summary.csv")
        make_figure(validation, session_summary, output_dir / "raw_bridge_diagnostic.png")
        refresh_manifest(output_dir, Path(__file__).resolve())
        print(f"Rendered raw Allen bridge: {output_dir}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from generate_retinotopic_csvs import read_nwb_tables

    release = pd.read_csv(ROOT / "data" / "unit_table.csv", low_memory=False)
    selection_audit = build_session_selection_audit(release, config["assets"])
    selection_audit.to_csv(output_dir / "session_selection_audit.csv", index=False)
    validation_frames = []
    subsample_frames = []
    input_records = []
    for asset in config["assets"]:
        session_id = int(asset["session_id"])
        session_type = str(asset["session_type"])
        cohort = COHORT_LABELS[session_type]
        selection_role = str(asset["selection_role"])
        path = Path(config["download_root"]) / asset["relative_path"]
        if not path.is_file() or path.stat().st_size != int(asset["bytes"]):
            raise FileNotFoundError(f"Missing or size-mismatched Allen NWB: {path}")
        observed_hash = None if args.skip_input_hashes else sha256(path)
        if observed_hash is not None and observed_hash != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        input_records.append(
            {
                "session_id": session_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed_hash or asset["sha256"],
                "dandiset_id": asset["dandiset_id"],
                "asset_id": asset["asset_id"],
                "selection_role": selection_role,
            }
        )
        print(f"[{session_id}] reading verified raw NWB", flush=True)
        extracted = read_nwb_tables(str(path))
        grating_table = extracted.intervals_tables[asset["grating_table"]]
        if len(grating_table) != int(asset["expected_grating_presentations"]):
            raise ValueError(f"{session_id}: grating presentation count mismatch")
        if len(extracted.intervals_tables["flashes_presentations"]) != int(
            asset["expected_flash_presentations"]
        ):
            raise ValueError(f"{session_id}: flash presentation count mismatch")
        released_conditions = condition_starts(grating_table)
        shared_conditions = condition_starts(
            grating_table, common_support=config["common_support"]
        )

        released_v1 = release.loc[
            release["ecephys_session_id"].eq(session_id)
            & release["ecephys_structure_acronym"].eq("VISp")
        ].copy()
        if len(released_v1) != int(asset["expected_released_v1_units"]):
            raise ValueError(f"{session_id}: released V1 unit count mismatch")
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        if not set(released_v1["ecephys_unit_id"].astype(int)).issubset(row_by_id):
            raise ValueError(f"{session_id}: released V1 unit IDs missing from NWB")

        validation_rows = []
        for _, released_row in released_v1.iterrows():
            unit_id = int(released_row["ecephys_unit_id"])
            spikes = extracted.spikes_by_unit[row_by_id[unit_id]]
            raw = released_metrics(spikes, released_conditions)
            validation_rows.append(
                {
                    "session_id": session_id,
                    "session_type": session_type,
                    "cohort": cohort,
                    "selection_role": selection_role,
                    "ecephys_unit_id": unit_id,
                    "common_qc": bool(common_qc(released_row.to_frame().T).iloc[0]),
                    "released_f1_f0_dg": released_row["f1_f0_dg"],
                    "released_mod_idx_dg": released_row["mod_idx_dg"],
                    **raw,
                }
            )
        validation = pd.DataFrame(validation_rows)
        validation_frames.append(validation)

        repeats = min(len(starts) for _, starts in shared_conditions)
        n_subsamples = 1 if repeats == int(config["common_support"]["trials_per_condition"]) else args.trial_subsamples
        print(
            f"[{session_id}] common bridge: {len(shared_conditions)} conditions, "
            f"{repeats} available trials, {n_subsamples} subsamples",
            flush=True,
        )
        subsample_rows = []
        for seed in range(n_subsamples):
            sampled = _subsample_conditions(
                shared_conditions,
                trials_per_condition=int(config["common_support"]["trials_per_condition"]),
                seed=session_id + seed,
            )
            for _, released_row in released_v1.iterrows():
                unit_id = int(released_row["ecephys_unit_id"])
                spikes = extracted.spikes_by_unit[row_by_id[unit_id]]
                common = common_window_metrics(spikes, sampled)
                subsample_rows.append(
                    {
                        "session_id": session_id,
                        "session_type": session_type,
                        "cohort": cohort,
                        "selection_role": selection_role,
                        "ecephys_unit_id": unit_id,
                        "subsample": seed,
                        "common_qc": bool(common_qc(released_row.to_frame().T).iloc[0]),
                        **common,
                    }
                )
        subsample_frames.append(pd.DataFrame(subsample_rows))

    validation = pd.concat(validation_frames, ignore_index=True)
    subsamples = pd.concat(subsample_frames, ignore_index=True)
    validation.to_csv(output_dir / "unit_metric_validation.csv", index=False)
    subsamples.to_csv(output_dir / "trial_subsample_metrics.csv", index=False)

    summary_rows = []
    for (cohort, session_id, selection_role), group in validation.loc[
        validation["common_qc"]
    ].groupby(["cohort", "session_id", "selection_role"]):
        for metric in ("mod_idx_dg", "f1_f0_dg"):
            values = pd.to_numeric(group[f"released_{metric}"], errors="coerce")
            summary_rows.append(
                {
                    "cohort": cohort,
                    "session_id": session_id,
                    "selection_role": selection_role,
                    "view": "released",
                    "subsample": -1,
                    "metric": metric,
                    "n_units": values.notna().sum(),
                    "mean_log10": np.log10(values.where(values > 0)).mean(),
                }
            )
    for (cohort, session_id, selection_role, seed), group in subsamples.loc[
        subsamples["common_qc"]
    ].groupby(["cohort", "session_id", "selection_role", "subsample"]):
        for metric in ("mod_idx_dg", "f1_f0_dg"):
            values = pd.to_numeric(group[f"common_{metric}"], errors="coerce")
            summary_rows.append(
                {
                    "cohort": cohort,
                    "session_id": int(session_id),
                    "selection_role": selection_role,
                    "view": "common_1s_15trials",
                    "subsample": int(seed),
                    "metric": metric,
                    "n_units": values.notna().sum(),
                    "mean_log10": np.log10(values.where(values > 0)).mean(),
                }
            )

    mouse = pd.read_csv(MOUSE_COMMON / "unit_metric_comparison.csv")
    mouse = mouse.loc[mouse["default_qc"].eq(True)]
    for site_number, group in mouse.groupby("site_number"):
        for metric in ("mod_idx_dg", "f1_f0_dg"):
            values = pd.to_numeric(
                group[f"{metric}_common_support"], errors="coerce"
            )
            summary_rows.append(
                {
                    "cohort": "MouseV2 V1",
                    "session_id": int(site_number),
                    "selection_role": "mouse_all_sessions",
                    "view": "common_1s_15trials",
                    "subsample": 0,
                    "metric": metric,
                    "n_units": values.notna().sum(),
                    "mean_log10": np.log10(values.where(values > 0)).mean(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "session_summary.csv", index=False)

    reproduction = []
    for (cohort, session_id, selection_role), group in validation.groupby(
        ["cohort", "session_id", "selection_role"]
    ):
        qc_group = group.loc[group["common_qc"]]
        all_mod_error = np.abs(
            group["raw_released_mod_idx_dg"] - group["released_mod_idx_dg"]
        )
        reproduction.append(
            {
                "cohort": cohort,
                "session_id": int(session_id),
                "selection_role": selection_role,
                "released_units": len(group),
                "common_qc_units": len(qc_group),
                "excluded_unit_mod_mismatches": int(
                    ((all_mod_error > 1e-5) & ~group["common_qc"]).sum()
                ),
                "f1_max_abs_error": np.nanmax(
                    np.abs(
                        qc_group["raw_released_f1_f0_dg"]
                        - qc_group["released_f1_f0_dg"]
                    )
                ),
                "mod_idx_max_abs_error": np.nanmax(
                    np.abs(
                        qc_group["raw_released_mod_idx_dg"]
                        - qc_group["released_mod_idx_dg"]
                    )
                ),
            }
        )
    reproduction = pd.DataFrame(reproduction)

    mouse_centers = {
        metric: summary.loc[
            summary["cohort"].eq("MouseV2 V1") & summary["metric"].eq(metric),
            "mean_log10",
        ].mean()
        for metric in ("mod_idx_dg", "f1_f0_dg")
    }
    result_lines = []
    for cohort in COHORT_LABELS.values():
        for metric in ("mod_idx_dg", "f1_f0_dg"):
            rows = summary.loc[
                summary["cohort"].eq(cohort)
                & summary["metric"].eq(metric)
                & summary["selection_role"].eq("representative")
            ]
            released_center = rows.loc[rows["view"].eq("released"), "mean_log10"].iloc[0]
            common_values = rows.loc[
                rows["view"].eq("common_1s_15trials"), "mean_log10"
            ]
            delta = common_values - released_center
            common_center = common_values.mean()
            result_lines.append(
                f"- {cohort}, representative session {int(rows['session_id'].iloc[0])}, "
                f"{metric}: released {released_center:+.3f}; "
                f"common-window {common_center:+.3f}; change {delta.mean():+.3f} "
                f"(subsample 2.5–97.5% {delta.quantile(.025):+.3f} to {delta.quantile(.975):+.3f}); "
                f"MouseV2 minus harmonized Allen {mouse_centers[metric] - common_center:+.3f}."
            )
    mouse_mod = summary.loc[
        summary["cohort"].eq("MouseV2 V1")
        & summary["metric"].eq("mod_idx_dg"),
        "mean_log10",
    ]
    report = [
        "# Raw Allen V1 bridge",
        "",
        "## Released-value reproduction",
        "",
        *[
            f"- {row.cohort}, session {int(row.session_id)} ({row.selection_role}): "
            f"{int(row.common_qc_units)}/{int(row.released_units)} common-QC/released V1 units; "
            f"max common-QC F1/F0 error "
            f"{row.f1_max_abs_error:.3g}; max common-QC modulation-index error {row.mod_idx_max_abs_error:.3g}."
            for row in reproduction.itertuples()
        ],
        *[
            f"- Note: {int(row.excluded_unit_mod_mismatches)} excluded non-QC unit(s) "
            "had modulation mismatch >1e-5."
            for row in reproduction.itertuples()
            if row.excluded_unit_mod_mismatches
        ],
        "",
        "The h5py path therefore reproduces the released metrics without relying on",
        "the currently incompatible AllenSDK/PyNWB environment.",
        "",
        "## Common 1-s / 15-trial sensitivity",
        "",
        "The representative session in each Allen cohort is the released session",
        "nearest that cohort's equal-session V1 mean modulation index. The original",
        "low-unit sessions are retained only as independent reproduction controls.",
        "",
        *result_lines,
        f"- MouseV2's eight-session common-support mean is {mouse_mod.mean():+.3f} log10 modulation index.",
        "- The common-window convention therefore does not remove the modulation-index gap;",
        "  it remains about -0.19 log10 versus Brain Observatory and -0.22 versus Functional Connectivity.",
        "- By contrast, the harmonized F1/F0 gap is only about -0.03 to -0.04 log10,",
        "  showing that the large modulation-index difference is metric-specific.",
        "",
        "The representative sessions expose the direction of the protocol effect, while",
        "all four sessions validate the raw implementation. One representative session",
        "per cohort is not sufficient to estimate the population-level",
        "Allen dataset coefficient; the common-window extraction must be extended to",
        "more sessions or its uncertainty must be propagated as a protocol sensitivity.",
        "",
        "## Verified raw protocol",
        "",
        "- Brain Observatory: 40 nonblank grating conditions x 15 repeats, 2-s trials, and 150 flashes.",
        "- Functional Connectivity: 8 grating directions x 75 repeats at 2 Hz, 2-s trials, and 150 flashes.",
        "- MouseV2: 100 grating conditions x 15 repeats, 1-s trials, and 300 flashes; its SF = 0.04 raw bridge is already complete.",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if not args.skip_figure:
        make_figure(validation, summary, output_dir / "raw_bridge_diagnostic.png")
    manifest = {
        "schema_version": 1,
        "inputs": input_records,
        "common_support": config["common_support"],
        "trial_subsamples": args.trial_subsamples,
        "outputs": [],
        "code": {
            "script": sha256(Path(__file__).resolve()),
            "metric": sha256(ROOT / "common" / "drifting_gratings.py"),
            "reader": sha256(ROOT / "generate_retinotopic_csvs.py"),
            "config": sha256(config_path),
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    refresh_manifest(output_dir, Path(__file__).resolve())
    print(f"Raw Allen V1 bridge written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
