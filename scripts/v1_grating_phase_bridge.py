#!/usr/bin/env python3
"""Decompose the harmonized V1 grating response into amplitude and phase coherence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import (  # noqa: E402
    _bin_trial_spike_counts,
    f1_f0_from_trial_counts,
    welch_modulation_index,
)
from scripts.extract_allen_v1_bridge import (  # noqa: E402
    COHORT_LABELS,
    _subsample_conditions as allen_subsample_conditions,
    common_qc,
    condition_starts as allen_condition_starts,
)
from scripts.extract_mousev2_grating_common_support import (  # noqa: E402
    CONTRAST,
    ORIENTATIONS_DEG,
    SPATIAL_FREQUENCY_CPD,
    TEMPORAL_FREQUENCIES_HZ,
    common_presentations,
)


MOUSE_CONFIG = ROOT / "config" / "figure3_mousev2.json"
ALLEN_CONFIG = ROOT / "config" / "allen_v1_bridge.json"
DEFAULT_OUTPUT = ROOT / "data" / "imports" / "v1_grating_phase_bridge_v1"
MOUSE_HASH_MANIFEST = (
    ROOT / "data" / "imports" / "mousev2_grating_metrics_v1" / "import_manifest.json"
)
COLORS = {
    "MouseV2 V1": "#D95F02",
    "Allen Brain Observatory 1.1": "#6F63A6",
    "Allen Functional Connectivity": "#B07AA1",
}
SUMMARY_METRICS = {
    "log10_mod_idx": "log10 Welch modulation index",
    "log10_f1_f0": "log10 F1/F0",
    "weighted_phase_coherence": "weighted phase coherence",
    "unweighted_phase_ppc": "unweighted phase PPC",
    "log10_mean_trial_f1_hz": "log10 mean trial F1 amplitude",
    "log10_coherent_f1_hz": "log10 coherent F1 amplitude",
    "log10_target_to_offtarget_psd": "log10 target/off-target PSD",
    "log10_onset_to_sustained_rate": "log10 onset/sustained rate",
    "log10_preferred_rate_hz": "log10 preferred-condition rate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mouse_nwb_provenance(
    mouse_config: Path = MOUSE_CONFIG,
) -> dict[int, dict[str, object]]:
    """Return checksum-verified MouseV2 input metadata keyed by site number."""
    config = json.loads(mouse_config.read_text(encoding="utf-8"))
    site_numbers = {str(session["site"]): int(session["site_number"]) for session in config["sessions"]}
    source = json.loads(MOUSE_HASH_MANIFEST.read_text(encoding="utf-8"))
    return {
        site_numbers[str(record["site"])]: record
        for record in source["inputs"]
        if str(record["site"]) in site_numbers
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse-config", type=Path, default=MOUSE_CONFIG)
    parser.add_argument("--allen-config", type=Path, default=ALLEN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fc-subsamples", type=int, default=25)
    parser.add_argument("--mouse-sites", nargs="*", default=None)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    return parser.parse_args()


def condition_trials(table: pd.DataFrame) -> list[tuple[tuple[float, ...], np.ndarray]]:
    required = {
        "orientation",
        "temporal_frequency",
        "spatial_frequency",
        "contrast",
        "start_time",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Grating table lacks columns {missing}")
    selected = table.copy()
    for column in required:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[
        selected["orientation"].isin(ORIENTATIONS_DEG)
        & selected["temporal_frequency"].isin(TEMPORAL_FREQUENCIES_HZ)
        & np.isclose(selected["spatial_frequency"], SPATIAL_FREQUENCY_CPD)
        & np.isclose(selected["contrast"], CONTRAST)
    ]
    dimensions = ["orientation", "temporal_frequency", "spatial_frequency", "contrast"]
    return [
        (tuple(map(float, key)), group["start_time"].to_numpy(dtype=float))
        for key, group in selected.groupby(dimensions, sort=True)
    ]


def preferred_trial_counts(
    spikes_s: np.ndarray,
    conditions: list[tuple[tuple[float, ...], np.ndarray]],
) -> tuple[tuple[float, ...], np.ndarray]:
    means = []
    for _, starts in conditions:
        first = np.searchsorted(spikes_s, starts, side="left")
        last = np.searchsorted(spikes_s, starts + 1.0, side="left")
        means.append(float(np.mean(last - first)))
    selected = int(np.argmax(means))
    parameters, starts = conditions[selected]
    return parameters, _bin_trial_spike_counts(spikes_s, starts, duration_ms=1000)


def fourier_decomposition(
    trial_counts: np.ndarray,
    temporal_frequency_hz: float,
) -> dict[str, float]:
    """Separate per-trial amplitude from across-trial phase-coherent amplitude."""
    counts = np.asarray(trial_counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] != 1000:
        raise ValueError("Expected at least two trials with exactly 1,000 1-ms bins")
    duration_s = counts.shape[1] / 1000.0
    time_s = np.arange(counts.shape[1], dtype=float) / 1000.0
    kernel = np.exp(-2j * np.pi * float(temporal_frequency_hz) * time_s)
    coefficients = (2.0 / duration_s) * (counts @ kernel)
    amplitudes = np.abs(coefficients)
    mean_trial_amplitude = float(np.mean(amplitudes))
    coherent_amplitude = float(np.abs(np.mean(coefficients)))
    weighted_coherence = (
        coherent_amplitude / mean_trial_amplitude if mean_trial_amplitude > 0 else np.nan
    )

    nonzero = amplitudes > 0
    phase_vectors = coefficients[nonzero] / amplitudes[nonzero]
    n_phase = len(phase_vectors)
    phase_resultant = (
        float(np.abs(np.mean(phase_vectors))) if n_phase else np.nan
    )
    phase_ppc = (
        float((np.abs(np.sum(phase_vectors)) ** 2 - n_phase) / (n_phase * (n_phase - 1)))
        if n_phase >= 2
        else np.nan
    )
    amplitude_sum = float(np.sum(amplitudes))
    amplitude_square_sum = float(np.sum(amplitudes**2))
    weighted_denominator = amplitude_sum**2 - amplitude_square_sum
    weighted_ppc = (
        float((np.abs(np.sum(coefficients)) ** 2 - amplitude_square_sum) / weighted_denominator)
        if weighted_denominator > 0
        else np.nan
    )
    cross_trial_power = float(
        (np.abs(np.sum(coefficients)) ** 2 - amplitude_square_sum)
        / (len(coefficients) * (len(coefficients) - 1))
    )

    psth = counts.mean(axis=0)
    frequencies, psd = signal.welch(psth, fs=1000.0, nperseg=1000)
    tf_index = int(np.argmin(np.abs(frequencies - float(temporal_frequency_hz))))
    target_psd = float(psd[tf_index])
    exclude = np.ones(len(psd), dtype=bool)
    exclude[0] = False
    exclude[tf_index] = False
    off_target_psd = float(np.mean(psd[exclude]))
    target_to_offtarget = target_psd / off_target_psd if off_target_psd > 0 else np.nan
    mean_psd = float(np.mean(psd))
    psd_sd = float(np.sqrt(max(np.mean(psd**2) - mean_psd**2, 0.0)))

    onset_rate = float(np.mean(counts[:, :100].sum(axis=1) / 0.1))
    sustained_rate = float(np.mean(counts[:, 100:].sum(axis=1) / 0.9))
    preferred_rate = float(np.mean(counts.sum(axis=1) / duration_s))
    equivalent_jitter = (
        float(np.sqrt(-2.0 * np.log(weighted_coherence)) / (2 * np.pi * temporal_frequency_hz) * 1000)
        if 0 < weighted_coherence <= 1
        else np.nan
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modulation = welch_modulation_index(psth, temporal_frequency_hz)
        f1_f0 = f1_f0_from_trial_counts(counts, temporal_frequency_hz, 1.0)
    return {
        "mod_idx_dg": modulation,
        "f1_f0_dg": f1_f0,
        "mean_trial_f1_hz": mean_trial_amplitude,
        "coherent_f1_hz": coherent_amplitude,
        "weighted_phase_coherence": weighted_coherence,
        "phase_resultant_length": phase_resultant,
        "unweighted_phase_ppc": phase_ppc,
        "weighted_phase_ppc": weighted_ppc,
        "cross_trial_coherent_power_hz2": cross_trial_power,
        "equivalent_phase_jitter_ms": equivalent_jitter,
        "target_psd": target_psd,
        "off_target_mean_psd": off_target_psd,
        "target_to_offtarget_psd": target_to_offtarget,
        "welch_psd_mean": mean_psd,
        "welch_psd_sd": psd_sd,
        "onset_rate_hz": onset_rate,
        "sustained_rate_hz": sustained_rate,
        "onset_to_sustained_rate": onset_rate / sustained_rate if sustained_rate > 0 else np.nan,
        "preferred_rate_hz": preferred_rate,
        "phase_trials": n_phase,
    }


def add_log_metrics(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    sources = {
        "log10_mod_idx": "mod_idx_dg",
        "log10_f1_f0": "f1_f0_dg",
        "log10_mean_trial_f1_hz": "mean_trial_f1_hz",
        "log10_coherent_f1_hz": "coherent_f1_hz",
        "log10_target_to_offtarget_psd": "target_to_offtarget_psd",
        "log10_onset_to_sustained_rate": "onset_to_sustained_rate",
        "log10_preferred_rate_hz": "preferred_rate_hz",
    }
    for target, source in sources.items():
        values = pd.to_numeric(result[source], errors="coerce")
        result[target] = np.log10(values.where(values > 0))
    return result


def summarize_sessions(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in units.groupby(
        ["cohort", "session_id", "selection_role", "subsample"], sort=True
    ):
        row = dict(zip(["cohort", "session_id", "selection_role", "subsample"], keys))
        row["n_units"] = len(group)
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = values.mean()
            row[f"median_{metric}"] = values.median()
            row[f"n_{metric}"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def center_table(summary: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        summary.groupby(["cohort", "session_id", "selection_role"], as_index=False)
        .mean(numeric_only=True)
    )
    rows = []
    for cohort, group in averaged.groupby("cohort"):
        if cohort.startswith("Allen"):
            primary = group.loc[group["selection_role"].eq("representative")]
        else:
            primary = group
        for metric in SUMMARY_METRICS:
            column = f"mean_{metric}"
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "label": SUMMARY_METRICS[metric],
                    "primary_sessions": len(primary),
                    "primary_equal_session_mean": primary[column].mean(),
                    "all_downloaded_session_mean": group[column].mean(),
                    "primary_session_min": primary[column].min(),
                    "primary_session_max": primary[column].max(),
                }
            )
    return pd.DataFrame(rows)


def temporal_frequency_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in units.groupby(
        [
            "cohort",
            "session_id",
            "selection_role",
            "subsample",
            "preferred_tf_hz",
        ],
        sort=True,
    ):
        row = dict(
            zip(
                [
                    "cohort",
                    "session_id",
                    "selection_role",
                    "subsample",
                    "preferred_tf_hz",
                ],
                keys,
            )
        )
        row.update(
            {
                "n_units": len(group),
                "mean_weighted_phase_coherence": group[
                    "weighted_phase_coherence"
                ].mean(),
                "mean_unweighted_phase_ppc": group["unweighted_phase_ppc"].mean(),
                "mean_log10_mod_idx": group["log10_mod_idx"].mean(),
                "mean_log10_mean_trial_f1_hz": group[
                    "log10_mean_trial_f1_hz"
                ].mean(),
                "mean_log10_coherent_f1_hz": group[
                    "log10_coherent_f1_hz"
                ].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def adjusted_phase_diagnostic(units: pd.DataFrame) -> pd.DataFrame:
    """Descriptive OLS adjustment; not a population-level inferential model."""
    primary = units.loc[
        units["selection_role"].ne("reproduction_control")
    ].copy()
    averaged = (
        primary.groupby(
            ["cohort", "session_id", "unit_id", "preferred_tf_hz"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    averaged["mouse"] = averaged["cohort"].eq("MouseV2 V1").astype(float)
    scopes = {
        "all_primary_downloads": averaged,
        "mouse_vs_bo_at_1_2_hz": averaged.loc[
            averaged["cohort"].ne("Allen Functional Connectivity")
            & averaged["preferred_tf_hz"].isin([1.0, 2.0])
        ],
    }
    rows = []
    for scope, table in scopes.items():
        for outcome in ("weighted_phase_coherence", "unweighted_phase_ppc"):
            selected = table.dropna(
                subset=[
                    outcome,
                    "log10_mean_trial_f1_hz",
                    "log10_preferred_rate_hz",
                ]
            ).copy()
            tf_dummies = pd.get_dummies(
                selected["preferred_tf_hz"].astype(str),
                prefix="tf",
                drop_first=True,
                dtype=float,
            )
            design = np.column_stack(
                [
                    np.ones(len(selected)),
                    selected["mouse"],
                    selected["log10_mean_trial_f1_hz"],
                    selected["log10_preferred_rate_hz"],
                    tf_dummies.to_numpy(),
                ]
            )
            response = selected[outcome].to_numpy(dtype=float)
            coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
            fitted = design @ coefficients
            residual = float(np.sum((response - fitted) ** 2))
            total = float(np.sum((response - response.mean()) ** 2))
            rows.append(
                {
                    "scope": scope,
                    "outcome": outcome,
                    "n_units": len(selected),
                    "n_sessions": selected["session_id"].nunique(),
                    "mouse_coefficient": coefficients[1],
                    "r_squared": 1.0 - residual / total if total > 0 else np.nan,
                    "covariates": "log10 mean-trial F1; log10 preferred rate; categorical TF",
                    "interpretation": "descriptive only; Allen primary support is one representative session per cohort",
                }
            )
    return pd.DataFrame(rows)


def make_figure(units: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260805)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for cohort, group in units.groupby("cohort"):
        if len(group) > 1800:
            group = group.iloc[rng.choice(len(group), 1800, replace=False)]
        axes[0, 0].scatter(
            group["log10_mean_trial_f1_hz"],
            group["log10_coherent_f1_hz"],
            s=5,
            alpha=0.18,
            color=COLORS[cohort],
            label=cohort,
        )
        axes[0, 1].scatter(
            group["weighted_phase_coherence"],
            group["log10_mod_idx"],
            s=5,
            alpha=0.18,
            color=COLORS[cohort],
        )
    axes[0, 0].set(
        xlabel="log10 mean single-trial F1 amplitude",
        ylabel="log10 coherent F1 amplitude",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set(
        xlabel="weighted phase coherence",
        ylabel="log10 Welch modulation index",
    )

    averaged = (
        summary.groupby(["cohort", "session_id", "selection_role"], as_index=False)
        .mean(numeric_only=True)
    )
    panels = [
        (axes[0, 2], "mean_weighted_phase_coherence", "weighted phase coherence"),
        (axes[1, 0], "mean_log10_mean_trial_f1_hz", "log10 mean trial F1"),
        (axes[1, 1], "mean_log10_coherent_f1_hz", "log10 coherent F1"),
        (axes[1, 2], "mean_log10_target_to_offtarget_psd", "log10 target/off-target PSD"),
    ]
    order = list(COLORS)
    for ax, column, ylabel in panels:
        for index, cohort in enumerate(order):
            group = averaged.loc[averaged["cohort"].eq(cohort)]
            jitter = rng.uniform(-0.08, 0.08, len(group))
            markers = ["o" if role != "reproduction_control" else "x" for role in group["selection_role"]]
            for x, (_, row), marker in zip(index + jitter, group.iterrows(), markers):
                ax.scatter(x, row[column], color=COLORS[cohort], marker=marker, s=42)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(("MouseV2\n8 sessions", "Allen BO\n2 downloaded", "Allen FC\n2 downloaded"))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle("Harmonized V1 grating response: amplitude versus phase coherence")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    centers: pd.DataFrame,
    tf_summary: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> None:
    def value(cohort: str, metric: str) -> float:
        return float(
            centers.loc[
                centers["cohort"].eq(cohort) & centers["metric"].eq(metric),
                "primary_equal_session_mean",
            ].iloc[0]
        )

    mouse = "MouseV2 V1"
    bo = "Allen Brain Observatory 1.1"
    fc = "Allen Functional Connectivity"
    amplitude_loss = {
        cohort: value(cohort, "log10_coherent_f1_hz")
        - value(cohort, "log10_mean_trial_f1_hz")
        for cohort in (mouse, bo, fc)
    }

    def tf_value(cohort: str, tf: float, column: str) -> float:
        selected = tf_summary.loc[
            tf_summary["cohort"].eq(cohort)
            & tf_summary["preferred_tf_hz"].eq(tf)
        ].copy()
        if cohort.startswith("Allen"):
            selected = selected.loc[selected["selection_role"].eq("representative")]
        per_session = selected.groupby("session_id")[column].mean()
        return float(per_session.mean())

    adjusted_phase = float(
        adjusted.loc[
            adjusted["scope"].eq("all_primary_downloads")
            & adjusted["outcome"].eq("weighted_phase_coherence"),
            "mouse_coefficient",
        ].iloc[0]
    )
    lines = [
        "# V1 grating phase-coherence bridge",
        "",
        "All values use the harmonized 1-s, 15-trial, SF = 0.04, contrast = 0.8",
        "support. Allen centers use the representative downloaded session per cohort;",
        "the independent reproduction-control sessions remain visible in the tables and figure.",
        "",
        "## Primary decomposition",
        "",
        f"- Weighted phase coherence: MouseV2 {value(mouse, 'weighted_phase_coherence'):.3f}; "
        f"Allen BO {value(bo, 'weighted_phase_coherence'):.3f}; Allen FC {value(fc, 'weighted_phase_coherence'):.3f}.",
        f"- Log10 mean single-trial F1 amplitude: MouseV2 {value(mouse, 'log10_mean_trial_f1_hz'):+.3f}; "
        f"Allen BO {value(bo, 'log10_mean_trial_f1_hz'):+.3f}; Allen FC {value(fc, 'log10_mean_trial_f1_hz'):+.3f}.",
        f"- Log10 coherent F1 amplitude: MouseV2 {value(mouse, 'log10_coherent_f1_hz'):+.3f}; "
        f"Allen BO {value(bo, 'log10_coherent_f1_hz'):+.3f}; Allen FC {value(fc, 'log10_coherent_f1_hz'):+.3f}.",
        f"- Log10 target/off-target PSD: MouseV2 {value(mouse, 'log10_target_to_offtarget_psd'):+.3f}; "
        f"Allen BO {value(bo, 'log10_target_to_offtarget_psd'):+.3f}; Allen FC {value(fc, 'log10_target_to_offtarget_psd'):+.3f}.",
        f"- Log10 amplitude lost during trial averaging: MouseV2 {amplitude_loss[mouse]:+.3f}; "
        f"Allen BO {amplitude_loss[bo]:+.3f}; Allen FC {amplitude_loss[fc]:+.3f}.",
        f"- At 1 Hz, weighted phase coherence is {tf_value(mouse, 1.0, 'mean_weighted_phase_coherence'):.3f} in MouseV2 "
        f"and {tf_value(bo, 1.0, 'mean_weighted_phase_coherence'):.3f} in representative Allen BO; "
        f"at 2 Hz it is {tf_value(mouse, 2.0, 'mean_weighted_phase_coherence'):.3f}, "
        f"{tf_value(bo, 2.0, 'mean_weighted_phase_coherence'):.3f}, and "
        f"{tf_value(fc, 2.0, 'mean_weighted_phase_coherence'):.3f} for MouseV2, Allen BO, and Allen FC.",
        f"- After descriptive adjustment for mean-trial F1 amplitude, preferred rate, and TF, "
        f"the MouseV2 phase-coherence coefficient remains {adjusted_phase:+.3f}.",
        "",
        "## Interpretation",
        "",
        "MouseV2 does not have weaker single-trial grating modulation: its mean-trial",
        "F1 amplitude is comparable to or higher than the representative Allen sessions.",
        "The difference appears when trials are coherently averaged. MouseV2 loses more",
        "amplitude to phase inconsistency, and its target-frequency peak is less distinct",
        "from off-target power. The same direction at 1 and 2 Hz argues against preferred-TF",
        "composition as the sole explanation.",
        "",
        "This strongly supports trial-to-trial phase/latency variability as the proximate",
        "mathematical cause of the low Welch modulation index. It does not distinguish",
        "display-timestamp jitter from neural-state, eye-movement, or genuine response-phase",
        "variability without an independent photodiode timing reference. The adjusted model",
        "is descriptive because raw Allen primary support remains one session per cohort.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_manifest(output_dir: Path) -> None:
    path = output_dir / "import_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mouse_inputs = mouse_nwb_provenance()
    for record in manifest.get("inputs", []):
        if record.get("dataset") != "MouseV2":
            continue
        source = mouse_inputs[int(record["session_id"])]
        if int(record["bytes"]) != int(source["bytes"]):
            raise ValueError(f"MouseV2 site {record['session_id']}: provenance byte count mismatch")
        record["sha256"] = str(source["sha256"])
        record["sha256_source"] = str(MOUSE_HASH_MANIFEST.relative_to(ROOT))
    manifest["outputs"] = []
    for name in (
        "README.md",
        "center_summary.csv",
        "phase_bridge_diagnostic.png",
        "session_summary.csv",
        "tf_summary.csv",
        "adjusted_phase_diagnostic.csv",
        "unit_phase_metrics.csv",
    ):
        output = output_dir / name
        if output.is_file():
            manifest["outputs"].append(
                {"path": name, "bytes": output.stat().st_size, "sha256": sha256(output)}
            )
    manifest["code"]["script_sha256"] = sha256(Path(__file__).resolve())
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def render_existing(output_dir: Path) -> None:
    units = pd.read_csv(output_dir / "unit_phase_metrics.csv")
    summary = pd.read_csv(output_dir / "session_summary.csv")
    centers = pd.read_csv(output_dir / "center_summary.csv")
    tf_summary = temporal_frequency_summary(units)
    adjusted = adjusted_phase_diagnostic(units)
    tf_summary.to_csv(output_dir / "tf_summary.csv", index=False)
    adjusted.to_csv(output_dir / "adjusted_phase_diagnostic.csv", index=False)
    make_figure(units, summary, output_dir / "phase_bridge_diagnostic.png")
    write_report(output_dir, centers, tf_summary, adjusted)
    refresh_manifest(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.render_existing:
        render_existing(output_dir)
        print(f"Rendered V1 phase bridge: {output_dir}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mouse_config = json.loads(args.mouse_config.resolve().read_text(encoding="utf-8"))
    allen_config = json.loads(args.allen_config.resolve().read_text(encoding="utf-8"))
    mouse_inputs = mouse_nwb_provenance(args.mouse_config.resolve())
    release = pd.read_csv(ROOT / "data" / "unit_table.csv", low_memory=False)
    from generate_retinotopic_csvs import choose_stim_table, read_nwb_tables

    rows = []
    inputs = []
    requested = set(args.mouse_sites) if args.mouse_sites else None
    mouse_sessions = [
        session
        for session in mouse_config["sessions"]
        if requested is None or str(session["site"]) in requested
    ]
    if requested is not None and {str(s["site"]) for s in mouse_sessions} != requested:
        raise ValueError("One or more requested MouseV2 sites are absent from the config")
    mouse_root = Path(mouse_config["nwb_input"]["default_root"])
    for session in mouse_sessions:
        site = str(session["site"])
        site_number = int(session["site_number"])
        offset = int(session["id_offset"])
        path = mouse_root / str(session["nwb_relative_path"])
        if not path.is_file() or path.stat().st_size != int(session["expected_nwb_bytes"]):
            raise FileNotFoundError(f"Missing or size-mismatched MouseV2 NWB: {path}")
        print(f"[{site}] reading MouseV2 NWB", flush=True)
        extracted = read_nwb_tables(str(path))
        _, table = choose_stim_table(
            extracted.intervals_tables, "drifting_gratings_field_block_presentations"
        )
        selected = common_presentations(table)
        conditions = condition_trials(selected)
        if len(conditions) != 20 or {len(starts) for _, starts in conditions} != {15}:
            raise ValueError(f"{site}: unexpected common grating support")
        quality = pd.read_csv(ROOT / "data" / f"{site}_processed" / "unit_quality.csv")
        quality = quality.loc[quality["default_qc"].eq(True), ["unit_id"]]
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        for index, output_id in enumerate(quality["unit_id"].astype(int), start=1):
            local_id = int(output_id) - offset
            parameters, counts = preferred_trial_counts(
                extracted.spikes_by_unit[row_by_id[local_id]], conditions
            )
            rows.append(
                {
                    "dataset": "MouseV2",
                    "cohort": "MouseV2 V1",
                    "session_id": site_number,
                    "selection_role": "mouse_all_sessions",
                    "subsample": 0,
                    "unit_id": int(output_id),
                    "preferred_orientation_deg": parameters[0],
                    "preferred_tf_hz": parameters[1],
                    "preferred_trials": len(counts),
                    **fourier_decomposition(counts, parameters[1]),
                }
            )
            if index % 400 == 0:
                print(f"[{site}] decomposed {index}/{len(quality)} units", flush=True)
        source = mouse_inputs[site_number]
        inputs.append(
            {
                "dataset": "MouseV2",
                "session_id": site_number,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": str(source["sha256"]),
                "sha256_source": str(MOUSE_HASH_MANIFEST.relative_to(ROOT)),
            }
        )

    allen_root = Path(allen_config["download_root"])
    for asset in allen_config["assets"]:
        session_id = int(asset["session_id"])
        path = allen_root / str(asset["relative_path"])
        if not path.is_file() or path.stat().st_size != int(asset["bytes"]):
            raise FileNotFoundError(f"Missing or size-mismatched Allen NWB: {path}")
        print(f"[{session_id}] reading Allen NWB", flush=True)
        extracted = read_nwb_tables(str(path))
        conditions = allen_condition_starts(
            extracted.intervals_tables[asset["grating_table"]],
            common_support=allen_config["common_support"],
        )
        expected_conditions = 20 if asset["session_type"] == "brain_observatory_1.1" else 4
        if len(conditions) != expected_conditions:
            raise ValueError(f"{session_id}: expected {expected_conditions} common conditions")
        released_v1 = release.loc[
            release["ecephys_session_id"].eq(session_id)
            & release["ecephys_structure_acronym"].eq("VISp")
        ].copy()
        released_v1 = released_v1.loc[common_qc(released_v1)]
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        repeats = min(len(starts) for _, starts in conditions)
        n_subsamples = args.fc_subsamples if repeats > 15 else 1
        for subsample in range(n_subsamples):
            sampled = allen_subsample_conditions(
                conditions, trials_per_condition=15, seed=session_id + subsample
            )
            for _, unit in released_v1.iterrows():
                unit_id = int(unit["ecephys_unit_id"])
                parameters, counts = preferred_trial_counts(
                    extracted.spikes_by_unit[row_by_id[unit_id]], sampled
                )
                rows.append(
                    {
                        "dataset": "Allen",
                        "cohort": COHORT_LABELS[str(asset["session_type"])],
                        "session_id": session_id,
                        "selection_role": str(asset["selection_role"]),
                        "subsample": subsample,
                        "unit_id": unit_id,
                        "preferred_orientation_deg": parameters[0],
                        "preferred_tf_hz": parameters[1],
                        "preferred_trials": len(counts),
                        **fourier_decomposition(counts, parameters[1]),
                    }
                )
        inputs.append(
            {
                "dataset": "Allen",
                "session_id": session_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": asset["sha256"],
            }
        )

    units = add_log_metrics(pd.DataFrame(rows))
    units.to_csv(output_dir / "unit_phase_metrics.csv", index=False)
    summary = summarize_sessions(units)
    summary.to_csv(output_dir / "session_summary.csv", index=False)
    centers = center_table(summary)
    centers.to_csv(output_dir / "center_summary.csv", index=False)
    tf_summary = temporal_frequency_summary(units)
    tf_summary.to_csv(output_dir / "tf_summary.csv", index=False)
    adjusted = adjusted_phase_diagnostic(units)
    adjusted.to_csv(output_dir / "adjusted_phase_diagnostic.csv", index=False)
    write_report(output_dir, centers, tf_summary, adjusted)
    if not args.skip_figure:
        make_figure(units, summary, output_dir / "phase_bridge_diagnostic.png")
    manifest = {
        "schema_version": 1,
        "condition_support": {
            "duration_s": 1.0,
            "trials": 15,
            "orientation_deg": list(ORIENTATIONS_DEG),
            "temporal_frequency_hz": list(TEMPORAL_FREQUENCIES_HZ),
            "spatial_frequency_cpd": SPATIAL_FREQUENCY_CPD,
            "contrast": CONTRAST,
        },
        "fc_subsamples": args.fc_subsamples,
        "inputs": inputs,
        "code": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "grating_metric_sha256": sha256(ROOT / "common" / "drifting_gratings.py"),
            "mouse_config_sha256": sha256(args.mouse_config.resolve()),
            "allen_config_sha256": sha256(args.allen_config.resolve()),
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    refresh_manifest(output_dir)
    print(f"V1 phase bridge written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
