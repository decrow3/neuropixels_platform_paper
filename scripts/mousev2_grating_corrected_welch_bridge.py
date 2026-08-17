#!/usr/bin/env python3
"""Run source-phase correction through the released Welch modulation index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mousev2_grating_start_phase_bridge import (  # noqa: E402
    MOUSE_CONFIG,
    STIMULUS_MANIFEST,
    phase_aware_conditions,
    phase_schedule,
    preferred_condition_counts,
)


START_PHASE_IMPORT = (
    ROOT / "data" / "imports" / "mousev2_grating_start_phase_bridge_v1"
)
START_PHASE_MANIFEST = START_PHASE_IMPORT / "import_manifest.json"
START_PHASE_UNITS = START_PHASE_IMPORT / "unit_start_phase_metrics.csv"
PHASE_BRIDGE_IMPORT = ROOT / "data" / "imports" / "v1_grating_phase_bridge_v1"
PHASE_BRIDGE_MANIFEST = PHASE_BRIDGE_IMPORT / "import_manifest.json"
PHASE_BRIDGE_UNITS = PHASE_BRIDGE_IMPORT / "unit_phase_metrics.csv"
PHASE_BRIDGE_CENTERS = PHASE_BRIDGE_IMPORT / "center_summary.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "imports" / "mousev2_grating_corrected_welch_bridge_v1"
)
BASE_SEED = 20260809
VIEWS = (
    "raw",
    "source_corrected",
    "phase_permutation",
    "opposite_sign",
)
SPECTRAL_FIELDS = (
    "mod_idx",
    "target_psd",
    "off_target_mean_psd",
    "psd_mean",
    "psd_sd",
    "target_to_offtarget_psd",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MOUSE_CONFIG)
    parser.add_argument("--stimulus-manifest", type=Path, default=STIMULUS_MANIFEST)
    parser.add_argument("--nwb-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase-permutations", type=int, default=100)
    parser.add_argument("--permutation-seed", type=int, default=BASE_SEED)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    return parser.parse_args()


def target_component(
    coefficient: np.ndarray | complex,
    temporal_frequency_hz: float,
    *,
    samples: int = 1000,
    sample_rate_hz: float = 1000.0,
) -> np.ndarray:
    """Reconstruct only the real target-frequency DFT component."""
    coefficients = np.asarray(coefficient, dtype=complex)
    time_s = np.arange(samples, dtype=float) / float(sample_rate_hz)
    carrier = np.exp(2j * np.pi * float(temporal_frequency_hz) * time_s)
    return np.real(coefficients[..., None] * carrier) / samples


def target_coefficients(
    trial_counts: np.ndarray,
    temporal_frequency_hz: float,
    *,
    sample_rate_hz: float = 1000.0,
) -> np.ndarray:
    counts = np.asarray(trial_counts, dtype=float)
    time_s = np.arange(counts.shape[1], dtype=float) / float(sample_rate_hz)
    kernel = np.exp(-2j * np.pi * float(temporal_frequency_hz) * time_s)
    return 2.0 * (counts @ kernel)


def welch_spectral_metrics(
    responses: np.ndarray,
    temporal_frequency_hz: float,
    *,
    sample_rate_hz: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Vectorized released modulation index plus its spectral ingredients."""
    values = np.asarray(responses, dtype=float)
    was_one_dimensional = values.ndim == 1
    if was_one_dimensional:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Expected response vectors shaped observations x time")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        frequencies, psd = signal.welch(
            values,
            fs=float(sample_rate_hz),
            nperseg=1024,
            axis=-1,
        )
    target_index = int(np.searchsorted(frequencies, float(temporal_frequency_hz)))
    if not 0 <= target_index < psd.shape[1]:
        raise ValueError("Temporal frequency lies outside the Welch grid")
    mean_psd = np.mean(psd, axis=1)
    psd_sd = np.sqrt(np.maximum(np.mean(psd**2, axis=1) - mean_psd**2, 0.0))
    target_psd = psd[:, target_index]
    modulation = np.divide(
        np.abs(target_psd - mean_psd),
        psd_sd,
        out=np.full(len(values), np.nan),
        where=psd_sd > 0,
    )
    # Match the released scalar function's explicit silent-response branch.
    modulation[mean_psd == 0.0] = 0.0
    exclude = np.ones(psd.shape[1], dtype=bool)
    exclude[0] = False
    exclude[target_index] = False
    off_target = np.mean(psd[:, exclude], axis=1)
    target_to_offtarget = np.divide(
        target_psd,
        off_target,
        out=np.full(len(values), np.nan),
        where=off_target > 0,
    )
    result = {
        "mod_idx": modulation,
        "target_psd": target_psd,
        "off_target_mean_psd": off_target,
        "psd_mean": mean_psd,
        "psd_sd": psd_sd,
        "target_to_offtarget_psd": target_to_offtarget,
    }
    if was_one_dimensional:
        return {key: value[0] for key, value in result.items()}
    return result


def corrected_welch_metrics(
    trial_counts: np.ndarray,
    temporal_frequency_hz: float,
    start_phase_cycles: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    """Rotate only the carrier component, preserving every other PSTH component."""
    counts = np.asarray(trial_counts, dtype=float)
    phases = np.asarray(start_phase_cycles, dtype=float)
    if counts.ndim != 2 or counts.shape != (len(phases), 1000):
        raise ValueError("Expected one start phase per 1,000-bin trial")
    if permutations < 1:
        raise ValueError("At least one phase permutation is required")

    coefficients = target_coefficients(counts, temporal_frequency_hz)
    raw_psth = np.mean(counts, axis=0)
    raw_mean_coefficient = np.mean(coefficients)
    noncarrier_psth = raw_psth - target_component(
        raw_mean_coefficient, temporal_frequency_hz
    )
    phase_factors = np.exp(-2j * np.pi * phases)
    source_mean_coefficient = np.mean(coefficients * phase_factors)
    opposite_mean_coefficient = np.mean(coefficients * np.conjugate(phase_factors))
    source_psth = noncarrier_psth + target_component(
        source_mean_coefficient, temporal_frequency_hz
    )
    opposite_psth = noncarrier_psth + target_component(
        opposite_mean_coefficient, temporal_frequency_hz
    )
    raw_reconstructed = noncarrier_psth + target_component(
        raw_mean_coefficient, temporal_frequency_hz
    )

    primary_responses = np.stack([raw_psth, source_psth, opposite_psth])
    primary = welch_spectral_metrics(primary_responses, temporal_frequency_hz)
    unique_phases = len(np.unique(np.round(phases, decimals=9)))
    if unique_phases == 1:
        permutation_coefficients = np.full(
            permutations, source_mean_coefficient, dtype=complex
        )
    else:
        rng = np.random.default_rng(seed)
        orders = np.stack([rng.permutation(len(phases)) for _ in range(permutations)])
        permutation_coefficients = np.mean(
            coefficients[None, :] * phase_factors[orders], axis=1
        )
    permutation_responses = noncarrier_psth[None, :] + target_component(
        permutation_coefficients, temporal_frequency_hz
    )
    permutation = welch_spectral_metrics(
        permutation_responses, temporal_frequency_hz
    )

    result: dict[str, float | int] = {
        "source_start_phase_count": unique_phases,
        "raw_reconstruction_max_abs_error": float(
            np.max(np.abs(raw_reconstructed - raw_psth))
        ),
        "source_mean_rate_change": float(np.mean(source_psth) - np.mean(raw_psth)),
        "raw_coherent_f1_hz": float(np.abs(raw_mean_coefficient)),
        "source_corrected_coherent_f1_hz": float(
            np.abs(source_mean_coefficient)
        ),
        "opposite_sign_coherent_f1_hz": float(
            np.abs(opposite_mean_coefficient)
        ),
    }
    for field in SPECTRAL_FIELDS:
        result[f"raw_{field}"] = float(primary[field][0])
        result[f"source_corrected_{field}"] = float(primary[field][1])
        result[f"opposite_sign_{field}"] = float(primary[field][2])
        result[f"phase_permutation_{field}_mean"] = float(
            np.nanmean(permutation[field])
        )
        result[f"phase_permutation_{field}_025"] = float(
            np.nanquantile(permutation[field], 0.025)
        )
        result[f"phase_permutation_{field}_975"] = float(
            np.nanquantile(permutation[field], 0.975)
        )
    for view in ("raw", "source_corrected", "opposite_sign"):
        for field in (
            "mod_idx",
            "target_psd",
            "off_target_mean_psd",
            "psd_sd",
            "target_to_offtarget_psd",
        ):
            value = float(result[f"{view}_{field}"])
            result[f"log10_{view}_{field}"] = (
                float(np.log10(value)) if value > 0 else np.nan
            )
    for field in (
        "mod_idx",
        "target_psd",
        "off_target_mean_psd",
        "psd_sd",
        "target_to_offtarget_psd",
    ):
        values = np.asarray(permutation[field], dtype=float)
        logged = np.log10(values[values > 0])
        result[f"log10_phase_permutation_{field}_mean"] = (
            float(np.mean(logged)) if len(logged) else np.nan
        )
    result["source_log10_mod_idx_gain"] = float(
        result["log10_source_corrected_mod_idx"] - result["log10_raw_mod_idx"]
    )
    result["source_log10_target_psd_gain"] = float(
        result["log10_source_corrected_target_psd"]
        - result["log10_raw_target_psd"]
    )
    result["source_log10_psd_sd_gain"] = float(
        result["log10_source_corrected_psd_sd"]
        - result["log10_raw_psd_sd"]
    )
    return result


def summarize_sessions(units: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_common_qc_units": np.ones(len(units), dtype=bool),
        "source_phase_varies_1_2_15_hz": units["preferred_tf_hz"]
        .isin([1.0, 2.0, 15.0])
        .to_numpy(),
        "source_phase_stable_4_8_hz": units["preferred_tf_hz"]
        .isin([4.0, 8.0])
        .to_numpy(),
    }
    metrics = [
        "log10_raw_mod_idx",
        "log10_source_corrected_mod_idx",
        "log10_phase_permutation_mod_idx_mean",
        "log10_opposite_sign_mod_idx",
        "source_log10_mod_idx_gain",
        "source_log10_target_psd_gain",
        "source_log10_psd_sd_gain",
        "log10_raw_target_to_offtarget_psd",
        "log10_source_corrected_target_to_offtarget_psd",
    ]
    rows = []
    for scope, mask in scopes.items():
        selected = units.loc[mask]
        for (site, session_id), group in selected.groupby(
            ["site", "session_id"], sort=True
        ):
            row: dict[str, object] = {
                "scope": scope,
                "site": site,
                "session_id": int(session_id),
                "n_units": len(group),
            }
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = values.mean()
                row[f"valid_{metric}"] = values.notna().sum()
            gains = pd.to_numeric(
                group["source_log10_mod_idx_gain"], errors="coerce"
            )
            row["units_source_mod_idx_increases"] = int(
                (gains > 1e-12).sum()
            )
            row["units_source_mod_idx_decreases"] = int(
                (gains < -1e-12).sum()
            )
            row["units_source_mod_idx_unchanged"] = int(
                (gains.abs() <= 1e-12).sum()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_session_tests(session_summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        (
            "source_corrected_minus_raw",
            "mean_log10_source_corrected_mod_idx",
            "mean_log10_raw_mod_idx",
        ),
        (
            "source_corrected_minus_phase_permutation",
            "mean_log10_source_corrected_mod_idx",
            "mean_log10_phase_permutation_mod_idx_mean",
        ),
        (
            "source_corrected_minus_opposite_sign",
            "mean_log10_source_corrected_mod_idx",
            "mean_log10_opposite_sign_mod_idx",
        ),
    )
    rows = []
    for scope, group in session_summary.groupby("scope", sort=False):
        for comparison, left, right in comparisons:
            difference = (
                pd.to_numeric(group[left], errors="coerce")
                - pd.to_numeric(group[right], errors="coerce")
            ).dropna()
            effective = difference.loc[difference.abs() > 1e-12]
            positives = int((effective > 0).sum())
            negatives = int((effective < 0).sum())
            if len(effective):
                sign_p = float(
                    stats.binomtest(positives, len(effective), 0.5).pvalue
                )
                wilcoxon_p = float(
                    stats.wilcoxon(effective, alternative="two-sided").pvalue
                )
            else:
                sign_p = np.nan
                wilcoxon_p = np.nan
            rows.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "sessions": len(difference),
                    "mean_paired_difference_log10": difference.mean(),
                    "median_paired_difference_log10": difference.median(),
                    "positive_sessions": positives,
                    "negative_sessions": negatives,
                    "unchanged_sessions": int(len(difference) - len(effective)),
                    "exact_sign_test_two_sided_p": sign_p,
                    "wilcoxon_signed_rank_two_sided_p": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


def summarize_temporal_frequency(
    units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "log10_raw_mod_idx",
        "log10_source_corrected_mod_idx",
        "log10_phase_permutation_mod_idx_mean",
        "log10_opposite_sign_mod_idx",
        "source_log10_mod_idx_gain",
        "source_log10_target_psd_gain",
        "source_log10_psd_sd_gain",
    ]
    rows = []
    for (site, session_id, temporal_frequency_hz), group in units.groupby(
        ["site", "session_id", "preferred_tf_hz"], sort=True
    ):
        row: dict[str, object] = {
            "site": site,
            "session_id": int(session_id),
            "preferred_tf_hz": float(temporal_frequency_hz),
            "n_units": len(group),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = pd.to_numeric(
                group[metric], errors="coerce"
            ).mean()
        rows.append(row)
    sessions = pd.DataFrame(rows)
    center_rows = []
    for temporal_frequency_hz, group in sessions.groupby(
        "preferred_tf_hz", sort=True
    ):
        row = {
            "preferred_tf_hz": float(temporal_frequency_hz),
            "sessions": group["session_id"].nunique(),
            "units": int(group["n_units"].sum()),
        }
        for metric in metrics:
            row[f"equal_session_{metric}"] = group[f"mean_{metric}"].mean()
        center_rows.append(row)
    return sessions, pd.DataFrame(center_rows)


def analysis_centers(session_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in session_summary.groupby("scope", sort=False):
        for view in VIEWS:
            source = (
                f"mean_log10_{view}_mod_idx"
                if view != "phase_permutation"
                else "mean_log10_phase_permutation_mod_idx_mean"
            )
            rows.append(
                {
                    "cohort": "MouseV2 V1",
                    "view": view,
                    "scope": scope,
                    "sessions": group["session_id"].nunique(),
                    "equal_session_log10_mod_idx": group[source].mean(),
                }
            )
    allen = pd.read_csv(PHASE_BRIDGE_CENTERS)
    for cohort in (
        "Allen Brain Observatory 1.1",
        "Allen Functional Connectivity",
    ):
        value = float(
            allen.loc[
                allen["cohort"].eq(cohort)
                & allen["metric"].eq("log10_mod_idx"),
                "primary_equal_session_mean",
            ].iloc[0]
        )
        rows.append(
            {
                "cohort": cohort,
                "view": "allen_representative_common_1s_15trials",
                "scope": "all_common_qc_units",
                "sessions": 1,
                "equal_session_log10_mod_idx": value,
            }
        )
    return pd.DataFrame(rows)


def center_value(
    centers: pd.DataFrame,
    cohort: str,
    view: str,
    scope: str = "all_common_qc_units",
) -> float:
    return float(
        centers.loc[
            centers["cohort"].eq(cohort)
            & centers["view"].eq(view)
            & centers["scope"].eq(scope),
            "equal_session_log10_mod_idx",
        ].iloc[0]
    )


def make_figure(
    session_summary: pd.DataFrame,
    tf_centers: pd.DataFrame,
    centers: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    current = session_summary.loc[
        session_summary["scope"].eq("all_common_qc_units")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    ax = axes[0, 0]
    for row in current.itertuples():
        ax.plot(
            [0, 1],
            [row.mean_log10_raw_mod_idx, row.mean_log10_source_corrected_mod_idx],
            color="#888888",
            alpha=0.8,
        )
    ax.scatter(
        np.zeros(len(current)), current["mean_log10_raw_mod_idx"], color="#D95F02"
    )
    ax.scatter(
        np.ones(len(current)),
        current["mean_log10_source_corrected_mod_idx"],
        color="#1B9E77",
    )
    ax.set(
        xticks=[0, 1],
        xticklabels=["raw", "source-phase\ncarrier corrected"],
        ylabel="session mean log10 modulation index",
        title="A. Released Welch estimator after carrier correction",
    )

    ax = axes[0, 1]
    x = np.arange(len(tf_centers))
    for column, label, color, style in (
        ("equal_session_log10_raw_mod_idx", "raw", "#D95F02", "o-"),
        (
            "equal_session_log10_source_corrected_mod_idx",
            "source corrected",
            "#1B9E77",
            "o-",
        ),
        (
            "equal_session_log10_phase_permutation_mod_idx_mean",
            "phase permutation",
            "#777777",
            "o--",
        ),
        (
            "equal_session_log10_opposite_sign_mod_idx",
            "opposite sign",
            "#7570B3",
            "o:",
        ),
    ):
        ax.plot(x, tf_centers[column], style, color=color, label=label)
    ax.set(
        xticks=x,
        xticklabels=[f"{value:g}" for value in tf_centers["preferred_tf_hz"]],
        xlabel="preferred temporal frequency (Hz)",
        ylabel="equal-session log10 modulation index",
        title="B. Temporal-frequency prediction and controls",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    gains = [
        current["mean_source_log10_target_psd_gain"].mean(),
        current["mean_source_log10_psd_sd_gain"].mean(),
        current["mean_source_log10_mod_idx_gain"].mean(),
    ]
    ax.bar(
        np.arange(3),
        gains,
        color=["#E69F00", "#56B4E9", "#1B9E77"],
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xticks=np.arange(3),
        xticklabels=["target PSD", "PSD SD", "modulation index"],
        ylabel="source corrected − raw (log10)",
        title="C. Numerator and denominator decomposition",
    )

    ax = axes[1, 1]
    labels = [
        "Multi-site V1\nraw",
        "Multi-site V1\ncorrected",
        "Allen\nBO",
        "Allen\nFC",
    ]
    values = [
        center_value(centers, "MouseV2 V1", "raw"),
        center_value(centers, "MouseV2 V1", "source_corrected"),
        center_value(
            centers,
            "Allen Brain Observatory 1.1",
            "allen_representative_common_1s_15trials",
        ),
        center_value(
            centers,
            "Allen Functional Connectivity",
            "allen_representative_common_1s_15trials",
        ),
    ]
    ax.bar(
        np.arange(4),
        values,
        color=["#D95F02", "#1B9E77", "#6F63A6", "#B07AA1"],
    )
    ax.set(
        xticks=np.arange(4),
        xticklabels=labels,
        ylabel="equal-session log10 modulation index",
        title="D. Residual gap to representative Allen sessions",
    )
    for current_ax in axes.flat:
        current_ax.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Multi-site V1 source-phase correction through Welch modulation index"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    session_summary: pd.DataFrame,
    tf_centers: pd.DataFrame,
    centers: pd.DataFrame,
    paired_tests: pd.DataFrame,
) -> None:
    raw = center_value(centers, "MouseV2 V1", "raw")
    corrected = center_value(centers, "MouseV2 V1", "source_corrected")
    permutation = center_value(centers, "MouseV2 V1", "phase_permutation")
    opposite = center_value(centers, "MouseV2 V1", "opposite_sign")
    affected_raw = center_value(
        centers, "MouseV2 V1", "raw", "source_phase_varies_1_2_15_hz"
    )
    affected_corrected = center_value(
        centers,
        "MouseV2 V1",
        "source_corrected",
        "source_phase_varies_1_2_15_hz",
    )
    stable_raw = center_value(
        centers, "MouseV2 V1", "raw", "source_phase_stable_4_8_hz"
    )
    stable_corrected = center_value(
        centers,
        "MouseV2 V1",
        "source_corrected",
        "source_phase_stable_4_8_hz",
    )
    bo = center_value(
        centers,
        "Allen Brain Observatory 1.1",
        "allen_representative_common_1s_15trials",
    )
    fc = center_value(
        centers,
        "Allen Functional Connectivity",
        "allen_representative_common_1s_15trials",
    )
    current = session_summary.loc[
        session_summary["scope"].eq("all_common_qc_units")
    ]
    session_gain = (
        current["mean_log10_source_corrected_mod_idx"]
        - current["mean_log10_raw_mod_idx"]
    )
    target_gain = current["mean_source_log10_target_psd_gain"].mean()
    denominator_gain = current["mean_source_log10_psd_sd_gain"].mean()
    affected = session_summary.loc[
        session_summary["scope"].eq("source_phase_varies_1_2_15_hz")
    ]
    stable = session_summary.loc[
        session_summary["scope"].eq("source_phase_stable_4_8_hz")
    ]
    units_increase = affected["units_source_mod_idx_increases"].sum()
    units_decrease = affected["units_source_mod_idx_decreases"].sum()
    units_unchanged = affected["units_source_mod_idx_unchanged"].sum()
    units_total = affected["n_units"].sum()
    stable_units_increase = stable["units_source_mod_idx_increases"].sum()
    stable_units_decrease = stable["units_source_mod_idx_decreases"].sum()
    stable_units_unchanged = stable["units_source_mod_idx_unchanged"].sum()
    stable_units_total = stable["n_units"].sum()
    primary_test = paired_tests.loc[
        paired_tests["scope"].eq("all_common_qc_units")
        & paired_tests["comparison"].eq("source_corrected_minus_raw")
    ].iloc[0]
    tf_lines = []
    for row in tf_centers.itertuples():
        tf_lines.append(
            f"- {row.preferred_tf_hz:g} Hz: raw {row.equal_session_log10_raw_mod_idx:+.3f}, "
            f"source corrected {row.equal_session_log10_source_corrected_mod_idx:+.3f}, "
            f"permutation {row.equal_session_log10_phase_permutation_mod_idx_mean:+.3f}, "
            f"opposite sign {row.equal_session_log10_opposite_sign_mod_idx:+.3f}."
        )
    lines = [
        "# MouseV2 source-corrected Welch modulation-index bridge",
        "",
        "For each trial, only the preferred temporal-frequency DFT component is",
        "removed, rotated by the source-defined starting phase, and added back. All",
        "non-carrier temporal structure—including the onset transient—is unchanged.",
        "The resulting condition-averaged PSTH is evaluated with the unchanged released",
        "Welch modulation-index function. Phase-permuted and opposite-sign rotations are",
        "mechanism controls; 4/8-Hz units are source-predicted negative controls.",
        "",
        "## Primary result",
        "",
        f"- Equal-session log10 modulation index changes from {raw:+.3f} to "
        f"{corrected:+.3f} after source-phase carrier correction, versus "
        f"{permutation:+.3f} for phase permutation and {opposite:+.3f} for the opposite sign.",
        f"- The affected 1/2/15-Hz center changes from {affected_raw:+.3f} to "
        f"{affected_corrected:+.3f}; the phase-stable 4/8-Hz center changes from "
        f"{stable_raw:+.3f} to {stable_corrected:+.3f}.",
        f"- All {int(primary_test.positive_sessions)}/{int(primary_test.sessions)} "
        f"sessions increase (exact two-sided sign-test p="
        f"{primary_test.exact_sign_test_two_sided_p:.4f}; paired session gains "
        f"{session_gain.min():+.3f} to {session_gain.max():+.3f} log10).",
        f"- Among the {units_total:,} affected 1/2/15-Hz units, {units_increase:,} "
        f"increase, {units_decrease:,} decrease, and {units_unchanged:,} are unchanged "
        f"within 1e-12 log10. Among {stable_units_total:,} stable-phase 4/8-Hz units, "
        f"{stable_units_increase:,} increase, {stable_units_decrease:,} decrease, and "
        f"{stable_units_unchanged:,} are unchanged.",
        f"- Source correction changes target PSD by {target_gain:+.3f} log10 and the "
        f"spectrum-wide PSD SD denominator by {denominator_gain:+.3f} log10 on average.",
        f"- Representative single-session common-window Allen centers are {bo:+.3f} (BO) and "
        f"{fc:+.3f} (FC), leaving corrected MouseV2 gaps of "
        f"{corrected - bo:+.3f} and {corrected - fc:+.3f} log10.",
        "",
        "## Temporal-frequency controls",
        "",
        *tf_lines,
        "",
        "## Interpretation boundary",
        "",
        "This is a target-component mechanism diagnostic, not a replacement field for",
        "released `mod_idx_dg`. It preserves the non-carrier PSTH exactly and asks how",
        "the released estimator responds when the acquisition-defined carrier phase is",
        "made comparable across trials. Any residual gap still requires multi-session",
        "Allen replication and homologous RF/layer/rate population matching.",
        "Session-level tests are recorded in `paired_session_tests.csv`; the unit counts",
        "describe heterogeneity and are not treated as independent inferential replicates.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_manifest(output_dir: Path) -> None:
    path = output_dir / "import_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    upstream = {
        record["site"]: record
        for record in json.loads(START_PHASE_MANIFEST.read_text(encoding="utf-8"))[
            "inputs"
        ]
    }
    for record in manifest.get("inputs", []):
        source = upstream[record["site"]]
        if int(record["bytes"]) != int(source["bytes"]):
            raise ValueError(f"{record['site']}: input byte-count drift")
        record["sha256"] = source["sha256"]
        record["sha256_source"] = str(START_PHASE_MANIFEST.relative_to(ROOT))
    manifest["outputs"] = []
    for name in (
        "README.md",
        "analysis_centers.csv",
        "corrected_welch_bridge.png",
        "paired_session_tests.csv",
        "session_summary.csv",
        "tf_center_summary.csv",
        "tf_session_summary.csv",
        "unit_corrected_welch_metrics.csv",
    ):
        output = output_dir / name
        if output.is_file():
            manifest["outputs"].append(
                {"path": name, "bytes": output.stat().st_size, "sha256": sha256(output)}
            )
    manifest["code"]["render_script_sha256"] = sha256(Path(__file__).resolve())
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def render_existing(output_dir: Path) -> None:
    units = pd.read_csv(output_dir / "unit_corrected_welch_metrics.csv")
    session_summary = summarize_sessions(units)
    tf_sessions, tf_centers = summarize_temporal_frequency(units)
    centers = analysis_centers(session_summary)
    tests = paired_session_tests(session_summary)
    session_summary.to_csv(output_dir / "session_summary.csv", index=False)
    tf_sessions.to_csv(output_dir / "tf_session_summary.csv", index=False)
    tf_centers.to_csv(output_dir / "tf_center_summary.csv", index=False)
    centers.to_csv(output_dir / "analysis_centers.csv", index=False)
    tests.to_csv(output_dir / "paired_session_tests.csv", index=False)
    make_figure(
        session_summary,
        tf_centers,
        centers,
        output_dir / "corrected_welch_bridge.png",
    )
    write_report(output_dir, session_summary, tf_centers, centers, tests)
    refresh_manifest(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.render_existing:
        render_existing(output_dir)
        print(f"Rendered corrected-Welch bridge: {output_dir}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    stimulus_manifest_path = args.stimulus_manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schedule = phase_schedule(stimulus_manifest_path)
    nwb_root = (
        args.nwb_root.resolve()
        if args.nwb_root is not None
        else Path(config["nwb_input"]["default_root"]).resolve()
    )
    requested = set(args.sites) if args.sites else None
    sessions_config = [
        session
        for session in config["sessions"]
        if requested is None or str(session["site"]) in requested
    ]
    if requested is not None and {str(item["site"]) for item in sessions_config} != requested:
        raise ValueError("One or more requested sites are absent from the config")

    from generate_retinotopic_csvs import choose_stim_table, read_nwb_tables

    start_units = pd.read_csv(START_PHASE_UNITS)
    phase_units = pd.read_csv(PHASE_BRIDGE_UNITS)
    phase_units = phase_units.loc[
        phase_units["cohort"].eq("MouseV2 V1"),
        ["unit_id", "mod_idx_dg"],
    ].drop_duplicates("unit_id")
    upstream_manifest = json.loads(START_PHASE_MANIFEST.read_text(encoding="utf-8"))
    upstream_inputs = {record["site"]: record for record in upstream_manifest["inputs"]}
    rows = []
    input_rows = []

    for session in sessions_config:
        site = str(session["site"])
        session_id = int(session["site_number"])
        offset = int(session["id_offset"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file() or nwb_path.stat().st_size != int(
            session["expected_nwb_bytes"]
        ):
            raise FileNotFoundError(f"Missing or size-mismatched MouseV2 NWB: {nwb_path}")
        print(f"[{site}] loading spikes and grating presentations", flush=True)
        extracted = read_nwb_tables(str(nwb_path))
        _, grating_table = choose_stim_table(
            extracted.intervals_tables, "drifting_gratings_field_block_presentations"
        )
        conditions = phase_aware_conditions(grating_table, schedule)
        expected = start_units.loc[start_units["site"].eq(site)].set_index("unit_id")
        quality = pd.read_csv(ROOT / "data" / f"{site}_processed" / "unit_quality.csv")
        quality = quality.loc[quality["default_qc"].eq(True), "unit_id"].astype(int)
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        if set(quality) != set(expected.index.astype(int)):
            raise ValueError(f"{site}: QC population drift from start-phase bridge")

        for index, unit_id in enumerate(quality, start=1):
            local_id = int(unit_id) - offset
            parameters, counts, phases = preferred_condition_counts(
                extracted.spikes_by_unit[row_by_id[local_id]], conditions
            )
            upstream = expected.loc[int(unit_id)]
            if not (
                np.isclose(parameters[0], upstream["preferred_orientation_deg"])
                and np.isclose(parameters[1], upstream["preferred_tf_hz"])
                and np.isclose(parameters[2], upstream["preferred_sf_cpd"])
            ):
                raise ValueError(f"{site} unit {unit_id}: preferred-condition drift")
            metrics = corrected_welch_metrics(
                counts,
                float(parameters[1]),
                phases,
                permutations=args.phase_permutations,
                seed=int(
                    np.random.SeedSequence(
                        [args.permutation_seed, session_id, int(unit_id)]
                    ).generate_state(1)[0]
                ),
            )
            rows.append(
                {
                    "site": site,
                    "session_id": session_id,
                    "unit_id": int(unit_id),
                    "preferred_orientation_deg": float(parameters[0]),
                    "preferred_tf_hz": float(parameters[1]),
                    "preferred_sf_cpd": float(parameters[2]),
                    "preferred_trials": len(phases),
                    "source_phase_expected_to_vary": float(parameters[1])
                    in {1.0, 2.0, 15.0},
                    **metrics,
                }
            )
            if index % 400 == 0:
                print(f"[{site}] corrected {index}/{len(quality)} units", flush=True)
        source = upstream_inputs[site]
        input_rows.append(
            {
                "site": site,
                "path": str(nwb_path),
                "bytes": nwb_path.stat().st_size,
                "sha256": source["sha256"],
                "sha256_source": str(START_PHASE_MANIFEST.relative_to(ROOT)),
            }
        )

    units = pd.DataFrame(rows).merge(
        phase_units.rename(columns={"mod_idx_dg": "upstream_raw_mod_idx"}),
        on="unit_id",
        how="left",
        validate="one_to_one",
    )
    units["raw_mod_idx_absolute_difference"] = np.abs(
        units["raw_mod_idx"] - units["upstream_raw_mod_idx"]
    )
    raw_values = units["raw_mod_idx"].to_numpy(dtype=float)
    upstream_values = units["upstream_raw_mod_idx"].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(raw_values), np.isnan(upstream_values)):
        raise ValueError("Raw Welch modulation index NaN pattern drift")
    finite_raw = np.isfinite(raw_values) & np.isfinite(upstream_values)
    max_difference = (
        float(np.max(np.abs(raw_values[finite_raw] - upstream_values[finite_raw])))
        if np.any(finite_raw)
        else 0.0
    )
    if max_difference > 1e-10:
        raise ValueError(
            f"Raw Welch modulation index drift from phase bridge: {max_difference}"
        )
    if units["raw_reconstruction_max_abs_error"].max() > 1e-12:
        raise ValueError("Target decomposition does not reconstruct the raw PSTH")
    if units["source_mean_rate_change"].abs().max() > 1e-12:
        raise ValueError("Target-only correction changed mean firing rate")

    session_summary = summarize_sessions(units)
    tf_sessions, tf_centers = summarize_temporal_frequency(units)
    centers = analysis_centers(session_summary)
    tests = paired_session_tests(session_summary)
    units.to_csv(output_dir / "unit_corrected_welch_metrics.csv", index=False)
    session_summary.to_csv(output_dir / "session_summary.csv", index=False)
    tf_sessions.to_csv(output_dir / "tf_session_summary.csv", index=False)
    tf_centers.to_csv(output_dir / "tf_center_summary.csv", index=False)
    centers.to_csv(output_dir / "analysis_centers.csv", index=False)
    tests.to_csv(output_dir / "paired_session_tests.csv", index=False)
    write_report(output_dir, session_summary, tf_centers, centers, tests)
    if not args.skip_figure:
        make_figure(
            session_summary,
            tf_centers,
            centers,
            output_dir / "corrected_welch_bridge.png",
        )

    script_hash = sha256(Path(__file__).resolve())
    manifest = {
        "schema_version": 1,
        "phase_permutations": args.phase_permutations,
        "permutation_seed": args.permutation_seed,
        "condition_support": {
            "duration_s": 1.0,
            "trials": 15,
            "spatial_frequency_cpd": 0.04,
            "contrast": 0.8,
        },
        "correction_method": {
            "component": "preferred temporal-frequency DFT component only",
            "source_rotation": "exp(-2j*pi*source_start_phase_cycles)",
            "noncarrier_psth_preserved": True,
            "mean_rate_preserved": True,
            "estimator": "unchanged released Welch modulation index",
            "controls": ["phase permutation", "opposite sign", "4/8-Hz stable phase"],
        },
        "inputs": input_rows,
        "upstream_source_provenance": upstream_manifest["source_provenance"],
        "code": {
            "script_sha256": script_hash,
            "analysis_script_sha256": script_hash,
            "render_script_sha256": script_hash,
            "start_phase_script_sha256": sha256(
                ROOT / "scripts" / "mousev2_grating_start_phase_bridge.py"
            ),
            "welch_metric_sha256": sha256(ROOT / "common" / "drifting_gratings.py"),
            "config_sha256": sha256(config_path),
            "stimulus_manifest_sha256": sha256(stimulus_manifest_path),
            "phase_bridge_manifest_sha256": sha256(PHASE_BRIDGE_MANIFEST),
        },
        "validation": {
            "max_raw_mod_idx_absolute_difference": float(max_difference),
            "max_raw_reconstruction_absolute_error": float(
                units["raw_reconstruction_max_abs_error"].max()
            ),
            "max_absolute_mean_rate_change": float(
                units["source_mean_rate_change"].abs().max()
            ),
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    refresh_manifest(output_dir)
    print(f"Corrected-Welch bridge written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
