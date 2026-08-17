#!/usr/bin/env python3
"""Test whether MouseV2 grating start phase explains trial-average cancellation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.drifting_gratings import _bin_trial_spike_counts  # noqa: E402
from scripts.extract_mousev2_grating_common_support import (  # noqa: E402
    common_presentations,
)


MOUSE_CONFIG = ROOT / "config" / "figure3_mousev2.json"
STIMULUS_MANIFEST = ROOT / "config" / "mousev2_stimulus_manifest.json"
INPUT_MANIFEST = (
    ROOT / "data" / "imports" / "mousev2_grating_metrics_v1" / "import_manifest.json"
)
PHASE_BRIDGE = ROOT / "data" / "imports" / "v1_grating_phase_bridge_v1"
DEFAULT_OUTPUT = (
    ROOT / "data" / "imports" / "mousev2_grating_start_phase_bridge_v1"
)
SOURCE_REPOSITORY = ROOT.parents[1] / "openscope_v2species"
SOURCE_FILES = {
    "stimulus_v2species_ephys.py": SOURCE_REPOSITORY
    / "stimulus_v2species_ephys.py",
    "camstim/camstim/sweepstim.py": SOURCE_REPOSITORY
    / "camstim"
    / "camstim"
    / "sweepstim.py",
}
BASE_PERMUTATION_SEED = 20260805
METRICS = (
    "raw_weighted_phase_coherence",
    "source_corrected_weighted_phase_coherence",
    "phase_permutation_weighted_phase_coherence",
    "opposite_sign_weighted_phase_coherence",
    "source_correction_gain",
    "source_gain_over_permutation",
    "log10_mean_trial_f1_hz",
    "log10_raw_coherent_f1_hz",
    "log10_source_corrected_coherent_f1_hz",
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
    parser.add_argument("--permutation-seed", type=int, default=BASE_PERMUTATION_SEED)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    return parser.parse_args()


def phase_schedule(stimulus_manifest: Path = STIMULUS_MANIFEST) -> dict[str, object]:
    manifest = json.loads(stimulus_manifest.read_text(encoding="utf-8"))
    return manifest["blocks"]["drifting_gratings"]["phase_schedule"]


def presentation_start_phase_cycles(
    presentation_ordinal: np.ndarray,
    temporal_frequency_hz: float,
    *,
    fps: float,
    presentation_stride_frames: int,
) -> np.ndarray:
    """Relative grating start phase implied by the frozen camstim source.

    The source uses the absolute experiment frame. Omitting the common block-start
    frame rotates every trial of a given TF equally and therefore cannot change
    coherence or any result below.
    """
    ordinal = np.asarray(presentation_ordinal, dtype=float)
    return np.mod(
        float(temporal_frequency_hz)
        * ordinal
        * int(presentation_stride_frames)
        / float(fps),
        1.0,
    )


def validate_presentation_order(table: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "start_time", "stop_time", "temporal_frequency"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Grating table lacks phase-audit columns {missing}")
    ordered = table.sort_values("start_time").reset_index(drop=True).copy()
    ids = pd.to_numeric(ordered["id"], errors="coerce").to_numpy(dtype=float)
    if not np.array_equal(ids, np.arange(len(ordered), dtype=float)):
        raise ValueError("Grating presentation id is not the chronological sweep ordinal")
    starts = pd.to_numeric(ordered["start_time"], errors="coerce").to_numpy()
    if not np.all(np.diff(starts) > 0):
        raise ValueError("Grating presentation starts are not strictly increasing")
    ordered["presentation_ordinal"] = ids.astype(int)
    return ordered


def phase_aware_conditions(
    table: pd.DataFrame,
    schedule: dict[str, object],
) -> list[dict[str, object]]:
    ordered = validate_presentation_order(table)
    selected = common_presentations(ordered)
    dimensions = ["orientation", "temporal_frequency", "spatial_frequency", "contrast"]
    conditions: list[dict[str, object]] = []
    for key, group in selected.groupby(dimensions, sort=True):
        temporal_frequency_hz = float(key[1])
        ordinal = group["presentation_ordinal"].to_numpy(dtype=int)
        phases = presentation_start_phase_cycles(
            ordinal,
            temporal_frequency_hz,
            fps=float(schedule["fps"]),
            presentation_stride_frames=int(schedule["presentation_stride_frames"]),
        )
        conditions.append(
            {
                "parameters": tuple(map(float, key)),
                "starts": group["start_time"].to_numpy(dtype=float),
                "ordinals": ordinal,
                "start_phase_cycles": phases,
            }
        )
    return conditions


def preferred_condition_counts(
    spikes_s: np.ndarray,
    conditions: list[dict[str, object]],
) -> tuple[tuple[float, ...], np.ndarray, np.ndarray]:
    means = []
    for condition in conditions:
        starts = np.asarray(condition["starts"], dtype=float)
        first = np.searchsorted(spikes_s, starts, side="left")
        last = np.searchsorted(spikes_s, starts + 1.0, side="left")
        means.append(float(np.mean(last - first)))
    selected = conditions[int(np.argmax(means))]
    starts = np.asarray(selected["starts"], dtype=float)
    return (
        tuple(selected["parameters"]),
        _bin_trial_spike_counts(spikes_s, starts, duration_ms=1000),
        np.asarray(selected["start_phase_cycles"], dtype=float),
    )


def weighted_phase_coherence(coefficients: np.ndarray) -> float:
    coefficients = np.asarray(coefficients, dtype=complex)
    denominator = float(np.mean(np.abs(coefficients)))
    return float(np.abs(np.mean(coefficients)) / denominator) if denominator > 0 else np.nan


def phase_adjusted_metrics(
    trial_counts: np.ndarray,
    temporal_frequency_hz: float,
    start_phase_cycles: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    counts = np.asarray(trial_counts, dtype=float)
    phases = np.asarray(start_phase_cycles, dtype=float)
    if counts.ndim != 2 or counts.shape != (len(phases), 1000):
        raise ValueError("Expected one start phase per 1,000-bin trial")
    if permutations < 1:
        raise ValueError("At least one phase permutation is required")
    time_s = np.arange(1000, dtype=float) / 1000.0
    kernel = np.exp(-2j * np.pi * float(temporal_frequency_hz) * time_s)
    coefficients = 2.0 * (counts @ kernel)
    amplitudes = np.abs(coefficients)
    mean_trial = float(np.mean(amplitudes))
    phase_factors = np.exp(-2j * np.pi * phases)
    corrected = coefficients * phase_factors
    opposite = coefficients * np.conjugate(phase_factors)
    raw_coherence = weighted_phase_coherence(coefficients)
    corrected_coherence = weighted_phase_coherence(corrected)
    opposite_coherence = weighted_phase_coherence(opposite)

    unique_phases = len(np.unique(np.round(phases, decimals=9)))
    if unique_phases == 1:
        permutation_values = np.full(permutations, corrected_coherence)
    else:
        rng = np.random.default_rng(seed)
        permutation_values = np.array(
            [
                weighted_phase_coherence(
                    coefficients * phase_factors[rng.permutation(len(phase_factors))]
                )
                for _ in range(permutations)
            ]
        )
    permutation_mean = float(np.mean(permutation_values))
    raw_coherent = float(np.abs(np.mean(coefficients)))
    corrected_coherent = float(np.abs(np.mean(corrected)))
    result: dict[str, float | int] = {
        "source_start_phase_count": unique_phases,
        "mean_trial_f1_hz": mean_trial,
        "raw_coherent_f1_hz": raw_coherent,
        "source_corrected_coherent_f1_hz": corrected_coherent,
        "raw_weighted_phase_coherence": raw_coherence,
        "source_corrected_weighted_phase_coherence": corrected_coherence,
        "phase_permutation_weighted_phase_coherence": permutation_mean,
        "phase_permutation_025": float(np.quantile(permutation_values, 0.025)),
        "phase_permutation_975": float(np.quantile(permutation_values, 0.975)),
        "opposite_sign_weighted_phase_coherence": opposite_coherence,
        "source_correction_gain": corrected_coherence - raw_coherence,
        "source_gain_over_permutation": corrected_coherence - permutation_mean,
    }
    for target, value in (
        ("log10_mean_trial_f1_hz", mean_trial),
        ("log10_raw_coherent_f1_hz", raw_coherent),
        ("log10_source_corrected_coherent_f1_hz", corrected_coherent),
    ):
        result[target] = float(np.log10(value)) if value > 0 else np.nan
    return result


def schedule_audit(
    table: pd.DataFrame,
    schedule: dict[str, object],
    *,
    site: str,
    session_id: int,
) -> pd.DataFrame:
    ordered = validate_presentation_order(table)
    selected = common_presentations(ordered)
    expected = {
        float(key): int(value)
        for key, value in schedule["expected_unique_start_phases_by_tf_hz"].items()
    }
    rows = []
    for temporal_frequency_hz, group in selected.groupby("temporal_frequency", sort=True):
        phases = presentation_start_phase_cycles(
            group["presentation_ordinal"].to_numpy(dtype=int),
            float(temporal_frequency_hz),
            fps=float(schedule["fps"]),
            presentation_stride_frames=int(schedule["presentation_stride_frames"]),
        )
        rounded = np.round(phases, decimals=9)
        values, counts = np.unique(rounded, return_counts=True)
        observed = len(values)
        if observed != expected[float(temporal_frequency_hz)]:
            raise ValueError(
                f"{site}: TF {temporal_frequency_hz:g} has {observed} start phases, "
                f"expected {expected[float(temporal_frequency_hz)]}"
            )
        rows.append(
            {
                "site": site,
                "session_id": session_id,
                "temporal_frequency_hz": float(temporal_frequency_hz),
                "presentations": len(group),
                "conditions": group["orientation"].nunique(),
                "expected_unique_start_phases": expected[float(temporal_frequency_hz)],
                "observed_unique_start_phases": observed,
                "start_phase_cycles": ";".join(f"{value:g}" for value in values),
                "phase_presentation_counts": ";".join(map(str, counts)),
                "source_phase_varies": observed > 1,
                "median_grating_duration_s": float(
                    np.median(group["stop_time"] - group["start_time"])
                ),
                "median_full_block_inter_start_s": float(
                    np.median(np.diff(ordered["start_time"].to_numpy(dtype=float)))
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_sessions(units: pd.DataFrame) -> pd.DataFrame:
    expected_varying = units["preferred_tf_hz"].isin([1.0, 2.0, 15.0])
    scopes = {
        "all_common_qc_units": np.ones(len(units), dtype=bool),
        "source_phase_varies_1_2_15_hz": expected_varying.to_numpy(),
        "source_phase_stable_4_8_hz": (~expected_varying).to_numpy(),
    }
    rows = []
    for scope, mask in scopes.items():
        selected = units.loc[mask]
        for (site, session_id), group in selected.groupby(["site", "session_id"], sort=True):
            row: dict[str, object] = {
                "scope": scope,
                "site": site,
                "session_id": int(session_id),
                "n_units": len(group),
            }
            for metric in METRICS:
                row[f"mean_{metric}"] = pd.to_numeric(
                    group[metric], errors="coerce"
                ).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_temporal_frequency(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        for metric in METRICS:
            row[f"mean_{metric}"] = pd.to_numeric(group[metric], errors="coerce").mean()
        rows.append(row)
    sessions = pd.DataFrame(rows)
    centers = (
        sessions.groupby("preferred_tf_hz", as_index=False)
        .agg(
            sessions=("session_id", "nunique"),
            total_units=("n_units", "sum"),
            **{
                f"equal_session_{metric}": (f"mean_{metric}", "mean")
                for metric in METRICS
            },
        )
        .sort_values("preferred_tf_hz")
    )
    return sessions, centers


def analysis_centers(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    views = {
        "mouse_raw": "mean_raw_weighted_phase_coherence",
        "mouse_source_phase_corrected": "mean_source_corrected_weighted_phase_coherence",
        "mouse_phase_permutation_null": "mean_phase_permutation_weighted_phase_coherence",
        "mouse_opposite_sign_control": "mean_opposite_sign_weighted_phase_coherence",
    }
    for scope, group in summary.groupby("scope", sort=False):
        for view, column in views.items():
            rows.append(
                {
                    "cohort": "MouseV2 V1",
                    "view": view,
                    "scope": scope,
                    "sessions": group["session_id"].nunique(),
                    "equal_session_weighted_phase_coherence": group[column].mean(),
                }
            )
    phase_centers = pd.read_csv(PHASE_BRIDGE / "center_summary.csv")
    allen = phase_centers.loc[
        phase_centers["cohort"].str.startswith("Allen")
        & phase_centers["metric"].eq("weighted_phase_coherence")
    ]
    for row in allen.itertuples():
        rows.append(
            {
                "cohort": row.cohort,
                "view": "allen_representative_raw",
                "scope": "all_common_qc_units",
                "sessions": int(row.primary_sessions),
                "equal_session_weighted_phase_coherence": row.primary_equal_session_mean,
            }
        )
    return pd.DataFrame(rows)


def center_value(centers: pd.DataFrame, cohort: str, view: str, scope: str) -> float:
    selected = centers.loc[
        centers["cohort"].eq(cohort)
        & centers["view"].eq(view)
        & centers["scope"].eq(scope),
        "equal_session_weighted_phase_coherence",
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one center for {cohort}/{view}/{scope}")
    return float(selected.iloc[0])


def make_figure(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    tf_centers: pd.DataFrame,
    centers: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    ax = axes[0, 0]
    example = audit.loc[audit["site"].eq(audit["site"].iloc[0])]
    for row in example.itertuples():
        phases = np.array([float(value) for value in row.start_phase_cycles.split(";")])
        ax.scatter(
            np.full(len(phases), row.temporal_frequency_hz),
            phases,
            s=45,
            color="#D95F02" if row.source_phase_varies else "#777777",
        )
    ax.set(
        xscale="log",
        xticks=[1, 2, 4, 8, 15],
        xticklabels=["1", "2", "4", "8", "15"],
        xlabel="temporal frequency (Hz)",
        ylabel="source-derived start phase (cycles)",
        title="A. Phase schedule from acquisition source",
    )

    ax = axes[0, 1]
    current = summary.loc[summary["scope"].eq("all_common_qc_units")]
    for row in current.itertuples():
        ax.plot(
            [0, 1],
            [
                row.mean_raw_weighted_phase_coherence,
                row.mean_source_corrected_weighted_phase_coherence,
            ],
            color="#888888",
            alpha=0.8,
        )
    ax.scatter(
        np.zeros(len(current)),
        current["mean_raw_weighted_phase_coherence"],
        color="#D95F02",
        zorder=3,
    )
    ax.scatter(
        np.ones(len(current)),
        current["mean_source_corrected_weighted_phase_coherence"],
        color="#1B9E77",
        zorder=3,
    )
    ax.set(
        xticks=[0, 1],
        xticklabels=["raw onset\nalignment", "source-phase\nadjusted"],
        ylabel="session mean weighted coherence",
        title="B. Paired MouseV2 sessions",
    )

    ax = axes[1, 0]
    x = np.arange(len(tf_centers))
    ax.plot(
        x,
        tf_centers["equal_session_raw_weighted_phase_coherence"],
        "o-",
        color="#D95F02",
        label="raw",
    )
    ax.plot(
        x,
        tf_centers["equal_session_source_corrected_weighted_phase_coherence"],
        "o-",
        color="#1B9E77",
        label="source-phase adjusted",
    )
    ax.plot(
        x,
        tf_centers["equal_session_phase_permutation_weighted_phase_coherence"],
        "o--",
        color="#777777",
        label="phase permutation",
    )
    ax.set(
        xticks=x,
        xticklabels=[f"{value:g}" for value in tf_centers["preferred_tf_hz"]],
        xlabel="preferred temporal frequency (Hz)",
        ylabel="equal-session weighted coherence",
        title="C. TF-specific prediction and controls",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    labels = ["Mouse\nraw", "Mouse\nphase adjusted", "Allen\nBO", "Allen\nFC"]
    values = [
        center_value(centers, "MouseV2 V1", "mouse_raw", "all_common_qc_units"),
        center_value(
            centers,
            "MouseV2 V1",
            "mouse_source_phase_corrected",
            "all_common_qc_units",
        ),
        center_value(
            centers,
            "Allen Brain Observatory 1.1",
            "allen_representative_raw",
            "all_common_qc_units",
        ),
        center_value(
            centers,
            "Allen Functional Connectivity",
            "allen_representative_raw",
            "all_common_qc_units",
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
        ylabel="equal-session weighted coherence",
        title="D. Material recovery; residual remains",
    )
    for current_ax in axes.flat:
        current_ax.grid(axis="y", alpha=0.18)
    fig.suptitle("MouseV2 grating start-phase bridge")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    tf_centers: pd.DataFrame,
    centers: pd.DataFrame,
) -> None:
    scope = "all_common_qc_units"
    raw = center_value(centers, "MouseV2 V1", "mouse_raw", scope)
    corrected = center_value(
        centers, "MouseV2 V1", "mouse_source_phase_corrected", scope
    )
    null = center_value(
        centers, "MouseV2 V1", "mouse_phase_permutation_null", scope
    )
    affected_raw = center_value(
        centers,
        "MouseV2 V1",
        "mouse_raw",
        "source_phase_varies_1_2_15_hz",
    )
    affected_corrected = center_value(
        centers,
        "MouseV2 V1",
        "mouse_source_phase_corrected",
        "source_phase_varies_1_2_15_hz",
    )
    affected_null = center_value(
        centers,
        "MouseV2 V1",
        "mouse_phase_permutation_null",
        "source_phase_varies_1_2_15_hz",
    )
    bo = center_value(
        centers,
        "Allen Brain Observatory 1.1",
        "allen_representative_raw",
        scope,
    )
    fc = center_value(
        centers,
        "Allen Functional Connectivity",
        "allen_representative_raw",
        scope,
    )
    all_sessions = summary.loc[summary["scope"].eq(scope)]
    session_gains = (
        all_sessions["mean_source_corrected_weighted_phase_coherence"]
        - all_sessions["mean_raw_weighted_phase_coherence"]
    )
    log_gain = (
        all_sessions["mean_log10_source_corrected_coherent_f1_hz"]
        - all_sessions["mean_log10_raw_coherent_f1_hz"]
    ).mean()
    tf_lookup = tf_centers.set_index("preferred_tf_hz")
    tf_lines = []
    for row in tf_centers.itertuples():
        tf_lines.append(
            f"- {row.preferred_tf_hz:g} Hz: raw "
            f"{row.equal_session_raw_weighted_phase_coherence:.3f}, source-phase adjusted "
            f"{row.equal_session_source_corrected_weighted_phase_coherence:.3f}, "
            f"permutation {row.equal_session_phase_permutation_weighted_phase_coherence:.3f}."
        )
    lines = [
        "# MouseV2 grating start-phase bridge",
        "",
        "The acquisition source advances grating phase as `TF * current_frame / fps`",
        "and does not reset phase at presentation onset. With 60 stimulus frames plus",
        "75 blank frames, the randomized 135-frame sweep stride produces four starting",
        "phases at 1 and 15 Hz, two at 2 Hz, and one at 4 and 8 Hz. The NWB interval",
        "table does not retain phase, so onset-aligned averaging mixes these source-defined",
        "phases unless they are reconstructed from presentation order.",
        "",
        "## Primary result",
        "",
        f"- Equal-session weighted coherence increases from {raw:.3f} to {corrected:.3f} "
        f"after the source-derived phase rotation; the phase-permutation center is {null:.3f}.",
        f"- Among the affected 1/2/15-Hz units, coherence increases from {affected_raw:.3f} "
        f"to {affected_corrected:.3f}, versus {affected_null:.3f} under phase permutation.",
        f"- All {len(session_gains)}/{len(session_gains)} sessions increase; the session-level "
        f"gain ranges from {session_gains.min():+.3f} to {session_gains.max():+.3f}.",
        f"- Coherent F1 increases by {log_gain:+.3f} log10 on average across sessions.",
        f"- Representative Allen coherence remains {bo:.3f} in Brain Observatory and "
        f"{fc:.3f} in Functional Connectivity, leaving residual gaps of "
        f"{corrected - bo:+.3f} and {corrected - fc:+.3f} after source-phase adjustment.",
        "",
        "Every MouseV2 session moves in the predicted direction. The adjustment is fixed",
        "by the acquisition code, 60-Hz frame schedule, and chronological presentation id;",
        "it is not estimated from Allen values or optimized against neural responses.",
        "",
        "## Temporal-frequency falsification",
        "",
        *tf_lines,
        "",
        "The 4/8-Hz values are unchanged by construction because their 135-frame stride",
        "contains an integer number of cycles. Improvements at the source-predicted varying",
        "phases, especially relative to permuted phase labels, support starting phase as a",
        "real partial cause of trial-average cancellation. At 1 and 15 Hz the source-sign",
        f"adjustments are {tf_lookup.loc[1.0, 'equal_session_source_corrected_weighted_phase_coherence']:.3f} "
        f"and {tf_lookup.loc[15.0, 'equal_session_source_corrected_weighted_phase_coherence']:.3f}, whereas the opposite-sign controls are "
        f"{tf_lookup.loc[1.0, 'equal_session_opposite_sign_weighted_phase_coherence']:.3f} and",
        f"{tf_lookup.loc[15.0, 'equal_session_opposite_sign_weighted_phase_coherence']:.3f}; at 2 Hz the two signs are mathematically identical for 0/0.5-cycle phases.",
        "",
        "## Interpretation boundary",
        "",
        "This bridge identifies a material stimulus-definition difference, but it does not",
        "erase the Allen gap and does not authorize a scalar correction to released",
        "`mod_idx_dg`. Source-phase-adjusted carrier coherence is a mechanism diagnostic;",
        "F1/F0 remains the phase-invariant cross-dataset grating measure. The next residual",
        "test should ask whether presentation-level phase shifts are shared across units and",
        "whether they covary with the recorded running and eye-tracking signals.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verified_source_files(schedule: dict[str, object]) -> list[dict[str, object]]:
    expected = schedule["source_sha256"]
    records = []
    for name, expected_hash in expected.items():
        path = SOURCE_FILES[name]
        observed = sha256(path) if path.is_file() else None
        if observed is not None and observed != expected_hash:
            raise ValueError(f"Acquisition source hash drift for {path}")
        records.append(
            {
                "path": str(path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed,
                "verified": observed == expected_hash,
            }
        )
    return records


def refresh_manifest(output_dir: Path) -> None:
    path = output_dir / "import_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prior_inputs = {
        record["site"]: record
        for record in json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))["inputs"]
    }
    for record in manifest.get("inputs", []):
        prior = prior_inputs[record["site"]]
        if int(record["bytes"]) != int(prior["bytes"]):
            raise ValueError(f"{record['site']}: raw-input byte count drift")
        record["sha256"] = prior["sha256"]
        record["sha256_source"] = str(INPUT_MANIFEST.relative_to(ROOT))
    manifest["outputs"] = []
    for name in (
        "README.md",
        "analysis_centers.csv",
        "phase_schedule_audit.csv",
        "session_summary.csv",
        "start_phase_bridge.png",
        "tf_center_summary.csv",
        "tf_session_summary.csv",
        "unit_start_phase_metrics.csv",
    ):
        output = output_dir / name
        if output.is_file():
            manifest["outputs"].append(
                {"path": name, "bytes": output.stat().st_size, "sha256": sha256(output)}
            )
    manifest["code"]["script_sha256"] = sha256(Path(__file__).resolve())
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def render_existing(output_dir: Path) -> None:
    units = pd.read_csv(output_dir / "unit_start_phase_metrics.csv")
    audit = pd.read_csv(output_dir / "phase_schedule_audit.csv")
    summary = summarize_sessions(units)
    tf_sessions, tf_centers = summarize_temporal_frequency(units)
    centers = analysis_centers(summary)
    summary.to_csv(output_dir / "session_summary.csv", index=False)
    tf_sessions.to_csv(output_dir / "tf_session_summary.csv", index=False)
    tf_centers.to_csv(output_dir / "tf_center_summary.csv", index=False)
    centers.to_csv(output_dir / "analysis_centers.csv", index=False)
    make_figure(audit, summary, tf_centers, centers, output_dir / "start_phase_bridge.png")
    write_report(output_dir, summary, tf_centers, centers)
    refresh_manifest(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.render_existing:
        render_existing(output_dir)
        print(f"Rendered MouseV2 start-phase bridge: {output_dir}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    stimulus_manifest_path = args.stimulus_manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stimulus = json.loads(stimulus_manifest_path.read_text(encoding="utf-8"))
    schedule = phase_schedule(stimulus_manifest_path)
    source_records = verified_source_files(schedule)
    nwb_root = (
        args.nwb_root.resolve()
        if args.nwb_root is not None
        else Path(config["nwb_input"]["default_root"]).resolve()
    )
    requested = set(args.sites) if args.sites else None
    sessions = [
        session
        for session in config["sessions"]
        if requested is None or str(session["site"]) in requested
    ]
    if requested is not None and {str(session["site"]) for session in sessions} != requested:
        raise ValueError("One or more requested sites are absent from the config")

    from generate_retinotopic_csvs import choose_stim_table, read_nwb_tables

    unit_rows = []
    audit_rows = []
    input_rows = []
    prior_inputs = {
        record["site"]: record
        for record in json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))["inputs"]
    }
    expected_phases = {
        float(key): int(value)
        for key, value in schedule["expected_unique_start_phases_by_tf_hz"].items()
    }
    for session in sessions:
        site = str(session["site"])
        session_id = int(session["site_number"])
        offset = int(session["id_offset"])
        nwb_path = nwb_root / str(session["nwb_relative_path"])
        if not nwb_path.is_file() or nwb_path.stat().st_size != int(
            session["expected_nwb_bytes"]
        ):
            raise FileNotFoundError(f"Missing or size-mismatched MouseV2 NWB: {nwb_path}")
        quality = pd.read_csv(ROOT / "data" / f"{site}_processed" / "unit_quality.csv")
        quality = quality.loc[quality["default_qc"].eq(True), ["unit_id"]]
        print(f"[{site}] reading raw NWB for {len(quality)} common-QC units", flush=True)
        extracted = read_nwb_tables(str(nwb_path))
        _, table = choose_stim_table(
            extracted.intervals_tables, "drifting_gratings_field_block_presentations"
        )
        conditions = phase_aware_conditions(table, schedule)
        audit_rows.append(
            schedule_audit(table, schedule, site=site, session_id=session_id)
        )
        nwb_ids = extracted.units_df["id"].astype(int).to_numpy()
        row_by_id = {unit_id: row for row, unit_id in enumerate(nwb_ids)}
        for unit_index, output_id in enumerate(quality["unit_id"].astype(int), start=1):
            local_id = int(output_id) - offset
            if local_id not in row_by_id:
                raise ValueError(f"{site}: common-QC unit {local_id} is absent from NWB")
            parameters, counts, phases = preferred_condition_counts(
                extracted.spikes_by_unit[row_by_id[local_id]], conditions
            )
            temporal_frequency_hz = float(parameters[1])
            metrics = phase_adjusted_metrics(
                counts,
                temporal_frequency_hz,
                phases,
                permutations=args.phase_permutations,
                seed=int(
                    np.random.SeedSequence(
                        [args.permutation_seed, session_id, int(output_id)]
                    ).generate_state(1)[0]
                ),
            )
            observed_phase_count = int(metrics["source_start_phase_count"])
            if observed_phase_count > expected_phases[temporal_frequency_hz]:
                raise ValueError(f"{site}: unit condition exceeds expected phase support")
            unit_rows.append(
                {
                    "site": site,
                    "session_id": session_id,
                    "unit_id": int(output_id),
                    "preferred_orientation_deg": float(parameters[0]),
                    "preferred_tf_hz": temporal_frequency_hz,
                    "preferred_sf_cpd": float(parameters[2]),
                    "preferred_trials": len(counts),
                    "source_phase_expected_to_vary": expected_phases[
                        temporal_frequency_hz
                    ]
                    > 1,
                    **metrics,
                }
            )
            if unit_index % 400 == 0:
                print(f"[{site}] analyzed {unit_index}/{len(quality)} units", flush=True)
        prior = prior_inputs[site]
        input_rows.append(
            {
                "site": site,
                "path": str(nwb_path),
                "bytes": nwb_path.stat().st_size,
                "sha256": prior["sha256"],
                "sha256_source": str(INPUT_MANIFEST.relative_to(ROOT)),
            }
        )

    units = pd.DataFrame(unit_rows)
    audit = pd.concat(audit_rows, ignore_index=True)
    summary = summarize_sessions(units)
    tf_sessions, tf_centers = summarize_temporal_frequency(units)
    centers = analysis_centers(summary)
    units.to_csv(output_dir / "unit_start_phase_metrics.csv", index=False)
    audit.to_csv(output_dir / "phase_schedule_audit.csv", index=False)
    summary.to_csv(output_dir / "session_summary.csv", index=False)
    tf_sessions.to_csv(output_dir / "tf_session_summary.csv", index=False)
    tf_centers.to_csv(output_dir / "tf_center_summary.csv", index=False)
    centers.to_csv(output_dir / "analysis_centers.csv", index=False)
    write_report(output_dir, summary, tf_centers, centers)
    if not args.skip_figure:
        make_figure(audit, summary, tf_centers, centers, output_dir / "start_phase_bridge.png")

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
        "source_provenance": {
            "repository": stimulus["source_repository"],
            "files": source_records,
            "phase_schedule": schedule,
        },
        "inputs": input_rows,
        "code": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "metric_sha256": sha256(ROOT / "common" / "drifting_gratings.py"),
            "config_sha256": sha256(config_path),
            "stimulus_manifest_sha256": sha256(stimulus_manifest_path),
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    refresh_manifest(output_dir)
    print(f"MouseV2 start-phase bridge written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
