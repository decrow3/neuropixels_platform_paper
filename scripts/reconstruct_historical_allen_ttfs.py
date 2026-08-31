#!/usr/bin/env python3
"""Test plausible reconstructions of the undocumented 2019 Allen TTFS metric."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from aggregate_response_verified_early_v1_psths import read_targeted_nwb
from crossvalidated_v1_flash_timing import all_unit_inputs

OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance"
VARIANTS = ("pooled", "bright", "dark", "preferred_0_250")
COLORS = {"pooled": "#4C78A8", "bright": "#F2CF5B", "dark": "#444444", "preferred_0_250": "#E45756"}


def first_spike_ms(spikes: np.ndarray, starts: np.ndarray) -> float:
    """Median first occupied 1-ms bin in [30, 200] ms; empty trials excluded."""
    if len(starts) == 0:
        return np.nan
    lo = np.searchsorted(spikes, starts + 0.030, side="left")
    hi = np.searchsorted(spikes, starts + 0.201, side="left")
    valid = lo < hi
    if not valid.any():
        return np.nan
    # Match a binned implementation rather than returning sub-ms spike times.
    bins = np.floor((spikes[lo[valid]] - starts[valid]) * 1000 + 1e-9)
    return float(np.median(bins))


def preferred_polarity(spikes: np.ndarray, starts: np.ndarray, polarity: np.ndarray) -> float:
    """Historical Flashes preference: largest mean spike count over its 250-ms trial."""
    means = []
    for value in (1.0, -1.0):
        selected = starts[polarity == value]
        left = np.searchsorted(spikes, selected, side="left")
        right = np.searchsorted(spikes, selected + 0.250, side="left")
        means.append(float(np.mean(right - left)))
    return (1.0, -1.0)[int(np.argmax(means))]


def process_session(cohort, sid, units, path, flash_table):
    spikes_by_id, starts, polarity = read_targeted_nwb(
        path, flash_table, units["source_unit_id"].astype(int).to_numpy()
    )
    rows = []
    for _, unit in units.iterrows():
        uid, source = int(unit["analysis_unit_id"]), int(unit["source_unit_id"])
        spikes = spikes_by_id[source]
        preferred = preferred_polarity(spikes, starts, polarity)
        masks = {
            "pooled": np.ones(len(starts), dtype=bool),
            "bright": polarity == 1.0,
            "dark": polarity == -1.0,
            "preferred_0_250": polarity == preferred,
        }
        released = 1000 * float(unit["time_to_first_spike_fl"]) if pd.notna(unit["time_to_first_spike_fl"]) else np.nan
        row = {"cohort": cohort, "session_id": sid, "unit_id": uid, "source_unit_id": source,
               "released_ttfs_ms": released, "preferred_polarity": "bright" if preferred == 1 else "dark"}
        for name, mask in masks.items():
            estimate = first_spike_ms(spikes, starts[mask])
            row[f"{name}_ttfs_ms"] = estimate
            row[f"{name}_signed_error_ms"] = estimate - released
            row[f"{name}_absolute_error_ms"] = abs(estimate - released)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in data.groupby("cohort"):
        for variant in VARIANTS:
            both = group[["released_ttfs_ms", f"{variant}_ttfs_ms"]].dropna()
            error = both[f"{variant}_ttfs_ms"] - both["released_ttfs_ms"]
            rows.append({
                "cohort": cohort, "variant": variant, "n": len(both),
                "exact_percent": 100 * np.mean(error == 0),
                "median_absolute_error_ms": np.median(np.abs(error)),
                "mean_absolute_error_ms": np.mean(np.abs(error)),
                "p95_absolute_error_ms": np.percentile(np.abs(error), 95),
                "mean_signed_error_ms": np.mean(error),
                "pearson_r": both.corr().iloc[0, 1],
            })
    return pd.DataFrame(rows)


def render(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    cohorts = list(data["cohort"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, cohort in zip(axes[0], cohorts):
        group = data[data.cohort == cohort]
        sample = group.sample(min(1200, len(group)), random_state=20260825)
        for variant in ("pooled", "preferred_0_250"):
            ax.scatter(sample.released_ttfs_ms, sample[f"{variant}_ttfs_ms"], s=8, alpha=.22,
                       color=COLORS[variant], label=variant.replace("_0_250", " polarity"))
        ax.plot([30, 200], [30, 200], color="k", lw=1, ls=":")
        ax.set(xlabel="Released TTFS (ms)", ylabel="Reconstructed TTFS (ms)", title=cohort)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    for ax, cohort in zip(axes[1], cohorts):
        group = data[data.cohort == cohort]
        for variant in VARIANTS:
            values = group[f"{variant}_absolute_error_ms"].dropna().sort_values().to_numpy()
            ax.plot(values, np.arange(1, len(values)+1)/len(values), color=COLORS[variant], lw=2,
                    label=variant.replace("_0_250", " polarity"))
        ax.set(xlim=(0, 80), xlabel="Absolute error from released TTFS (ms)", ylabel="Cumulative fraction",
               title=f"{cohort}: error distribution")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Reconstructing the undocumented Allen TTFS estimator\nAll common-QC V1 units; no latency-based unit selection")
    fig.savefig(OUT / "historical_ttfs_reconstruction.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "historical_ttfs_reconstruction.pdf", bbox_inches="tight")
    plt.close(fig)

    pivot = summary.pivot(index="variant", columns="cohort", values="median_absolute_error_ms")
    pivot.to_csv(OUT / "historical_ttfs_reconstruction_median_error.csv")


def main() -> None:
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen.area_coarse.eq("V1")].copy()
    # Needed only to reuse the audited session router; MouseV2 outputs are skipped.
    mouse = load_mousev2_units(
        apply_qc=False, grating_metrics_dir=ROOT / "data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT / "data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile="common_qc",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    frames, status = [], []
    for cohort, sid, units, path, flash_table in all_unit_inputs(allen, mouse):
        if cohort == "MouseV2":
            continue
        print(f"[{cohort} {sid}] {len(units)} units", flush=True)
        try:
            frames.append(process_session(cohort, sid, units, path, flash_table))
            status.append({"cohort": cohort, "session_id": sid, "status": "ok", "units": len(units), "error": ""})
        except Exception as exc:
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            status.append({"cohort": cohort, "session_id": sid, "status": "error", "units": 0,
                           "error": f"{type(exc).__name__}: {exc}"})
    data = pd.concat(frames, ignore_index=True)
    summary = summarize(data)
    data.to_csv(OUT / "historical_ttfs_reconstruction_unit_level.csv", index=False)
    summary.to_csv(OUT / "historical_ttfs_reconstruction_summary.csv", index=False)
    pd.DataFrame(status).to_csv(OUT / "historical_ttfs_reconstruction_status.csv", index=False)
    render(data, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
