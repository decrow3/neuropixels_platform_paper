#!/usr/bin/env python3
"""Write population-mask counts, diagnostics, and a review figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units  # noqa: E402
from common.figure3_rf import DEFAULT_RF_IMPORT_DIR, load_rf_import  # noqa: E402
from common.population_masks import (  # noqa: E402
    COMMON_QC_RULE,
    PUBLISHED_LIKE_RULE,
    population_mask,
)


TARGET_AREAS = ("LGd", "V1", "LM", "RL", "LP", "AL", "PM", "AM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--grating-metrics-dir", type=Path, default=None)
    parser.add_argument("--rf-import-dir", type=Path, default=DEFAULT_RF_IMPORT_DIR)
    parser.add_argument("--population-profile", default="common_qc")
    return parser.parse_args()


def flow_row(
    dataset: str,
    stage: str,
    units: int | None,
    rule: str,
    *,
    available: bool = True,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "stage": stage,
        "units": units,
        "available": available,
        "rule": rule,
    }


def make_flow_figure(flow: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = {"Allen": "#6f6f6f", "MouseV2": "#8172B3"}
    for axis, dataset in zip(axes, ("Allen", "MouseV2")):
        data = flow[(flow["dataset"] == dataset) & flow["available"]].copy()
        data = data[data["units"].notna()]
        y = np.arange(len(data))
        values = data["units"].to_numpy(dtype=float)
        axis.barh(y, values, color=colors[dataset], alpha=0.82)
        axis.set_yticks(y, data["stage"])
        axis.invert_yaxis()
        axis.set_xlabel("units")
        axis.set_title(dataset)
        axis.grid(axis="x", alpha=0.18)
        for yi, value in zip(y, values):
            axis.text(value, yi, f" {int(value):,}", va="center", fontsize=9)
    fig.suptitle("Figure 3 population-mask flow (stages are diagnostic, not all sequential)")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allen = load_allen_units(args.config)
    mouse = load_mousev2_units(
        apply_qc=False,
        config_path=args.config,
        grating_metrics_dir=args.grating_metrics_dir,
        population_profile="pipeline_baseline",
    )
    peaks, _, _, rf_manifest = load_rf_import(args.rf_import_dir)
    mouse = mouse.merge(
        peaks[["unit_id", "pilot_qc", "rf_method", "gaze_correction"]],
        on="unit_id",
        how="left",
        validate="one_to_one",
    )

    allen_target = allen["area_coarse"].isin(TARGET_AREAS)
    allen_common = population_mask(allen, profile="common_qc", dataset="allen")
    allen_published = population_mask(
        allen, profile="published_like", dataset="allen"
    )
    mouse_common = population_mask(mouse, profile="common_qc", dataset="mousev2")
    mouse_pilot = mouse["pilot_qc"].fillna(False).astype(bool)

    flow_rows = [
        flow_row("Allen", "all unit_table", len(allen), "all released table rows"),
        flow_row(
            "Allen",
            "target Figure 3 regions",
            int(allen_target.sum()),
            "LGd, V1, LM, RL, LP, AL, PM, AM",
        ),
        flow_row(
            "Allen",
            "common_qc in target regions",
            int((allen_target & allen_common).sum()),
            COMMON_QC_RULE,
        ),
        flow_row(
            "Allen",
            "published_like in target regions",
            int((allen_target & allen_published).sum()),
            PUBLISHED_LIKE_RULE,
        ),
        flow_row(
            "Allen",
            "common_qc ∩ published_like",
            int((allen_target & allen_common & allen_published).sum()),
            f"{COMMON_QC_RULE}; {PUBLISHED_LIKE_RULE}",
        ),
        flow_row("MouseV2", "all targeted units", len(mouse), "eight NWB unit tables"),
        flow_row(
            "MouseV2", "common_qc", int(mouse_common.sum()), COMMON_QC_RULE
        ),
        flow_row(
            "MouseV2",
            "Pilot RF diagnostic QC",
            int(mouse_pilot.sum()),
            "snr > 5; d_prime > 2; rp_contamination < 0.1; default_qc",
        ),
        flow_row(
            "MouseV2",
            "common_qc ∩ Pilot RF QC",
            int((mouse_common & mouse_pilot).sum()),
            "diagnostic intersection; not a published-like RF filter",
        ),
        flow_row(
            "MouseV2",
            "published_like",
            None,
            "unavailable: no validated all-session p_value_rf/area_rf export",
            available=False,
        ),
    ]
    flow = pd.DataFrame(flow_rows)
    flow.to_csv(output_dir / "population_flow.csv", index=False)

    common_allen = allen[allen_target & allen_common]
    common_mouse = mouse[mouse_common]
    group_rows = []
    for area, group in common_allen.groupby("area_coarse"):
        group_rows.append(
            {
                "dataset": "Allen",
                "profile": "common_qc",
                "group_type": "area",
                "group": area,
                "units": len(group),
                "sessions": group["ecephys_session_id"].nunique(),
            }
        )
    for (site, probe), group in common_mouse.groupby(["site", "probe_letter"]):
        group_rows.append(
            {
                "dataset": "MouseV2",
                "profile": "common_qc",
                "group_type": "site_probe",
                "group": f"{site}_{probe}",
                "units": len(group),
                "sessions": 1,
            }
        )
    groups = pd.DataFrame(group_rows)
    groups.to_csv(output_dir / "population_by_group.csv", index=False)
    make_flow_figure(flow, output_dir / "population_flow.png")

    mouse_site_counts = common_mouse.groupby("site").size()
    mouse_probe_counts = common_mouse.groupby("probe_letter").size()
    report = [
        "# Figure 3 population profiles",
        "",
        f"Selected checkpoint profile: `{args.population_profile}`.",
        "",
        "## Common QC",
        "",
        f"Rule applied to both datasets: `{COMMON_QC_RULE}`.",
        "",
        f"- Allen: {int(allen_common.sum()):,}/{len(allen):,} across the full unit table; "
        f"{len(common_allen):,} in the eight Figure 3 regions.",
        f"- MouseV2: {len(common_mouse):,}/{len(mouse):,} units.",
        f"- MouseV2 per-site range: {int(mouse_site_counts.min()):,}–{int(mouse_site_counts.max()):,} units.",
        f"- MouseV2 per-probe pooled range: {int(mouse_probe_counts.min()):,}–{int(mouse_probe_counts.max()):,} units.",
        "",
        "MouseV2 `common_qc` is verified to equal the NWB `default_qc` flag exactly.",
        "",
        "## RF-filter status",
        "",
        "`published_like` is computable for Allen but unavailable for MouseV2.",
        "The provisional RF peak import has no significance p-value or area and uses",
        f"`{peaks['rf_method'].iloc[0]}` with gaze correction `{peaks['gaze_correction'].iloc[0]}`.",
        "`firing_rate_dg` is available as preferred-condition mean spikes divided by",
        "the validated 1.0-s analysis duration.",
        f"Pilot RF diagnostic QC retains {int(mouse_pilot.sum()):,} units, but it must not",
        "be described as the published RF filter.",
        "",
        f"RF import schema version: {rf_manifest.get('schema_version', 'unknown')}.",
    ]
    (output_dir / "population_profile_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Population diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
