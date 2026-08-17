#!/usr/bin/env python3
"""Spatial map of agreement/disagreement between the naive (single pooled offset) mapping and
the per-session mapping. Since both add an offset to the same raw RF value, their difference at
any cell is exactly (that session's final offset - the pooled offset) -- constant within a
session, so this shows, spatially, which sessions/regions the per-session correction moved the
most (and in which direction) relative to treating the whole cohort as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
PER_SESSION_DIR = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_rf_offset"
OUTPUT = PER_SESSION_DIR


def main() -> None:
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    offset_manifest = json.loads((PER_SESSION_DIR / "per_session_offset_manifest.json").read_text())
    pooled_offset_az = offset_manifest["pooled_offset_az_deg"]
    pooled_offset_el = offset_manifest["pooled_offset_el_deg"]

    offsets = pd.read_csv(PER_SESSION_DIR / "per_session_rf_offset.csv")
    offsets["disagreement_az"] = offsets.final_offset_az - pooled_offset_az
    offsets["disagreement_el"] = offsets.final_offset_el - pooled_offset_el
    offsets["disagreement_magnitude_deg"] = np.hypot(offsets.disagreement_az, offsets.disagreement_el)

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0
    cells = cells.merge(
        offsets[["ecephys_session_id", "disagreement_az", "disagreement_el", "disagreement_magnitude_deg", "capped"]],
        on="ecephys_session_id", how="left", validate="many_to_one",
    )
    if cells.disagreement_az.isna().any():
        raise RuntimeError("cells missing a matching session disagreement value")

    diverging_norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25)
    magnitude_norm = Normalize(vmin=0, vmax=cells.disagreement_magnitude_deg.quantile(0.98))

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.4), constrained_layout=True)
    panels = (
        ("disagreement_az", "Azimuth disagreement\n(per-session offset - pooled offset)", "RdBu_r", diverging_norm),
        ("disagreement_el", "Elevation disagreement\n(per-session offset - pooled offset)", "RdBu_r", diverging_norm),
        ("disagreement_magnitude_deg", "Total disagreement magnitude\n(vector norm, deg)", "magma_r", magnitude_norm),
    )
    for ax, (col, title, cmap, norm) in zip(axes, panels):
        scatter = ax.scatter(cells.ccf_ml_mm, cells.ccf_ap_mm, c=cells[col], cmap=cmap, norm=norm,
                              s=6, alpha=0.75, linewidths=0, rasterized=True)
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("degrees")
        ax.set(title=title, xlabel="Medial-lateral CCF (mm)", ylabel="Anterior-posterior CCF (mm)")
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    n_capped = int(offsets.capped.sum())
    fig.suptitle(
        f"Agreement/disagreement between naive (single pooled offset = az{pooled_offset_az:+.1f}, "
        f"el{pooled_offset_el:+.1f}) and per-session mapping\n"
        f"(constant within a session by construction; {n_capped}/{len(offsets)} sessions capped, "
        f"median disagreement magnitude={offsets.disagreement_magnitude_deg.median():.1f} deg)",
        fontsize=12.5,
    )
    figure_path = OUTPUT / "Figure_naive_vs_per_session_agreement.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(figure_path)

    # session-level summary, sorted by disagreement magnitude, for the printed record
    summary = offsets[["ecephys_session_id", "disagreement_az", "disagreement_el",
                        "disagreement_magnitude_deg", "capped"]].sort_values(
        "disagreement_magnitude_deg", ascending=False)
    summary.to_csv(OUTPUT / "naive_vs_per_session_disagreement_by_session.csv", index=False)
    print(f"median disagreement magnitude: {offsets.disagreement_magnitude_deg.median():.2f} deg")
    print(f"max disagreement magnitude: {offsets.disagreement_magnitude_deg.max():.2f} deg "
          f"(session {int(summary.iloc[0].ecephys_session_id)})")
    print(OUTPUT / "naive_vs_per_session_disagreement_by_session.csv")


if __name__ == "__main__":
    main()
