#!/usr/bin/env python3
"""Naive cross-session RF-by-CCF-position map, V1-median-centered per session.

The simplest possible baseline against the Zhuang/Garrett atlas comparisons: no atlas, no
fitted affine, no smoothing model -- just pool every cell's (CCF, RF) pair across sessions,
after subtracting that session's OWN median V1 RF from every cell in that session (including
V1 itself). This cancels each session's unknown translation exactly the same way the atlas
comparisons do (a constant offset subtracted from both sides of a difference), so it works
with the raw Allen-native RF convention directly -- no +50/+10 shift needed, since any
additive convention cancels in the subtraction.

This is deliberately "naive": a single global per-session anchor point (V1 median), not a
locally-varying correction. It's a sanity-check baseline, not a competitor to the local-linear
Jacobian approach -- if this naive pooled map doesn't show recognizable retinotopic structure
at all, that's informative on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_joint_multistructure_dispersion_likelihood import load_all  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint" / "naive_v1_centered_cross_session_map"
CCF2 = ["anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate"]
MIN_V1_CELLS = 5


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pop = load_all()
    pop = pop.loc[~pop.center_bound & pop[CCF2 + ["rf_x", "rf_y"]].notna().all(axis=1)].copy()

    v1_median = (
        pop.loc[pop.structure_group.eq("V1")]
        .groupby("ecephys_session_id")
        .agg(v1_median_rf_x=("rf_x", "median"), v1_median_rf_y=("rf_y", "median"), v1_cells=("rf_x", "size"))
    )
    v1_median = v1_median.loc[v1_median.v1_cells >= MIN_V1_CELLS]
    print(f"sessions with usable V1 anchor (>={MIN_V1_CELLS} V1 cells): {len(v1_median)} / {pop.ecephys_session_id.nunique()}")

    cortex = pop.loc[pop.structure_group.isin(("V1", "HVA"))].merge(
        v1_median, left_on="ecephys_session_id", right_index=True, how="inner"
    )
    cortex["normalized_rf_x"] = cortex.rf_x - cortex.v1_median_rf_x
    cortex["normalized_rf_y"] = cortex.rf_y - cortex.v1_median_rf_y

    keep_cols = ["ecephys_session_id", "ecephys_probe_id", "structure_group", "map_area",
                 *CCF2, "rf_x", "rf_y", "normalized_rf_x", "normalized_rf_y", "v1_cells"]
    result = cortex[keep_cols].copy()
    result.to_csv(OUTPUT / "naive_v1_centered_cells.csv.gz", index=False, compression="gzip")
    print(f"pooled cells: {len(result)} across {result.ecephys_session_id.nunique()} sessions, "
          f"{result.map_area.nunique()} areas")

    audit = {
        "min_v1_cells_per_session": MIN_V1_CELLS,
        "sessions_total": int(pop.ecephys_session_id.nunique()),
        "sessions_with_v1_anchor": int(len(v1_median)),
        "pooled_cells": int(len(result)),
        "normalized_rf_x_percentiles": {str(q): float(result.normalized_rf_x.quantile(q)) for q in (.02, .5, .98)},
        "normalized_rf_y_percentiles": {str(q): float(result.normalized_rf_y.quantile(q)) for q in (.02, .5, .98)},
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, col, title in ((axes[0], "normalized_rf_x", "V1-centered azimuth"), (axes[1], "normalized_rf_y", "V1-centered elevation")):
        vmax = float(result[col].abs().quantile(.98))
        norm = Normalize(vmin=-vmax, vmax=vmax)
        scatter = ax.scatter(result[CCF2[1]], result[CCF2[0]], c=result[col], cmap="coolwarm", norm=norm, s=6, alpha=.5, rasterized=True)
        ax.set(xlabel="left-right CCF (um)", ylabel="anterior-posterior CCF (um)", title=title, aspect="equal")
        fig.colorbar(scatter, ax=ax, label="deg, relative to session V1 median")
    fig.suptitle(f"Naive cross-session pooled map, V1-median-centered per session (n={len(result)} cells, "
                 f"{result.ecephys_session_id.nunique()} sessions)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_naive_v1_centered_pooled_scatter.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
