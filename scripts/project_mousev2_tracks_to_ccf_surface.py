#!/usr/bin/env python3
"""Project MouseV2 V1 probe entry points onto the Allen CCFv3 dorsal atlas.

The AIND electrode coordinates use CCF AP/DV/ML order (NWB x/y/z).  Allen's
``top.nrrd`` is a dorsal surface atlas with array axes ML/AP at 10-um spacing,
so the surface projection is the direct mapping (ML, AP) = (z, x).
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = (
    ROOT / "artifacts/figure3/06r_mousev2_probe_tracks_from_ccf/"
    "mousev2_probe_track_points.csv"
)
DEFAULT_ATLAS = ROOT / "data/reference/allen_ccf_2017_surface/top.nrrd"
DEFAULT_LABELS = (
    ROOT / "data/reference/allen_ccf_2017_surface/"
    "labelDescription_ITKSNAPColor.txt"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/figure3/06s_mousev2_ccf_surface_projection"
)
PROBE_ORDER = ("ProbeA", "ProbeE", "ProbeC", "ProbeB")
PROBE_LABELS = {
    "ProbeA": "Anterior", "ProbeE": "Lateral",
    "ProbeC": "Posterior", "ProbeB": "Medial",
}
PROBE_COLORS = {
    "ProbeA": "#d73027", "ProbeE": "#fc8d59",
    "ProbeC": "#1a9850", "ProbeB": "#4575b4",
}
VISUAL_AREAS = (
    "VISp", "VISal", "VISam", "VISl", "VISpl", "VISpm",
    "VISa", "VISli", "VISpor", "VISrl",
)
ATLAS_URL = (
    "https://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/"
    "master_updated/top.nrrd"
)


def read_nrrd(path: Path) -> np.ndarray:
    """Read the small gzip-encoded 2-D uint16 NRRD without an extra dependency."""
    header, payload = path.read_bytes().split(b"\n\n", 1)
    sizes_match = re.search(rb"^sizes:\s+([0-9 ]+)$", header, re.MULTILINE)
    if sizes_match is None or b"type: uint16" not in header or b"encoding: gzip" not in header:
        raise ValueError(f"Unsupported NRRD header in {path}")
    sizes = tuple(int(value) for value in sizes_match.group(1).split())
    return np.frombuffer(gzip.decompress(payload), dtype="<u2").reshape(sizes, order="F")


def read_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    pattern = re.compile(
        r'^\s*(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+"([^"]+)"'
    )
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            labels[int(match.group(1))] = match.group(2)
    return labels


def derive_entry_points(points: pd.DataFrame, atlas: np.ndarray,
                        labels: dict[int, str]) -> pd.DataFrame:
    """Use the most superficial VISp-labelled AP contact as the entry proxy."""
    visp = points.loc[points["location"].astype(str).str.startswith("VISp")].copy()
    entries = (
        visp.sort_values("rel_y")
        .groupby(["subject_id", "probe"], as_index=False, sort=True)
        .tail(1)
        .copy()
    )
    if len(entries) != 32 or entries.groupby("subject_id")["probe"].nunique().ne(4).any():
        raise ValueError(f"Expected 32 entries (8 sessions x 4 probes), found {len(entries)}")
    entries["ccf_ap_um"] = entries["x"].astype(float)
    entries["ccf_dv_um"] = entries["y"].astype(float)
    entries["ccf_ml_um"] = entries["z"].astype(float)
    entries["surface_ap_voxel"] = np.floor(entries["ccf_ap_um"] / 10).astype(int)
    entries["surface_ml_voxel"] = np.floor(entries["ccf_ml_um"] / 10).astype(int)
    entries["surface_atlas_index"] = [
        int(atlas[ml, ap])
        for ml, ap in zip(entries.surface_ml_voxel, entries.surface_ap_voxel)
    ]
    entries["surface_atlas_acronym"] = entries.surface_atlas_index.map(labels).fillna("void")
    entries["surface_agrees_visp"] = entries.surface_atlas_acronym.eq("VISp")
    entries["probe_display"] = entries.probe.map(PROBE_LABELS)
    return entries[
        ["subject_id", "site", "probe", "probe_display", "channel_name",
         "rel_y", "location", "ccf_ap_um", "ccf_dv_um", "ccf_ml_um",
         "surface_ap_voxel", "surface_ml_voxel", "surface_atlas_index",
         "surface_atlas_acronym", "surface_agrees_visp"]
    ].sort_values(["subject_id", "probe"])


def _area_index(labels: dict[int, str], acronym: str) -> int:
    return next(index for index, label in labels.items() if label == acronym)


def plot_surface(ax: plt.Axes, entries: pd.DataFrame, atlas: np.ndarray,
                 labels: dict[int, str], *, title_prefix: str = "A1",
                 show_mean_positions: bool = False) -> None:
    """Draw the visual-cortex crop of the genuine Allen dorsal surface atlas."""
    # top.nrrd has axes (ML, AP). Transposition gives conventional image rows AP,
    # columns ML; extent reverses AP so anterior is at the top.
    backdrop = np.zeros(atlas.shape, dtype=float)
    for acronym in VISUAL_AREAS:
        index = _area_index(labels, acronym)
        backdrop[atlas == index] = 1.0 if acronym == "VISp" else 0.55
    masked = np.ma.masked_where(backdrop == 0, backdrop)
    ax.imshow(
        masked.T, origin="upper", extent=(0, 11.4, 13.2, 0),
        cmap=matplotlib.colors.ListedColormap(["#f5f5f3", "#dedce9"]),
        vmin=0.45, vmax=1.05, interpolation="nearest", zorder=0,
    )
    ml = np.arange(atlas.shape[0]) * 0.01
    ap = np.arange(atlas.shape[1]) * 0.01
    for acronym in VISUAL_AREAS:
        mask = atlas == _area_index(labels, acronym)
        ax.contour(ml, ap, mask.T.astype(float), levels=[0.5], colors="#565961",
                   linewidths=0.65 if acronym != "VISp" else 1.25, zorder=1)
        yy, xx = np.where(mask.T)
        if len(xx):
            # Medians are more robust than centroids for concave atlas polygons.
            ax.text(np.median(xx) * 0.01, np.median(yy) * 0.01,
                    acronym.removeprefix("VIS") if acronym != "VISp" else "V1",
                    ha="center", va="center", fontsize=6.2,
                    fontweight="bold" if acronym == "VISp" else "normal",
                    color="#45474c", zorder=2)
    for probe in PROBE_ORDER:
        local = entries.loc[entries.probe.eq(probe)]
        if show_mean_positions:
            ax.scatter([local.ccf_ml_um.mean() / 1000], [local.ccf_ap_um.mean() / 1000],
                       s=190, facecolor=matplotlib.colors.to_rgba(PROBE_COLORS[probe], .30),
                       edgecolor=PROBE_COLORS[probe], linewidth=1.6, zorder=3)
        ax.scatter(local.ccf_ml_um / 1000, local.ccf_ap_um / 1000,
                   s=14 if show_mean_positions else 31,
                   color=PROBE_COLORS[probe], edgecolor="white",
                   linewidth=0.45 if show_mean_positions else 0.65, alpha=0.92, zorder=4)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.6,
               markerfacecolor=PROBE_COLORS[p], markeredgecolor="white",
               label=PROBE_LABELS[p])
        for p in PROBE_ORDER
    ]
    ax.legend(handles=handles, title="V1 probe", ncol=2, loc="lower left",
              frameon=False, fontsize=6.8, title_fontsize=7.2,
              columnspacing=0.75, handletextpad=0.3, borderaxespad=0.2)
    # Keep enough surrounding visual cortex in view to orient the V1 sampling
    # locations relative to the neighbouring higher visual areas.
    ax.set_xlim(1.25, 4.50)
    ax.set_ylim(10.50, 6.75)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral  ←  CCF ML (mm)  →  Medial", fontsize=7.3)
    ax.set_ylabel("CCF AP (mm)\nAnterior ↑", fontsize=7.3)
    ax.tick_params(labelsize=6.5, length=2)
    ax.set_title(f"{title_prefix}  MouseV2 entry points on CCFv3", loc="left",
                 fontweight="bold", fontsize=11.2, pad=4)
    ax.text(0.0, 0.995,
            f"Allen dorsal surface · V1/HVA borders · {len(entries)} probes · "
            f"{entries.subject_id.nunique()} animals",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.4,
            color="#62666d")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    atlas = read_nrrd(args.atlas)
    labels = read_labels(args.labels)
    entries = derive_entry_points(pd.read_csv(args.points), atlas, labels)
    entries.to_csv(args.output / "mousev2_ccf_surface_entry_points.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    plot_surface(ax, entries, atlas, labels)
    fig.tight_layout()
    fig.savefig(args.output / "mousev2_ccf_surface_projection.png", dpi=300)
    fig.savefig(args.output / "mousev2_ccf_surface_projection.pdf")
    plt.close(fig)
    agreement = int(entries.surface_agrees_visp.sum())
    mismatches = entries.loc[~entries.surface_agrees_visp,
                             ["subject_id", "probe", "surface_atlas_acronym"]]
    report = [
        "# MouseV2 CCF dorsal-surface projection", "",
        f"- Entry points: {len(entries)} (8 animals x 4 probes).",
        "- Entry proxy: most superficial AP electrode whose NWB location starts with `VISp`.",
        "- Projection: NWB `(x, z)` = CCF `(AP, ML)`, directly overlaid on Allen `top.nrrd`.",
        f"- Independent surface-label agreement with VISp: {agreement}/{len(entries)}.",
        f"- Atlas source: {ATLAS_URL}", "", "## Border cases", "",
    ]
    report.extend(
        f"- subject {row.subject_id}, {row.probe}: {row.surface_atlas_acronym}"
        for row in mismatches.itertuples()
    )
    (args.output / "README.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
