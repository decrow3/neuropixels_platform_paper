#!/usr/bin/env python3
"""Do Allen's 6 probes converge toward a common point in CCF space?

For each session, find the 3D point that minimizes total squared perpendicular distance to all
fitted probe lines (the standard least-squares line-intersection: for lines with unit direction
d_i through point p_i, minimize sum_i ||(I - d_i d_i^T)(x - p_i)||^2, a 3x3 linear solve). Real
skew lines never exactly meet, so this is the best common point, with a residual (RMS
perpendicular distance) quantifying how good "converge" actually is.

If Allen's multi-probe holder mechanically aims every probe at one shared virtual target (a
sensible design to avoid probe collisions as they enter from a shared manipulator stage at
different angles), this point should (a) be reasonably consistent in absolute CCF coordinates
across sessions/animals, and (b) sit close to where each individual probe's own electrodes
physically stop (near or just past the deepest recorded unit), not at some arbitrary depth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_allen_probe_insertion_angle_from_ccf import (  # noqa: E402
    CCF_COLUMNS,
    MIN_UNITS_PER_PROBE,
    PROBE_LETTER_COLORS,
    fit_probe_line,
)

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data" / "unit_table.csv"
OUTPUT = ROOT / "artifacts" / "figure3" / "06n_allen_probe_convergence_point"
MIN_PROBES_PER_SESSION = 4


def per_probe_fits(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for probe_id, group in units.groupby("ecephys_probe_id", observed=True):
        if len(group) < MIN_UNITS_PER_PROBE:
            continue
        points = group[list(CCF_COLUMNS)].to_numpy(float)
        depth = group["probe_vertical_position"].to_numpy(float)
        fit = fit_probe_line(points, depth)
        t = (points - fit["centroid"]) @ fit["direction"]
        rows.append(
            {
                "probe_id": int(probe_id),
                "session": int(group["ecephys_session_id"].iat[0]),
                "session_type": group["session_type"].iat[0],
                "letter": group["name"].iat[0],
                "direction": fit["direction"],
                "centroid": fit["centroid"],
                "t_min": t.min(),
                "t_max": t.max(),
            }
        )
    return pd.DataFrame(rows)


def line_intersection(directions: list[np.ndarray], centroids: list[np.ndarray]) -> tuple[np.ndarray, float]:
    projectors = [np.eye(3) - np.outer(d, d) for d in directions]
    A = sum(projectors)
    b = sum(M @ p for M, p in zip(projectors, centroids))
    x = np.linalg.solve(A, b)
    residual = float(np.sqrt(np.mean([np.sum((M @ (x - p)) ** 2) for M, p in zip(projectors, centroids)])))
    return x, residual


def build_convergence_table(fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_rows = []
    fraction_rows = []
    for session_id, sess in fits.groupby("session"):
        if len(sess) < MIN_PROBES_PER_SESSION:
            continue
        x, residual = line_intersection(sess.direction.tolist(), sess.centroid.tolist())
        session_rows.append(
            {
                "session": session_id,
                "session_type": sess.session_type.iat[0],
                "n_probes": len(sess),
                "AP_um": x[0], "DV_um": x[1], "LR_um": x[2],
                "residual_um": residual,
            }
        )
        for _, row in sess.iterrows():
            t_conv = (x - row.centroid) @ row.direction
            fraction_rows.append(
                {
                    "letter": row.letter,
                    "frac_along_recorded_span": (t_conv - row.t_min) / (row.t_max - row.t_min),
                }
            )
    return pd.DataFrame(session_rows), pd.DataFrame(fraction_rows)


def make_figure(conv: pd.DataFrame, frac: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(conv.AP_um / 1000, conv.LR_um / 1000, conv.DV_um / 1000,
               s=25, alpha=0.6, color="#4575b4", edgecolor="none")
    mean_point = conv[["AP_um", "LR_um", "DV_um"]].mean().to_numpy() / 1000
    ax.scatter([mean_point[0]], [mean_point[1]], [mean_point[2]], s=140, color="black",
               marker="*", zorder=5, label="mean")
    ax.invert_zaxis()
    ax.set(xlabel="AP (mm)", ylabel="LR (mm)", zlabel="DV (mm)")
    ax.set_title(f"Per-session convergence point\n(n={len(conv)} sessions)", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(1, 3, 2)
    ax.hist(conv.residual_um / 1000, bins=18, color="#4575b4", alpha=0.8, edgecolor="white")
    ax.axvline((conv.residual_um / 1000).median(), color="black", linewidth=2,
               label=f"median = {conv.residual_um.median():.0f} um")
    ax.set(xlabel="RMS perpendicular residual (mm)", ylabel="sessions",
           title="Convergence quality\n(how close lines really come to meeting)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    ax = fig.add_subplot(1, 3, 3)
    letters = [f"probe{L}" for L in "ABCDEF" if f"probe{L}" in frac.letter.values]
    rng = np.random.default_rng(0)
    for index, letter in enumerate(letters):
        values = frac.loc[frac.letter.eq(letter), "frac_along_recorded_span"].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, len(values))
        ax.scatter(index + jitter, values, s=18, alpha=0.6,
                   color=PROBE_LETTER_COLORS[letter], edgecolor="none")
        ax.hlines(np.median(values), index - 0.28, index + 0.28, color="black", lw=2, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="deepest recorded unit")
    ax.axhline(1, color="gray", linestyle=":", linewidth=1, label="surface entry")
    ax.set_xticks(range(len(letters)))
    ax.set_xticklabels([letter.replace("probe", "") for letter in letters])
    ax.set(ylabel="position of convergence point\nalong probe's own recorded span (0=tip, 1=surface)",
           title="Where the convergence point falls\nrelative to each probe's electrodes")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Do Allen's probes converge on a common CCF coordinate?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(conv: pd.DataFrame, frac: pd.DataFrame, output_path: Path) -> None:
    mean_mm = conv[["AP_um", "DV_um", "LR_um"]].mean() / 1000
    sd_um = conv[["AP_um", "DV_um", "LR_um"]].std()
    by_letter = frac.groupby("letter")["frac_along_recorded_span"].median().reindex(
        [f"probe{L}" for L in "ABCDEF"]
    )
    lines = [
        "# Do Allen's probes converge on a common CCF coordinate?",
        "",
        f"Least-squares line-intersection point computed per session ({len(conv)} sessions with "
        f">= {MIN_PROBES_PER_SESSION} fitted probes).",
        "",
        f"- Mean convergence point: AP={mean_mm.AP_um:.2f} mm, DV={mean_mm.DV_um:.2f} mm, "
        f"LR={mean_mm.LR_um:.2f} mm.",
        f"- Cross-session SD: AP={sd_um.AP_um:.0f} um, DV={sd_um.DV_um:.0f} um, LR={sd_um.LR_um:.0f} um.",
        f"- Median RMS perpendicular residual: {conv.residual_um.median():.0f} um "
        "(lines don't exactly meet, but come reasonably close for real skew lines through separate animals).",
        "",
        "## Where the convergence point sits relative to each probe's own recorded electrodes "
        "(median fraction along recorded span, 0=deepest unit, 1=surface entry)",
        "",
        by_letter.to_string(),
        "",
        "Negative values mean the convergence point is extrapolated slightly PAST the deepest recorded unit "
        "(deeper than any recorded electrode); values near 0 mean it lands almost exactly at the tip.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "ecephys_unit_id", "ecephys_probe_id", "ecephys_session_id", "name", "session_type",
        "probe_vertical_position", "ecephys_structure_acronym", *CCF_COLUMNS,
    ]
    units = pd.read_csv(UNITS, usecols=columns, low_memory=False).dropna(subset=list(CCF_COLUMNS))

    fits = per_probe_fits(units)
    conv, frac = build_convergence_table(fits)
    conv.to_csv(OUTPUT / "per_session_convergence_point.csv", index=False, float_format="%.2f")
    frac.to_csv(OUTPUT / "convergence_point_fraction_along_probe.csv", index=False, float_format="%.4f")

    make_figure(conv, frac, OUTPUT / "Figure_allen_probe_convergence_point.png")
    write_report(conv, frac, OUTPUT / "ALLEN_PROBE_CONVERGENCE_POINT.md")

    print(conv[["AP_um", "DV_um", "LR_um", "residual_um"]].describe())
    print(f"wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
