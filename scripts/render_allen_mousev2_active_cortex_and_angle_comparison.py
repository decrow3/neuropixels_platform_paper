#!/usr/bin/env python3
"""Compare active-cortex depth span and estimated insertion angle: Allen V1 vs. MouseV2.

Reuses two already-computed pipelines rather than recomputing them:
  - Allen VISp probes: RF-significant-unit depth span along the probe (`compare_rf_depth_span_
    mousev2_vs_allen.py`, artifacts/figure3/06h.../allen_rf_depth_span.csv) as the "active cortex"
    proxy, and the DIRECT CCF-trace angle from vertical restricted to VISp-primary probes
    (`compute_allen_probe_insertion_angle_from_ccf.py`, artifacts/figure3/06k.../
    allen_probe_insertion_angle_from_ccf.csv) -- a real geometric measurement.
  - MouseV2 probes: the same RF-significant-unit depth span metric (MouseV2 has no CCF, so this
    is the best available "active cortex" proxy) and the ratio-derived INDIRECT angle estimate
    from that same script (artifacts/figure3/06h.../mousev2_rf_depth_span.csv) -- inferred from
    span ratio vs. the Allen reference under a symmetric-ratio assumption, not measured directly.

Allen's angle here is restricted to VISp-targeting (probeC) probes specifically, not the full
6-probe/6-area population, because MouseV2's 4 probes all target V1 itself -- the fair comparison
is Allen's own V1 probe geometry, not its population average across V1 and 5 HVAs.

The angle panel is explicitly a SCHEMATIC: it draws each group's median +/- IQR as an angular
wedge from a shared vertical reference, to make the two numbers visually comparable. It is not a
literal trajectory reconstruction (unlike the earlier true-3D Allen-only figures), and the
MouseV2 wedge is visually distinguished (dashed, hatched) from Allen's solid wedge to keep the
direct-measurement vs. indirect-estimate distinction visible in the figure itself, not just the
caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALLEN_ANGLE_CSV = ROOT / "artifacts/figure3/06k_allen_probe_insertion_angle_from_ccf/allen_probe_insertion_angle_from_ccf.csv"
RF_SPAN_DIR = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle"
ALLEN_SPAN_CSV = RF_SPAN_DIR / "allen_rf_depth_span.csv"
MOUSEV2_SPAN_CSV = RF_SPAN_DIR / "mousev2_rf_depth_span.csv"
OUTPUT = ROOT / "artifacts/figure3/06o_allen_mousev2_active_cortex_and_angle_comparison"

ALLEN_COLOR = "#4575b4"
MOUSEV2_COLOR = "#d73027"
ALLEN_LABEL = "'Brain Observatory 1.1 & Functional Connectivity'"
MOUSEV2_LABEL = "'MouseV2'"


def load_data():
    allen_angle = pd.read_csv(ALLEN_ANGLE_CSV)
    allen_visp_angle = allen_angle.loc[allen_angle.primary_structure.eq("VISp"), "angle_from_vertical_deg"]

    allen_span = pd.read_csv(ALLEN_SPAN_CSV)["depth_span_um"]
    mousev2 = pd.read_csv(MOUSEV2_SPAN_CSV)
    mousev2_span = mousev2["depth_span_um"]
    mousev2_angle = mousev2["estimated_angle_from_vertical_deg"]

    return allen_visp_angle, allen_span, mousev2_span, mousev2_angle


def make_span_panel(ax, allen_span: pd.Series, mousev2_span: pd.Series) -> None:
    bins = np.linspace(0, max(allen_span.max(), mousev2_span.max()) * 1.05, 22)
    ax.hist(allen_span, bins=bins, alpha=0.55, density=True, color=ALLEN_COLOR,
            label=f"{ALLEN_LABEL} V1 (n={len(allen_span)} probes)")
    ax.hist(mousev2_span, bins=bins, alpha=0.55, density=True, color=MOUSEV2_COLOR,
            label=f"{MOUSEV2_LABEL} (n={len(mousev2_span)} probes)")
    ax.axvline(allen_span.median(), color=ALLEN_COLOR, linestyle="--", linewidth=1.5)
    ax.axvline(mousev2_span.median(), color=MOUSEV2_COLOR, linestyle="--", linewidth=1.5)
    ax.set(xlabel="along-probe span of RF-significant (\"active\") units, 5-95th pct (um)",
           ylabel="density",
           title="Active cortex sites along probe\n(RF-significant-unit depth span, same criterion both datasets)")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.2)


def _draw_angle_wedge(ax, angle_center_deg: float, angle_lo_deg: float, angle_hi_deg: float,
                       color: str, label: str, hatch: str | None, linestyle: str) -> None:
    length = 1.0
    theta_center = np.radians(90 - angle_center_deg)
    ax.plot([0, length * np.cos(theta_center)], [0, length * np.sin(theta_center)],
            color=color, linewidth=2.5, linestyle=linestyle, zorder=4,
            label=f"{label}: median {angle_center_deg:.0f}° (IQR {angle_lo_deg:.0f}-{angle_hi_deg:.0f}°)")
    thetas = np.radians(90 - np.linspace(angle_lo_deg, angle_hi_deg, 40))
    wedge_x = np.concatenate([[0], length * np.cos(thetas), [0]])
    wedge_y = np.concatenate([[0], length * np.sin(thetas), [0]])
    ax.fill(wedge_x, wedge_y, color=color, alpha=0.18, hatch=hatch, zorder=2, edgecolor=color, linewidth=0.5)


def make_angle_schematic(ax, allen_visp_angle: pd.Series, mousev2_angle: pd.Series) -> None:
    ax.plot([0, 0], [0, 1.05], color="gray", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(0.01, 1.06, "vertical", fontsize=8, color="gray", ha="left")

    _draw_angle_wedge(
        ax, allen_visp_angle.median(), allen_visp_angle.quantile(0.25), allen_visp_angle.quantile(0.75),
        ALLEN_COLOR, f"{ALLEN_LABEL} V1 (probeC): DIRECT CCF measurement", hatch=None, linestyle="-",
    )
    _draw_angle_wedge(
        ax, mousev2_angle.median(), mousev2_angle.quantile(0.25), mousev2_angle.quantile(0.75),
        MOUSEV2_COLOR, f"{MOUSEV2_LABEL}: INDIRECT RF-span-ratio estimate", hatch="//", linestyle="--",
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8.5, frameon=False)
    ax.set_title(
        "Schematic: estimated insertion angle from vertical\n(wedge = IQR across probes; NOT a literal trajectory)",
        fontsize=11,
    )


def write_report(allen_visp_angle, allen_span, mousev2_span, mousev2_angle, output_path: Path) -> None:
    lines = [
        f"# {ALLEN_LABEL} vs. {MOUSEV2_LABEL}: active cortex span and estimated insertion angle",
        "",
        "## Active cortex sites along probe (RF-significant-unit depth span, 5-95th pct)",
        "",
        f"- {ALLEN_LABEL} V1: median {allen_span.median():.0f} um (IQR {allen_span.quantile(.25):.0f}-"
        f"{allen_span.quantile(.75):.0f}), n={len(allen_span)} probes.",
        f"- {MOUSEV2_LABEL}: median {mousev2_span.median():.0f} um (IQR {mousev2_span.quantile(.25):.0f}-"
        f"{mousev2_span.quantile(.75):.0f}), n={len(mousev2_span)} probes.",
        "",
        "## Estimated insertion angle from vertical",
        "",
        f"- {ALLEN_LABEL} V1 (probeC only, DIRECT measurement from per-unit CCF trace): median "
        f"{allen_visp_angle.median():.1f} deg (IQR {allen_visp_angle.quantile(.25):.1f}-"
        f"{allen_visp_angle.quantile(.75):.1f}), n={len(allen_visp_angle)} probes.",
        f"- {MOUSEV2_LABEL} (INDIRECT estimate from RF-significant depth-span ratio vs. Allen reference, "
        "symmetric ratio assumption): median "
        f"{mousev2_angle.median():.1f} deg (IQR {mousev2_angle.quantile(.25):.1f}-"
        f"{mousev2_angle.quantile(.75):.1f}), n={len(mousev2_angle)} probes.",
        "",
        "## Caveats",
        "",
        "- The two angle numbers are NOT the same kind of measurement: Allen's comes directly from "
        "real per-unit CCF coordinates; MouseV2's is inferred from how much longer its active-cortex "
        "span is than Allen's, which conflates true insertion angle with any other reason the spans "
        "might differ (unmatched RF-significance yield, registration/QC asymmetries -- see "
        "`compare_rf_depth_span_mousev2_vs_allen.py` docstring).",
        "- MouseV2's larger active-cortex span is consistent with, but does not on its own prove, a "
        "larger angle from vertical (a more oblique/shallower insertion, in the conventional sense where "
        "'steep' means close to vertical) -- it is also consistent with MouseV2 simply retaining more "
        "RF-significant units per unit of true cortical thickness.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    allen_visp_angle, allen_span, mousev2_span, mousev2_angle = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    make_span_panel(axes[0], allen_span, mousev2_span)
    make_angle_schematic(axes[1], allen_visp_angle, mousev2_angle)
    fig.suptitle(f"{ALLEN_LABEL} vs. {MOUSEV2_LABEL}:\nactive cortex span and estimated insertion angle",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUTPUT / "Figure_allen_mousev2_active_cortex_and_angle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    write_report(allen_visp_angle, allen_span, mousev2_span, mousev2_angle,
                 OUTPUT / "ALLEN_MOUSEV2_COMPARISON.md")
    print(f"wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
