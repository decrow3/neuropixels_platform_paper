#!/usr/bin/env python3
"""Check and correct a dynamic-range mismatch between the Zhuang atlas fields and the naive
pooled RF map, before any individual-session fitting is attempted against the atlas.

Direct comparison of offset-corrected naive azimuth/elevation against the domain-patched
Zhuang smoothed fields:

  axis       naive IQR   Zhuang IQR   IQR ratio   naive std   Zhuang std   std ratio
  azimuth    ~27.1       ~27.4        ~0.99       23.8        21.1         1.13
  elevation  ~21.3       ~17.2        ~1.24       19.5        13.6         1.43

Azimuth's *core* spread (IQR, robust to outlier units) already matches almost exactly --
the std/p5-p95 gap there is attributable to a heavier naive tail (noisy RF fits), not a true
range mismatch, so azimuth is left unscaled (gain=1). Elevation's IQR, std, and p5-p95 ratios
all agree elevation is compressed by roughly 1.2-1.45x, consistent with the previously
diagnosed dynamic-range saturation + widefield blur (Zhuang -25/+30 vs. real data's much wider
span). We use the IQR ratio specifically (not std or p5-p95) because it is the most robust to
outliers AND -- per the earlier finding that blur mainly "rounds off the tips" of the map --
the interior of the map is expected to be the more faithful part to calibrate against; matching
the already-known-to-diverge tails would overcorrect.

The rescale is applied as a linear gain anchored at the Zhuang value at the V1 seed pixel
(row=240, col=200; same point `render_naive_map_over_zhuang_rough_bbox.py` and
`fit_translation_rotation_naive_to_zhuang.py` already use as the placement/offset anchor), so
composing this with the existing offset fit leaves the V1 anchor point unchanged and only
stretches the rest of the map outward/inward to match naive's dynamic range:

    rescaled = anchor_value + gain * (field - anchor_value)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
ZHUANG_SMOOTH = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched.npz"
)
FIT_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
V1_SEED_ROW_COL = (240, 200)


def robust_span(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    q1, q3 = np.percentile(values, [25, 75])
    p5, p95 = np.percentile(values, [5, 95])
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
        "iqr": float(q3 - q1),
        "p5_95_span": float(p95 - p5),
        "median": float(np.median(values)),
    }


def main() -> None:
    fit = json.loads(FIT_MANIFEST.read_text())
    offset_az, offset_el = fit["fitted_rf_offset_deg"]

    cells = pd.read_csv(NAIVE_CELLS)
    naive_az = (cells["normalized_rf_x"] + offset_az).to_numpy(float)
    naive_el = (cells["normalized_rf_y"] + offset_el).to_numpy(float)

    smoothed = {k: v for k, v in np.load(ZHUANG_SMOOTH).items()}
    zhuang_az = smoothed["azimuth_smoothed_for_gradient_deg"]
    zhuang_el = smoothed["elevation_smoothed_for_gradient_deg"]

    naive_az_stats = robust_span(naive_az)
    naive_el_stats = robust_span(naive_el)
    zhuang_az_stats = robust_span(zhuang_az)
    zhuang_el_stats = robust_span(zhuang_el)

    gain_az = naive_az_stats["iqr"] / zhuang_az_stats["iqr"]
    gain_el = naive_el_stats["iqr"] / zhuang_el_stats["iqr"]
    # Azimuth's IQR already matches within a few percent -- treat as unscaled rather than
    # chasing noise; only apply a gain when the IQR mismatch is large enough to matter.
    applied_gain_az = gain_az if abs(gain_az - 1.0) > 0.10 else 1.0
    applied_gain_el = gain_el

    row, col = V1_SEED_ROW_COL
    anchor_az = float(zhuang_az[row, col])
    anchor_el = float(zhuang_el[row, col])

    rescaled_az = np.where(np.isfinite(zhuang_az), anchor_az + applied_gain_az * (zhuang_az - anchor_az), np.nan)
    rescaled_el = np.where(np.isfinite(zhuang_el), anchor_el + applied_gain_el * (zhuang_el - anchor_el), np.nan)

    rescaled_az_stats = robust_span(rescaled_az)
    rescaled_el_stats = robust_span(rescaled_el)

    print("azimuth: naive vs. Zhuang (raw) vs. Zhuang (span-matched)")
    for label, stats in (("naive", naive_az_stats), ("zhuang_raw", zhuang_az_stats), ("zhuang_span_matched", rescaled_az_stats)):
        print(f"  {label:20s} n={stats['n']:6d} min={stats['min']:6.1f} max={stats['max']:6.1f} "
              f"median={stats['median']:6.1f} iqr={stats['iqr']:5.1f} std={stats['std']:5.1f} p5_95={stats['p5_95_span']:5.1f}")
    print(f"  IQR ratio (naive/zhuang_raw) = {gain_az:.3f}; applied gain = {applied_gain_az:.3f}")

    print("\nelevation: naive vs. Zhuang (raw) vs. Zhuang (span-matched)")
    for label, stats in (("naive", naive_el_stats), ("zhuang_raw", zhuang_el_stats), ("zhuang_span_matched", rescaled_el_stats)):
        print(f"  {label:20s} n={stats['n']:6d} min={stats['min']:6.1f} max={stats['max']:6.1f} "
              f"median={stats['median']:6.1f} iqr={stats['iqr']:5.1f} std={stats['std']:5.1f} p5_95={stats['p5_95_span']:5.1f}")
    print(f"  IQR ratio (naive/zhuang_raw) = {gain_el:.3f}; applied gain = {applied_gain_el:.3f}")
    print(f"\nanchor (V1 seed row={row}, col={col}): azimuth={anchor_az:.1f} deg, elevation={anchor_el:.1f} deg")

    smoothed["azimuth_span_matched_deg"] = rescaled_az.astype(np.float32)
    smoothed["elevation_span_matched_deg"] = rescaled_el.astype(np.float32)
    output_npz = OUTPUT / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
    np.savez_compressed(output_npz, **smoothed)
    print(f"\nwrote {output_npz}")

    manifest = {
        "source_domain_patched_field": str(ZHUANG_SMOOTH),
        "v1_seed_row_col": list(V1_SEED_ROW_COL),
        "gain_metric": "IQR ratio (naive offset-corrected / Zhuang raw), robust to outlier tails",
        "azimuth": {
            "iqr_ratio": gain_az, "applied_gain": applied_gain_az, "anchor_deg": anchor_az,
            "naive": naive_az_stats, "zhuang_raw": zhuang_az_stats, "zhuang_span_matched": rescaled_az_stats,
        },
        "elevation": {
            "iqr_ratio": gain_el, "applied_gain": applied_gain_el, "anchor_deg": anchor_el,
            "naive": naive_el_stats, "zhuang_raw": zhuang_el_stats, "zhuang_span_matched": rescaled_el_stats,
        },
        "note": (
            "Azimuth IQR already matched (~1.0) so left unscaled; azimuth's larger std/p5-p95 "
            "ratio is attributed to a heavier naive outlier tail, not a true range mismatch. "
            "Elevation gain applied via IQR ratio specifically (not std or p5-p95) because the "
            "map's core is the more faithful part to calibrate against -- the tails are already "
            "expected to diverge from widefield blur rounding off the extremes."
        ),
    }
    manifest_path = OUTPUT / "domain_patched_span_match_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, naive_vals, zhuang_raw_vals, zhuang_rescaled_vals, title in (
        (axes[0], naive_az, zhuang_az, rescaled_az, "Azimuth (deg)"),
        (axes[1], naive_el, zhuang_el, rescaled_el, "Elevation (deg)"),
    ):
        naive_vals = np.asarray(naive_vals, dtype=np.float64)
        naive_vals = naive_vals[np.isfinite(naive_vals)]
        zhuang_raw_vals = np.asarray(zhuang_raw_vals, dtype=np.float64)
        zhuang_raw_vals = zhuang_raw_vals[np.isfinite(zhuang_raw_vals)]
        zhuang_rescaled_vals = np.asarray(zhuang_rescaled_vals, dtype=np.float64)
        zhuang_rescaled_vals = zhuang_rescaled_vals[np.isfinite(zhuang_rescaled_vals)]
        bins = np.linspace(
            min(naive_vals.min(), zhuang_raw_vals.min()),
            max(naive_vals.max(), zhuang_raw_vals.max()),
            60,
        )
        ax.hist(naive_vals, bins=bins, density=True, histtype="step", linewidth=1.8, color="#222222", label="naive (offset-corrected)")
        ax.hist(zhuang_raw_vals, bins=bins, density=True, histtype="step", linewidth=1.4, color="#b33f62", label="Zhuang (raw)")
        ax.hist(zhuang_rescaled_vals, bins=bins, density=True, histtype="step", linewidth=1.4, linestyle="--", color="#2864a8", label="Zhuang (span-matched)")
        ax.set(title=title, xlabel="deg", ylabel="density")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Naive vs. Zhuang value-span comparison (before individual-session fitting)", fontsize=12)
    figure_path = OUTPUT / "Figure_naive_vs_zhuang_span_comparison.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
