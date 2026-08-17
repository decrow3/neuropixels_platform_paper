#!/usr/bin/env python3
"""Summarize held-out benefits and failures of rotated RF fits."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 746083955
INPUT = ROOT / "artifacts" / "allen_population_rotated_rf" / f"session_{SESSION_ID}"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_population_rotation_summary" / f"session_{SESSION_ID}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visualization-html", type=Path, default=None)
    return parser.parse_args()


def bootstrap_interval(values, statistic, rng, repetitions=10000):
    values = np.asarray(values)
    boot = np.empty(repetitions)
    for index in range(repetitions):
        sample = values[rng.integers(0, len(values), len(values))]
        boot[index] = statistic(sample)
    return float(statistic(values)), *np.quantile(boot, [0.025, 0.975]).astype(float)


def make_summary(evaluation):
    rng = np.random.default_rng(746083955)
    rows = []
    for model in ("point", "aperture"):
        for group in ("V1", "HVA", "all"):
            local = evaluation.loc[evaluation["spatial_model"].eq(model)]
            if group != "all":
                local = local.loc[local["group"].eq(group)]
            clean = local.loc[~local["sigma_lower_bound"] & ~local["sigma_upper_bound"]]
            median, median_low, median_high = bootstrap_interval(
                local["rotation_test_gain"], np.median, rng
            )
            fraction, fraction_low, fraction_high = bootstrap_interval(
                local["rotation_test_gain"].gt(0).to_numpy(float), np.mean, rng
            )
            clean_median, clean_low, clean_high = bootstrap_interval(
                clean["rotation_test_gain"], np.median, rng
            )
            rows.append({
                "spatial_model": model,
                "group": group,
                "units": len(local),
                "median_rotation_test_gain": median,
                "median_gain_ci_low": median_low,
                "median_gain_ci_high": median_high,
                "fraction_helped": fraction,
                "fraction_helped_ci_low": fraction_low,
                "fraction_helped_ci_high": fraction_high,
                "clean_units": len(clean),
                "clean_median_rotation_test_gain": clean_median,
                "clean_median_gain_ci_low": clean_low,
                "clean_median_gain_ci_high": clean_high,
                "median_log2_area_change": local["rotation_log2_area_ratio"].median(),
            })
    return pd.DataFrame(rows)


def render(evaluation, summary, path, dark=False):
    if dark:
        plt.style.use("dark_background")
        colors = {"V1": "#60a5fa", "HVA": "#fb923c"}
        neutral = "#cbd5e1"
    else:
        plt.style.use("default")
        colors = {"V1": "#2563a6", "HVA": "#c44e22"}
        neutral = "#475569"
    model_markers = {"point": "o", "aperture": "^"}
    model_lines = {"point": "-", "aperture": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    wide = evaluation.pivot(index="ecephys_unit_id", columns="spatial_model", values="rotation_test_gain")
    metadata = evaluation.drop_duplicates("ecephys_unit_id").set_index("ecephys_unit_id")
    limit = max(np.abs(wide.to_numpy()).max() * 1.08, 0.001)
    for group in ("V1", "HVA"):
        ids = wide.index[metadata.loc[wide.index, "group"].eq(group)]
        axes[0, 0].scatter(
            wide.loc[ids, "point"], wide.loc[ids, "aperture"], s=19,
            alpha=0.62, color=colors[group], marker="o", label=group,
        )
    axes[0, 0].plot([-limit, limit], [-limit, limit], linestyle="--", color=neutral, linewidth=1)
    axes[0, 0].axhline(0, color=neutral, linewidth=0.7, alpha=0.6)
    axes[0, 0].axvline(0, color=neutral, linewidth=0.7, alpha=0.6)
    axes[0, 0].set(xscale="symlog", yscale="symlog", xlim=(-limit, limit), ylim=(-limit, limit),
                   xlabel="Point-model held-out gain", ylabel="Aperture-model held-out gain",
                   title="Point and aperture models usually agree")
    axes[0, 0].legend(frameon=False)

    for model in ("point", "aperture"):
        for group in ("V1", "HVA"):
            values = np.sort(evaluation.loc[
                evaluation["spatial_model"].eq(model) & evaluation["group"].eq(group),
                "rotation_test_gain",
            ].to_numpy())
            cumulative = np.arange(1, len(values) + 1) / len(values)
            axes[0, 1].plot(
                values, cumulative, color=colors[group], linestyle=model_lines[model],
                linewidth=1.7, label=f"{group} · {model}",
            )
    axes[0, 1].axvline(0, color=neutral, linewidth=1)
    axes[0, 1].set(xscale="symlog", xlabel="Held-out deviance gain from rotation",
                   ylabel="Cumulative fraction of units", title="Rotation helps some units and hurts others")
    axes[0, 1].legend(frameon=False, fontsize=8)

    positions, labels = [], []
    for model_index, model in enumerate(("point", "aperture")):
        for group_index, group in enumerate(("V1", "HVA")):
            row = summary.loc[
                summary["spatial_model"].eq(model) & summary["group"].eq(group)
            ].iloc[0]
            position = model_index * 3 + group_index
            positions.append(position)
            labels.append(f"{model}\n{group}")
            axes[1, 0].errorbar(
                position, row["fraction_helped"],
                yerr=[[row["fraction_helped"] - row["fraction_helped_ci_low"]],
                      [row["fraction_helped_ci_high"] - row["fraction_helped"]]],
                fmt=model_markers[model], color=colors[group], capsize=4, markersize=7,
            )
    axes[1, 0].axhline(0.5, color=neutral, linestyle="--", linewidth=1)
    axes[1, 0].set(xticks=positions, xticklabels=labels, ylim=(0.2, 0.85),
                   ylabel="Fraction with positive held-out gain",
                   title="Bootstrap 95% intervals include modest effects")

    informative = evaluation.loc[
        evaluation["spatial_model"].eq("aperture")
        & evaluation["axis_ratio"].ge(1.2)
        & ~evaluation["sigma_lower_bound"]
        & ~evaluation["sigma_upper_bound"]
    ].copy()
    informative["absolute_angle_deg"] = informative["major_axis_angle_deg"].abs()
    for group in ("V1", "HVA"):
        local = informative.loc[informative["group"].eq(group)]
        axes[1, 1].scatter(
            local["absolute_angle_deg"], local["rotation_test_gain"],
            s=20, alpha=0.62, color=colors[group], label=group,
        )
    axes[1, 1].axhline(0, color=neutral, linewidth=1)
    axes[1, 1].set(yscale="symlog", xlim=(0, 90), xlabel="Absolute major-axis tilt (deg)",
                   ylabel="Aperture held-out gain", title="Large fitted tilts are not uniformly beneficial")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        f"Session {SESSION_ID}: does a free RF rotation generalize? · evaluation units only",
        fontsize=14,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_html(light_path, dark_path, html_path):
    light = base64.b64encode(light_path.read_bytes()).decode("ascii")
    dark = base64.b64encode(dark_path.read_bytes()).decode("ascii")
    fragment = f'''<div id="allen-population-rotation-summary">
  <h2>Population effect of freely rotated RF fits</h2>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{dark}" />
    <img src="data:image/png;base64,{light}" alt="Four population plots compare held-out deviance gains from rotated point and aperture RF fits across 160 evaluation units, including model agreement, cumulative gain distributions, bootstrap fractions helped, and aperture gain versus fitted angle." />
  </picture>
</div>
<style>
#allen-population-rotation-summary {{ width: 100%; background: transparent; color: var(--foreground); }}
#allen-population-rotation-summary img {{ display: block; width: 100%; height: auto; }}
</style>
'''
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(fragment, encoding="utf-8")


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(INPUT / "rotated_population_fits.csv", low_memory=False)
    evaluation = table.loc[table["unit_split"].eq("evaluation")].copy()
    summary = make_summary(evaluation)
    summary.to_csv(output / "rotation_bootstrap_summary.csv", index=False, float_format="%.9g")
    light = output / "population_rotation_summary.png"
    dark = output / "population_rotation_summary_dark.png"
    render(evaluation, summary, light, dark=False)
    render(evaluation, summary, dark, dark=True)
    if args.visualization_html is not None:
        write_html(light, dark, args.visualization_html.resolve())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
