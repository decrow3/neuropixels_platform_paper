"""Build companion C/D/E figures at neuron and session-mean resolution.

Run: MPLCONFIGDIR=/tmp/mpl-figure4-cells MPLBACKEND=Agg python scripts/build_figure4_cell_location_controls.py
The existing candidate is never overwritten. Cell estimates weight neurons equally;
bootstrap and shuffle units remain complete sessions and location blocks.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_figure4_variant_review as style
from scripts.figure3_robust_spread_comparison import ALLEN_HARMONIZED_F1_F0, omega_squared, session_group_table
from common.figure3_mousev2 import load_allen_units

OUT = ROOT / "Figure3"
KEYS = ["dataset", "metric", "session_id", "group"]
CELL_KEYS = KEYS + ["unit_id"]
TTFS_AUDIT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance/figure3_response_filtered_ttfs_all_areas_unit_audit.csv"


def collapse_cells(raw):
    """One observation per neuron, averaging its available valid fit draws."""
    if raw.duplicated(CELL_KEYS + ["draw_id"]).any():
        raise ValueError("Duplicate neuron/draw input")
    if not np.isfinite(raw.value).all():
        raise ValueError("Nonfinite cell measurement")
    return raw.groupby(CELL_KEYS, as_index=False).agg(
        value=("value", "mean"), n_valid_fits=("value", "size")
    )


def load_allen_v1(reference):
    ttfs = pd.read_csv(TTFS_AUDIT, dtype={"session_id": str, "unit_id": str})
    ttfs = ttfs.loc[ttfs.area_coarse.eq("V1") & ttfs.selected_positive_responder_area
                    & ttfs.preferred_0_250_ttfs_ms.lt(100)].copy()
    ttfs = ttfs.rename(columns={"preferred_0_250_ttfs_ms": "value"})
    ttfs["metric"] = style.METRICS[0]
    f1 = pd.read_csv(ALLEN_HARMONIZED_F1_F0,
                     dtype={"ecephys_session_id": str, "ecephys_unit_id": str})
    f1 = f1.loc[f1.area_coarse.eq("V1") & f1.f1_f0_dg_harmonized.gt(0)].copy()
    f1["value"] = np.log10(f1.f1_f0_dg_harmonized)
    f1["metric"] = style.METRICS[1]
    tau = load_allen_units(population_profile="common_qc")
    valid = tau.area_coarse.eq("V1") & tau.timescale_ac.between(1, 300)
    valid &= tau.spike_count_ac.gt(50) & tau.err_ac.lt(20)
    tau = tau.loc[valid].copy()
    tau["value"] = tau.timescale_ac
    tau["metric"] = style.METRICS[2]
    tables = []
    for table in (ttfs, f1, tau):
        table = table.rename(columns={"ecephys_session_id": "session_id",
                                      "ecephys_unit_id": "unit_id"})
        table["session_id"] = table.session_id.astype(str)
        table["unit_id"] = table.unit_id.astype(str)
        tables.append(table[["metric", "session_id", "unit_id", "value"]])
    cells = pd.concat(tables, ignore_index=True)
    ref = reference.copy()
    ref["session_id"] = ref.session_id.astype(str)
    matched = pd.read_csv(OUT / "Figure4_location_controls_inputs.csv", dtype={"session_id": str})
    matched = matched.loc[matched.comparison.eq("control"), ["metric", "session_id"]].drop_duplicates()
    ref = ref.merge(matched, on=["metric", "session_id"], validate="one_to_one")
    cells = cells.merge(ref[["metric", "session_id"]], on=["metric", "session_id"],
                        validate="many_to_one")
    counts = cells.groupby(["metric", "session_id"]).value.transform("size")
    cells = cells.loc[counts.ge(np.where(cells.metric.eq("TTFS (ms)"), 10, 5))].copy()
    rebuilt = cells.groupby(["metric", "session_id"], as_index=False).agg(
        rebuilt_mean=("value", "mean"), rebuilt_n=("unit_id", "size"))
    audit = ref.merge(rebuilt, on=["metric", "session_id"], how="outer", validate="one_to_one", indicator=True)
    # The frozen reference is legacy, unfiltered Allen data, whereas the current
    # HVA inputs use common QC and response-filtered TTFS / harmonized F1/F0.
    # Export the discrepancy; do not pretend this is a pure resolution change.
    audit["retained_current_processing"] = audit._merge.eq("both")
    audit["mean_change"] = audit.rebuilt_mean - audit["mean"]
    audit["count_change"] = audit.rebuilt_n - audit.n_units
    legacy_units = load_allen_units()
    legacy = []
    for index, (column, metric) in enumerate(zip(
        ["time_to_first_spike_fl", "f1_f0_dg", "timescale_ac"], style.METRICS
    )):
        legacy.append(session_group_table(
            legacy_units, dataset="Allen-V1", session_column="ecephys_session_id",
            group_column="area_coarse", groups=["V1"], metric=column,
            metric_label=metric, metric_index=index, min_units=5))
    legacy = pd.concat(legacy, ignore_index=True)
    legacy["session_id"] = legacy.session_id.astype(str)
    legacy = legacy[["metric", "session_id", "mean", "n_units"]].rename(
        columns={"mean": "legacy_rebuilt_mean", "n_units": "legacy_rebuilt_n"})
    audit = audit.merge(legacy, on=["metric", "session_id"], how="left", validate="one_to_one")
    np.testing.assert_allclose(audit["mean"], audit.legacy_rebuilt_mean, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(audit.n_units, audit.legacy_rebuilt_n)
    cells["dataset"], cells["group"], cells["n_valid_fits"] = "Allen-V1", "V1", 1
    assert not cells.duplicated(CELL_KEYS).any()
    return cells, audit.drop(columns="_merge")


def cell_inputs():
    groups, allen_ref, _ = style.load_inputs()
    groups, raw, _ = style.apply_anatomical_v1_filter(groups, style.load_figure_cells())
    # Retain exactly the session/location support used by the current C/D/E.
    raw = raw.merge(groups[KEYS], on=KEYS, validate="many_to_one")
    cells = collapse_cells(raw)
    allen, audit = load_allen_v1(allen_ref)
    # HVA cell means and counts must reconstruct the current session summaries.
    hva = cells.loc[cells.dataset.eq("Post-V1")]
    rebuilt = hva.groupby(KEYS, as_index=False).agg(rebuilt_mean=("value", "mean"), rebuilt_n=("unit_id", "size"))
    check = groups.loc[groups.dataset.eq("Post-V1")].merge(rebuilt, on=KEYS, how="outer", indicator=True)
    assert check._merge.eq("both").all()
    np.testing.assert_allclose(check["mean"], check.rebuilt_mean, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(check.n_units, check.rebuilt_n)
    comparisons = [cells.assign(comparison=np.where(cells.dataset.eq("Within-V1"), "v1", "hva"))]
    for metric in style.METRICS:
        v = allen.loc[allen.metric.eq(metric)]
        h = hva.loc[hva.metric.eq(metric)]
        matched = set(v.session_id) & set(h.session_id)
        control = pd.concat([v.loc[v.session_id.isin(matched)].assign(group="v1"),
                             h.loc[h.session_id.isin(matched)].assign(group="hva")])
        comparisons.append(control.assign(comparison="control"))
    return pd.concat(comparisons, ignore_index=True), audit


def sufficient_blocks(table):
    """Keep N, sum(y), sum(y²) for each session/location, without averaging cells."""
    names = sorted(table.group.unique())
    moments = table.assign(squared=table.value ** 2).groupby(["session_id", "group"], as_index=False).agg(
        n=("value", "size"), total=("value", "sum"), squares=("squared", "sum"))
    blocks = []
    for _, part in moments.groupby("session_id", sort=True):
        block = np.zeros((len(names), 3))
        for row in part.itertuples():
            block[names.index(row.group)] = [row.n, row.total, row.squares]
        blocks.append(block)
    return np.asarray(blocks)


def omega_moments(moments):
    """Vectorized original one-way omega², exactly matching expanded cell rows."""
    n, total, squares = np.moveaxis(np.asarray(moments), -1, 0)
    keep = n >= 2
    n, total, squares = (np.where(keep, a, 0) for a in (n, total, squares))
    count, ngroups = n.sum(axis=-1), keep.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = total.sum(axis=-1) ** 2 / count
        ss_total = squares.sum(axis=-1) - correction
        ss_between = np.divide(total ** 2, n, out=np.zeros_like(total), where=n > 0).sum(axis=-1) - correction
        ms_within = np.maximum(0, ss_total - ss_between) / (count - ngroups)
        omega = (ss_between - (ngroups - 1) * ms_within) / (ss_total + ms_within)
    return np.where((ngroups >= 2) & (count > ngroups) & (ss_total > 0), omega, np.nan)


def analyze(table, n_draws, seed):
    rng = np.random.default_rng(seed)
    records, draws_out = [], []
    for metric in style.METRICS:
        bootstraps, nulls, observed = {}, {}, {}
        for comparison in ("v1", "hva", "control"):
            local = table.loc[table.metric.eq(metric) & table.comparison.eq(comparison)]
            blocks = sufficient_blocks(local)
            point = float(omega_moments(blocks.sum(axis=0)))
            np.testing.assert_allclose(point, omega_squared(local.group.to_numpy(), local.value.to_numpy()), atol=1e-12)
            selection = rng.integers(0, len(blocks), size=(n_draws, len(blocks)))
            bootstrap = omega_moments(blocks[selection].sum(axis=1))
            null_moments = np.zeros((n_draws, blocks.shape[1], 3))
            for block in blocks:
                available = np.flatnonzero(block[:, 0])
                # Every neuron from a session/location moves together. Labels
                # absent from a session stay absent; one-location sessions stay fixed.
                permutations = np.argsort(rng.random((n_draws, len(available))), axis=1)
                null_moments[:, available, :] += block[available][permutations]
            null = omega_moments(null_moments)
            if not np.isfinite(bootstrap).all() or not np.isfinite(null).all():
                raise ValueError("Degenerate bootstrap or shuffle replicate")
            bootstraps[comparison], nulls[comparison], observed[comparison] = bootstrap, null, point
            records.append(dict(metric=metric, comparison=comparison, observed=point,
                                bootstrap_low=np.quantile(bootstrap, .025), bootstrap_high=np.quantile(bootstrap, .975),
                                shuffle_median=np.median(null), shuffle_low=np.quantile(null, .025), shuffle_high=np.quantile(null, .975),
                                n_sessions=local.session_id.nunique(), n_observations=len(local),
                                permutation_p_greater=(1 + (null >= point).sum()) / (n_draws + 1)))
        delta = bootstraps["hva"] - bootstraps["v1"]
        delta_null = nulls["hva"] - nulls["v1"]
        records.append(dict(metric=metric, comparison="delta", observed=observed["hva"] - observed["v1"],
                            bootstrap_low=np.quantile(delta, .025), bootstrap_high=np.quantile(delta, .975),
                            shuffle_median=np.median(delta_null), shuffle_low=np.quantile(delta_null, .025),
                            shuffle_high=np.quantile(delta_null, .975), n_sessions=np.nan, n_observations=np.nan,
                            permutation_p_greater=np.nan))
        for comparison in bootstraps:
            draws_out.append(pd.DataFrame(dict(metric=metric, comparison=comparison, replicate=np.arange(n_draws),
                                              bootstrap=bootstraps[comparison], shuffle=nulls[comparison])))
        draws_out.append(pd.DataFrame(dict(metric=metric, comparison="delta", replicate=np.arange(n_draws),
                                          bootstrap=delta, shuffle=delta_null)))
    return pd.DataFrame(records), pd.concat(draws_out, ignore_index=True)


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.grid(axis="y", color=style.GRID, lw=.7)
    ax.set_axisbelow(True)


def distribution(ax, table, order, colors, limits, cell_level, seed):
    rng = np.random.default_rng(seed)
    for x, group in enumerate(order):
        part = table.loc[table.group.eq(group)]
        values = part.value.to_numpy()
        grid = np.linspace(values.min(), values.max(), 240)
        density = gaussian_kde(values)(grid)
        width = .32 * density / density.max()
        ax.fill_betweenx(grid, x, x + width, color=colors[group], alpha=.18, lw=.8)
        ax.scatter(x + rng.uniform(-.23, -.025, len(values)), values, s=3 if cell_level else 16,
                   color=colors[group], alpha=.22 if cell_level else .65, linewidth=0, rasterized=cell_level)
        ax.plot([x - .16, x + .2], [values.mean()] * 2, color=colors[group], lw=2.6)
        label = f"{len(values):,} / {part.session_id.nunique()}" if cell_level else str(len(values))
        ax.text(x, .99, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.6, color=style.MUTED)
    ax.set_xticks(range(len(order)), style.display_group_labels(order))
    if any(group in style.PROBE_ORDER for group in order):
        plt.setp(ax.get_xticklabels(), rotation=23, ha="right", rotation_mode="anchor")
    ax.set_xlim(-.5, len(order) - .5)
    ax.set_ylim(limits)
    clean_axis(ax)


def significance_label(record):
    if record.comparison == "delta":
        excludes_zero = record.bootstrap_low > 0 or record.bootstrap_high < 0
        return "*†" if excludes_zero else "n.s.†"
    p = record.permutation_p_greater
    if not np.isfinite(p):
        raise ValueError("Missing shuffle p-value for effect annotation")
    label = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    return label + ("‡" if getattr(record, "conditional_shuffle", False) else "")


def effect_axis(ax, record, color, limits, label, ylabel):
    for x, point, low, high, marker, c, fill in (
        (-.10, record.observed, record.bootstrap_low, record.bootstrap_high, "o", color, color),
        (.10, record.shuffle_median, record.shuffle_low, record.shuffle_high, "s", "#999999", "white"),
    ):
        ax.vlines(x, low * 100, high * 100, color=c, lw=1.8, zorder=3)
        ax.plot(x, point * 100, marker=marker, mfc=fill, mec=c, ms=5, zorder=4)
    ax.axhline(0, color="#858585", lw=1, ls="--")
    ax.set_xlim(-.4, .4)
    ax.set_ylim(limits)
    ax.set_xticks([0], [label])
    ax.set_ylabel(ylabel, fontsize=9)
    value_label = f"{record.observed * 100:+.1f}" if record.comparison == "delta" else f"{record.observed * 100:.1f}"
    ax.text(.5, .99, f"{value_label}  {significance_label(record)}",
            transform=ax.transAxes, ha="center", va="top", color=color, fontsize=9)
    clean_axis(ax)


def render(table, stats, prefix, cell_level, *, central_mode=None, rf_adjusted=False, rf_subset=False):
    fig = plt.figure(figsize=(16.25, 10.2))
    # Explicit gutters keep C/D spacing while giving E more separation.
    outer = fig.add_gridspec(3, 5, width_ratios=[1.45, .34, 1.60, .48, 1.03],
                             left=.052, right=.985, top=.85, bottom=.15 if central_mode == "include" else .125, wspace=0, hspace=.48)
    fig.suptitle("Functional variation across V1 locations and higher visual areas", x=.052, ha="left", y=.988, fontsize=17, fontweight="bold")
    subtitle = ("Single-cell version · one value per neuron · equal cell weights · uncertainty resampled by session"
                if cell_level else "Session-mean version · one value per session/location · uncertainty resampled by session")
    if central_mode:
        subtitle = "Single-cell version · older Allen V1 added as Central · equal cell weights · session bootstrap"
    if rf_adjusted:
        subtitle = "Single-cell version with Central · adjusted for RF azimuth/elevation · RF model refitted in every session bootstrap"
    elif rf_subset:
        subtitle = "Single-cell version with Central · RF-qualified neurons only · unadjusted companion on the same cells"
    fig.text(.052, .944, subtitle, fontsize=10.5, color=style.MUTED)
    handles = [Line2D([], [], marker="o", color="#444444", lw=1.5, label="Observed + 95% session bootstrap interval"),
               Line2D([], [], marker="s", mfc="white", color="#999999", lw=1.5, label="Location-block shuffle: median + central 95%")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.047, .929), frameon=False, ncol=2, fontsize=9)
    for row, metric in enumerate(style.METRICS):
        local = table.loc[table.metric.eq(metric)]
        recs = stats.loc[stats.metric.eq(metric)].set_index("comparison", drop=False)
        response = local.loc[local.comparison.isin(["v1", "hva"]), "value"]
        pad = (response.max() - response.min()) * .12
        limits = (response.min() - pad * .3, response.max() + pad)
        vals = stats.loc[stats.metric.eq(metric), ["observed", "bootstrap_low", "bootstrap_high", "shuffle_low", "shuffle_high"]].to_numpy() * 100
        effect_limits = (min(0, vals.min()) - 2, vals.max() + max(2, np.ptp(vals) * .19))
        v1_order = [*style.PROBE_ORDER[:2], "Central", *style.PROBE_ORDER[2:]] if central_mode else style.PROBE_ORDER
        v1_colors = {**style.PROBE_COLORS, "Central": "#555555"}
        for col, (comparison, order, colors, color, title) in enumerate([
            ("v1", v1_order, v1_colors, style.V1_COLOR, "C  Within V1"),
            ("hva", style.AREA_ORDER, style.AREA_COLORS, style.HVA_COLOR, "D  Across HVAs"),
        ]):
            sub = outer[row, col * 2].subgridspec(1, 2, width_ratios=[4.0, 1.0], wspace=.58)
            ax = fig.add_subplot(sub[0, 0])
            distribution(ax, local.loc[local.comparison.eq(comparison)], order, colors, limits, cell_level, 50 + row + col)
            if col == 0:
                ax.set_ylabel(("RF-adjusted " if rf_adjusted else "") + metric)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=20)
            ea = fig.add_subplot(sub[0, 1])
            effect_axis(ea, recs.loc[comparison], color, effect_limits, "Location\neffect", "Partial ω² after RF (%)" if rf_adjusted else "Variance explained (%)")
        sub = outer[row, 4].subgridspec(1, 2, wspace=.9)
        control = fig.add_subplot(sub[0, 0])
        effect_axis(control, recs.loc["control"], "#9a641e", effect_limits, "V1 vs\npooled HVAs", "Partial ω² after RF (%)" if rf_adjusted else "Variance explained (%)")
        delta = fig.add_subplot(sub[0, 1])
        effect_axis(delta, recs.loc["delta"], style.DELTA_COLOR, effect_limits, "Across HVAs\n− within V1", "Difference (percentage points)")
        if row == 0:
            control.set_title("E  Control and main comparison", loc="left", fontsize=12, fontweight="bold", pad=20)
    foot = ("C/D: points = neurons; bars = cell-weighted means; counts = cells / sessions. Repeated timescale fits are averaged within neuron.\n"
            "E control: matched Allen sessions; HVAs pooled with equal cell weights. Effects use bias-corrected ω²; negative estimates reflect the correction."
            if cell_level else
            "C/D: points = session/location means; bars = means across sessions; counts = sessions. Location effects use session-mean ω².\n"
            "E control: matched Allen V1 and equal-area HVA session means. Effects use bias-corrected ω²; negative estimates reflect the correction.")
    control_note = ("Cell control uses HVA-matched processing; the original session control uses legacy V1 inputs (see notes)."
                    if cell_level else "Original control retained: legacy V1 processing differs from the HVA processing (see notes).")
    if central_mode:
        control_note = ("Central = legacy Allen V1, a separate cohort/processing pipeline; "
                        + ("included in V1 effects and E differences." if central_mode == "include" else "reference only; excluded from effect estimates."))
    if rf_adjusted:
        foot = ("C/D: neuron measurements adjusted to the pooled RF-feature mean; bars = cell means; counts = cells / sessions.\n"
                "Quadratic display RF azimuth/elevation adjustment; no gaze correction. Effects = bias-corrected partial ω², which can be negative.")
    significance_note = ("Uncorrected one-sided shuffle tests: * p < .05; ** p < .01; *** p < .001; n.s. p ≥ .05.\n"
                         "† Difference: * = 95% bootstrap interval excludes zero; n.s. = includes zero (not an equivalence test or shuffle p-value).")
    if central_mode == "include":
        significance_note += "\n‡ V1 shuffle holds Central fixed and tests only the four MouseV2 labels. Shared Allen sessions are bootstrapped jointly."
    fig.text(.052, .057 if central_mode == "include" else .048, foot + "\n" + control_note + "\n" + significance_note, fontsize=8.3, va="center", color=style.MUTED, linespacing=1.5)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    cells, audit = cell_inputs()
    cell_prefix = OUT / "Figure4_CDE_single_cell_candidate"
    cells.to_csv(cell_prefix.with_name(cell_prefix.name + "_inputs.csv"), index=False)
    audit.to_csv(cell_prefix.with_name(cell_prefix.name + "_allen_v1_reconstruction.csv"), index=False)
    summaries = []
    for cell_level in (True, False):
        prefix = cell_prefix if cell_level else OUT / "Figure4_CDE_session_candidate"
        table = cells if cell_level else pd.read_csv(OUT / "Figure4_location_controls_inputs.csv", dtype={"session_id": str}).rename(columns={"mean": "value"})
        stats, draws = analyze(table, args.draws, args.seed)
        stats.to_csv(prefix.with_name(prefix.name + "_statistics.csv"), index=False)
        draws.to_csv(prefix.with_name(prefix.name + "_resamples.csv.gz"), index=False)
        render(table, stats, prefix, cell_level)
        summaries.append(stats.assign(resolution="single cell" if cell_level else "session mean"))
        print(stats[["metric", "comparison", "observed", "bootstrap_low", "bootstrap_high"]].to_string(index=False), flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / "Figure4_CDE_resolution_comparison.csv", index=False)
    metadata = dict(draws=args.draws, seed=args.seed, cell_weighting="equal neuron",
                    cell_timescale="arithmetic mean of available valid fits per neuron; no fit-draw resampling",
                    bootstrap="whole sessions, preserving all cells and locations; separate V1 and HVA resamples",
                    shuffle="permute whole location labels within each session; swap whole V1/HVA pools for control",
                    control_change="cell control uses common QC, response-filtered TTFS and harmonized F1/F0; session control preserves legacy inputs",
                    sources=[str(p.relative_to(ROOT)) for p in [style.TTFS_CELL_INPUT, style.EXTENSION_CELL_INPUT,
                             style.MOUSEV2_UNIT_CCF_LOCATIONS, TTFS_AUDIT, ALLEN_HARMONIZED_F1_F0,
                             ROOT / "data/unit_table.csv", OUT / "Figure4_location_controls_inputs.csv"]])
    (OUT / "Figure4_CDE_single_cell_method.json").write_text(json.dumps(metadata, indent=2) + "\n")
    notes = [
        "# Figure 4 C/D/E: single-cell companion and session-layout comparison",
        "",
        "Generated by `scripts/build_figure4_cell_location_controls.py`. The original candidate is unchanged.",
        "",
        "## Reading the layout",
        "C shows four V1 locations and their location effect. D shows five HVAs and their area-identity effect. E places the V1-versus-pooled-HVA control beside the HVA-minus-V1 effect difference. All variation and effect intervals run vertically. Response axes match between C and D; effect axes share numerical limits within a metric, with percentage points explicitly distinguished from variance explained (%).",
        "",
        "## Cell-level estimand and uncertainty",
        "Each plotted dot is one eligible neuron. Repeated valid MouseV2 timescale fits are averaged within neuron before plotting and analysis; fit-draw uncertainty is not propagated. All valid neurons belonging to the original eligible session/location groups are retained. The original V1 extremum-channel VISp filter and session/location support floor remain in force. Timescale cell counts are counts of unique neurons, not the original rounded mean count per fit draw.",
        "",
        "Neurons receive equal weight. A recording with more cells contributes more, and a pooled HVA control with more cells from an area weights that area more. This differs from equal session or equal area weighting. C/D bars show cell-weighted means; half-violins show cell distributions, independently normalized to a common maximum width. All observations are displayed, with rasterized scatter points in vector exports.",
        "",
        "The same one-way omega-squared formula as the existing figure is evaluated on individual cell values, rather than on session/location means. The denominator therefore includes within-location cell heterogeneity. Its algebraic bias correction uses the number of neurons; it is not a multilevel variance-component model or a correction for session/probe confounding. Negative estimates are retained. This measures the association with a stable location/area label, not total within-location variance or distances between every pair of cells.",
        "",
        f"Intervals use {args.draws:,} whole-session bootstrap replicates (seed {args.seed}), retaining all cells and locations from each selected session. The V1 and HVA bootstrap samples are independent; their replicate-wise difference gives the main-result interval. Gray intervals show the median and central 95% range of within-session block-label shuffles. All neurons at one session/location move together; the control exchanges entire V1 and HVA pools. Single-location sessions cannot change under the shuffle. No independent-cell inferential test is used. These controls assume exchangeability of available location labels within sessions. Session/animal nesting beyond the supplied session identifiers is not modeled.",
        "",
        "Significance labels for V1, HVA, and the control use uncorrected one-sided block-shuffle p-values: * p < .05, ** p < .01, *** p < .001, and n.s. p ≥ .05. Difference labels carry †: *† means the 95% bootstrap interval excludes zero, while n.s.† means it includes zero; these labels are not shuffle p-values, and no higher star tiers are inferred for the difference. A bootstrap interval crossing zero is not an equivalence result; a cell-level effect is not interchangeable with the original fraction of variance among session means. Cell-level estimates remain conditional on the metric filters, including TTFS <100 ms and valid timescale fits.",
        "",
        "## Positive-control provenance finding",
        "The original Allen V1 reference means and counts reconstruct exactly from the unfiltered legacy `data/unit_table.csv` with native TTFS/F1/F0 and existing timescale validity rules. They do not reconstruct from the current HVA-matched common-QC/response-filtered/harmonized inputs. The saved reconstruction audit verifies the legacy match for every original matched-control session and records the new V1 counts and mean changes.",
        "",
        "The new cell control uses common-QC Allen V1 measurements with the same metric processing as the HVA cells: response-selected preferred TTFS, harmonized F1/F0, and valid common-QC timescales. V1 groups require at least 10 TTFS neurons or 5 neurons for other metrics. Only sessions with both V1 and eligible HVA cells are retained. Thus control changes between figures reflect processing, cohort, weighting, and resolution changes; they must not be attributed solely to switching to cells. The original session control is reproduced for layout comparison and remains flagged for a separate correction. The cell timescale control is weak and should not be described as a demonstrated positive result.",
        "",
        "## Results",
        "",
        "| Metric | Within V1 (%) | Across HVAs (%) | HVA − V1 (pp), 95% bootstrap interval | Cell-control sessions |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in style.METRICS:
        st = summary.loc[summary.resolution.eq("single cell") & summary.metric.eq(metric)].set_index("comparison")
        d = st.loc["delta"]
        notes.append(f"| {metric} | {100 * st.loc['v1', 'observed']:.2f} | {100 * st.loc['hva', 'observed']:.2f} | {100*d.observed:+.2f} [{100*d.bootstrap_low:+.2f}, {100*d.bootstrap_high:+.2f}] | {int(st.loc['control', 'n_sessions'])} |")
    notes += [
        "",
        "The timescale difference interval is above zero under this cell-weighted analysis; TTFS and phase modulation intervals cross zero. The small fractions concern stable group identity, not an absence of biologically meaningful cell diversity. Only eight MouseV2 sessions underlie the V1 benchmark.",
        "",
        "The session-layout companion preserves the original observations and point estimates; bootstrap and shuffle intervals are regenerated with the seed above and may differ slightly by Monte Carlo variation. It is a layout companion, not a repair of the legacy control.",
        "",
        "## Verification and saved evidence",
        "Original MouseV2 session summaries are checked by the reused anatomical-filter loader. Current HVA cell means and counts reconstruct their frozen session summaries. Every effect is checked against the original omega-squared function evaluated on the expanded observations. Tests cover unequal cell counts, repeated bootstrap blocks, repeated neuron fits, and whole-location shuffling. PDF/PNG/SVG companions, per-neuron analysis inputs, statistics, resamples, source/method JSON, and the Allen V1 provenance audit are saved alongside this note.",
    ]
    (OUT / "Figure4_CDE_single_cell_notes.md").write_text("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
