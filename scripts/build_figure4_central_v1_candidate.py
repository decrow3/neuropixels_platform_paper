"""Add legacy Allen V1 as Central to the single-cell C/D/E comparison.

Run: MPLCONFIGDIR=/tmp/mpl-figure4-cells MPLBACKEND=Agg python scripts/build_figure4_central_v1_candidate.py
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.figure3_mousev2 import load_allen_units
from scripts import build_figure4_cell_location_controls as base
from scripts.figure3_robust_spread_comparison import omega_squared

PREFIX = base.OUT / "Figure4_CDE_single_cell_central_candidate"


def legacy_central_cells():
    """Reconstruct every legacy Allen V1 session mean using its original cells."""
    units = load_allen_units()
    units = units.loc[units.area_coarse.eq("V1")].copy()
    reference = pd.read_csv(base.OUT / "Figure3_Allen_V1_session_means.csv", dtype={"session_id": str})
    tables = []
    for index, (column, metric) in enumerate(zip(
        ["time_to_first_spike_fl", "f1_f0_dg", "timescale_ac"], base.style.METRICS
    )):
        values = pd.to_numeric(units[column], errors="coerce")
        keep = values.notna()
        if index == 0:
            keep &= values.lt(.1)
            values = values * 1000
        elif index == 1:
            keep &= values.gt(0)
            values = np.log10(values.clip(lower=1e-6))
        else:
            keep &= values.between(1, 300) & units.spike_count_ac.gt(50) & units.err_ac.lt(20)
        table = units.loc[keep, ["ecephys_session_id", "ecephys_unit_id"]].rename(
            columns={"ecephys_session_id": "session_id", "ecephys_unit_id": "unit_id"})
        table["value"] = values.loc[keep]
        table["metric"] = metric
        table["session_id"] = table.session_id.astype(str)
        table["unit_id"] = table.unit_id.astype(str)
        table = table.loc[table.groupby("session_id").value.transform("size").ge(5)]
        tables.append(table)
    cells = pd.concat(tables, ignore_index=True)
    assert np.isfinite(cells.value).all()
    assert not cells.duplicated(["metric", "session_id", "unit_id"]).any()
    rebuilt = cells.groupby(["metric", "session_id"], as_index=False).agg(
        reconstructed_mean=("value", "mean"), reconstructed_n=("value", "size"))
    audit = reference.merge(rebuilt, on=["metric", "session_id"], how="outer", validate="one_to_one", indicator=True)
    assert audit._merge.eq("both").all()
    np.testing.assert_allclose(audit["mean"], audit.reconstructed_mean, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(audit.n_units, audit.reconstructed_n)
    cells = cells.assign(dataset="Allen-V1-legacy", group="Central", comparison="v1", n_valid_fits=1)
    return cells, audit.drop(columns="_merge")


def moment_cube(table, sessions, groups):
    """Session × label sufficient statistics, with zeros for absent groups."""
    cube = np.zeros((len(sessions), len(groups), 3))
    session_index = {session: i for i, session in enumerate(sessions)}
    group_index = {group: i for i, group in enumerate(groups)}
    moments = table.assign(squared=table.value ** 2).groupby(["session_id", "group"], as_index=False).agg(
        n=("value", "size"), total=("value", "sum"), squares=("squared", "sum"))
    for row in moments.itertuples():
        cube[session_index[row.session_id], group_index[row.group]] = [row.n, row.total, row.squares]
    return cube


def session_multiplicities(n_sessions, n_draws, rng):
    """Counts of complete session copies in each ordinary cluster bootstrap."""
    return rng.multinomial(n_sessions, np.full(n_sessions, 1 / n_sessions), size=n_draws)


def joint_bootstraps(local, n_draws, rng):
    """Preserve cohort sizes and shared-session dependence across comparisons."""
    moments = {}
    groups = {comparison: sorted(local.loc[local.comparison.eq(comparison), "group"].unique())
              for comparison in ("v1", "hva", "control")}
    for comparison in groups:
        moments[comparison] = np.zeros((n_draws, len(groups[comparison]), 3))
    for source, part in local.groupby("source", sort=True):
        sessions = sorted(part.session_id.unique())
        multiplicities = session_multiplicities(len(sessions), n_draws, rng)
        # The same Allen multiplicities act on Central, HVAs, and the control.
        for comparison in groups:
            selected = part.loc[part.comparison.eq(comparison)]
            cube = moment_cube(selected, sessions, groups[comparison])
            moments[comparison] += np.einsum("bs,sgm->bgm", multiplicities, cube)
    draws = {key: base.omega_moments(value) for key, value in moments.items()}
    if not all(np.isfinite(value).all() for value in draws.values()):
        raise ValueError("Degenerate joint bootstrap replicate")
    return draws


def conditional_shuffle(local, n_draws, rng):
    """Shuffle only labels that co-occur in a source/session, fixing Central."""
    groups = sorted(local.group.unique())
    combined = np.zeros((n_draws, len(groups), 3))
    for _, source in local.groupby("source", sort=True):
        cube = moment_cube(source, sorted(source.session_id.unique()), groups)
        for block in cube:
            available = np.flatnonzero(block[:, 0])
            permutations = np.argsort(rng.random((n_draws, len(available))), axis=1)
            combined[:, available, :] += block[available][permutations]
    return base.omega_moments(combined)


def analyze(table, n_draws, seed):
    rng = np.random.default_rng(seed)
    records, draws = [], []
    for metric in base.style.METRICS:
        local = table.loc[table.metric.eq(metric)]
        bootstraps = joint_bootstraps(local, n_draws, rng)
        observed, nulls = {}, {}
        for comparison in ("v1", "hva", "control"):
            selected = local.loc[local.comparison.eq(comparison)]
            observed[comparison] = omega_squared(selected.group.to_numpy(), selected.value.to_numpy())
            # Independent algebraic reconstruction of each observed effect.
            cube = moment_cube(selected, sorted(selected.session_id.unique()), sorted(selected.group.unique()))
            np.testing.assert_allclose(base.omega_moments(cube.sum(axis=0)), observed[comparison], atol=1e-12)
            nulls[comparison] = conditional_shuffle(selected, n_draws, rng)
            records.append(dict(metric=metric, comparison=comparison, observed=observed[comparison],
                                bootstrap_low=np.quantile(bootstraps[comparison], .025),
                                bootstrap_high=np.quantile(bootstraps[comparison], .975),
                                shuffle_median=np.median(nulls[comparison]),
                                shuffle_low=np.quantile(nulls[comparison], .025),
                                shuffle_high=np.quantile(nulls[comparison], .975),
                                permutation_p_greater=(1 + (nulls[comparison] >= observed[comparison]).sum()) / (n_draws + 1),
                                conditional_shuffle=comparison == "v1",
                                n_sessions=selected[["source", "session_id"]].drop_duplicates().shape[0],
                                n_observations=len(selected)))
        bootstraps["delta"] = bootstraps["hva"] - bootstraps["v1"]
        nulls["delta"] = nulls["hva"] - nulls["v1"]
        records.append(dict(metric=metric, comparison="delta", observed=observed["hva"] - observed["v1"],
                            bootstrap_low=np.quantile(bootstraps["delta"], .025),
                            bootstrap_high=np.quantile(bootstraps["delta"], .975),
                            shuffle_median=np.median(nulls["delta"]), shuffle_low=np.quantile(nulls["delta"], .025),
                            shuffle_high=np.quantile(nulls["delta"], .975), permutation_p_greater=np.nan,
                            conditional_shuffle=False, n_sessions=np.nan, n_observations=np.nan))
        for comparison in bootstraps:
            draws.append(pd.DataFrame(dict(metric=metric, comparison=comparison, replicate=np.arange(n_draws),
                                           bootstrap=bootstraps[comparison], shuffle=nulls[comparison])))
    return pd.DataFrame(records), pd.concat(draws, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    previous_path = base.OUT / "Figure4_CDE_single_cell_candidate_inputs.csv"
    previous = pd.read_csv(previous_path, dtype={"session_id": str, "unit_id": str})
    central, audit = legacy_central_cells()
    table = pd.concat([previous, central], ignore_index=True)
    table["source"] = np.where(table.dataset.eq("Within-V1"), "MouseV2", "Allen")
    assert not table.duplicated(["comparison", "source", "metric", "session_id", "unit_id"]).any()
    stats, draws = analyze(table, args.draws, args.seed)
    table.to_csv(PREFIX.with_name(PREFIX.name + "_inputs.csv"), index=False)
    audit.to_csv(PREFIX.with_name(PREFIX.name + "_central_reconstruction.csv"), index=False)
    stats.to_csv(PREFIX.with_name(PREFIX.name + "_statistics.csv"), index=False)
    draws.to_csv(PREFIX.with_name(PREFIX.name + "_resamples.csv.gz"), index=False)
    counts = central.groupby("metric").agg(cells=("unit_id", "size"), sessions=("session_id", "nunique"))
    base.render(table, stats, PREFIX, True, central_mode="include")
    old = pd.read_csv(base.OUT / "Figure4_CDE_single_cell_candidate_statistics.csv")
    comparison = old.merge(stats, on=["metric", "comparison"], suffixes=("_four_locations", "_with_central"))
    stable = comparison.loc[comparison.comparison.isin(["hva", "control"])]
    np.testing.assert_allclose(stable.observed_four_locations, stable.observed_with_central, atol=1e-12)
    comparison.to_csv(PREFIX.with_name(PREFIX.name + "_effect_changes.csv"), index=False)
    method = dict(seed=args.seed, draws=args.draws,
                  central_source="data/unit_table.csv; legacy metric processing, not harmonized/common-QC",
                  central_eligibility="Original metric validity rules and >=5 cells per session; all original Allen V1 reference sessions",
                  central_label="User-requested display label; anatomical centrality was not verified from probe coordinates",
                  bootstrap="Whole sessions stratified by source. Allen session multiplicities shared across Central, HVAs and control, using union of eligible sessions per metric",
                  shuffle="Within source/session location-label blocks; Central is fixed because never co-recorded with four MouseV2 locations",
                  control="Same cell input population as prior single-cell control; refreshed bootstrap/shuffle draws",
                  sources=[str(previous_path.relative_to(ROOT)), "data/unit_table.csv", "Figure3/Figure3_Allen_V1_session_means.csv"])
    PREFIX.with_name(PREFIX.name + "_method.json").write_text(json.dumps(method, indent=2) + "\n")
    lines = [
        "# Single-cell C/D/E with older Allen V1 added as Central", "",
        "Central is included in C's five-group V1 effect and E's HVA-minus-V1 difference. Original figures are preserved.", "",
        "## Central source and meaning", "",
        "Central uses the original legacy Allen V1 cells from `data/unit_table.csv`, with native metrics and the original metric-specific validity filters and minimum of five cells per session. All original reference sessions are included, rather than restricting Central to E's matched control sessions. Per-session means and counts reconstruct `Figure3_Allen_V1_session_means.csv` exactly. TTFS is converted to ms, F1/F0 is log10-transformed, and timescale remains in ms. No additional common-QC or harmonization filters are applied to Central.", "",
        "Central is the requested display label, not a coordinate-verified anatomical location. Its dataset and processing pipeline are confounded with its group label. The five-group estimate is consequently a sensitivity analysis of combined V1 populations; differences cannot be attributed specifically to spatial position. This version should not be presented as five matched recording locations.", "",
        "| Metric | Central cells | Central sessions |", "|---|---:|---:|",
    ]
    for metric, row in counts.iterrows():
        lines.append(f"| {metric} | {int(row.cells):,} | {int(row.sessions)} |")
    lines += ["", "## Estimation and resampling", "",
              "Every eligible neuron receives equal weight. The original four-location and HVA cells, repeated-fit handling, and E control inputs are retained. Cell-level one-way omega-squared is recomputed after adding Central to V1. It is bias-corrected and can be negative; it is not a multilevel variance-component estimate.", "",
              f"{args.draws:,} bootstrap replicates (seed {args.seed}) resample whole sessions separately within MouseV2 and Allen. Within each metric, the union of eligible Allen sessions is resampled, with identical multiplicities applied to Central, HVAs, and the control. Missing session/population combinations remain missing. This preserves dependence between V1 and HVA effect estimates from shared Allen sessions. E's difference is computed within each joint replicate. The two datasets are not resampled as one exchangeable cohort.", "",
              "The V1 shuffle holds Central fixed, since its sessions never contain the other four location labels. Its stars/n.s. carry ‡ and test only the assignment of the four MouseV2 labels conditional on the fixed Central distribution; they are not an omnibus test of all five groups or of the Central-versus-peripheral distinction. Gray V1 null estimates may therefore be far from zero. HVA/control shuffles retain their prior interpretation. All shuffle p-values are one-sided and uncorrected. Difference labels † use the 95% bootstrap interval, not a shuffle p-value; intervals including zero do not establish equivalence.", "",
              "E's V1–pooled-HVA control retains the prior HVA-matched processing and populations. Central's legacy cells are not substituted into that separate control. HVA and control point estimates are unchanged; intervals and shuffle p-values are refreshed under the joint bootstrap and new random stream.", "",
              "## Effects", "", "| Metric | V1 including Central (%) | Across HVAs (%) | HVA − V1 (pp), 95% interval |", "|---|---:|---:|---:|"]
    for metric in base.style.METRICS:
        rows = stats.loc[stats.metric.eq(metric)].set_index("comparison")
        d = rows.loc["delta"]
        lines.append(f"| {metric} | {100*rows.loc['v1','observed']:.2f} | {100*rows.loc['hva','observed']:.2f} | {100*d.observed:+.2f} [{100*d.bootstrap_low:+.2f}, {100*d.bootstrap_high:+.2f}] |")
    lines += ["", "Reproduce with `MPLCONFIGDIR=/tmp/mpl-figure4-cells MPLBACKEND=Agg python scripts/build_figure4_central_v1_candidate.py`. Inputs, reconstruction audit, effects, resamples, method JSON, and PDF/PNG/SVG exports are saved with the same filename prefix."]
    PREFIX.with_name(PREFIX.name + "_notes.md").write_text("\n".join(lines) + "\n")
    print(counts.to_string())
    print(stats[["metric", "comparison", "observed", "bootstrap_low", "bootstrap_high", "permutation_p_greater"]].to_string(index=False))


if __name__ == "__main__":
    main()
