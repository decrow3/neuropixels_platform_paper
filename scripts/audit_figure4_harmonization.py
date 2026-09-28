"""Audit current Figure 4 inputs against accepted cross-dataset bridges.

Read-only with respect to figures and data; saves an audit report and tables.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.figure3_mousev2 import load_allen_units
from common.population_masks import common_qc_mask
from scripts import build_figure4_variant_review as style
from scripts.figure3_robust_spread_comparison import (
    MOUSE_HARMONIZED_F1_F0, ALLEN_HARMONIZED_F1_F0, MOUSE_TIMESCALE_TRIAL_BRIDGE,
)
from scripts.figure3_full_cell_multilevel_model import MOUSE_TTFS_UNITS, ALLEN_TTFS_AUDIT

OUT = ROOT / "Figure3/Figure4_harmonization_audit"


def main():
    current = pd.read_csv(ROOT / "Figure3/Figure4_CDE_single_cell_central_candidate_inputs.csv",
                          dtype={"session_id": str, "unit_id": str})
    rf = pd.read_csv(ROOT / "Figure3/Figure4_CDE_single_cell_central_RF_adjusted_candidate_raw_inputs.csv",
                     dtype={"session_id": str, "unit_id": str})
    checks = []

    def compare(label, cells, expected, source):
        expected = expected[["unit_id", "value"]].rename(columns={"value": "expected_value"})
        assert not expected.unit_id.duplicated().any()
        joined = cells.merge(expected, on="unit_id", how="left", validate="one_to_one", indicator=True)
        assert joined._merge.eq("both").all(), label
        error = float((joined.value - joined.expected_value).abs().max())
        np.testing.assert_allclose(joined.value, joined.expected_value, rtol=0, atol=1e-9, err_msg=label)
        checks.append(dict(check=label, neurons=len(cells), max_absolute_error=error, source=str(source.relative_to(ROOT)), status="exact_match"))

    def population(dataset, metric, comparison=None):
        keep = current.dataset.eq(dataset) & current.metric.eq(metric)
        if comparison:
            keep &= current.comparison.eq(comparison)
        return current.loc[keep].copy()

    mouse_ttfs = pd.read_csv(MOUSE_TTFS_UNITS, dtype={"unit_id": str})
    mouse_ttfs = mouse_ttfs.loc[mouse_ttfs.cohort.eq("MouseV2") & mouse_ttfs.selected
                                 & mouse_ttfs.preferred_0_250_ttfs_ms.lt(100)].rename(columns={"preferred_0_250_ttfs_ms": "value"})
    compare("MouseV2 preferred response-selected TTFS", population("Within-V1", style.METRICS[0]), mouse_ttfs, MOUSE_TTFS_UNITS)
    allen_ttfs = pd.read_csv(ALLEN_TTFS_AUDIT, dtype={"unit_id": str})
    allen_ttfs = allen_ttfs.loc[allen_ttfs.selected_positive_responder_area & allen_ttfs.preferred_0_250_ttfs_ms.lt(100)].rename(columns={"preferred_0_250_ttfs_ms": "value"})
    compare("HVA preferred response-selected TTFS", population("Post-V1", style.METRICS[0], "hva"), allen_ttfs, ALLEN_TTFS_AUDIT)
    compare("Control Allen V1 preferred response-selected TTFS", population("Allen-V1", style.METRICS[0]), allen_ttfs, ALLEN_TTFS_AUDIT)

    mouse_f1 = pd.read_csv(MOUSE_HARMONIZED_F1_F0, dtype={"unit_id": str})
    mouse_f1 = mouse_f1.loc[mouse_f1.default_qc & mouse_f1.f1_f0_dg_common_support.gt(0)].copy()
    mouse_f1["value"] = np.log10(mouse_f1.f1_f0_dg_common_support)
    allen_f1 = pd.read_csv(ALLEN_HARMONIZED_F1_F0, dtype={"ecephys_unit_id": str})
    assert allen_f1.population_profile.eq("common_qc").all()
    allen_f1 = allen_f1.loc[allen_f1.f1_f0_dg_harmonized.gt(0)].rename(columns={"ecephys_unit_id": "unit_id"})
    allen_f1["value"] = np.log10(allen_f1.f1_f0_dg_harmonized)
    compare("MouseV2 common-support F1/F0", population("Within-V1", style.METRICS[1]), mouse_f1, MOUSE_HARMONIZED_F1_F0)
    compare("HVA harmonized F1/F0", population("Post-V1", style.METRICS[1], "hva"), allen_f1, ALLEN_HARMONIZED_F1_F0)
    compare("Control Allen V1 harmonized F1/F0", population("Allen-V1", style.METRICS[1]), allen_f1, ALLEN_HARMONIZED_F1_F0)

    bridge = pd.read_csv(MOUSE_TIMESCALE_TRIAL_BRIDGE, dtype={"unit_id": str})
    matched = bridge.loc[bridge.view.eq("mouse_matched_150") & bridge.valid_timescale].copy()
    assert matched.flash_trials.eq(150).all() and matched.bright_trials.eq(75).all() and matched.dark_trials.eq(75).all()
    tau = matched.groupby("unit_id", as_index=False).agg(value=("timescale_ms", "mean"), n_fits=("timescale_ms", "size"))
    mouse_tau = population("Within-V1", style.METRICS[2])
    compare("MouseV2 mean of valid 75+75 flash timescale fits", mouse_tau, tau, MOUSE_TIMESCALE_TRIAL_BRIDGE)
    fit_counts = mouse_tau.merge(tau[["unit_id", "n_fits"]], on="unit_id", validate="one_to_one")
    np.testing.assert_array_equal(fit_counts.n_valid_fits, fit_counts.n_fits)

    allen = load_allen_units()
    allen["unit_id"] = allen.ecephys_unit_id.astype(str)
    allen["common_qc"] = common_qc_mask(allen, dataset="allen")
    valid_tau = allen.loc[allen.common_qc & allen.timescale_ac.between(1, 300)
                          & allen.spike_count_ac.gt(50) & allen.err_ac.lt(20)].copy()
    valid_tau["value"] = valid_tau.timescale_ac
    compare("HVA common-QC valid timescale", population("Post-V1", style.METRICS[2], "hva"), valid_tau, ROOT / "data/unit_table.csv")
    compare("Control Allen V1 common-QC valid timescale", population("Allen-V1", style.METRICS[2]), valid_tau, ROOT / "data/unit_table.csv")
    # Establish that Central actually uses native values rather than merely
    # bearing a legacy dataset label.
    for metric, column in zip(style.METRICS, ["time_to_first_spike_fl", "f1_f0_dg", "timescale_ac"]):
        native = allen.copy()
        native["value"] = native[column] * 1000 if metric == style.METRICS[0] else np.log10(native[column].clip(lower=1e-6)) if metric == style.METRICS[1] else native[column]
        compare("Central native " + metric, population("Allen-V1-legacy", metric), native, ROOT / "data/unit_table.csv")

    peaks = pd.read_csv(ROOT / "data/imports/pilot_rf_peaks_v1/rf_unit_peaks.csv", dtype={"unit_id": str})
    mouse_qc = peaks[["unit_id", "default_qc"]].rename(columns={"default_qc": "common_qc"}).assign(source="MouseV2")
    qcs = pd.concat([mouse_qc, allen[["unit_id", "common_qc"]].assign(source="Allen")], ignore_index=True)
    qc_records = []
    for stage, table in [("Before RF selection", current), ("RF-qualified subset", rf)]:
        joined = table.merge(qcs, on=["source", "unit_id"], how="left", validate="many_to_one")
        assert joined.common_qc.notna().all()
        summary = joined.groupby(["comparison", "dataset", "metric"], as_index=False).agg(
            cells=("unit_id", "size"), common_qc_pass=("common_qc", "sum"), sessions=("session_id", "nunique"))
        summary["common_qc_fail"] = summary.cells - summary.common_qc_pass
        qc_records.append(summary.assign(stage=stage))
    qc = pd.concat(qc_records, ignore_index=True)
    assert qc.loc[qc.dataset.ne("Allen-V1-legacy"), "common_qc_fail"].eq(0).all()
    keys = ["comparison", "dataset", "metric", "session_id", "group", "unit_id"]
    preserved = rf.merge(current[keys + ["value"]], on=keys, how="left", suffixes=("_rf", "_original"), validate="one_to_one")
    np.testing.assert_allclose(preserved.value_rf, preserved.value_original, rtol=0, atol=1e-12)

    central_f1 = population("Allen-V1-legacy", style.METRICS[1])
    accepted_v1 = allen_f1.loc[allen_f1.area_coarse.eq("V1")]
    paired = central_f1.merge(accepted_v1[["unit_id", "value"]], on="unit_id", suffixes=("_native", "_harmonized"), validate="one_to_one")
    paired["harmonized_minus_native"] = paired.value_harmonized - paired.value_native
    pd.DataFrame(checks).to_csv(OUT.with_name(OUT.name + "_source_checks.csv"), index=False)
    qc.to_csv(OUT.with_name(OUT.name + "_qc.csv"), index=False)
    paired.to_csv(OUT.with_name(OUT.name + "_central_f1_paired.csv"), index=False)
    fit_counts.to_csv(OUT.with_name(OUT.name + "_timescale_fit_counts.csv"), index=False)
    lines = ["# Figure 4 harmonization audit", "",
             "## Finding", "",
             "The newer figures do not uniformly preserve the established cross-dataset harmonization. The four MouseV2 locations and HVA response inputs retain the accepted aligned metrics. The single-cell E control also uses aligned Allen V1 inputs. Adding Central reintroduced native Allen metrics and broader QC, and the RF-adjusted version imposed a legacy Pilot-QC fit-coverage restriction on MouseV2. Timescale trial-count matching remains, but the single-cell aggregation and uncertainty differ from the earlier validated draw-aware analysis.", "",
             "No figures, source data, filters, or scientific conclusions were changed by this audit. Exact input-level checks and QC counts are saved alongside it.", "",
             "## What was preserved", "",
             "All current non-Central response values match their accepted source tables to numerical precision: response-selected preferred-polarity TTFS; harmonized conventional F1/F0; valid Allen timescales; and means of valid MouseV2 matched-150-flash timescale fits. All non-Central plotted populations pass common waveform QC. The RF-qualified subset preserves those incoming raw values exactly before regression.", "",
             "The F1/F0 contract specifies a common 1-s window, 15 trials per orientation×TF, orientations 0/45/90/135°, TFs 1/2/4/8/15 Hz, SF=.04 cycles/degree, contrast=.8, and the 28 complete Allen Brain Observatory sessions. The current HVA and non-Central control values still use that table. MouseV2 timescale input fits still use 75 bright + 75 dark flashes, matching Allen's trial count.", "",
             "## Regression 1: Central uses native, not harmonized, Allen data", "",
             "The Central builder reads `data/unit_table.csv` without common QC, transforms native F1/F0 and released TTFS, and applies the old metric-validity/minimum-cell rules. Its values are confirmed to match those native columns. This is not the harmonized Allen V1 reference already available for comparison with MouseV2. The original session-figure control also read the legacy `Figure3_Allen_V1_session_means.csv`; the later non-Central single-cell control corrected that source, but the Central branch reintroduced it.", "",
             "| Central metric | Cells | Common QC pass | Common QC fail | Sessions |", "|---|---:|---:|---:|---:|"]
    for row in qc.loc[qc.stage.eq("Before RF selection") & qc.dataset.eq("Allen-V1-legacy")].itertuples():
        lines.append(f"| {row.metric} | {row.cells} | {row.common_qc_pass} | {row.common_qc_fail} | {row.sessions} |")
    lines += ["",
              f"For {len(paired):,} Central F1/F0 neurons shared with the accepted harmonized Allen V1 table, the native mean is {paired.value_native.mean():.6f} log10 versus {paired.value_harmonized.mean():.6f} harmonized, a paired mean change of {paired.harmonized_minus_native.mean():+.6f}. These are the same cells, so that change is directly attributable to metric processing. This comparison is cell-weighted and is not the historical equal-session bridge mean. Central includes {central_f1.session_id.nunique()} sessions, while the harmonized F1/F0 reference is restricted to 28 complete Brain Observatory sessions. Its apparent difference from MouseV2 must not be read as within-V1 spatial variation.", "",
              "## Regression 2: RF coverage/selection is not aligned", "",
              "The MouseV2 RF builder fits only the earlier Pilot-QC subset and then demands supported Gaussian fits (including reliability, fit quality, on-screen centers, and interior widths). Allen uses released RF centers with published-like RF significance/area/SNR/rate gates. These are different estimator/selection contracts, not a validated matched RF population. The separate `Figure4_RF_TTFS_probe_audit.md` establishes that all 1,090 original MouseV2 TTFS cells have raw RF records but only 515 have entries in the parametric-fit table, exactly matching legacy Pilot-QC eligibility. 189 pass support and 158 survive the five-cell floor. Missing parametric fits must not be labeled unreliable RFs.", "",
              "The new adjustment also differs from the earlier validated Allen-only RF workflow: it uses absolute display coordinates (+50/+10 for MouseV2), a common quadratic surface per comparison, no common-support restriction, and in-sample partial omega-squared. The earlier Allen workflow uses session-V1-relative RF positions, strict overlap support, session-centered outcomes, and held-out-session prediction. The new approach is explicitly exploratory; its stars cannot be treated as reproducing the earlier analysis. The coordinate conversion is not gaze correction.", "",
              "## Regression 3: timescale draw handling and weighting changed", "",
              f"Current single-cell C/D/E uses one mean of available valid fits per MouseV2 neuron, with equal neuron weights and session-only resampling. Of {len(fit_counts)} current MouseV2 timescale neurons, {int(fit_counts.n_fits.eq(10).sum())} have all ten valid fits, {int(fit_counts.n_fits.lt(10).sum())} have fewer, and {int(fit_counts.n_fits.eq(1).sum())} have only one. Allen contributes one estimate per neuron. The trial-matched response sources are preserved, but equal weighting of one-valid-draw and ten-valid-draw neurons and omission of trial-draw uncertainty are changes to the measurement/uncertainty comparison.", "",
              "The earlier full-cell extension fit each matched draw separately and sampled one draw per resampled MouseV2 session. Its sensitivity work explicitly found that complete-draw inclusion and fitted-error selection changed the timescale contrast. Those checks were not carried forward into the simplified new C/D/E analysis. Anatomical V1 filtering also changes the population relative to that older checkpoint, so its numerical results should not be copied directly.", "",
              "## Pre-existing limits, not new regressions", "",
              "Matching a preferred-response TTFS estimator does not establish an independent physical light-onset calibration between datasets. The prior roadmap explicitly excludes interpreting uncalibrated absolute cross-dataset latency shifts as biology. Adding Allen V1 as a fifth MouseV2 location reintroduces sensitivity to dataset-wide offsets. Anatomical V1 restriction, common waveform QC, and metric harmonization also do not match cell type, layer, behavior, or sampling density.", "",
              "The earlier timescale interim conclusion was that a systematic HVA hierarchy was not robust across fit-population sensitivities. New categorical-effect stars estimate a different quantity and do not overturn that result. The V1 shuffle in Central-inclusive plots holds Central fixed; it is not an omnibus cross-dataset group test.", "",
              "## Concrete repair sequence", "",
              "1. Use the harmonized Allen V1 unit sources for Central, with the same waveform QC and metric processing as the HVA/control branches. Use every eligible session for that reference, not only the subset retained for E's paired control; retain metric-specific session eligibility.",
              "2. Fit and validate RFs for the current MouseV2 analysis population missing from the legacy fit table. Review comparable RF estimator/support criteria and coordinate alignment before interpreting RF-adjusted cross-dataset effects; do not substitute raw peak centers or relax thresholds silently.",
              "3. Restore trial-draw-aware timescale estimation/uncertainty and its complete-draw/fit-error sensitivity checks on the current anatomical population. Keep any mean-per-neuron display explicitly separate from inferential weighting.",
              "4. Regenerate unadjusted and RF-qualified/adjusted figures from a shared provenance-checked loader, keeping same-cell controls and the original exploratory versions.", "",
              "## Inspected evidence", "",
              "- `artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/07_figure3_harmonized_f1_f0/FIGURE3_HARMONIZED_F1_F0_CHECKPOINT.md`",
              "- `data/imports/mousev2_timescale_trial_bridge_v1/README.md`",
              "- `artifacts/figure3/07_big_picture_concrete_first/timescale_population_sensitivity/TIMESCALE_INTERIM_CONCLUSION.md`",
              "- `artifacts/figure3/07_big_picture_concrete_first/retinotopy_adjusted_hva_models/VALIDATION_REPORT.md`",
              "- `scripts/figure3_full_cell_multilevel_extension.py` and current Figure 4 builders",
              "- `ANALYSIS_ROADMAP.md` and the source tables recorded in `_source_checks.csv`",
              "", "Reproduce with `MPLCONFIGDIR=/tmp/mpl-figure4-cells python scripts/audit_figure4_harmonization.py`."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(checks).to_string(index=False))
    print(qc.loc[qc.dataset.eq("Allen-V1-legacy")].to_string(index=False))
    print(f"Paired Central F1 native-to-harmonized change: {paired.harmonized_minus_native.mean():+.6f} ({len(paired)} cells)")
    print(OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
