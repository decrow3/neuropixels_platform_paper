"""Trace Figure 4 MouseV2 TTFS RF exclusions without changing populations."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Figure3/Figure4_RF_TTFS_probe_audit"


def main():
    source = pd.read_csv(ROOT / "Figure3/Figure4_CDE_single_cell_central_candidate_inputs.csv",
                         dtype={"session_id": str, "unit_id": str})
    source = source.loc[source.source.eq("MouseV2") & source.metric.eq("TTFS (ms)")].copy()
    peaks = pd.read_csv(ROOT / "data/imports/pilot_rf_peaks_v1/rf_unit_peaks.csv", dtype={"unit_id": str})
    fits = pd.read_csv(ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv", dtype={"unit_id": str})
    peaks = peaks[["unit_id", "site_number", "probe", "pilot_qc", "default_qc", "rf_center_x_deg", "rf_center_y_deg"]].rename(
        columns={c: "peak_"+c for c in ["site_number", "probe", "pilot_qc", "default_qc", "rf_center_x_deg", "rf_center_y_deg"]})
    table = source.merge(peaks, on="unit_id", how="left", validate="one_to_one", indicator="peak_match")
    table = table.merge(fits, on="unit_id", how="left", validate="one_to_one", indicator="fit_match")
    table["raw_peak_available"] = table.peak_match.eq("both")
    table["raw_peak_finite"] = np.isfinite(table[["peak_rf_center_x_deg", "peak_rf_center_y_deg"]]).all(axis=1)
    table["fit_available"] = table.fit_match.eq("both")
    table["finite_fitted_center"] = np.isfinite(table[["rf_center_x_deg", "rf_center_y_deg"]]).all(axis=1)
    assert table.raw_peak_available.all() and table.raw_peak_finite.all()
    assert table.peak_site_number.astype(str).eq(table.session_id).all()
    assert table.peak_probe.eq(table.group).all()
    present = table.fit_available
    assert table.loc[present, "site_number"].astype(int).astype(str).eq(table.loc[present, "session_id"]).all()
    assert table.loc[present, "probe"].eq(table.loc[present, "group"]).all()
    assert table.fit_available.eq(table.peak_pilot_qc).all()

    manifest = json.loads((ROOT / "data/imports/mousev2_parametric_rf_v1/run_manifest.json").read_text())
    params = manifest["parameters"]
    gates = {
        "fit_success": table.rf_fit_success.eq(True),
        "center_on_screen": table.rf_center_on_screen.eq(True),
        "model_fdr": table.rf_lrt_q.le(params["fdr_alpha"]),
        "reliability_fdr": table.rf_reliability_q.le(params["fdr_alpha"]),
        "reliability_ge_0p3": table.rf_split_half_spearman_brown.ge(params["minimum_split_half_spearman_brown"]),
        "pseudo_r2_ge_0p1": table.rf_pseudo_r2.ge(params["minimum_pseudo_r2"]),
        "major_width_interior": table.rf_sigma_major_deg.between(3.05, 79.5),
        "minor_width_interior": table.rf_sigma_minor_deg.between(3.05, 79.5),
    }
    supported = present.copy()
    for name, passed in gates.items():
        table["passes_"+name] = passed
        supported &= passed
    table["supported_rf"] = table.rf_model_supported.eq(True)
    assert supported.eq(table.supported_rf).all()
    table["supported_count_in_session_probe"] = table.groupby(["session_id", "group"]).supported_rf.transform("sum")
    table["retained"] = table.supported_rf & table.supported_count_in_session_probe.ge(5)
    table["exclusion_stage"] = np.select(
        [~present, present & ~supported, supported & ~table.retained],
        ["not_fit_due_to_legacy_pilot_qc", "fitted_but_unsupported", "below_five_supported_ttfs_cells"], default="retained")
    existing = pd.read_csv(ROOT / "Figure3/Figure4_CDE_single_cell_central_RF_adjusted_candidate_raw_inputs.csv", dtype={"unit_id": str})
    existing = existing.loc[existing.source.eq("MouseV2") & existing.metric.eq("TTFS (ms)")]
    assert set(existing.unit_id) == set(table.loc[table.retained, "unit_id"])

    summary = table.groupby(["session_id", "group"]).agg(
        ttfs_cells=("unit_id", "size"), raw_rf_peaks=("raw_peak_available", "sum"),
        parametric_fits=("fit_available", "sum"), finite_fitted_centers=("finite_fitted_center", "sum"),
        supported_rf=("supported_rf", "sum"), final_cells=("retained", "sum"))
    inventory = pd.MultiIndex.from_product([[str(i) for i in range(2, 10)], ["A", "E", "C", "B"]], names=["session_id", "group"])
    summary = summary.reindex(inventory, fill_value=0).reset_index()
    config = json.loads((ROOT / "config/figure3_mousev2.json").read_text())
    subjects = {str(s["site_number"]): s["subject_id"] for s in config["sessions"]}
    summary["subject_id"] = summary.session_id.map(subjects)
    summary["location"] = summary.group.map({"A": "Anterior", "E": "Lateral", "C": "Posterior", "B": "Medial"})
    table["failed_support_gates"] = [";".join(name for name in gates if not row["passes_"+name]) if row.fit_available else "not_fitted"
                                      for _, row in table.iterrows()]
    gate_rows = []
    for group, part in table.loc[present].groupby("group"):
        for name in gates:
            gate_rows.append(dict(group=group, gate=name, fitted_cells=len(part),
                                  failed_cells=int((~part["passes_"+name]).sum())))
    gates_summary = pd.DataFrame(gate_rows)
    table.to_csv(OUT.with_name(OUT.name + "_cells.csv"), index=False)
    summary.to_csv(OUT.with_name(OUT.name + "_sessions.csv"), index=False)
    gates_summary.to_csv(OUT.with_name(OUT.name + "_support_failures.csv"), index=False)
    focus = summary.loc[summary.group.isin(["E", "B"])]
    lines = ["# Lateral and medial TTFS RF attrition audit", "",
             "All current lateral/medial TTFS cells match raw per-neuron RF-peak records. No session/probe-ID mismatch was found. The parametric RF fit table covers only the older Pilot-QC-selected population; fit availability matches that legacy eligibility flag exactly. Thus absence from the fit table is not a failed RF fit and not missing RF stimulation.", "",
             "The current plot applied supported-parametric-RF eligibility, followed by at least five supported TTFS cells per session/probe. The reconstructed retained neuron IDs match the plotted population exactly.", "",
             "| Session | Animal | Location | Original TTFS | Raw RF records | Parametric fits | Supported fits | Final cells |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in focus.itertuples():
        lines.append(f"| site{row.session_id} | {row.subject_id} | {row.location} | {row.ttfs_cells} | {row.raw_rf_peaks} | {row.parametric_fits} | {row.supported_rf} | {row.final_cells} |")
    anatomy = pd.read_csv(ROOT / "Figure3/Figure4_mousev2_visp_filter_audit.csv")
    prior = anatomy.loc[anatomy.metric.eq("TTFS (ms)") & anatomy.session_id.eq(7) & anatomy.group.eq("B")].iloc[0]
    assert prior.n_units_visp_prefloor == 2 and not prior.retained_after_visp_filter
    lines += ["", "Site7 medial already had only two anatomically eligible TTFS neurons before RF selection and failed the pre-existing five-cell floor. The pre-anatomy session mean had 38 units. Its zero in this audit means absent from the incoming figure population, not absence of RF data at that probe.", "",
              "## Where cells disappear", "",
              "| Location | No parametric fit (legacy Pilot-QC) | Fitted but unsupported | Supported but below session/probe floor | Retained |", "|---|---:|---:|---:|---:|"]
    for group, label in [("E", "Lateral"), ("B", "Medial")]:
        p = table.loc[table.group.eq(group)].exclusion_stage.value_counts()
        lines.append(f"| {label} | {p.get('not_fit_due_to_legacy_pilot_qc',0)} | {p.get('fitted_but_unsupported',0)} | {p.get('below_five_supported_ttfs_cells',0)} | {p.get('retained',0)} |")
    lines += ["", "## RF support criteria", "",
              "The saved criteria require fit success, an on-screen center, model and reliability BH-FDR q≤.05, Spearman–Brown split-half reliability≥.3, pseudo-R²≥.1, and both width parameters in [3.05,79.5] degrees. The support-failure CSV counts each failed gate separately; failures overlap and must not be added as exclusive causes. The cell CSV retains every gate and original fit values.", "",
              "## Interpretation and next step", "",
              "The two-session result is a property of the figure's inherited fit coverage and selection pipeline, not a statement that only two sessions recorded RFs. In particular, more than half of the lateral/medial TTFS neurons were never included in this parametric-fit table. Their RF validity has not been assessed by these models. The current figure cannot establish that those cells lack reliable RFs.", "",
              "Before interpreting or relaxing the RF-adjusted TTFS analysis, the targeted next step is to fit/validate parametric RFs for the current TTFS population missing from the legacy fit table, retaining explicit quality criteria. Existing raw argmax RF centers are not automatically interchangeable with supported Gaussian centers. This audit does not refit RFs, relax thresholds, or change the figures.", "",
              "Provenance: `scripts/extract_mousev2_parametric_rf.py` selects `pilot_qc` before fitting; `data/imports/mousev2_parametric_rf_v1/run_manifest.json` records the fit population and thresholds; `scripts/build_figure4_rf_adjusted_candidate.py:attach_rf` applies supported-fit selection and the five-cell V1 floor."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(focus[["session_id", "location", "ttfs_cells", "raw_rf_peaks", "parametric_fits", "supported_rf", "final_cells"]].to_string(index=False))
    print(gates_summary.loc[gates_summary.group.isin(["E", "B"])].to_string(index=False))
    print(table.groupby(["group", "exclusion_stage"]).size().to_string())
    print(OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
