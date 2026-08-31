#!/usr/bin/env python3
"""Build the canonical artifact payload for the V1 TTFS diagnostic report."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/figure3/06r_probe_flash_photoartifact_pilot"
DATA=BASE/"ttfs_metric_decomposition"
OUT=DATA/"report"


def source(sid,label,path,description,filters,definitions):
    return {"id":sid,"label":label,"path":path,"query":{"description":description,"language":"sql","engine":"DuckDB","sql":f"SELECT * FROM read_csv_auto('{path}')","tables_used":[path],"filters":filters,"metric_definitions":definitions}}


def main():
    reproduction=pd.read_csv(DATA/"ttfs_reproduction_by_cohort.csv").round(6)
    gaps=pd.read_csv(DATA/"ttfs_response_composition_gaps.csv")
    gaps=gaps.loc[gaps.metric.eq("released_ttfs_ms")].copy()
    gaps["population_label"]=gaps.population.map({"all common-QC V1":"All common-QC","independently selected positive responders":"Independent positive responders"})
    gaps["allen_label"]=gaps.cohort.map({"Allen Brain Observatory":"Allen BO","Allen Functional Connectivity":"Allen FC"})
    alignment=pd.read_csv(BASE/"crossvalidated_latency_independent_psths/heldout_alignment_summary.csv")
    alignment=alignment.loc[alignment["view"].isin(["raw","source_predicted","best_fit"])].copy().round(4)
    centers=pd.read_csv(DATA/"ttfs_response_composition_centers.csv")
    centers=centers.loc[centers.metric.eq("released_ttfs_ms")].copy().round(4)
    now=datetime.now(timezone.utc).isoformat()
    sources=[
        source("reproduction","TTFS raw-spike reproduction","ttfs_reproduction_by_cohort.csv","Released fields compared with direct first-spike recomputation from local raw NWB spikes.",["common-QC V1","all finite released/recomputed TTFS"],["Absolute error = |recomputed TTFS - released TTFS|","Exact fraction uses error <0.001 ms"]),
        source("composition","Response-composition sensitivity","ttfs_response_composition_gaps.csv","Equal-session TTFS centers before and after trial-independent positive-response selection.",["released TTFS <100 ms","odd trials select 30-180 ms positive response","even trials reserved for timing"],["Gap = MouseV2 equal-session mean - Allen equal-session mean","Gap reduction = 1 - selected gap / all-unit gap"]),
        source("alignment","Cross-validated waveform timing","heldout_alignment_summary.csv","Held-out, session-balanced waveform alignment with no TTFS or onset selection.",["all common-QC V1 eligible","odd/even trial split","alignment window 20-120 ms"],["Best shift minimizes peak-normalized waveform RMSE","Negative shift moves MouseV2 earlier"]),
    ]
    blocks=[
        {"id":"title","type":"markdown","body":"# V1 Flash TTFS Discrepancy: Metric and Population Audit","layout":"full"},
        {"id":"summary","type":"markdown","body":"## Technical summary\n\nThe approximately 7–9 ms released MouseV2–Allen TTFS gap is not evidence for a matching delay in flash-evoked V1 response onset. Held-out PSTHs selected without latency information align best at approximately zero shift, and shifting MouseV2 by −8.34 ms worsens waveform agreement. Population composition explains most of the scalar gap: restricting the comparison to positive flash responders selected on independent trials reduces the released gap to about 3.05 ms, a 58% reduction versus Allen Brain Observatory and 65% versus Functional Connectivity. The residual cannot yet be decomposed cleanly because MouseV2 TTFS reproduces exactly from local raw spikes whereas the released Allen field does not.","layout":"full"},
        {"id":"finding1","type":"markdown","body":"## The released TTFS fields are not computationally interchangeable\n\nDirect application of the MouseV2 first-spike definition reproduces all 11,063 finite MouseV2 unit values exactly. In Allen, the median absolute unit error is 4 ms, only 12–13% of values reproduce exactly, and the 95th-percentile absolute error is approximately 49–56 ms. The disagreement is unit-specific and occurs across sessions, so it cannot be repaired by one constant timestamp correction.","sourceId":"reproduction","layout":"full"},
        {"id":"reprochart","type":"chart","chartId":"reproduction_chart","layout":"full"},
        {"id":"finding2","type":"markdown","body":"## Nonresponding units account for most of the released cohort gap\n\nThe published TTFS filter requires only TTFS <100 ms; it does not require a positive flash-evoked response. Selecting positive responders on odd trials and evaluating timing independently reduces the MouseV2-minus-Allen gap from 7.20 to 3.05 ms for Brain Observatory and from 8.70 to 3.06 ms for Functional Connectivity. This explains 58–65% of the released discrepancy without imposing matched onset times.","sourceId":"composition","layout":"full"},
        {"id":"gapchart","type":"chart","chartId":"gap_chart","layout":"full"},
        {"id":"finding3","type":"markdown","body":"## Held-out response timing rejects a global −8.34 ms correction\n\nWith no TTFS or onset filter, odd trials select response amplitude over a broad window and even trials estimate timing. Bright-response waveform fits prefer MouseV2 shifts of −0.25 to −0.50 ms; dark-response fits prefer −1.25 to +1.25 ms. Applying the source-predicted −8.34 ms correction worsens every RMSE and correlation comparison.","sourceId":"alignment","layout":"full"},
        {"id":"align_table_block","type":"table","tableId":"alignment_table","layout":"full"},
        {"id":"scope","type":"markdown","body":"## Scope, data, and metric definitions\n\nThe audit uses 32 locally available Allen Brain Observatory sessions, 24 Functional Connectivity sessions, and all eight MouseV2 sessions. The population is common-QC V1. Released TTFS is retained exactly as plotted in Figure 3 and filtered at <100 ms for cohort centers. Raw recomputation uses the median first occupied 1 ms bin from 30–200 ms, omitting trials without a spike in that interval. Session-level centers receive equal weight.","layout":"full"},
        {"id":"methods","type":"markdown","body":"## Methodology and validation design\n\nRaw-spike decomposition measures pre-flash firing, 30–180 ms evoked rate, 30–60 ms spike occupancy, first-spike dispersion, and PSTH peak timing per unit. The response-composition sensitivity avoids circular latency selection: odd trials test positive response amplitude with FDR q<0.01 and effect ≥2 Hz, while even trials estimate PSTHs. Bright and dark flashes are analyzed separately. The exact MouseV2 reproduction is an internal positive control for spike extraction and binning.","layout":"full"},
        {"id":"limitations","type":"markdown","body":"## Limitations, uncertainty, and robustness\n\nThe Allen mismatch blocks a fully additive decomposition of the remaining ~3 ms released gap. The local Allen NWBs, current AllenSDK environment, and historical released unit table may differ in stimulus-time correction, unit inclusion, SDK version, or metric-export provenance. Unit-level regression on baseline rate, evoked amplitude, and early occupancy does not remove the cohort coefficient and should not be interpreted causally while the outcome fields are non-equivalent. Dark-flash waveform shapes differ more than bright responses, although their best shifts remain near zero.","layout":"full"},
        {"id":"next","type":"markdown","body":"## Recommended next steps\n\n1. Recover or identify the exact Allen TTFS generation artifact: SDK version, stimulus table, session files, unit inclusion, and export transform.\n2. Recompute TTFS for all datasets through one frozen raw-spike implementation and treat that harmonized field—not the mixed released fields—as the comparison metric.\n3. Require independently verified visual responsiveness, or report responder and nonresponder strata separately, when interpreting TTFS biologically.\n4. Retain the cross-validated PSTH timing analysis as the primary evidence about physical response onset.","layout":"full"},
        {"id":"questions","type":"markdown","body":"## Further questions\n\n- Which historical Allen session/stimulus timestamps produced the released TTFS field?\n- Does the residual harmonized responder gap persist after matching response amplitude and layer?\n- Why are dark-flash post-peak shapes less concordant despite near-zero best shifts?","layout":"full"},
    ]
    charts=[
        {"id":"reproduction_chart","title":"Raw-spike reproduction of released TTFS","subtitle":"Exact fraction by dataset; all finite common-QC V1 units","type":"bar","dataset":"reproduction","sourceId":"reproduction","intent":"comparison","encodings":{"x":{"field":"cohort","type":"nominal","label":"Dataset"},"y":{"field":"exact_fraction","type":"quantitative","format":"percent","label":"Exact fraction"}},"layout":"full"},
        {"id":"gap_chart","title":"MouseV2 minus Allen released TTFS","subtitle":"Equal-session means before and after response selection on independent trials","type":"line","dataset":"gaps","sourceId":"composition","intent":"comparison","encodings":{"x":{"field":"population_label","type":"nominal","label":"Population"},"y":{"field":"mouse_minus_allen_ms","type":"quantitative","label":"TTFS gap","unit":"ms"},"color":{"field":"allen_label","type":"nominal","label":"Allen cohort"}},"layout":"full"},
    ]
    tables=[{"id":"alignment_table","title":"Held-out waveform alignment","subtitle":"Raw, source-predicted, and best-fit MouseV2 shifts over 20–120 ms","dataset":"alignment","sourceId":"alignment","layout":"full","density":"spacious","defaultSort":{"field":"allen_cohort","direction":"asc"},"columns":[{"field":"allen_cohort","label":"Allen cohort","type":"text"},{"field":"polarity","label":"Polarity","type":"text"},{"field":"view","label":"View","type":"text"},{"field":"mouse_shift_ms","label":"Mouse shift (ms)","type":"number"},{"field":"normalized_rmse","label":"RMSE","type":"number"},{"field":"correlation","label":"Correlation","type":"number"}]}]
    artifact={"surface":"report","manifest":{"version":1,"surface":"report","title":"V1 Flash TTFS Discrepancy: Metric and Population Audit","description":"Technical audit of the MouseV2–Allen V1 flash TTFS discrepancy.","generatedAt":now,"blocks":blocks,"charts":charts,"tables":tables,"sources":sources},"snapshot":{"version":1,"generatedAt":now,"status":"ready","datasets":{"reproduction":reproduction.to_dict("records"),"gaps":gaps[["allen_label","population_label","mouse_minus_allen_ms","gap_reduction_fraction"]].round(6).to_dict("records"),"alignment":alignment[["allen_cohort","polarity","view","mouse_shift_ms","normalized_rmse","correlation"]].to_dict("records"),"centers":centers.to_dict("records")},"accessIssues":[]},"sources":sources}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"artifact.json").write_text(json.dumps(artifact,indent=2)+"\n")


if __name__=="__main__":main()
