#!/usr/bin/env python3
"""Build consistent response-filtered preferred-TTFS session means for Figure 3."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from aggregate_response_verified_early_v1_psths import session_inputs
from crossvalidated_v1_flash_timing import crossvalidated_session
from plot_response_verified_early_v1_psths import bh_adjust
from reconstruct_historical_allen_ttfs import process_session

BASE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance"
OUT = BASE / "figure3_response_filtered_ttfs_override.csv"
AUDIT_OUT = BASE / "figure3_response_filtered_ttfs_all_areas_unit_audit.csv"
STATUS_OUT = BASE / "figure3_response_filtered_ttfs_all_areas_status.csv"
TARGETS = {"V1", "LM", "RL", "LP", "AL", "PM", "AM"}
MIN_UNITS = 10


def main():
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen.area_coarse.isin(TARGETS)].copy()
    mouse = load_mousev2_units(
        apply_qc=False, grating_metrics_dir=ROOT/"data/imports/mousev2_grating_metrics_v1",
        flash_metrics_dir=ROOT/"data/imports/mousev2_flash_metrics_v1",
        flash_variant="pooled", population_profile="common_qc")
    # Make every unit pass session_inputs' legacy TTFS candidate routing.
    ac = allen.copy(); mc = mouse.copy()
    ac["_released"] = ac.time_to_first_spike_fl; mc["_released"] = mc.time_to_first_spike_fl
    ac.time_to_first_spike_fl = 0.; mc.time_to_first_spike_fl = 0.
    audits=[]; status=[]
    for cohort,sid,units,path,flash_table in session_inputs(ac,mc):
        if cohort == "MouseV2": continue
        units=units.copy(); units["time_to_first_spike_fl"] = units["_released"]
        print(f"[{cohort} {sid}] {len(units)} visual units",flush=True)
        try:
            response,_ = crossvalidated_session(cohort,sid,units,path,flash_table)
            latency = process_session(cohort,sid,units,path,flash_table)
            metadata = units[["analysis_unit_id","area_coarse"]].rename(columns={"analysis_unit_id":"unit_id"})
            response=response.merge(metadata,on="unit_id",validate="many_to_one")
            response["selection_fdr_q_area"] = np.nan
            for _,idx in response.groupby(["area_coarse","polarity"]).groups.items():
                response.loc[idx,"selection_fdr_q_area"] = bh_adjust(response.loc[idx,"selection_wilcoxon_p"].to_numpy())
            merged=response.merge(latency[["unit_id","preferred_polarity","preferred_0_250_ttfs_ms"]],on="unit_id",validate="many_to_one")
            merged=merged.loc[merged.polarity.eq(merged.preferred_polarity)].copy()
            merged["selected_positive_responder_area"]=(merged.selection_response_minus_baseline_hz.ge(2)&merged.selection_fdr_q_area.lt(.01))
            audits.append(merged)
            status.append({"cohort":cohort,"session_id":sid,"status":"ok","units":len(units),"error":""})
        except Exception as exc:
            print(f" ERROR {type(exc).__name__}: {exc}",flush=True)
            status.append({"cohort":cohort,"session_id":sid,"status":"error","units":0,"error":f"{type(exc).__name__}: {exc}"})
    audit=pd.concat(audits,ignore_index=True)
    chosen=audit.loc[audit.selected_positive_responder_area & audit.preferred_0_250_ttfs_ms.lt(100)].copy()
    allen_session=chosen.groupby(["session_id","area_coarse"],as_index=False).agg(
        mean=("preferred_0_250_ttfs_ms","mean"),n_units=("unit_id","size"))
    allen_session=allen_session.loc[allen_session.n_units.ge(MIN_UNITS)].copy()
    allen_session["dataset"]=np.where(allen_session.area_coarse.eq("V1"),"Allen-V1","Post-V1")
    allen_session["group"]=allen_session.area_coarse.replace({"V1":"Visual Coding VISp"})
    allen_session["metric"]="TTFS (ms)"

    mouse_session=pd.read_csv(BASE/"figure3_response_filtered_preferred_ttfs_sessions.csv")
    mouse_session=mouse_session.loc[
        mouse_session.cohort.eq("MouseV2") & mouse_session.included_min_units
    ].copy()
    mouse_session=mouse_session.rename(columns={"mean_ttfs_ms":"mean","units":"n_units","location":"group"})
    mouse_session["dataset"]="Within-V1"; mouse_session["metric"]="TTFS (ms)"
    override=pd.concat([
        mouse_session[["dataset","metric","session_id","group","mean","n_units"]],
        allen_session[["dataset","metric","session_id","group","mean","n_units"]]
    ],ignore_index=True)
    override["session_id"]=override.session_id.astype(str)
    override["session_mean"]=override.groupby(["dataset","metric","session_id"])["mean"].transform("mean")
    override["centered_mean"]=override["mean"]-override["session_mean"]
    override["n_groups_in_session"]=override.groupby(["dataset","metric","session_id"])["group"].transform("size")
    override.to_csv(OUT,index=False); audit.to_csv(AUDIT_OUT,index=False); pd.DataFrame(status).to_csv(STATUS_OUT,index=False)
    print(override.groupby(["dataset","group"]).agg(sessions=("session_id","nunique"),units=("n_units","sum"),mean=("mean","mean")).to_string())

if __name__ == "__main__": main()
