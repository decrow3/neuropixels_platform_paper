#!/usr/bin/env python3
"""Decompose V1 flash TTFS into baseline, response, and spike-occupancy inputs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units
from aggregate_response_verified_early_v1_psths import (
    COHORTS, COLORS, CENTERS_MS, EDGES_S, read_targeted_nwb,
)
from crossvalidated_v1_flash_timing import all_unit_inputs

OUT = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_metric_decomposition"
BASELINE = (CENTERS_MS >= -100) & (CENTERS_MS < 0)
RESPONSE = (CENTERS_MS >= 30) & (CENTERS_MS < 180)
PEAK = (CENTERS_MS >= 20) & (CENTERS_MS < 120)


def trial_counts(spikes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(spikes, starts[:, None] + EDGES_S[None, :], side="left")
    return np.diff(indices, axis=1).astype(float)


def first_spike_components(spikes: np.ndarray, starts: np.ndarray):
    start30 = starts + .030; stop60 = starts + .060; stop200 = starts + .200
    first_idx = np.searchsorted(spikes, start30, side="left")
    stop60_idx = np.searchsorted(spikes, stop60, side="left")
    stop200_idx = np.searchsorted(spikes, stop200, side="left")
    valid = first_idx < stop200_idx
    early = first_idx < stop60_idx
    if valid.any():
        first_bins = ((spikes[first_idx[valid]] - starts[valid]) * 1000).astype(np.int64)
        ttfs = float(np.median(first_bins))
        q25, q75 = np.percentile(first_bins, [25, 75])
    else:
        ttfs = q25 = q75 = np.nan
    return ttfs, float(valid.mean()), float(early.mean()), float(q25), float(q75)


def extract_session(cohort, sid, units, path, flash_table):
    spikes_by_id, starts, _ = read_targeted_nwb(
        path, flash_table, units["source_unit_id"].astype(int).to_numpy()
    )
    rows = []
    for _, unit in units.iterrows():
        uid, source = int(unit["analysis_unit_id"]), int(unit["source_unit_id"])
        spikes = spikes_by_id[source]
        counts = trial_counts(spikes, starts)
        rate = counts.mean(axis=0) / np.diff(EDGES_S)
        baseline = float(rate[BASELINE].mean())
        response = float(rate[RESPONSE].mean())
        delta = response - baseline
        smooth = gaussian_filter1d(rate - baseline, 1.0)
        peak_time = float(CENTERS_MS[PEAK][np.argmax(smooth[PEAK])])
        peak_delta = float(np.max(smooth[PEAK]))
        recomputed, occupancy, early_occupancy, q25, q75 = first_spike_components(spikes, starts)
        released = float(1000 * unit["time_to_first_spike_fl"]) if pd.notna(unit["time_to_first_spike_fl"]) else np.nan
        rows.append({
            "cohort": cohort, "session_id": sid, "unit_id": uid,
            "source_unit_id": source, "flash_trials": len(starts),
            "released_ttfs_ms": released, "recomputed_ttfs_ms": recomputed,
            "ttfs_abs_error_ms": abs(released - recomputed) if np.isfinite(released) and np.isfinite(recomputed) else np.nan,
            "baseline_rate_hz": baseline, "response_rate_30_180_hz": response,
            "evoked_delta_30_180_hz": delta, "positive_evoked_delta_hz": max(delta, 0),
            "peak_delta_rate_hz": peak_delta, "psth_peak_time_ms": peak_time,
            "occupancy_30_200": occupancy, "occupancy_30_60": early_occupancy,
            "first_spike_iqr_ms": q75-q25, "first_spike_q25_ms": q25, "first_spike_q75_ms": q75,
        })
    return pd.DataFrame(rows)


def session_balanced_summary(data):
    metric_cols = [
        "released_ttfs_ms", "recomputed_ttfs_ms", "baseline_rate_hz",
        "positive_evoked_delta_hz", "peak_delta_rate_hz", "psth_peak_time_ms",
        "occupancy_30_200", "occupancy_30_60", "first_spike_iqr_ms",
    ]
    valid = data.loc[data.released_ttfs_ms.lt(100)].copy()
    session = valid.groupby(["cohort", "session_id"])[metric_cols].mean().reset_index()
    summary = session.groupby("cohort")[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    return valid, session, summary.reset_index()


def standardized_gap(data, predictors):
    """Return cohort coefficients from transparent OLS with session-cluster bootstrap."""
    d = data.loc[data.released_ttfs_ms.lt(100)].copy()
    d["bo"] = d.cohort.eq("Allen Brain Observatory").astype(float)
    d["fc"] = d.cohort.eq("Allen Functional Connectivity").astype(float)
    columns = ["intercept", "bo", "fc"] + predictors
    xparts = [np.ones(len(d)), d.bo.to_numpy(), d.fc.to_numpy()]
    for p in predictors:
        values = d[p].to_numpy(float)
        values = (values - np.nanmean(values)) / np.nanstd(values)
        xparts.append(values)
    X = np.column_stack(xparts); y = d.released_ttfs_ms.to_numpy(float)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    return pd.DataFrame({"term": columns, "coefficient_ms": beta, "predictors": "+".join(predictors) or "none", "n_units": int(keep.sum())})


def correlations(data):
    rows=[]
    metrics=["baseline_rate_hz","positive_evoked_delta_hz","peak_delta_rate_hz","occupancy_30_200","occupancy_30_60","psth_peak_time_ms"]
    d=data.loc[data.released_ttfs_ms.lt(100)]
    for cohort,g in d.groupby("cohort"):
        for metric in metrics:
            z=g[["released_ttfs_ms",metric]].dropna()
            rho,p=spearmanr(z.released_ttfs_ms,z[metric])
            rows.append({"cohort":cohort,"metric":metric,"n_units":len(z),"spearman_rho":rho,"p_value":p})
    return pd.DataFrame(rows)


def select_cases(data):
    d=data.loc[data.released_ttfs_ms.lt(100)].copy()
    d["ttfs_rank"] = d.groupby("cohort").released_ttfs_ms.rank(pct=True)
    d["occupancy_rank"] = d.groupby("cohort").occupancy_30_60.rank(pct=True)
    roles=[]
    definitions={
        "early TTFS / high early occupancy": (d.ttfs_rank<.15)&(d.occupancy_rank>.85),
        "late TTFS / low early occupancy": (d.ttfs_rank>.85)&(d.occupancy_rank<.15),
        "early TTFS despite weak evoked response": (d.ttfs_rank<.15)&(d.positive_evoked_delta_hz<1),
        "late TTFS despite strong evoked response": (d.ttfs_rank>.85)&(d.positive_evoked_delta_hz>d.positive_evoked_delta_hz.quantile(.75)),
    }
    for cohort in COHORTS:
        for role,mask in definitions.items():
            candidates=d.loc[d.cohort.eq(cohort)&mask]
            if candidates.empty: continue
            if "high early" in role: row=candidates.sort_values("occupancy_30_60",ascending=False).iloc[0]
            elif "low early" in role: row=candidates.sort_values("occupancy_30_60").iloc[0]
            elif "weak" in role: row=candidates.sort_values("positive_evoked_delta_hz").iloc[0]
            else: row=candidates.sort_values("positive_evoked_delta_hz",ascending=False).iloc[0]
            item=row.to_dict();item["selection_role"]=role;roles.append(item)
    return pd.DataFrame(roles)


def render(data, session, corr, models, cases):
    valid=data.loc[data.released_ttfs_ms.lt(100)].copy()
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for cohort in COHORTS:
        g=valid.loc[valid.cohort.eq(cohort)]
        axes[0,0].hexbin(g.occupancy_30_60,g.released_ttfs_ms,gridsize=28,mincnt=1,cmap="Greys",alpha=.35)
        sample=g.sample(min(700,len(g)),random_state=20260825)
        axes[0,0].scatter(sample.occupancy_30_60,sample.released_ttfs_ms,s=8,alpha=.18,color=COLORS[cohort],label=cohort)
        axes[0,1].scatter(sample.baseline_rate_hz,sample.released_ttfs_ms,s=8,alpha=.18,color=COLORS[cohort],label=cohort)
        axes[0,2].scatter(sample.positive_evoked_delta_hz,sample.released_ttfs_ms,s=8,alpha=.18,color=COLORS[cohort],label=cohort)
    axes[0,0].set(xlabel="Probability of a spike from 30–60 ms",ylabel="Released TTFS (ms)",title="Early spike occupancy is the direct TTFS input")
    axes[0,1].set(xlabel="Pre-flash baseline firing rate (Hz)",ylabel="Released TTFS (ms)",title="Baseline firing can create an early first spike")
    axes[0,2].set(xlabel="Positive evoked mean-rate increase (Hz)",ylabel="Released TTFS (ms)",title="Response amplitude also advances first-spike timing")
    axes[0,0].legend(frameon=False,fontsize=7)
    metrics=[("released_ttfs_ms","TTFS (ms)"),("occupancy_30_60","30–60 ms occupancy"),("baseline_rate_hz","baseline rate (Hz)"),("positive_evoked_delta_hz","positive evoked Δ (Hz)")]
    x=np.arange(len(metrics)); width=.24
    for i,cohort in enumerate(COHORTS):
        g=session.loc[session.cohort.eq(cohort)]
        vals=[g[m].mean() for m,_ in metrics]
        # Normalize heterogeneous metrics to the Allen BO center for a compact driver view.
        refs=[session.loc[session.cohort.eq("Allen Brain Observatory"),m].mean() for m,_ in metrics]
        axes[1,0].bar(x+(i-1)*width,np.array(vals)/np.array(refs),width,color=COLORS[cohort],label=cohort)
    axes[1,0].set(xticks=x,xticklabels=[label for _,label in metrics],ylabel="Ratio to Allen BO session mean",title="Dataset-level TTFS and estimator inputs")
    axes[1,0].tick_params(axis="x",rotation=25);axes[1,0].legend(frameon=False,fontsize=7)
    coef=models.pivot(index="predictors",columns="term",values="coefficient_ms")
    for term,color in (("bo",COLORS["Allen Brain Observatory"]),("fc",COLORS["Allen Functional Connectivity"])):
        axes[1,1].plot(np.arange(len(coef)),coef[term],marker="o",lw=2,color=color,label=term.upper())
    axes[1,1].axhline(0,color="0.4",lw=.8);axes[1,1].set(xticks=np.arange(len(coef)),xticklabels=["raw","+baseline","+evoked","+occupancy","+all"],ylabel="Allen minus MouseV2 coefficient (ms)",title="Gap after sequential unit-level adjustment")
    axes[1,1].tick_params(axis="x",rotation=25);axes[1,1].legend(frameon=False)
    axes[1,2].axis("off")
    med=valid.groupby("cohort").agg(ttfs=("released_ttfs_ms","mean"),occ=("occupancy_30_60","mean"),base=("baseline_rate_hz","mean"),evoked=("positive_evoked_delta_hz","mean"))
    lines=["Unit-level decomposition (TTFS <100 ms)",""]
    for cohort,row in med.iterrows(): lines.append(f"{cohort}:\n  TTFS {row.ttfs:.1f} ms; early occupancy {row.occ:.3f}\n  baseline {row.base:.2f} Hz; positive evoked Δ {row.evoked:.2f} Hz")
    lines += ["",f"Released vs recomputed median |error|: {data.ttfs_abs_error_ms.median():.3f} ms",f"Inspectable diagnostic cases saved: {len(cases)}"]
    axes[1,2].text(0,1,"\n".join(lines),va="top",family="monospace",fontsize=8.5)
    for ax in axes.flat[:5]: ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Why released V1 TTFS differs when held-out response timing does not")
    fig.savefig(OUT/"ttfs_metric_decomposition.png",dpi=220,bbox_inches="tight")
    fig.savefig(OUT/"ttfs_metric_decomposition.pdf",bbox_inches="tight")
    plt.close(fig)


def main():
    allen=load_allen_units(population_profile="common_qc");allen=allen.loc[allen.area_coarse.eq("V1")].copy()
    mouse=load_mousev2_units(apply_qc=False,grating_metrics_dir=ROOT/"data/imports/mousev2_grating_metrics_v1",flash_metrics_dir=ROOT/"data/imports/mousev2_flash_metrics_v1",flash_variant="pooled",population_profile="common_qc")
    OUT.mkdir(parents=True,exist_ok=True)
    frames=[];status=[]
    for cohort,sid,units,path,flash_table in all_unit_inputs(allen,mouse):
        print(f"[{cohort} {sid}] {len(units)} units",flush=True)
        try:
            frame=extract_session(cohort,sid,units,path,flash_table);frames.append(frame)
            status.append({"cohort":cohort,"session_id":sid,"status":"ok","units":len(frame),"error":""})
        except Exception as exc:
            status.append({"cohort":cohort,"session_id":sid,"status":"error","units":0,"error":f"{type(exc).__name__}: {exc}"})
    data=pd.concat(frames,ignore_index=True)
    valid,session,summary=session_balanced_summary(data)
    corr=correlations(data)
    model_specs=[[],["baseline_rate_hz"],["baseline_rate_hz","positive_evoked_delta_hz"],["occupancy_30_60"],["baseline_rate_hz","positive_evoked_delta_hz","occupancy_30_60"]]
    models=pd.concat([standardized_gap(data,p) for p in model_specs],ignore_index=True)
    cases=select_cases(data)
    data.to_csv(OUT/"unit_level_ttfs_components.csv",index=False)
    valid.to_csv(OUT/"unit_level_ttfs_under_100ms.csv",index=False)
    session.to_csv(OUT/"session_level_ttfs_components.csv",index=False)
    summary.to_csv(OUT/"cohort_session_balanced_summary.csv",index=False)
    corr.to_csv(OUT/"ttfs_component_correlations.csv",index=False)
    models.to_csv(OUT/"sequential_gap_adjustment.csv",index=False)
    cases.to_csv(OUT/"diagnostic_cases.csv",index=False)
    pd.DataFrame(status).to_csv(OUT/"session_processing_status.csv",index=False)
    render(data,session,corr,models,cases)


if __name__=="__main__": main()
