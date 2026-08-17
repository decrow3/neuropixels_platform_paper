#!/usr/bin/env python3
"""Cross-session population gaze calibration using bounded analytic-aperture RFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from compare_allen_point_vs_aperture_rf import fit_unit


ROOT=Path(__file__).resolve().parents[1]
SESSIONS=(746083955,755434585,760693773,798911424)
GAINS=(0.0,0.5,1.0,1.5)
DEFAULT_CACHE=ROOT/"artifacts"/"allen_population_gaze_rf"
DEFAULT_OUTPUT=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"/"04_gaze"
SEED=20260815


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions",nargs="+",type=int,default=SESSIONS)
    p.add_argument("--cache-root",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--calibration-units",type=int,default=64)
    p.add_argument("--evaluation-units",type=int,default=80)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--resume",action="store_true")
    return p.parse_args()


def balanced_subset(population,split,limit):
    local=population.loc[population.published_like_qc.astype(bool)&population.unit_split.eq(split)]
    if limit is None or limit>=len(local):return local
    pools={g:list(x.index) for g,x in local.groupby("group",observed=True)}; indices=[]
    while len(indices)<limit and any(pools.values()):
        for g in ("V1","HVA"):
            if pools.get(g) and len(indices)<limit:indices.append(pools[g].pop(0))
    return local.loc[indices].sort_index()


def candidates(trials,session_id):
    rows=[{"candidate":f"gx_{gx:g}_gy_{gy:g}","gain_x":gx,"gain_y":gy,"control":False,"shuffle":False}
          for gx in GAINS for gy in GAINS]
    rows += [{"candidate":"sign_reversed","gain_x":-1.0,"gain_y":-1.0,"control":True,"shuffle":False},
             {"candidate":"time_shuffled","gain_x":1.0,"gain_y":1.0,"control":True,"shuffle":True}]
    rng=np.random.default_rng(SEED+session_id); permutation=np.arange(len(trials))
    valid=np.flatnonzero(trials.valid_gaze.to_numpy(bool));permutation[valid]=rng.permutation(valid)
    return pd.DataFrame(rows),permutation


def coordinates(trials,row,permutation):
    dx=trials.gaze_dx_deg.to_numpy(float);dy=trials.gaze_dy_deg.to_numpy(float)
    if row.shuffle:dx=dx[permutation];dy=dy[permutation]
    return (trials.x_position.to_numpy(float)-float(row.gain_x)*dx,
            trials.y_position.to_numpy(float)-float(row.gain_y)*dy)


def metric_record(session_id,unit,row,metrics,nominal):
    return {"session_id":session_id,"ecephys_unit_id":int(unit.ecephys_unit_id),"group":unit.group,
            "ecephys_structure_acronym":unit.ecephys_structure_acronym,"unit_split":unit.unit_split,
            "candidate":row.candidate,"gain_x":row.gain_x,"gain_y":row.gain_y,
            "control":row.control,"shuffle":row.shuffle,
            "test_deviance":metrics["test_poisson_deviance"],
            "train_deviance":metrics["train_poisson_deviance"],
            "area_deg2":metrics["latent_halfmax_area_deg2"],
            "amplitude_spikes":metrics["mean_amplitude_spikes"],
            "center_x_deg":metrics["center_x_deg"],"center_y_deg":metrics["center_y_deg"],
            "sigma_x_deg":metrics["sigma_x_deg"],"sigma_y_deg":metrics["sigma_y_deg"],
            "censored":metrics["censored"],
            "test_deviance_improvement":nominal["test_poisson_deviance"]-metrics["test_poisson_deviance"],
            "log2_area_ratio":np.log2(metrics["latent_halfmax_area_deg2"]/nominal["latent_halfmax_area_deg2"]),
            "log2_amplitude_ratio":np.log2((metrics["mean_amplitude_spikes"]+1e-6)/
                                            (nominal["mean_amplitude_spikes"]+1e-6))}


def fit_units(session_id,units,population,trials,counts,candidate_table,permutation,chosen=None):
    valid=trials.valid_gaze.to_numpy(bool);train=valid&trials.trial_split.eq("train").to_numpy(bool)
    test=valid&trials.trial_split.eq("test").to_numpy(bool);orientation=trials.orientation_index.to_numpy(int)
    nominal_row=candidate_table.loc[candidate_table.candidate.eq("gx_0_gy_0")].iloc[0]
    rows=[]
    for progress,unit in enumerate(units.itertuples(),start=1):
        unit_counts=counts[unit.Index].astype(float);x0,y0=coordinates(trials,nominal_row,permutation)
        nominal_parameters,nominal=fit_unit(unit_counts,x0,y0,orientation,train,test,unit.group,"aperture")
        use=candidate_table if chosen is None else candidate_table.loc[
            candidate_table.candidate.isin(["gx_0_gy_0",chosen,"sign_reversed","time_shuffled"])]
        for candidate in use.itertuples(index=False):
            if candidate.candidate=="gx_0_gy_0":parameters,metrics=nominal_parameters,nominal
            else:
                x,y=coordinates(trials,candidate,permutation)
                parameters,metrics=fit_unit(unit_counts,x,y,orientation,train,test,unit.group,
                                            "aperture",start=nominal_parameters)
            rows.append(metric_record(session_id,unit,candidate,metrics,nominal))
        if progress%8==0 or progress==len(units):
            print(f"Session {session_id} {units.unit_split.iloc[0]}: {progress}/{len(units)} units",flush=True)
    return pd.DataFrame(rows)


def select_candidate(calibration):
    summary=calibration.groupby(["candidate","gain_x","gain_y","control"],observed=True).agg(
        units=("ecephys_unit_id","nunique"),
        median_test_gain=("test_deviance_improvement","median"),
        fraction_test_gain_positive=("test_deviance_improvement",lambda x:(x>0).mean()),
        median_log2_area_ratio=("log2_area_ratio","median"),
        median_log2_amplitude_ratio=("log2_amplitude_ratio","median"),
        censored_fraction=("censored","mean")).reset_index()
    chosen=summary.loc[~summary.control.astype(bool)].sort_values(
        ["median_test_gain","fraction_test_gain_positive"],ascending=False).iloc[0].candidate
    return chosen,summary


def render(session_id,summary,evaluation,trials,path,chosen):
    grid=summary.loc[~summary.control.astype(bool)].pivot(index="gain_y",columns="gain_x",values="median_test_gain")
    selected=evaluation.loc[evaluation.candidate.eq(chosen)]
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.5),constrained_layout=True)
    lim=max(float(np.nanmax(np.abs(grid))),1e-6)
    im=axes[0,0].imshow(grid.values,origin="lower",cmap="coolwarm",norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    axes[0,0].set(xticks=np.arange(len(grid.columns)),xticklabels=[f"{v:g}" for v in grid.columns],
                  yticks=np.arange(len(grid.index)),yticklabels=[f"{v:g}" for v in grid.index],
                  xlabel="Horizontal gain",ylabel="Vertical gain",title="Calibration-population held-out gain")
    crow=summary.loc[summary.candidate.eq(chosen)].iloc[0]
    axes[0,0].scatter(list(grid.columns).index(crow.gain_x),list(grid.index).index(crow.gain_y),marker="*",s=170,color="#222")
    fig.colorbar(im,ax=axes[0,0],label="Median test-deviance improvement")
    colors={"V1":"#3366aa","HVA":"#d97736"}
    for group in ("V1","HVA"):
        local=selected.loc[selected.group.eq(group)]
        axes[0,1].hist(local.test_deviance_improvement,bins=25,histtype="step",lw=2,color=colors[group],label=f"{group} n={len(local)}")
        axes[1,0].scatter(local.log2_area_ratio,local.log2_amplitude_ratio,s=20,alpha=.6,color=colors[group],label=group)
    axes[0,1].axvline(0,color="#555",ls="--");axes[0,1].set(xlabel="Held-out deviance improvement",ylabel="Evaluation units",title="Unseen-neuron prediction")
    axes[1,0].axvline(0,color="#555",ls="--");axes[1,0].axhline(0,color="#555",ls="--")
    axes[1,0].set(xlabel="log₂ corrected / nominal latent area",ylabel="log₂ corrected / nominal amplitude",title="Sharpening and magnitude")
    controls=evaluation.groupby("candidate",observed=True).test_deviance_improvement.median().sort_values()
    axes[1,1].barh(np.arange(len(controls)),controls.values,color=["#3366aa" if c==chosen else "#999" for c in controls.index])
    axes[1,1].axvline(0,color="#555",ls="--");axes[1,1].set(yticks=np.arange(len(controls)),yticklabels=controls.index,
                  xlabel="Median evaluation-neuron test gain",title="Chosen trace versus controls")
    for axis in axes.ravel():axis.grid(alpha=.14);axis.legend(frameon=False) if axis in (axes[0,1],axes[1,0]) else None
    fig.suptitle(f"Session {session_id}: population gaze correction with analytic aperture RF",fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def main():
    args=parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    session_summaries=[]
    for sid in args.sessions:
        out=output/f"session_{sid}";out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()) and not (args.overwrite or args.resume):raise FileExistsError(f"{out} exists; use --overwrite or --resume")
        cache=args.cache_root.resolve()/f"session_{sid}"
        population=pd.read_csv(cache/"visual_unit_population.csv",low_memory=False)
        trials=pd.read_csv(cache/"gabor_trial_gaze_table.csv",low_memory=False)
        counts=np.load(cache/"gabor_spike_counts.npz")["counts"]
        table,permutation=candidates(trials,sid)
        calibration_units=balanced_subset(population,"calibration",args.calibration_units)
        evaluation_units=balanced_subset(population,"evaluation",args.evaluation_units)
        saved=[out/"calibration_unit_sweep.csv",out/"calibration_candidate_summary.csv",out/"evaluation_unit_results.csv"]
        if args.resume and all(path.exists() for path in saved):
            calibration=pd.read_csv(saved[0],low_memory=False)
            cal_summary=pd.read_csv(saved[1],low_memory=False)
            chosen,_=select_candidate(calibration)
            evaluation=pd.read_csv(saved[2],low_memory=False)
        else:
            calibration=fit_units(sid,calibration_units,population,trials,counts,table,permutation)
            chosen,cal_summary=select_candidate(calibration)
            evaluation=fit_units(sid,evaluation_units,population,trials,counts,table,permutation,chosen)
            calibration.to_csv(saved[0],index=False,float_format="%.9g")
            cal_summary.to_csv(saved[1],index=False,float_format="%.9g")
            evaluation.to_csv(saved[2],index=False,float_format="%.9g")
        render(sid,cal_summary,evaluation,trials,out/"Figure_aperture_gaze_validation.png",chosen)
        selected=evaluation.loc[evaluation.candidate.eq(chosen)]
        session_summaries.append({"session_id":sid,"chosen_candidate":chosen,
            "chosen_gain_x":float(cal_summary.loc[cal_summary.candidate.eq(chosen),"gain_x"].iloc[0]),
            "chosen_gain_y":float(cal_summary.loc[cal_summary.candidate.eq(chosen),"gain_y"].iloc[0]),
            "calibration_units":len(calibration_units),"evaluation_units":len(evaluation_units),
            "median_evaluation_test_gain":selected.test_deviance_improvement.median(),
            "fraction_evaluation_gain_positive":selected.test_deviance_improvement.gt(0).mean(),
            "median_evaluation_log2_area_ratio":selected.log2_area_ratio.median(),
            "median_evaluation_log2_amplitude_ratio":selected.log2_amplitude_ratio.median(),
            "valid_gaze_fraction":trials.valid_gaze.mean(),
            "gaze_dx_sd_deg":trials.loc[trials.valid_gaze,"gaze_dx_deg"].std(),
            "gaze_dy_sd_deg":trials.loc[trials.valid_gaze,"gaze_dy_deg"].std()})
    summary=pd.DataFrame(session_summaries);summary.to_csv(output/"all_session_gaze_summary.csv",index=False,float_format="%.9g")
    print(summary.to_string(index=False))


if __name__=="__main__":main()
