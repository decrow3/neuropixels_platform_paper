#!/usr/bin/env python3
"""Extract and fit raw LGd Gabor maps in concrete cross-session cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compare_allen_point_vs_aperture_rf import fit_unit, model_prediction, poisson_deviance


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = Path("/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions/session_inventory.json")
UNIT_TABLE = ROOT / "data" / "unit_table.csv"
OUTPUT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint" / "lgd_gabor_boundary_pilot"
SESSIONS = (755434585, 760345702, 754829445)
RESPONSE_WINDOW = .249


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=int, nargs="+", default=SESSIONS)
    p.add_argument("--all-lgd", action="store_true", help="Process every session containing LGd units.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def extract_session(nwb_path, population, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(nwb_path, "r") as nwb:
        g = nwb["intervals/gabors_presentations"]
        trials = pd.DataFrame({k: g[k][()] for k in ("start_time", "stop_time", "x_position", "y_position", "orientation")})
        trials["orientation_index"] = trials.orientation.map({v:i for i,v in enumerate(sorted(trials.orientation.unique()))})
        trials["repeat"] = trials.groupby(["x_position", "y_position", "orientation"], observed=True).cumcount()
        trials["trial_split"] = np.where(trials.repeat.mod(3).eq(2), "test", "train")
        nwb_ids = nwb["units/id"][()].astype(int)
        id_to_row = {uid:i for i,uid in enumerate(nwb_ids)}
        spike_index = nwb["units/spike_times_index"][()].astype(np.int64)
        spike_data = nwb["units/spike_times"]
        starts = trials.start_time.to_numpy(float); stops = starts + RESPONSE_WINDOW
        counts = np.zeros((len(population), len(trials)), dtype=np.int16)
        for out_row, uid in enumerate(population.ecephys_unit_id.astype(int)):
            row = id_to_row[uid]
            left = 0 if row == 0 else spike_index[row-1]
            right = spike_index[row]
            spikes = spike_data[left:right]
            counts[out_row] = np.searchsorted(spikes, stops, side="left") - np.searchsorted(spikes, starts, side="left")
    population.reset_index(drop=True).to_csv(cache_dir / "lgd_population.csv", index=False)
    trials.to_csv(cache_dir / "gabor_trials.csv", index=False)
    np.savez_compressed(cache_dir / "gabor_counts.npz", counts=counts, unit_ids=population.ecephys_unit_id.to_numpy(int))
    return trials, counts


def load_or_extract(sid, nwb_path, population, cache_dir, overwrite):
    paths = [cache_dir / "lgd_population.csv", cache_dir / "gabor_trials.csv", cache_dir / "gabor_counts.npz"]
    if overwrite or not all(p.exists() for p in paths):
        return extract_session(nwb_path, population, cache_dir)
    trials = pd.read_csv(paths[1]); payload = np.load(paths[2]); return trials, payload["counts"]


def null_deviance(counts, orientation, train, test):
    prediction = np.zeros(test.sum())
    test_orientation = orientation[test]
    for ori in np.unique(orientation):
        prediction[test_orientation == ori] = np.mean(counts[train & (orientation == ori)])
    return poisson_deviance(counts[test], prediction)


def fit_session(sid, population, trials, counts, output, overwrite):
    path = output / "lgd_aperture_fits.csv"
    if path.exists() and not overwrite:
        return pd.read_csv(path)
    x=trials.x_position.to_numpy(float); y=trials.y_position.to_numpy(float)
    ori=trials.orientation_index.to_numpy(int); train=trials.trial_split.eq("train").to_numpy(); test=~train
    rows=[]
    for i, unit in enumerate(population.itertuples()):
        c=counts[i].astype(float)
        parameters, fit=fit_unit(c,x,y,ori,train,test,"HVA","aperture")
        null=null_deviance(c,ori,train,test)
        edge=min(40-abs(fit["center_x_deg"]),40-abs(fit["center_y_deg"]))
        rows.append({
            "ecephys_session_id":sid,"ecephys_unit_id":unit.ecephys_unit_id,
            "center_x_deg":fit["center_x_deg"],"center_y_deg":fit["center_y_deg"],
            "sigma_x_deg":fit["sigma_x_deg"],"sigma_y_deg":fit["sigma_y_deg"],
            "edge_distance_deg":edge,"parameter_censored":fit["censored"],
            "test_deviance":fit["test_poisson_deviance"],"null_test_deviance":null,
            "heldout_spatial_gain":null-fit["test_poisson_deviance"],
            "mean_amplitude_spikes":fit["mean_amplitude_spikes"],
            "parameters":json.dumps(parameters.tolist()),
        })
        if (i+1)%20==0 or i+1==len(population):
            pd.DataFrame(rows).to_csv(path,index=False)
            print(f"{sid}: fit {i+1}/{len(population)}",flush=True)
    return pd.DataFrame(rows)


def response_map(counts, trials):
    frame=pd.DataFrame({"x":trials.x_position,"y":trials.y_position,"count":counts})
    pivot=frame.groupby(["y","x"],observed=True)["count"].mean().unstack("x")
    return pivot.sort_index(ascending=True)


def select_examples(fits):
    localized=fits.loc[fits.heldout_spatial_gain>0].copy()
    interior=localized.loc[localized.edge_distance_deg>=10].sort_values("heldout_spatial_gain",ascending=False)
    boundary=localized.loc[localized.edge_distance_deg<0].sort_values("heldout_spatial_gain",ascending=False)
    weak=fits.iloc[(fits.heldout_spatial_gain.abs()).argsort()]
    return {
        "strong interior": int(interior.iloc[0].ecephys_unit_id) if len(interior) else None,
        "strong boundary": int(boundary.iloc[0].ecephys_unit_id) if len(boundary) else None,
        "weak/control": int(weak.iloc[0].ecephys_unit_id) if len(weak) else None,
    }


def render_session(sid,population,trials,counts,fits,output):
    merged=population.merge(fits,on=["ecephys_session_id","ecephys_unit_id"])
    examples=select_examples(fits)
    (output/"selected_examples.json").write_text(json.dumps(examples,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    good=merged.loc[merged.heldout_spatial_gain>0]
    for col,(value,title,cmap,limits) in enumerate([
        ("center_x_deg","fitted Gabor-grid x","turbo",(-60,60)),
        ("center_y_deg","fitted Gabor-grid y","coolwarm",(-60,60)),
    ]):
        ax=axes[0,col]
        sc=ax.scatter(good.left_right_ccf_coordinate/1000,good.dorsal_ventral_ccf_coordinate/1000,
                      c=good[value],s=35+80*np.clip(good.heldout_spatial_gain,0,.2),cmap=cmap,vmin=limits[0],vmax=limits[1],edgecolor="k",linewidth=.2)
        ax.set_aspect("equal");ax.set_xlabel("CCF ML (mm)");ax.set_ylabel("CCF DV (mm)");ax.set_title(title)
        fig.colorbar(sc,ax=ax,shrink=.75,label="degrees")
    ax=axes[0,2]
    ax.scatter(merged.edge_distance_deg,merged.heldout_spatial_gain,s=22,alpha=.7)
    ax.axvline(0,color=".6",lw=1);ax.axhline(0,color=".6",lw=1)
    ax.set(xlabel="fitted center distance inside grid (deg)",ylabel="held-out spatial deviance gain",title="Boundary fits can generalize")
    for col,(role,uid) in enumerate(examples.items()):
        ax=axes[1,col]
        if uid is None: ax.axis("off");continue
        idx=population.index[population.ecephys_unit_id.eq(uid)][0]
        m=response_map(counts[idx],trials); row=fits.loc[fits.ecephys_unit_id.eq(uid)].iloc[0]
        im=ax.imshow(m.to_numpy(),origin="lower",extent=[m.columns.min()-5,m.columns.max()+5,m.index.min()-5,m.index.max()+5],cmap="magma",aspect="equal")
        ax.scatter(row.center_x_deg,row.center_y_deg,marker="x",s=90,color="cyan",linewidth=2,clip_on=False)
        ax.set_xlim(-65,65);ax.set_ylim(-65,65);ax.set_xlabel("grid x (deg)");ax.set_ylabel("grid y (deg)")
        ax.set_title(f"{role}\nunit {uid}; center=({row.center_x_deg:+.1f},{row.center_y_deg:+.1f})\ngain={row.heldout_spatial_gain:+.3f}")
        fig.colorbar(im,ax=ax,shrink=.7,label="mean spikes/presentation")
    fig.suptitle(f"LGd raw Gabor boundary pilot: session {sid}");fig.tight_layout()
    fig.savefig(output/"Figure_LGd_raw_boundary_maps.png",dpi=180);plt.close(fig)
    return examples


def main():
    args=parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    inventory={int(r["ecephys_session_id"]):r for r in json.loads(INVENTORY.read_text())}
    units=pd.read_csv(UNIT_TABLE,low_memory=False)
    sessions=(sorted(units.loc[units.ecephys_structure_acronym.eq("LGd"),"ecephys_session_id"].astype(int).unique()) if args.all_lgd else args.sessions)
    summary=[]; selections=[]
    for sid in sessions:
        out=OUTPUT/f"session_{sid}";cache=out/"cache";out.mkdir(parents=True,exist_ok=True)
        population=units.loc[units.ecephys_session_id.eq(sid)&units.ecephys_structure_acronym.eq("LGd")&units[["anterior_posterior_ccf_coordinate","left_right_ccf_coordinate","dorsal_ventral_ccf_coordinate"]].notna().all(axis=1)].copy().reset_index(drop=True)
        trials,counts=load_or_extract(sid,Path(inventory[sid]["nwb_path"]),population,cache,args.overwrite)
        fits=fit_session(sid,population,trials,counts,out,args.overwrite)
        examples=render_session(sid,population,trials,counts,fits,out)
        summary.append({"ecephys_session_id":sid,"lgd_units":len(population),"positive_heldout_gain":int((fits.heldout_spatial_gain>0).sum()),"boundary_positive_gain":int(((fits.heldout_spatial_gain>0)&(fits.edge_distance_deg<0)).sum()),"median_heldout_gain":fits.heldout_spatial_gain.median()})
        selections.extend({"ecephys_session_id":sid,"role":role,"ecephys_unit_id":uid} for role,uid in examples.items())
    pd.DataFrame(summary).to_csv(OUTPUT/"session_summary.csv",index=False)
    pd.DataFrame(selections).to_csv(OUTPUT/"selected_examples.csv",index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=="__main__":main()
