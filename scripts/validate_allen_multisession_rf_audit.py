#!/usr/bin/env python3
"""Independent integrity checks and visual scorecard for the RF audit bundle."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from compare_allen_point_vs_aperture_rf import model_prediction
from render_allen_rotated_point_aperture_rf_examples import rotated_prediction


ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"
CACHE=ROOT/"artifacts"/"allen_population_gaze_rf"
SESSIONS=(746083955,755434585,760693773,798911424)


def check(name,condition,evidence,rows):
    rows.append({"check":name,"passed":bool(condition),"evidence":str(evidence)})


def main():
    rows=[]
    for sid in SESSIONS:
        cache=CACHE/f"session_{sid}";population=pd.read_csv(cache/"visual_unit_population.csv",low_memory=False)
        trials=pd.read_csv(cache/"gabor_trial_gaze_table.csv",low_memory=False);spikes=np.load(cache/"gabor_spike_counts.npz")
        condition_counts=trials.groupby(["x_position","y_position","orientation"],observed=True).size()
        check(f"{sid} balanced Gabor design",len(trials)==3645 and len(condition_counts)==243 and condition_counts.eq(15).all(),
              f"trials={len(trials)}, conditions={len(condition_counts)}, repeats={sorted(condition_counts.unique())}",rows)
        check(f"{sid} spike/unit alignment",np.array_equal(spikes["unit_ids"].astype(int),population.ecephys_unit_id.to_numpy(int)) and spikes["counts"].shape==(len(population),len(trials)),
              f"counts_shape={spikes['counts'].shape}, units={len(population)}",rows)
        check(f"{sid} eye coverage",trials.valid_gaze.mean()>.89,f"valid={trials.valid_gaze.mean():.3%}",rows)

    baseline=pd.read_csv(BUNDLE/"02_allen_baseline"/"all_session_baseline_fits.csv",low_memory=False)
    check("Allen baseline population complete",len(baseline)==3006 and baseline.ecephys_unit_id.nunique()==3006,
          f"rows={len(baseline)}, unique_units={baseline.ecephys_unit_id.nunique()}",rows)
    finite_area=baseline.threshold_area_deg2.dropna()
    check("Threshold areas use 100 deg2 pixels",np.allclose(np.mod(finite_area,100),0,atol=1e-5),
          f"finite={len(finite_area)}, max_mod={np.mod(finite_area,100).max():.3g}",rows)

    geometry=pd.read_csv(BUNDLE/"03_geometry"/"all_session_unit_geometry_fits.csv",low_memory=False)
    duplicate=geometry.duplicated(["session_id","ecephys_unit_id","spatial_model"]).sum()
    check("Geometry population complete",len(geometry)==3680 and duplicate==0,
          f"rows={len(geometry)}, units={geometry.ecephys_unit_id.nunique()}, duplicates={duplicate}",rows)
    check("Axis fits finite",geometry[["axis_area_deg2","axis_test_deviance"]].notna().all().all(),
          f"null_cells={geometry[['axis_area_deg2','axis_test_deviance']].isna().sum().sum()}",rows)
    rotated=geometry.loc[geometry.rotation_test_gain.notna()]
    objective_differences=[]
    for sid in SESSIONS:
        cache=CACHE/f"session_{sid}";population=pd.read_csv(cache/"visual_unit_population.csv",low_memory=False)
        trials=pd.read_csv(cache/"gabor_trial_gaze_table.csv",low_memory=False);counts=np.load(cache/"gabor_spike_counts.npz")["counts"]
        train=trials.valid_gaze.to_numpy(bool)&trials.trial_split.eq("train").to_numpy(bool)
        x=trials.x_position.to_numpy(float);y=trials.y_position.to_numpy(float);orientation=trials.orientation_index.to_numpy(int)
        for fit in rotated.loc[rotated.session_id.eq(sid)].itertuples(index=False):
            index=population.index[population.ecephys_unit_id.eq(fit.ecephys_unit_id)][0];observed=counts[index].astype(float)
            axis_parameters=np.asarray(json.loads(fit.axis_parameters),float)
            rotation_parameters=np.asarray(json.loads(fit.rotation_parameters),float)
            axis_prediction=model_prediction(axis_parameters,x[train],y[train],orientation[train],fit.spatial_model)
            rotation_prediction=rotated_prediction(rotation_parameters,x[train],y[train],orientation[train],fit.spatial_model)
            transform=lambda prediction:2*(np.sqrt(np.maximum(prediction,0)+3/8)-np.sqrt(observed[train]+3/8))
            objective_differences.append(np.mean(transform(rotation_prediction)**2)-np.mean(transform(axis_prediction)**2))
    max_objective_difference=max(objective_differences)
    check("Rotation nested on training objective",max_objective_difference<1e-7,
          f"rotation_fits={len(rotated)}, maximum_Anscombe_MSE_difference={max_objective_difference:.3g}",rows)

    gaze_summary=pd.read_csv(BUNDLE/"04_gaze"/"all_session_gaze_summary.csv")
    for sid in SESSIONS:
        local=BUNDLE/"04_gaze"/f"session_{sid}"
        calibration=pd.read_csv(local/"calibration_unit_sweep.csv");evaluation=pd.read_csv(local/"evaluation_unit_results.csv")
        cal_ids=set(calibration.ecephys_unit_id);eval_ids=set(evaluation.ecephys_unit_id)
        check(f"{sid} gaze neuron split",len(cal_ids)==64 and len(eval_ids)==80 and not(cal_ids&eval_ids),
              f"calibration={len(cal_ids)}, evaluation={len(eval_ids)}, overlap={len(cal_ids&eval_ids)}",rows)
        nominal=evaluation.loc[evaluation.candidate.eq("gx_0_gy_0")]
        check(f"{sid} nominal gaze identity",np.allclose(nominal.test_deviance_improvement,0,atol=1e-12),
              f"max_abs={nominal.test_deviance_improvement.abs().max():.3g}",rows)

    synthetic=pd.read_csv(BUNDLE/"05_synthetic"/"synthetic_recovery_by_population_size.csv")
    control=json.loads((BUNDLE/"05_synthetic"/"control_summary.json").read_text())
    largest=synthetic.loc[synthetic.population_units.idxmax()]
    check("Synthetic trace recovery rises above chance",largest.x_correlation>.4 and largest.y_correlation>.4,
          f"1024-unit rho=({largest.x_correlation:.3f},{largest.y_correlation:.3f})",rows)
    check("Synthetic shuffled-label control",abs(control["x_correlation"])<.1 and abs(control["y_correlation"])<.1,
          f"shuffle rho=({control['x_correlation']:.3f},{control['y_correlation']:.3f})",rows)

    grid=pd.read_csv(BUNDLE/"07_registration_readiness"/"area_specific_surface_grid.csv")
    check("Registration surface support is bounded",grid.centered_log2_aperture_area.notna().any() and grid.effective_units.eq(0).any(),
          f"supported={grid.centered_log2_aperture_area.notna().sum()}, unsupported={(grid.effective_units==0).sum()}",rows)
    figures=sorted(BUNDLE.rglob("*.png"));bad=[]
    for path in figures:
        try:
            with Image.open(path) as image:
                if image.width<600 or image.height<350:bad.append(path.name)
        except Exception:bad.append(path.name)
    check("All audit figures readable",len(figures)>=20 and not bad,f"figures={len(figures)}, bad={bad}",rows)

    result=pd.DataFrame(rows);out=BUNDLE/"08_validation";out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"validation_checks.csv",index=False)
    summary={"assessment":"Share with caveats" if result.passed.all() else "Needs revision",
             "checks":len(result),"passed":int(result.passed.sum()),"failed":int((~result.passed).sum()),
             "required_caveats":[
                 "Direct Allen RF metric recomputation is not byte-for-byte identical to every released metric row.",
                 "Cross-session tilt uses a predeclared 80-neuron evaluation subset per session; session 746083955 has the prior full-population check.",
                 "Measured gaze correction selected zero gain in three of four sessions and did not materially sharpen RFs.",
                 "Area-specific corrected RF-size surfaces are a four-session pilot and are not stable enough to drive registration alone.",
             ]}
    (out/"validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    render(result,out/"Figure_validation_scorecard.png")
    print(result.to_string(index=False));print(summary)
    if not result.passed.all():raise SystemExit(1)


def render(result,path):
    fig,axis=plt.subplots(figsize=(10.5,7.2),constrained_layout=True);y=np.arange(len(result))[::-1]
    colors=np.where(result.passed,"#3366aa","#d97736")
    axis.barh(y,np.ones(len(result)),color=colors)
    axis.set(yticks=y,yticklabels=result.check,xlim=(0,1.02),xticks=[],title="RF audit integrity checks")
    for yy,passed in zip(y,result.passed):axis.text(.98,yy,"PASS" if passed else "FAIL",ha="right",va="center",color="white",weight="bold")
    axis.grid(False);fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


if __name__=="__main__":main()
