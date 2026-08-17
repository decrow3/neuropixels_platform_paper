#!/usr/bin/env python3
"""Package the multisession RF audit as a canonical portable report artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"artifacts"/"allen_multisession_rf_validation_v1"


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def source(source_id,label,path,description,filters=None,metrics=None):
    return {"id":source_id,"label":label,"path":path,"query":{
        "description":description,"language":"sql","engine":"DuckDB",
        "sql":f"SELECT * FROM read_csv_auto('{path}')", "tables_used":[path],
        "filters":filters or [],"metric_definitions":metrics or []}}


def main():
    generated=datetime.now(timezone.utc).isoformat()
    inventory=pd.read_csv(BUNDLE/"00_inventory"/"session_inventory.csv")
    baseline=pd.read_csv(BUNDLE/"02_allen_baseline"/"all_session_baseline_summary.csv")
    geometry=pd.read_csv(BUNDLE/"06_cross_session"/"cross_session_geometry_summary.csv")
    gaze=pd.read_csv(BUNDLE/"04_gaze"/"all_session_gaze_summary.csv")
    synthetic=pd.read_csv(BUNDLE/"05_synthetic"/"synthetic_recovery_by_population_size.csv")
    area=pd.read_csv(BUNDLE/"07_registration_readiness"/"area_specific_surface_summary.csv")
    validation=pd.read_csv(BUNDLE/"08_validation"/"validation_checks.csv")
    for frame in (inventory,baseline,geometry,gaze,area):
        frame["session_label"]=frame["session_id"].astype(str)
    baseline_long=baseline.melt(id_vars=["session_label"],value_vars=[
        "azimuth_exact_fraction","elevation_exact_fraction","threshold_area_exact_fraction"],
        var_name="metric",value_name="exact_fraction")
    baseline_long["metric"]=baseline_long.metric.map({"azimuth_exact_fraction":"Azimuth",
        "elevation_exact_fraction":"Elevation","threshold_area_exact_fraction":"Threshold area"})
    inventory_long=inventory.melt(id_vars=["session_label","nwb_gib","cache_mib","valid_gaze_fraction"],
        value_vars=["visual_units","qc_units"],var_name="cohort",value_name="units")
    inventory_long["cohort"]=inventory_long.cohort.map({"visual_units":"Visual units","qc_units":"RF/QC units"})
    synthetic_long=synthetic.melt(id_vars=["population_units","trials","vector_rmse_deg"],
        value_vars=["x_correlation","y_correlation"],var_name="component",value_name="correlation")
    synthetic_long["component"]=synthetic_long.component.map({"x_correlation":"Horizontal","y_correlation":"Vertical"})
    geometry_v1=geometry.loc[geometry.group.eq("V1")].drop(columns="session_id").copy()
    geometry_hva=geometry.loc[geometry.group.eq("HVA")].drop(columns="session_id").copy()
    gaze=gaze.drop(columns="session_id")
    area=area.drop(columns="session_id")

    sources=[
        source("inventory","Session inventory","00_inventory/session_inventory.csv","Local NWB sizes, compact cache sizes, unit counts, and gaze coverage."),
        source("baseline","Allen baseline summary","02_allen_baseline/all_session_baseline_summary.csv","Direct ReceptiveFieldMapping recomputation and bounded-baseline refit summary.",
               ["Published-like RF/QC for exact-match fractions"],["Exact fraction = rows numerically equal within declared tolerance / comparable rows"]),
        source("geometry","RF geometry summary","06_cross_session/cross_session_geometry_summary.csv","Held-out point/aperture and rotation results.",
               ["p_value_rf < 0.01","area_rf < 2500 deg2","snr > 1","firing_rate_dg > 0.1"],
               ["Half-max area = 2*pi*ln(2)*sigma_x*sigma_y","Rotation gain = axis test deviance - rotated test deviance"]),
        source("gaze","Gaze validation summary","04_gaze/all_session_gaze_summary.csv","Shared gain selected on calibration neurons and evaluated on disjoint neurons.",
               ["64 calibration neurons/session","80 evaluation neurons/session","analytic aperture RF model"],
               ["Positive test gain means lower held-out Poisson deviance"]),
        source("synthetic","Synthetic eye-trace recovery","05_synthetic/synthetic_recovery_by_population_size.csv","Known shared gaze trace recovered from simulated population spike counts.",
               ["800 Gabor trials","known RF parameters","0.5-deg displacement grid"],
               ["Vector RMSE = sqrt(mean(dx_error^2 + dy_error^2))"]),
        source("area","Area-specific RF-size surfaces","07_registration_readiness/area_specific_surface_summary.csv","Interior analytic-aperture size-surface support and spatial variation by area.",
               ["center > 10 deg from every edge","uncensored aperture fits","minimum 8 units/session-area"],
               ["Spatial SD is the supported surface SD of median-centered log2 RF area"]),
        source("validation","Validation checks","08_validation/validation_checks.csv","Independent completeness, identity, objective, control, and figure-readability checks."),
    ]

    charts=[
        {"id":"inventory_chart","title":"Visual and RF/QC units by session","subtitle":"Four local Brain Observatory 1.1 sessions; exact unit counts","type":"bar","dataset":"inventory","sourceId":"inventory","intent":"comparison",
         "encodings":{"x":{"field":"session_label","type":"nominal","label":"Session"},"y":{"field":"units","type":"quantitative","label":"Units"},"color":{"field":"cohort","type":"nominal","label":"Cohort"}},"layout":"full"},
        {"id":"baseline_chart","title":"Released threshold metrics reproduced","subtitle":"Exact-match fraction among published-like RF/QC units","type":"bar","dataset":"baseline_long","sourceId":"baseline","intent":"comparison",
         "encodings":{"x":{"field":"session_label","type":"nominal","label":"Session"},"y":{"field":"exact_fraction","type":"quantitative","format":"percent","label":"Exact fraction"},"color":{"field":"metric","type":"nominal","label":"Metric"}},"layout":"full"},
        {"id":"geometry_v1_chart","title":"V1 latent RF area by stimulus model","subtitle":"Inside-grid, uncensored fits; session medians in deg2","type":"bar","dataset":"geometry_v1","sourceId":"geometry","intent":"comparison",
         "encodings":{"x":{"field":"session_label","type":"nominal","label":"Session"},"y":{"field":"median_axis_area_deg2","type":"quantitative","label":"Median half-max area","unit":"deg2"},"color":{"field":"spatial_model","type":"nominal","label":"Model"}},"layout":"full"},
        {"id":"geometry_hva_chart","title":"Pooled-HVA latent RF area by stimulus model","subtitle":"Inside-grid, uncensored fits; session medians in deg2","type":"bar","dataset":"geometry_hva","sourceId":"geometry","intent":"comparison",
         "encodings":{"x":{"field":"session_label","type":"nominal","label":"Session"},"y":{"field":"median_axis_area_deg2","type":"quantitative","label":"Median half-max area","unit":"deg2"},"color":{"field":"spatial_model","type":"nominal","label":"Model"}},"layout":"full"},
        {"id":"gaze_chart","title":"Held-out benefit of the selected gaze transform","subtitle":"Median Poisson-deviance improvement on 80 unseen neurons per session","type":"bar","dataset":"gaze","sourceId":"gaze","intent":"comparison",
         "encodings":{"x":{"field":"session_label","type":"nominal","label":"Session"},"y":{"field":"median_evaluation_test_gain","type":"quantitative","label":"Median held-out gain"}},
         "referenceLines":[{"axis":"y","value":0,"label":"No improvement","color":"neutral","lineStyle":"dashed"}],"layout":"full"},
        {"id":"synthetic_chart","title":"Synthetic eye-trace recovery versus population size","subtitle":"Known shared trace; 800 trials; RF parameters supplied to the decoder","type":"line","dataset":"synthetic","sourceId":"synthetic","intent":"relationship",
         "encodings":{"x":{"field":"population_units","type":"quantitative","label":"Simultaneously recorded units"},"y":{"field":"correlation","type":"quantitative","label":"True-inferred correlation"},"color":{"field":"component","type":"nominal","label":"Component"}},"layout":"full"},
        {"id":"area_chart","title":"Spatial RF-size variation by visual area","subtitle":"Each point is one supported session-area surface; V1 is the comparison baseline","type":"scatter","dataset":"area","sourceId":"area","intent":"distribution",
         "encodings":{"x":{"field":"area","type":"nominal","label":"Area"},"y":{"field":"surface_spatial_sd_log2_area","type":"quantitative","label":"Spatial SD of centered log2 RF area"},"size":{"field":"source_units","type":"quantitative","label":"Interior units"},"tooltip":[{"field":"session_label","type":"nominal","label":"Session"},{"field":"supported_cells","type":"quantitative","label":"Supported cells"}]},"layout":"full"},
    ]
    tables=[{"id":"validation_table","title":"Integrity checks","subtitle":"All 29 declared checks passed after objective-aligned validation","dataset":"validation","sourceId":"validation","layout":"full","density":"dense","defaultSort":{"field":"check","direction":"asc"},
             "columns":[{"field":"check","label":"Check","type":"text"},{"field":"passed","label":"Passed","type":"text"},{"field":"evidence","label":"Evidence","type":"text"}]}]
    blocks=[
        {"id":"title","type":"markdown","body":"# Allen Neuropixels RF Re-estimation: Four-Session Audit","layout":"full"},
        {"id":"summary","type":"markdown","body":"## Technical summary\n\nThe original Allen RF code is runnable sequentially on all four native sessions, but its 13–22 GB transient memory footprint explains the earlier hard failures. Direct threshold metrics reproduce exactly for roughly 70–80% of released RF/QC rows rather than universally. Analytic aperture modeling reduces latent Gaussian area and weakens edge association, whereas free rotation produces small held-out gains mainly in HVAs and changes area little. Measured gaze correction does not provide the expected population-wide sharpening: three sessions select zero gain, and the fourth has a very small predictive gain with essentially unchanged area and amplitude. Corrected RF-size surfaces show potentially larger spatial variation in several HVAs than across V1, but cross-session HVA surface shapes are not yet stable enough to drive registration.","layout":"full"},
        {"id":"inventory_text","type":"markdown","body":"## Four complete local sessions make the pilot computationally feasible\n\nThe compact cache retains all 3,645 Gabor presentations, centered spherical gaze summaries, and spike counts for 3,006 visual units while avoiding repeated high-memory NWB loads. Static audit figures are organized in numbered folders `00_inventory` through `08_validation`.","sourceId":"inventory","layout":"full"},
        {"id":"inventory_block","type":"chart","chartId":"inventory_chart","layout":"full"},
        {"id":"baseline_text","type":"markdown","body":"## Allen's code runs, but released historical rows are not fully regenerated\n\nEvery canonical visual unit was processed with `ReceptiveFieldMapping`. Width and height align with the recomputed Gaussian only after respecting Allen's row/column convention. The remaining threshold-center and area mismatches are real historical-reproduction discrepancies, so a new pipeline should save its own derived metrics and provenance rather than assuming exact interchangeability with the released table.","sourceId":"baseline","layout":"full"},
        {"id":"baseline_block","type":"chart","chartId":"baseline_chart","layout":"full"},
        {"id":"geometry_text","type":"markdown","body":"## Aperture handling changes RF size more than tilt does\n\nAll 1,840 published-like RF/QC units received axis-aligned point and analytic circular-aperture fits. Rotation was tested on a predeclared balanced evaluation subset of 80 neurons per session, with the earlier 318-unit session retained as the full-population reference. Aperture fits generally yield smaller latent areas and lower point-model edge association. Rotation gives small, repeatable HVA prediction gains but nearly unchanged median areas; V1 tilt evidence is mixed.","sourceId":"geometry","layout":"full"},
        {"id":"geometry_v1_block","type":"chart","chartId":"geometry_v1_chart","layout":"full"},
        {"id":"geometry_hva_block","type":"chart","chartId":"geometry_hva_chart","layout":"full"},
        {"id":"gaze_text","type":"markdown","body":"## Measured gaze correction does not materially sharpen this four-session set\n\nA single gain pair was selected from a 4x4 grid on 64 calibration neurons and tested on 80 disjoint neurons per session using the bounded analytic-aperture RF. Sessions 755434585, 760693773, and 798911424 choose zero gain. Session 746083955 chooses vertical gain 0.5, but its median held-out improvement is only 0.000296 and median log2 changes in RF area and amplitude are about 0.0015 and 0.0009. The result argues against applying gaze correction wholesale.","sourceId":"gaze","layout":"full"},
        {"id":"gaze_block","type":"chart","chartId":"gaze_chart","layout":"full"},
        {"id":"synthetic_text","type":"markdown","body":"## Population-only eye inference is identifiable only weakly at very large scale\n\nIn an upper-bound simulation with known RF parameters and a truly shared gaze trace, recovery improves monotonically with population size. Even at 1,024 neurons, horizontal and vertical correlations are 0.44 and 0.48 and vector RMSE is 3.15 deg; shuffled-label correlations remain near zero. Real inference would be harder because RF parameters are estimated, so this should remain a secondary, strongly regularized goal.","sourceId":"synthetic","layout":"full"},
        {"id":"synthetic_block","type":"chart","chartId":"synthetic_chart","layout":"full"},
        {"id":"registration_text","type":"markdown","body":"## RF-size surfaces may reveal HVA differences, but are not ready as sole registration anchors\n\nInterior V1 surfaces have a narrow observed spatial-SD span of 0.08–0.12 centered log2 area. Several HVA session-area surfaces exceed that span, consistent with potentially richer size organization. However, pairwise HVA surface correlations are heterogeneous and often negative. RF location can remain a registration coordinate, while size should initially enter as a softly weighted feature with leave-one-session-out validation.","sourceId":"area","layout":"full"},
        {"id":"area_block","type":"chart","chartId":"area_chart","layout":"full"},
        {"id":"scope","type":"markdown","body":"## Scope, data, and metric definitions\n\n**Sessions.** 746083955, 755434585, 760693773, and 798911424; Brain Observatory 1.1 with 9x9 positions, three orientations, and 15 repeats. **RF/QC cohort.** `p_value_rf < 0.01`, released `area_rf < 2500 deg2`, `snr > 1`, and drifting-grating firing rate >0.1. **Point model.** Gaussian response evaluated at the Gabor center. **Aperture model.** Analytic overlap of the latent Gaussian with the known 10-deg-radius circular aperture; no carrier rasterization. **Area.** Gaussian half-maximum ellipse `2*pi*ln(2)*sigma_x*sigma_y`, distinct from Allen's threshold-component `area_rf`. **Interior surface.** Uncensored aperture fit with center >10 deg from all sampled edges and at least three units within 20 deg of a grid location.","layout":"full"},
        {"id":"methods","type":"markdown","body":"## Methodology\n\nModels use a nonnegative baseline, three orientation amplitudes, centers allowed 20 deg beyond the grid, and V1/HVA sigma ceilings of 40/50 deg. Training uses two of every three repeats within condition; test metrics use the remaining repeats. Tilt uses five angle starts. Gaze candidates use centered per-presentation median filtered spherical gaze and one shared transform per session. The synthetic decoder maximizes population Poisson likelihood over a 0.5-deg displacement grid. Surface comparisons use median-centered log2 aperture area and bounded local support.","layout":"full"},
        {"id":"limitations","type":"markdown","body":"## Limitations, uncertainty, and robustness\n\nThis is a four-session pilot, not the full Allen HVA dataset. Direct historical metric reproduction remains incomplete. Aperture correction models the circular aperture but not carrier phase or screen clipping. Rotation replication is subset-based outside the previously completed full session. Gaze gain selection is discrete and session-specific; the zero-gain candidate is allowed. The synthetic trace decoder is an upper bound because it receives true RF parameters. HVA surface stability is sensitive to area identity and coverage, so pooled-HVA surfaces should not be used for registration.","layout":"full"},
        {"id":"validation_text","type":"markdown","body":"## Validation status: share with caveats\n\nAll 29 declared integrity checks pass, including balanced stimulus conditions, spike/unit identity, disjoint gaze neuron splits, nominal-transform identity, independently recomputed rotation nesting on the actual Anscombe objective, synthetic shuffled controls, bounded surface support, and PNG readability. Passing these checks establishes internal consistency, not biological generalizability.","sourceId":"validation","layout":"full"},
        {"id":"validation_block","type":"table","tableId":"validation_table","layout":"full"},
        {"id":"next","type":"markdown","body":"## Recommended next steps\n\n1. Treat analytic aperture, bounded baseline, and explicit censor flags as the candidate production RF geometry; keep rotation only if leave-one-session-out HVA gains remain positive.\n2. Do not enable gaze correction by default. Expand to more tracked-eye sessions first and require nonzero gain, predictive benefit, and coherent sharpening to replicate at the session level.\n3. Scale RF re-estimation through compact extraction caches and sequential NWB loading; storage is ample, while RAM—not disk—is the operational constraint.\n4. Test RF-size surfaces as a softly weighted registration feature alongside RF location and tuning. Choose its weight only through leave-one-animal/session validation and benchmark HVA improvements against the natural V1 session span.\n5. Process additional sessions area-by-area before interpreting HVA surface variance as biological.","layout":"full"},
        {"id":"questions","type":"markdown","body":"## Further questions\n\n- Why do 20–30% of directly recomputed threshold rows differ from the released table despite native NWBs and Allen's code?\n- Do nonzero gaze gains replicate in sessions with larger calibrated spherical motion when the transform grid is refined around zero?\n- Are unstable HVA size surfaces caused by low spatial support, area composition, unit sampling, or genuine animal differences?\n- Does adding RF size improve held-out SF/TF tuning alignment beyond RF-center registration without degrading V1 consistency?","layout":"full"},
    ]
    artifact={"surface":"report","manifest":{"version":1,"surface":"report","title":"Allen Neuropixels RF Re-estimation: Four-Session Audit","description":"Technical validation of Allen RF reproduction, aperture/tilt geometry, gaze correction, synthetic trace inference, and registration readiness.","generatedAt":generated,"blocks":blocks,"charts":charts,"tables":tables,"sources":sources},
              "snapshot":{"version":1,"generatedAt":generated,"status":"ready","datasets":{"inventory":records(inventory_long),"baseline_long":records(baseline_long),"geometry_v1":records(geometry_v1),"geometry_hva":records(geometry_hva),"gaze":records(gaze),"synthetic":records(synthetic_long),"area":records(area),"validation":records(validation)},"accessIssues":[]},"sources":sources}
    (BUNDLE/"artifact.json").write_text(json.dumps(artifact,indent=2)+"\n",encoding="utf-8")
    chart_map=pd.DataFrame([{"section":chart["id"],"question":chart.get("question",chart["title"]),"type":chart["type"],"dataset":chart["dataset"],"source":chart["sourceId"]} for chart in charts])
    chart_map.to_csv(BUNDLE/"chart_map.csv",index=False)
    print(BUNDLE/"artifact.json")


if __name__=="__main__":main()
