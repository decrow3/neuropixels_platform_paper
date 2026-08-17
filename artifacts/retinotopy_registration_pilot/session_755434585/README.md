# Session 755434585 registration pilot — initial fit checkpoint

This pilot tests whether one global affine can register a single Allen session
to the Zhuang et al. (2017) Figure 9 population retinotopy using both CCF
surface coordinates and receptive-field centers.

## Case selection

Session 755434585 was selected algorithmically from the four
sessions displayed in the registration-readiness PDF: it has complete CCF
coverage for all 190 trusted units and the
largest usable set of named visual areas. Those units occupy only
4 independent probe penetrations, so the fit
uses penetration medians rather than treating neurons as independent cortical
landmarks.

## Models

1. **Joint anatomy + RF affine** penalizes leaving the Zhuang compartment
   corresponding to the Allen area acronym.
2. **RF-only affine** omits that compartment penalty. It is a diagnostic for
   whether RF agreement is obtained by placing penetrations in the wrong area.

For each model, both cortical handedness choices and both the native and
`100 - azimuth` retinal conventions were tried. The manifest preserves all
candidate scores.

## Initial result

- Joint model: median penetration RF error
  1.0°, with
  4/4
  penetration landmarks inside their named Zhuang compartments.
- RF-only model: median penetration RF error
  2.4°, with
  0/4
  landmarks inside their named compartments.

This is an exploratory registration, not a validated warp. A large improvement
in the RF-only model accompanied by anatomical violations would reject a
single global affine as the final model. The next model should only add local
deformation after this tradeoff is inspected, because six penetrations cannot
support a high-flexibility warp without strong regularization or widefield
landmarks.

## Evidence and derived layers

- `Figure_registration_pilot_QA.png`: CCF landmarks, both transforms, RF
  target/prediction pairs, and unit-level residual distributions.
- `penetration_landmarks.csv`: the six fit observations and both model outputs.
- `unit_registration_residuals.csv.gz`: neuron-level evaluation only.
- `candidate_model_summary.csv`: all handedness/convention candidates.
- `zhuang_interpolated_fields.npz`: explicit linear interpolation of the
  published 5° contours, named-area masks, and border/domain layers.
- `run_manifest.json`: complete provenance, fit parameters, and chart contract.

## Important limitations

1. The continuous Zhuang fields are derived by linear interpolation between
   published contours and nearest fill only at the small unsupported edge.
2. Zhuang area compartments were identified from labeled Figure 3C; they are
   not Allen CCF boundaries.
3. The source RF centers are trusted aperture fits, but within-penetration RF
   dispersion remains substantial in some HVAs.
4. This fit has no held-out penetration and should not be used for inference.
