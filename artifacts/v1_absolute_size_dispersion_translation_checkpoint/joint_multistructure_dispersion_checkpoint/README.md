# Joint V1/HVA/LGd dispersion-registration checkpoint

## Goal

Estimate a shared two-dimensional session translation from multiple
structure-specific relationships between RF location and anatomy-corrected local
RF dispersion. Complementary, non-degenerate likelihoods should resolve each
other's ambiguity if they genuinely encode the same screen/eye offset.

## Cohort preparation

Raw Gabor aperture fits were completed for all 2,582 LGd units in 33 sessions.
Only units with positive held-out spatial gain were eligible as RF coordinates.
Centers hitting the +/-60-degree optimization limit were excluded; extrapolated
40-60-degree centers were retained when their response flank generalized.

The matched descriptor was computed identically in V1, combined HVAs, and LGd:

1. estimate a leave-one-cell-out anatomical mean RF map at 250 um;
2. estimate each cell's RF residual from that map;
3. calculate covariance trace of residual RF vectors in a 15-degree RF
   neighborhood;
4. smooth log2 trace over RF location with a 12-degree kernel.

HVA anatomical mean maps were estimated separately by cortical area before HVA
residual dispersion was pooled. Sixteen sessions passed the fixed minimums of 30
valid V1, 50 valid HVA, and 10 valid LGd dispersion estimates.

## Concrete results

Exact-support shuffles keep RF locations fixed and permute dispersion values.
They test whether the observed descriptor-to-location organization matches the
leave-one-session-out population template better than arbitrary arrangements.

| Session | Component | Preferred shift (az, el) | Shuffle p | Interpretation |
|---|---|---:|---:|---|
| 715093703 | V1 | (-14, +20) | 0.020 | strong |
| 715093703 | HVA | (+30, -8) | 0.020 | strong but incompatible with V1 |
| 715093703 | LGd | (+8, +2) | 0.725 | unsupported |
| 760345702 | V1 | (+12, 0) | 0.020 | strong |
| 760345702 | HVA | (+26, +10) | 0.020 | strong; 17 degrees from V1 |
| 760345702 | LGd | (-30, -28) | 0.020 | strong but boundary optimum, incompatible |
| 754829445 | V1 | (-30, +30) | 0.020 | strong but boundary optimum |
| 754829445 | HVA | (+16, -20) | 0.176 | weak |
| 754829445 | LGd | (-6, +10) | 0.902 | unsupported |

The initial reliability-weighted joint optima are therefore exploratory
compromises, not registrations. For example, session 760345702 combines to
(0, +8) degrees even though none of the three structures independently prefers
that point.

## What this establishes

- V1 and combined HVA dispersion can each contain strong nonrandom spatial
  organization.
- Cohort-scale raw LGd fitting provides enough units for an LGd likelihood in 16
  simultaneous sessions.
- The individual loss surfaces have genuinely different shapes, so in principle
  they could provide complementary constraints.

## What remains unsupported

The preferred translations do not yet agree across structures. Multiplying or
averaging incompatible surfaces would create false precision. Before any joint
estimate is accepted, each component must pass split-half and held-out physical
section tests, and the training templates must be learned in a coordinate system
that accounts for unknown training-animal translations.

The current templates average sessions in raw screen coordinates. If those
sessions have real offsets, each structure-specific template is blurred and may
be displaced differently because its animals and RF support differ. A nested
alternating model is the appropriate next test:

1. initialize training-animal translations with a zero-mean constraint;
2. build the three dispersion templates after applying those translations;
3. update each training translation from the joint likelihood;
4. iterate using training animals only;
5. freeze the templates and infer the held-out animal's translation;
6. recompute descriptors and translations independently in cell and physical
   halves.

The global visual-field constant remains arbitrary; the estimand is relative
screen/eye translation across animals.

## Outputs

- `Figure_concrete_joint_dispersion_likelihoods.png`: separate and joint loss
  surfaces.
- `concrete_component_and_joint_results.csv`: optima and shuffle evidence.
- `session_structure_eligibility.csv`: fixed support audit.
- `three_structure_eligible_sessions.csv`: 16-session discovery cohort.
- `all_structure_dispersion_descriptors.csv.gz`: matched unit descriptors.
- `run_manifest.json`: exploratory model settings.
