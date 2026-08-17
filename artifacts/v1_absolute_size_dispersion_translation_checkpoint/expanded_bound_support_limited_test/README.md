# Expanded-bound joint-dispersion test

## Question

Did the exact V1–LGd agreement for session `760345702` at `(30°, −30°)` identify a shared visual-coordinate translation, or was it imposed by the original ±30° search bound?

## Test

- Expanded azimuth and elevation translation candidates to ±60° in 2° steps.
- Expanded the template grid to ±120°.
- Limited each training session's contribution to locations where its third-nearest observed descriptor was within 24°.
- Required at least five non-target sessions at every usable template pixel.
- Kept the damped alternating-model training translations fixed.
- Evaluated V1, combined HVA, LGd, V1+LGd, and all-component landscapes separately.

## Result

The apparent V1–LGd agreement does not survive.

| Component | Expanded-bound optimum |
|---|---:|
| V1 | (+22°, −24°) |
| HVA | (+18°, +18°) |
| LGd | (+18°, −60°) |
| V1+LGd | (+26°, −36°) |
| All | (+20°, −34°) |

LGd continues to the new negative-elevation boundary, while V1 and HVA prefer distinct interior regions. The earlier common `(30°, −30°)` optimum was therefore a search-bound artifact. The combined optima are compromises between incompatible component landscapes and should not be used as registrations.

The other two audited sessions also lack three-structure consensus. `715093703` retains widely separated V1, HVA, and LGd optima. In `754829445`, the all-component minimum follows LGd while V1 prefers a point 20° farther negative in azimuth and HVA prefers a different elevation.

## Outputs

- `Figure_expanded_bound_landscapes.png`
- `expanded_bound_component_optima.csv`
- `run_manifest.json`
- analysis script: `scripts/test_expanded_joint_dispersion_bounds.py`

