# 'Brain Observatory 1.1 & Functional Connectivity' vs. 'MouseV2': active cortex span and estimated insertion angle

## Active cortex sites along probe (RF-significant-unit depth span, 5-95th pct)

- 'Brain Observatory 1.1 & Functional Connectivity' V1: median 384 um (IQR 342-450), n=41 probes.
- 'MouseV2': median 580 um (IQR 398-727), n=28 probes.

## Estimated insertion angle from vertical

- 'Brain Observatory 1.1 & Functional Connectivity' V1 (probeC only, DIRECT measurement from per-unit CCF trace): median 20.7 deg (IQR 17.6-22.6), n=39 probes.
- 'MouseV2' (INDIRECT estimate from RF-significant depth-span ratio vs. Allen reference, symmetric ratio assumption): median 51.2 deg (IQR 30.3-58.9), n=28 probes.

## Caveats

- The two angle numbers are NOT the same kind of measurement: Allen's comes directly from real per-unit CCF coordinates; MouseV2's is inferred from how much longer its active-cortex span is than Allen's, which conflates true insertion angle with any other reason the spans might differ (unmatched RF-significance yield, registration/QC asymmetries -- see `compare_rf_depth_span_mousev2_vs_allen.py` docstring).
- MouseV2's larger active-cortex span is consistent with, but does not on its own prove, a larger angle from vertical (a more oblique/shallower insertion, in the conventional sense where 'steep' means close to vertical) -- it is also consistent with MouseV2 simply retaining more RF-significant units per unit of true cortical thickness.
