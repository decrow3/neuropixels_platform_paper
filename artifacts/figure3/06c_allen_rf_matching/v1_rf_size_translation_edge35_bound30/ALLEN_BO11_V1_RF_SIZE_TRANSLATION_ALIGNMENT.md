# Allen BO 1.1 interior V1 RF-size translation pilot

Each of 21 sessions was matched to a leave-one-session-out V1 RF-size template using only RF centers at least 35° from every RF-grid edge.
Only translation was allowed. RF size was log2 transformed and median/IQR standardized within session, so absolute between-session RF-size differences could not masquerade as position.
SF, TF, and HVA data were held out and evaluated afterward.

Median RF-size correlation gain: **+0.479**.
Median azimuth/elevation profile widths: **3.8° / 1.5°** (smaller means better identified).
Sessions at a ±30° bound: **0/21**.
Spatially identifiable session surfaces: **17/21**; non-identifiable sessions were retained with zero shift.

Split-half fitting was skipped because several sessions retain too few units after this exclusion.

| Group | Map | Median paired Δr versus raw |
| --- | --- | ---: |
| HVA pooled | SF | -0.047 |
| HVA pooled | TF | +0.020 |
| V1 | SF | -0.132 |
| V1 | TF | -0.043 |
