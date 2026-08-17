# Allen BO 1.1 interior V1 RF-size translation pilot

Each of 31 sessions was matched to a leave-one-session-out V1 RF-size template using only RF centers at least 20° from every RF-grid edge.
Only translation was allowed. RF size was log2 transformed and median/IQR standardized within session, so absolute between-session RF-size differences could not masquerade as position.
SF, TF, and HVA data were held out and evaluated afterward.

Median RF-size correlation gain: **+0.518**.
Median azimuth/elevation profile widths: **1.5° / 2.2°** (smaller means better identified).
Sessions at a ±30° bound: **0/31**.
Independent target-unit halves reproduce azimuth offsets at **rho = +0.082** and elevation offsets at **rho = +0.047**.
Median absolute half-to-half differences are **15.8°** azimuth and **8.0°** elevation.

| Group | Map | Median paired Δr versus raw |
| --- | --- | ---: |
| HVA pooled | SF | +0.114 |
| HVA pooled | TF | -0.148 |
| V1 | SF | -0.078 |
| V1 | TF | -0.082 |
