# Allen BO 1.1 interior V1 RF-size translation pilot

Each of 31 sessions was matched to a leave-one-session-out V1 RF-size template using only RF centers at least 20° from every RF-grid edge.
Only translation was allowed. RF size was log2 transformed and median/IQR standardized within session, so absolute between-session RF-size differences could not masquerade as position.
SF, TF, and HVA data were held out and evaluated afterward.

Median RF-size correlation gain: **+0.254**.
Median azimuth/elevation profile widths: **2.4° / 2.5°** (smaller means better identified).
Sessions at a ±10° bound: **12/31**.
Independent target-unit halves reproduce azimuth offsets at **rho = +0.261** and elevation offsets at **rho = -0.236**.
Median absolute half-to-half differences are **5.4°** azimuth and **6.7°** elevation.

| Group | Map | Median paired Δr versus raw |
| --- | --- | ---: |
| HVA pooled | SF | +0.082 |
| HVA pooled | TF | -0.063 |
| V1 | SF | -0.056 |
| V1 | TF | -0.043 |
