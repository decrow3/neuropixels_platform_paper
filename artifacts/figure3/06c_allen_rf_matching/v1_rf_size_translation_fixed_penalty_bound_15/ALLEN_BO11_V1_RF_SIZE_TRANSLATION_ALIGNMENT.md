# Allen BO 1.1 interior V1 RF-size translation pilot

Each of 31 sessions was matched to a leave-one-session-out V1 RF-size template using only RF centers at least 20° from every RF-grid edge.
Only translation was allowed. RF size was log2 transformed and median/IQR standardized within session, so absolute between-session RF-size differences could not masquerade as position.
SF, TF, and HVA data were held out and evaluated afterward.

Median RF-size correlation gain: **+0.442**.
Median azimuth/elevation profile widths: **1.1° / 2.2°** (smaller means better identified).
Sessions at a ±15° bound: **12/31**.
Independent target-unit halves reproduce azimuth offsets at **rho = -0.150** and elevation offsets at **rho = +0.184**.
Median absolute half-to-half differences are **12.0°** azimuth and **6.0°** elevation.

| Group | Map | Median paired Δr versus raw |
| --- | --- | ---: |
| HVA pooled | SF | +0.027 |
| HVA pooled | TF | -0.177 |
| V1 | SF | -0.024 |
| V1 | TF | +0.007 |
