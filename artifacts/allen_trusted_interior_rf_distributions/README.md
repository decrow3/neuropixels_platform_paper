# Trusted-interior Allen RF-size distributions

## Cohort

This exactly reconstructs the **4,090-unit** cohort used by the selected V1 `area_rf` alignment: the same 31 BO 1.1 sessions, published-like RF/QC support, and RF centers at least 20° from every 9×9 mapping-grid boundary. Unit identifiers match the saved alignment support exactly.

## Released threshold area

V1 contains **2,457 units** with median `area_rf` **400 deg²**, 90th percentile **900 deg²**, and 95th percentile **1100 deg²**. Pooled HVAs contain **1,633 units** with median **700 deg²**, 90th percentile **1500 deg²**, and 95th percentile **1800 deg²**.

The trusted source was already conditioned on `area_rf < 2500 deg²`. Under identical session, significance, SNR, firing-rate, area, and interior-center criteria but with that one cap removed, V1's 95th/99th percentiles are **1200/2200 deg²** and HVA's are **2400/3884 deg²**. The cap excludes **21 V1** and **84 HVA** interior units.

## Released Gaussian dimensions within the trusted cohort

After additionally requiring Allen's fitted Gaussian center to be on screen and both dimensions finite, **3,960/4,090 fits** remain. The major-sigma median/90th/95th percentiles are **23.0/60.7/76.7°** in V1 and **33.5/71.5/93.0°** in pooled HVAs.

The upper Gaussian tail remains implausibly broad even in this interior/on-screen subset, consistent with the previously identified no-baseline failure. These quantiles are descriptive evidence for choosing and testing a V1/HVA-specific regularizer; they should not by themselves be interpreted as biological upper limits.

## Use for the proposed DC ring

The central threshold-area distributions support a larger prior scale for HVAs than V1. The exact DC-ring radius still requires a declared size convention (sigma, FWHM diameter, equivalent diameter, or contour area). The held-out-repeat comparison should test candidate radii around these empirical central quantiles rather than fitting the contaminated Gaussian tail.
