# Multi-case artificial RF cropping with a DC-return ring

## Design

Four units were selected algorithmically before inspecting their raw maps: the units nearest the median and 75th percentile released major sigma among eligible interior V1 and HVA fits in native session 737581020. Eligibility required published-like QC, p_value_rf < .01, an on-screen released Gaussian center, a threshold center at least 20° from every grid edge, and released major sigma between 5° and 40°.

Each native 9×9 map was cropped by one to four rows or columns from all four directions. Every estimator is compared with its own full-map estimate. The DC-ring model fits a nonnegative baseline plus Gaussian, allows the center two pixels beyond the observed crop, and adds 24 soft pseudo-points whose target is the fitted DC baseline. The exploratory return-to-DC radii are 40° for V1 and 50° for HVA; the total ring weight equals four observed grid positions.

## Result

Across 64 cropped-map fits per model, the median absolute log2 major-sigma errors were **0.097** for Allen, **0.042** for the screen-bounded baseline model, **0.049** for the extended baseline model, and **0.035** for the DC ring.

The corresponding 90th-percentile center errors were **213.0°**, **7.1°**, **7.1°**, and **6.2°**. Failure-or-bound rates were **10.9%**, **21.9%**, **7.8%**, and **6.2%**.

When the original threshold component remained intact, Allen and the ring were both stable (median absolute log2 sigma error **0.061** and **0.021**). Once the original component was censored but its peak remained, Allen's median error rose to **0.239**; the extended-baseline and ring errors were **0.203** and **0.209**.

## Interpretation

This is an exploratory four-case stress test, not a population validation. It separates the benefit of adding a DC baseline from the incremental effect of the enclosing DC ring. The ring radius and weight remain hyperparameters and should be swept or cross-fit before production use.
