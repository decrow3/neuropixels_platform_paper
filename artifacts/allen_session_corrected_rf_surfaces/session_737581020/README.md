# Allen session 737581020: corrected RF Gaussian surfaces

All 369 units in canonical V1/HVA structures were fitted with Allen's no-baseline Gaussian and a corrected nonnegative-baseline Gaussian. No enclosing border or pseudo-observations were used.

The primary surfaces use 180 matched units that pass published-like RF/QC filters, have finite Allen parameters, and have a successful corrected fit without its center or upper sigma limit being reached. Allen's implementation returns and releases finite parameters even when its least-squares convergence flag is false, so that flag is retained as an audit field rather than used as a selection filter. RF location is the direct threshold-map center for both models, so only the Gaussian size estimate changes.

The plotted size is Gaussian half-maximum ellipse area, `2*pi*ln(2)*sigma_x*sigma_y`. V1 sigma is bounded at 40 degrees and HVA sigma at 50 degrees; fits reaching an upper sigma or center-extension bound are labeled censored and excluded from both matched surfaces.

In V1, median half-maximum area changes from 1576 to 341 deg2, and Spearman rho between log2 area and distance from the nearest sampled edge changes from -0.31 to -0.06.
In pooled HVAs, median half-maximum area changes from 14650 to 576 deg2. HVA support in this session is strongly boundary concentrated, so its spatial surface should be read within the effective-sample contours.
