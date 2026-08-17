# RF size evidence for the Allen BO 1.1 non-center registration

The plotted quantity is log2 released RF area, robustly standardized within each
anatomical area before smoothing. This is exactly the RF-size field used by the
selected non-center **similarity** model. Raw RF area is not a Gaussian width
estimate; released `width_rf` and `height_rf` were excluded because their scales are
not reliable as a size/shape decomposition.

The maps compare raw coordinates with the selected transform. The radial panel first
takes the median within session × eccentricity bin, then displays the median and IQR
across sessions; bins with fewer than ten sessions are omitted.

RF area is weakly associated with Allen-origin eccentricity and is not sufficient to identify
a two-dimensional translation. It was combined with directional CCF/probe fields and
weak latency regularizers in the fourth-row fit.

The radial relationship is non-monotonic and weak: size varies through intermediate
eccentricities and falls at the mapped periphery. The peripheral decline may partly reflect RFs
being truncated by the finite mapping display, causing released area to be underestimated.
RF area should therefore be treated as a reproducible scalar pattern, not an uncensored
biological size-versus-eccentricity calibration.
