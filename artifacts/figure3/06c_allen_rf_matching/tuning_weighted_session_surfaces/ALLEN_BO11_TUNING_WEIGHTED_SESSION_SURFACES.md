# Allen BO 1.1 tuning-quality-weighted session surfaces

Built SF/TF maps for 31 simultaneous V1/HVA sessions. Eligible units
retain the existing selectivity, response-rate, and unique-preference gates.

Each unit weight is the geometric mean of lifetime-sparseness tuning strength,
saturating stimulus response rate, and `1/(1 + Fano factor)`. The Fano term is
only a trial-variability proxy: it is not split-half tuning reliability. Weights
are clipped and renormalized to mean one within session × group × SF/TF, so this
changes relative unit influence without changing a map's total nominal weight.

Grid support uses weighted Kish effective local unit count within 1.5 bandwidths.
The resulting grid has the same contract as the equal-unit affine pilot input.
