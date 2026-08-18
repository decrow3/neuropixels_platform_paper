# Allen visual cortex DV thickness, from per-unit CCF coordinates

Per-probe robust (P5-P95) DV-coordinate span of VIS*-labeled units, 283 probes with >= 10 such units each.

- Mean per-probe DV thickness: 511.9 um (0.512 mm).
- Median per-probe DV thickness: 518.0 um (IQR 441.9-589.0 um).
- SD across probes: 117.0 um.

## By primary visual area (probe's modal structure label)

                    median        mean         std  count
primary_structure                                        
VISp               627.325  625.365625   91.978432     48
VISl               465.700  474.694286  125.050692     35
VISal              459.400  462.793182  143.436891     44
VISrl              432.000  440.716327   93.243608     49
VISam              534.400  529.120408   56.307217     49
VISpm              569.150  567.036207   55.444098     29
VISli              472.000  440.020000  121.198167      5
VISmma             562.500  573.300000   46.103362      6
VISmmp             460.400  469.187500   99.901388      4

## Pooled-by-area cross-check (mixes cross-session registration variability -- expect wider)

  area  n_units  n_probes  dv_thickness_um
 VISam     5780        49           577.00
VISmma      722         6           582.00
VISmmp      269         4           585.40
 VISpm     3414        29           614.00
  VISp     7412        48           749.45
 VISrl     4978        49           841.45
 VISal     5620        44           853.00
 VISli      658         5           926.20
  VISl     4088        35           929.60
   VIS     1623        14          1807.00

## Caveat

This treats the local pia/white-matter boundary as a horizontal (constant-DV) sheet, so the raw DV-coordinate span of cortically-labeled units equals perpendicular thickness without needing the probe's own insertion angle. That approximation is best near the dorsal vertex and degrades for areas where the true cortical surface normal tilts away from the DV axis.
