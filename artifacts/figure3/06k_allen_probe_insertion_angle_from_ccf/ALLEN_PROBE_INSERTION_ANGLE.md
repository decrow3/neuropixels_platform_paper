# Allen probe insertion angle from per-unit CCF trace

Fitted 284 probes (>= 20 CCF-complete good-quality units each) by total-least-squares line through each probe's own unit CCF coordinates.

- Overall median angle from vertical: 29.3 deg (IQR 19.6-40.2).
- Median fit quality (r2 colinearity): 0.9968; 5/284 probes fell below 0.95 and should be treated cautiously.

## Median angle from vertical by probe letter

                 median       std  count
probe_letter                            
probeA        24.360250  4.144608     49
probeB         5.978353  3.799445     47
probeC        21.086530  5.004902     47
probeD        38.095630  8.296825     47
probeE        41.954507  7.558295     51
probeF        39.622105  5.643901     43

## Outputs

- `allen_probe_insertion_angle_from_ccf.csv`: per-probe angle, azimuth, fit quality, and sanity-check columns.
- `Figure_allen_probe_insertion_angle.png`: summary figure (angle by probe letter, azimuth rose plot, session-type histogram, and fit sanity check).
- `Figure_allen_probe_insertion_angle_3d.png`: 3D view -- one example session's real probe tracks in CCF space, and every probe's insertion direction plotted as a vector from a common origin.
