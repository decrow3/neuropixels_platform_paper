# Allen BO 1.1 interior V1 RF size

RF centers within 20° of any released RF-grid boundary were excluded.
The retained population contains **2457/3186 units** from **31 sessions**.
RF area is log2 transformed and median/IQR standardized within session before spatial aggregation.
The map first estimates each session locally and then combines sessions, preventing unit-rich sessions from dominating.

At this cutoff, the median within-session association between RF size and Allen-origin eccentricity is **rho = -0.029**.
Separately, the median associations are **rho = -0.043** for azimuth and **rho = +0.126** for elevation.
The cutoff-sensitivity panel shows whether that relationship persists as progressively more boundary-adjacent RF centers are removed.
This removes center estimates near the sampled grid boundary; it cannot guarantee that a large RF centered inside the boundary was fully contained by the stimulus support.
