# MouseV2 Allen-condition-support grating bridge

Processed 8 sessions and 20,374 units from raw NWBs.
Each unit's preference was recomputed using only SF = 0.04 cycles/degree,
the four shared MouseV2 orientations, all five shared TFs, contrast 0.8,
15 trials per condition, and the unchanged 1-s MouseV2 response window.

## Result

- Equal-site common-support mean log10 modulation index: -0.098.
- Equal-site change in mean log10 modulation index: +0.009 (site range -0.019 to +0.036).
- Equal-site common-support mean log10 F1/F0: -0.069.
- Equal-site change in mean log10 F1/F0: +0.061 (site range +0.046 to +0.074).

This isolates the MouseV2 preferred-SF condition-space effect. It does not
yet test Allen's 2-s window, Welch grid, Functional Connectivity repeat
count, flash protocol, or population support; those require original Allen NWBs.
