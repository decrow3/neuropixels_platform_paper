# Allen Brain Observatory 1.1 simultaneous V1/HVA session atlas

The atlas contains 31 sessions with V1 and at least one simultaneously
recorded HVA contributing supported RF, SF, and TF populations. Each page pools all
simultaneous HVA probes and compares them with V1 in the same session.

RF, SF, and TF maps use independent eligible unit populations. RF comes from the
Gabor mapping block, SF from static gratings, and TF from drifting gratings. Probes
recorded simultaneously within each block, but the three blocks occurred sequentially.
No cross-session alignment is applied.

Area-specific LM/RL/AL/PM/AM surfaces are retained in the grid CSV even though the
primary atlas pools them for more stable within-session support.

## Outputs

- `Figure_allen_bo11_simultaneous_v1_hva_session_atlas.pdf`: one page per session.
- `session_figures/`: the same pages as individual PNG files.
- `allen_bo11_simultaneous_v1_hva_surface_grid.csv`: pooled and area-specific grids.
- `allen_bo11_simultaneous_v1_hva_population.csv`: exact populations and support.

## Interpretation boundary

This is a within-session V1-versus-HVA view, not an Allen analogue of four V1 probes.
Most Allen sessions contain one V1 probe; the HVA row can combine several probes and
areas. Differences may reflect area, probe targeting, RF coverage, or finite sampling.
