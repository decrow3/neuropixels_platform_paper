# Raw Allen V1 bridge

## Released-value reproduction

- Allen Brain Observatory 1.1: 50 V1 units; max absolute F1/F0 error 1.72e-09; max absolute modulation-index error 7.08e-08.
- Allen Functional Connectivity: 44 V1 units; max absolute F1/F0 error 5e-11; max absolute modulation-index error 1.31e-07.

The h5py path therefore reproduces the released metrics without relying on
the currently incompatible AllenSDK/PyNWB environment.

## Common 1-s / 15-trial sensitivity

- Allen Brain Observatory 1.1, mod_idx_dg: released +0.196; common-window +0.231; change +0.035 (subsample 2.5–97.5% +0.035 to +0.035).
- Allen Brain Observatory 1.1, f1_f0_dg: released -0.265; common-window -0.129; change +0.136 (subsample 2.5–97.5% +0.136 to +0.136).
- Allen Functional Connectivity, mod_idx_dg: released -0.099; common-window -0.133; change -0.034 (subsample 2.5–97.5% -0.233 to +0.201).
- Allen Functional Connectivity, f1_f0_dg: released -0.246; common-window -0.119; change +0.127 (subsample 2.5–97.5% +0.099 to +0.152).
- MouseV2's eight-session common-support mean is -0.098 log10 modulation index.

These two Allen sessions validate the raw bridge and expose the direction of
the protocol effect. They are not sufficient to estimate the population-level
Allen dataset coefficient; the common-window extraction must be extended to
more sessions or its uncertainty must be propagated as a protocol sensitivity.

## Verified raw protocol

- Brain Observatory: 40 nonblank grating conditions x 15 repeats, 2-s trials, and 150 flashes.
- Functional Connectivity: 8 grating directions x 75 repeats at 2 Hz, 2-s trials, and 150 flashes.
- MouseV2: 100 grating conditions x 15 repeats, 1-s trials, and 300 flashes; its SF = 0.04 raw bridge is already complete.
