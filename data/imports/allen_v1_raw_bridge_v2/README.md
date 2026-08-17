# Raw Allen V1 bridge

## Released-value reproduction

- Allen Brain Observatory 1.1, session 737581020 (representative): 50/119 common-QC/released V1 units; max common-QC F1/F0 error 1.98e-09; max common-QC modulation-index error 4.6e-08.
- Allen Brain Observatory 1.1, session 746083955 (reproduction_control): 14/50 common-QC/released V1 units; max common-QC F1/F0 error 6.01e-10; max common-QC modulation-index error 4.86e-08.
- Allen Functional Connectivity, session 789848216 (reproduction_control): 17/44 common-QC/released V1 units; max common-QC F1/F0 error 5e-11; max common-QC modulation-index error 1.31e-07.
- Allen Functional Connectivity, session 835479236 (representative): 84/159 common-QC/released V1 units; max common-QC F1/F0 error 5.91e-11; max common-QC modulation-index error 1.43e-07.
- Note: 1 excluded non-QC unit(s) had modulation mismatch >1e-5.

The h5py path therefore reproduces the released metrics without relying on
the currently incompatible AllenSDK/PyNWB environment.

## Common 1-s / 15-trial sensitivity

The representative session in each Allen cohort is the released session
nearest that cohort's equal-session V1 mean modulation index. The original
low-unit sessions are retained only as independent reproduction controls.

- Allen Brain Observatory 1.1, representative session 737581020, mod_idx_dg: released +0.034; common-window +0.088; change +0.053 (subsample 2.5–97.5% +0.053 to +0.053); MouseV2 minus harmonized Allen -0.186.
- Allen Brain Observatory 1.1, representative session 737581020, f1_f0_dg: released -0.237; common-window -0.037; change +0.200 (subsample 2.5–97.5% +0.200 to +0.200); MouseV2 minus harmonized Allen -0.032.
- Allen Functional Connectivity, representative session 835479236, mod_idx_dg: released +0.207; common-window +0.122; change -0.085 (subsample 2.5–97.5% -0.219 to +0.016); MouseV2 minus harmonized Allen -0.220.
- Allen Functional Connectivity, representative session 835479236, f1_f0_dg: released -0.180; common-window -0.034; change +0.146 (subsample 2.5–97.5% +0.130 to +0.160); MouseV2 minus harmonized Allen -0.035.
- MouseV2's eight-session common-support mean is -0.098 log10 modulation index.
- The common-window convention therefore does not remove the modulation-index gap;
  it remains about -0.19 log10 versus Brain Observatory and -0.22 versus Functional Connectivity.
- By contrast, the harmonized F1/F0 gap is only about -0.03 to -0.04 log10,
  showing that the large modulation-index difference is metric-specific.

The representative sessions expose the direction of the protocol effect, while
all four sessions validate the raw implementation. One representative session
per cohort is not sufficient to estimate the population-level
Allen dataset coefficient; the common-window extraction must be extended to
more sessions or its uncertainty must be propagated as a protocol sensitivity.

## Verified raw protocol

- Brain Observatory: 40 nonblank grating conditions x 15 repeats, 2-s trials, and 150 flashes.
- Functional Connectivity: 8 grating directions x 75 repeats at 2 Hz, 2-s trials, and 150 flashes.
- MouseV2: 100 grating conditions x 15 repeats, 1-s trials, and 300 flashes; its SF = 0.04 raw bridge is already complete.
