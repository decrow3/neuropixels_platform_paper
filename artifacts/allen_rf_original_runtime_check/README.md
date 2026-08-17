# Original Allen RF runtime checkpoint

Run date: 2026-08-14 (America/Los_Angeles)

## Question

Can the unmodified AllenSDK receptive-field analysis run through a full native
Brain Observatory 1.1 `EcephysSession`, rather than a lightweight session
adapter?

## Native input

- Allen ecephys session: `721123822`
- Warehouse well-known-file ID: `1026123696`
- Download URL: `http://api.brain-map.org/api/v2/well_known_file_download/1026123696`
- Bytes: `1,736,516,600`
- SHA-256: `4e284295a1be5c6cca49df84fab52ad38b4749d2361b2edebeb676051cf09921`
- NWB version: `2.2.2`
- NWB identifier: `721123822`
- Raw NWB unit rows: `1,603`
- RF stimulus table: `gabors_presentations`

The downloaded NWB was stored temporarily at
`/tmp/allen_rf_baseline/session_721123822.nwb` and is intentionally not copied
into the repository.

## Runtime

- Python `3.7.16`
- AllenSDK `2.2.0`
- PyNWB `1.5.0`
- HDMF `2.5.5`
- h5py `2.10.0`
- NumPy `1.18.5`
- pandas `0.25.3`
- user-site packages disabled with `PYTHONNOUSERSITE=1`

The test used the original public APIs without modifying AllenSDK:

```python
session = EcephysSession.from_nwb_path(
    nwb_path,
    api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf,
        "filter_by_validity": True,
    },
)
rf = ReceptiveFieldMapping(session, filter=[950907203])
metrics = rf.metrics
```

The documented unit filter restricted the expensive serial metric calculation
to one released unit, but the input remained the complete native session and
AllenSDK performed its normal full-session unit-table access.

## Result

- Exit status: `0`
- Python elapsed time through `rf.metrics`: `68.052 s`
- Total wall time including environment startup: `74.00 s`
- Peak resident memory: `2,071,124 KiB` (about `1.98 GiB`)
- Swaps: `0`
- Output: `session_721123822_unit_950907203_metrics.csv`

This establishes that the paper-era AllenSDK environment can load a complete
native historical session and execute the original RF metrics code. A separate
all-unit run was started and passed the same load path, then intentionally
interrupted after this checkpoint because the 1,000-shuffle significance test
is serial and would take hours over the full population.

## Constructor-versus-batch threshold discovery

The bare class constructor and Allen's batch runner do not use the same RF-mask
default in AllenSDK 2.2.0:

- `ReceptiveFieldMapping.__init__`: `mask_threshold=0.5`
- `_schemas.ReceptiveFieldMapping`: `mask_threshold=1.0`
- `stimulus_analysis.__main__` passes the schema values into the class.

The initial direct-constructor run therefore thresholded the smoothed RF at
`peak - 0.5 SD`. For unit 950907203 this selected a four-pixel top-edge
component with pixel center `(6.5, 0.0)`, producing `(75 deg, 50 deg)` and
`400 deg2`.

The same native 9 x 9 spike-count map thresholded with the batch-schema value,
`peak - 1.0 SD`, selected a ten-pixel component with pixel center `(5.8, 1.4)`.
That converts exactly to the released `(68 deg, 36 deg)` and `1000 deg2`.

## Batch-path reproduction

The one-unit calculation was repeated with the parameters supplied by Allen's
batch schema and session loader, including `mask_threshold=1.0` and
`filter_by_validity=False`. It completed successfully and reproduced the
released center and area exactly:

| Metric | Released | Bare constructor | Batch parameters |
| --- | ---: | ---: | ---: |
| azimuth_rf | 68 | 75 | 68 |
| elevation_rf | 36 | 50 | 36 |
| area_rf | 1000 | 400 | 1000 |
| width_rf | 2514.165622 | 2514.456346 | 2514.456346 |
| height_rf | 1574.699273 | 1574.749330 | 1574.749330 |
| p_value_rf | 0.600 | 0.622 | 0.589 |

The batch-path test was then repeated on the three original diagnostic units
from checksum-verified session 737581020. All three released centers and areas
were reproduced exactly:

| Unit | Released center | Bare-constructor center | Batch center | Released area | Bare area | Batch area |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 951867908 | (46.667, 13.333) | (50, 15) | (46.667, 13.333) | 300 | 200 | 300 |
| 951868026 | (90, 25) | (90, 25) | (90, 25) | 400 | 200 | 400 |
| 951870245 | (60, 50) | (20, -25) | (60, 50) | 100 | 600 | 100 |

All deterministic non-Gaussian RF outputs now match for four inspected units
across two native sessions. The p-value varies because the 1,000-shuffle
routine is unseeded. Two very broad Gaussian fits retain small numerical
width/height differences, while the well-conditioned fit for unit 951867908
matches to floating-point precision; this is consistent with optimizer or
dependency sensitivity and does not affect center or area.

The earlier conclusion that the released table used `peak - 0.5 SD` was based
on the bare constructor and is corrected here: the inspected released metrics
are consistent with Allen's batch schema, `peak - 1.0 SD`.

