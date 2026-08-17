# Allen BO 1.1 RF/gaze pilot download

The local storage audit found 3.8 TiB free on `/media/huklaban5/Data` and only
176 GiB free on the system volume. The dedicated pilot cache therefore lives at
`/media/huklaban5/Data/MouseV2/allen_bo11_rf_pilot_cache` and enforces a 250 GiB
minimum-free-space guard before any download begins.

Three new eye-tracked BO 1.1 sessions were selected by the largest combined
trusted-interior V1 and HVA populations, excluding sessions already local:

| Session | Trusted V1 | Trusted HVA | NWB size | SHA-256 |
|---:|---:|---:|---:|---|
| 760693773 | 129 | 93 | 2.67 GiB | `aa993f1b5164307047f94e5e2376244075d2e68c5e49075bae795a68b6aa8272` |
| 755434585 | 125 | 95 | 2.08 GiB | `1a50c37dd514ce753d7c0c0d64adc4ebd8233b9e49a5225b171e101ecfc07220` |
| 798911424 | 112 | 105 | 2.67 GiB | `87a30e67747c3a83c02f369fa6ae6923f23e985f3c85716f9e085bcedeb6fb7c` |

All three files open as HDF5/NWB and contain units, Gabor presentations, eye
tracking, and raw and filtered gaze mappings. Their session NWBs total 7.42 GiB.
No LFP NWBs or stimulus templates were requested, and no files were deleted.

Together with already local sessions 746083955 and 756029989, this gives a
five-session RF/gaze pilot cohort. The external cache contains the durable
`download_summary.json` and append-only `download_events.jsonl` audit records.

