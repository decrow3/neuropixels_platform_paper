# MouseV2 physical flash-onset calibration blocker

## What is established

All eight NWBs contain 300 full-field flash intervals: 150 bright
(`contrast == +1`) and 150 dark (`contrast == -1`). Each interval
`start_time` occurs exactly in the NWB processed-stimulus timestamp series.
Median flash duration is approximately 250.21 ms, median inter-start interval
is approximately 2.2519 s, and the implied display-frame period is
approximately 16.681 ms in every session.

This establishes a stable, synchronized **NWB timestamp reference**. It does
not establish when photons appeared on the display. None of the eight files
contains a photodiode dataset or metadata that identifies the interval start as
command time, first displayed frame, photodiode time, or an already corrected
time. Therefore the physical display latency and any session-to-session
variation remain unknown.

## Consequence for this repository

MouseV2 TTFS is reported as raw latency relative to NWB flash `start_time`.
Within-MouseV2 probe comparisons can use that invariant reference. Absolute
MouseV2-versus-Allen latency offsets must not be interpreted, and the earlier
display-only mean match to Allen V1 must not be described as a calibration.

The paper-facing extractor, polarity sensitivities, tests, figures, and timing
audit belong in this repository and are now versioned. A physical correction
must come from evidence independent of the recorded neural responses.

## Cross-repository work needed

### `openscope_v2species` / acquisition records

- Document the semantic meaning of the timestamp written as flash
  `start_time` for the ephys rig and stimulus software version used here.
- Locate the raw sync/photodiode channel, display-latency measurement, rig
  calibration, or stimulus log that links command time to the first visible
  frame.
- Record whether the correction is fixed by rig or varies by session, flash
  polarity, or refresh phase.

### `PilotAnalysis`

- If raw sync or photodiode data can be recovered, validate onset detection on
  all eight sessions and visualize command/frame/photodiode offsets.
- Export a versioned session-level table containing the raw source identity,
  detection method, latency estimate, uncertainty, units, polarity dependence,
  and coverage/failure flags.
- Do not derive the correction from V1 response latency or by matching the
  MouseV2 response distribution to Allen.

### This repository

- Import and validate that timing table without reimplementing raw sync
  exploration.
- Preserve raw TTFS and add a separately named calibrated TTFS column.
- Apply a correction only to absolute-latency displays; a constant shift does
  not change within-MouseV2 variance estimates.
- Regenerate all three flash checkpoints and report whether any non-constant
  correction changes the probe effect.

## Acceptance condition

Calibration is unblocked only when the correction is tied to an acquisition
signal or documented rig measurement, has explicit provenance and units, and
has session-level coverage and uncertainty. If those records do not exist, raw
NWB-relative TTFS remains the final defensible quantity and the paper must state
that absolute cross-dataset latency is not comparable.
