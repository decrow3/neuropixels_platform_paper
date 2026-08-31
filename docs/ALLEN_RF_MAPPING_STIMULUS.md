# Allen Visual Coding Neuropixels RF-mapping stimulus

This note documents the realized Gabor stimulus used by the Allen Visual Coding
Neuropixels sessions and the interpretation used by this repository's RF fits.
It distinguishes the rendered frame schedule from the misleading presentation-
level `phase` value stored in the released NWB tables.

## Carrier, aperture, grid, and timing

- Carrier: achromatic sinusoidal grating, spatial frequency 0.08 cycles/degree,
  temporal frequency 4 Hz, and contrast 0.8.
- Carrier orientations: 0, 45, and 90 degrees.
- Aperture: 20-degree-diameter circular mask on mean-luminance gray.
- Grid centers: `-40, -30, ..., +40` degrees on both axes, giving 81 positions
  with a 10-degree stride.
- Conditions: 81 positions x 3 orientations = 243.
- Repetitions: normally 15 per condition, or 3,645 presentations.
- Nominal presentation duration: 0.25 s with no blank interval.
- Nominal display rate: 60 frames/s, giving 15 rendered frames per presentation.

The 10-degree stride is half the aperture diameter. Adjacent cardinal positions
therefore overlap by 10 degrees along their center line (50% of the diameter;
about 39% overlap by circular area). The analysis treats the stimulus as a
10-degree-radius circular aperture rather than a Gaussian envelope.

## Exact framewise phase schedule

The public CamStim implementation updates a drifting grating on every rendered
frame using:

```python
self.stim.setPhase(TF * self.current_frame / self.fps)
```

For this stimulus, `TF = 4` and `fps = 60`, so the unwrapped phase on RF-block
frame `f` is

```text
phase_cycles(f) = 4 f / 60 = f / 15.
```

PsychoPy phase is measured in cycles and rendered modulo one. Presentation `k`
occupies frames `15k` through `15k + 14`, so its within-presentation phases are:

| Frame | Phase (cycles) | Phase (degrees) |
| ---: | ---: | ---: |
| 0 | 0/15 | 0 |
| 1 | 1/15 | 24 |
| 2 | 2/15 | 48 |
| 3 | 3/15 | 72 |
| 4 | 4/15 | 96 |
| 5 | 5/15 | 120 |
| 6 | 6/15 | 144 |
| 7 | 7/15 | 168 |
| 8 | 8/15 | 192 |
| 9 | 9/15 | 216 |
| 10 | 10/15 | 240 |
| 11 | 11/15 | 264 |
| 12 | 12/15 | 288 |
| 13 | 13/15 | 312 |
| 14 | 14/15 | 336 |

The next presentation begins one full cycle later. The code does not explicitly
reset the phase at a position/orientation transition, but the 15-frame duration
at 4 Hz returns the phase to zero modulo one. Consequently **every Gabor
presentation begins at phase 0 cycles (0 degrees)**. Starting phase was not
randomized across position, orientation, or repetition.

Source-code anchors:

- [Allen CamStim RF stimulus definition](https://github.com/AllenInstitute/openscope-crossv2species/blob/main/production-scripts/ephysrecording-FINAL.py#L449-L478)
- [CamStim per-frame temporal-frequency update](https://github.com/AllenInstitute/openscope-crossv2species/blob/main/camstim/camstim/sweepstim.py#L204-L252)
- [CamStim inclusive 15-frame sweep construction](https://github.com/AllenInstitute/openscope-crossv2species/blob/main/camstim/camstim/misc.py#L303-L311)
- [PsychoPy phase convention](https://devdocs.psychopy.org/api/visual/gratingstim.html)

## Why every NWB row says `[3644.93333333, 3644.93333333]`

That value is not the onset phase of each presentation. There are 3,645
presentations x 15 frames = 54,675 rendered frames, indexed 0 through 54,674.
The phase on the final rendered frame is

```text
4 * 54,674 / 60 = 3,644.93333333 cycles.
```

This exactly matches the value copied into every released Gabor presentation
row. It is the final unwrapped state of the shared PsychoPy phase object as
serialized into the stimulus metadata. Modulo one, it is 14/15 cycles = 336
degrees, the phase of the last rendered frame—not a per-presentation starting
phase. It must not be used as though every presentation began at 336 degrees.

Allen staff independently confirmed that the drifting-grating/Gabor phase was
the same rather than randomized:
[Allen Brain Map discussion](https://community.brain-map.org/t/gabor-stimuli-properties/1499).

## Relation to the RF fits in this repository

The current aperture RF model does not include carrier phase as a fitted
variable. It fits a baseline, three orientation-specific amplitudes, and latent
Gaussian spatial center/width parameters to individual presentation spike
counts. Because the Gabor onset phase is fixed, these measurements are not
phase-averaged RF estimates. The analytic aperture correction concerns the
20-degree circular spatial mask and does not alter the carrier's framewise
phase schedule.
