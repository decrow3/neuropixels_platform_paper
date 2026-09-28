# Figure 4 MouseV2 anatomical-V1 update

Updated 2026-09-25 from the refreshed protected DANDI:001568 draft.

## What changed

- The canonical CCF table contains all 32 localized probe entry points on the genuine Allen CCFv3 dorsal
  surface atlas, with the VISp and surrounding HVA borders supplied by that atlas. Entry points
  are the most superficial VISp-labelled AP contacts; 30/32 independently land in the VISp
  surface polygon, with one VISa and one VISpm border case retained without adjustment.
- Figure 4 A includes all 32 probes from eight animals. The corrected subject 815152 NWB resolves
  the earlier B/C/E SmartSPIM track-association conflict and agrees with the ephys and
  receptive-field probe identities.
- Every MouseV2 neuron in panels C, E, and F is now retained only when its NWB unit's extremum
  AP channel has an atlas `location` beginning with `VISp`.
- All 32 probe tracks intersect a VISp-labelled channel span. The two entry proxies that project
  just outside the independent dorsal VISp polygon (subject 810532 probes A and B) contribute
  only neurons whose extremum channels are both VISp-labelled and strictly deeper than the entry
  proxy. These conditions and the analysis-to-NWB probe-label match are enforced during rebuilds.
- The existing minimum of five eligible neurons per session × probe × metric cell is reapplied
  after anatomical filtering.
- Allen/HVA observations and all metric definitions are unchanged.

The NWB-to-analysis reconciliation is exact: all 20,374 analysis unit IDs map uniquely to an NWB
unit and extremum AP electrode. Of these, 8,314 (40.8%) are on VISp-labeled channels.

## Figure-population changes

| Metric | Unique MouseV2 neurons before | VISp-filtered | Retained | Session × probe cells |
| --- | ---: | ---: | ---: | ---: |
| TTFS | 1,413 | 1,063 | 75.2% | 31/32 |
| log10 F1/F0 | 11,242 | 4,771 | 42.4% | 32/32 |
| Response timescale | 681 | 576 | 84.6% | 29/30 |

Site 7 probe B falls below the five-neuron floor for TTFS (2 VISp neurons) and response timescale
(1 VISp neuron), so those two cells are omitted. The other previously eligible cells remain.

## Updated identity contrast

The observed bias-corrected identity contrast is `omega^2(HVA) - omega^2(V1)`; intervals use the
same 5,000-replicate whole-session bootstrap as before.

| Metric | Previous contrast | VISp-filtered contrast | VISp-filtered 95% interval |
| --- | ---: | ---: | ---: |
| TTFS | +0.032 | +0.163 | [-0.131, +0.313] |
| log10 F1/F0 | +0.062 | -0.063 | [-0.324, +0.087] |
| Response timescale | +0.190 | +0.147 | [-0.291, +0.266] |

All three updated intervals include zero, so the paper-facing conclusion is unchanged: these data
do not establish that HVA identity explains more variation than V1 location identity.

## Audit artifacts

- `Figure4_mousev2_visp_filter_audit.csv` records before/after counts and means for every
  session × probe × metric cell.
- `Figure4_mousev2_probe_v1_audit.csv` records the VISp contact span, surface-border status, and
  exact Figure 4 VISp-unit counts for every session × probe track.
- `../artifacts/figure3/06r_mousev2_probe_tracks_from_ccf/mousev2_unit_ccf_locations.csv` records
  the unit-to-extremum-electrode CCF and atlas-location mapping.
