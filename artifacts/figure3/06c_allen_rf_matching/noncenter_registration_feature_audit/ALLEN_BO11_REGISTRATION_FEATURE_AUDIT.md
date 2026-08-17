# Allen BO 1.1 non-center registration feature audit

Population: 10,919 RF-supported visual-cortical units across 31 simultaneous V1/HVA sessions.
Associations rank units within session × area before pooling, so gross session and
area differences do not create the reported gradients. These are screening statistics,
not evidence that any feature can identify a session transform.

| Candidate | Family | Coverage | Strongest coordinate | ρ | Group sign agreement |
| --- | --- | ---: | --- | ---: | ---: |
| Dorsal–ventral CCF coordinate | Anatomy / probe | 74.2% | RF azimuth | -0.172 | 68.5% |
| Cortical depth | Anatomy / probe | 100.0% | RF eccentricity | -0.151 | 70.3% |
| Probe vertical position | Anatomy / probe | 100.0% | RF eccentricity | +0.143 | 67.6% |
| Probe horizontal position | Anatomy / probe | 100.0% | RF eccentricity | +0.116 | 81.1% |
| RF lifetime sparseness | RF mapping | 100.0% | RF azimuth | +0.073 | 60.8% |
| RF area | RF mapping | 100.0% | RF eccentricity | -0.069 | 54.7% |
| Drifting-grating response rate | Grating response | 100.0% | RF eccentricity | -0.068 | 60.1% |
| Drifting-grating F1/F0 | Grating response | 100.0% | RF eccentricity | +0.064 | 58.8% |
| RF response time-to-peak | RF mapping | 100.0% | RF elevation | -0.058 | 69.6% |
| RF response rate | RF mapping | 100.0% | RF eccentricity | -0.057 | 60.1% |
| Flash sustained index | Flash / latency | 98.0% | RF eccentricity | -0.051 | 60.5% |
| Flash response rate | Flash / latency | 100.0% | RF azimuth | -0.047 | 57.8% |
| Left–right CCF coordinate | Anatomy / probe | 74.2% | RF eccentricity | +0.047 | 58.6% |
| Intrinsic timescale | Cell physiology | 85.8% | RF azimuth | +0.047 | 61.0% |
| Drifting-grating direction selectivity | Grating response | 100.0% | RF eccentricity | +0.039 | 56.8% |

## Interpretation

- RF area is the valid released RF-size measure. `width_rf` and `height_rf` are
  not used because their released values show implausible scales and do not form a
  reliable size/shape decomposition.
- RF-mapping response features can be used as non-center scalar fields, although
  they are not independent of the RF stimulus block.
- Flash latency is stimulus-independent of RF and grating tuning and is therefore
  attractive if its spatial gradient is sufficiently reproducible.
- Drifting-grating modulation/F1–F0 can inform an exploratory transform, but using
  them to fit a transform and then claiming improved TF agreement would be circular.
  They require SF evaluation or explicit cross-fitting.
- CCF/probe coordinates have the strongest available gradients but may encode probe
  trajectory and cortical depth rather than eye/screen displacement. They should be
  combined with physiological fields, not used alone.
