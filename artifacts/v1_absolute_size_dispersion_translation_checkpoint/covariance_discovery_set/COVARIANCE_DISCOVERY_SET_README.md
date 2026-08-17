# High-confidence V1 covariance discovery set

## Selection fixed without covariance-registration outcomes

Five sessions passed every independent quality gate:

- at least 80 V1 units;
- held-out CCF-to-RF gradient R2 at least 0.5;
- median anatomy-support correction below 10%;
- maximum consecutive tangential CCF step below 150 um;
- fewer than 30% parameter-bound RF fits;
- robust 10-90% RF-center span of at least 20 degrees;
- median held-out RF-fit deviance at least 0.9.

The selected sessions are 781842082, 798911424, 778240327, 760345702, and
771990200. Covariance objective shape, optimum, and reproducibility played no role
in selection.

## Support-matched evaluation

Local covariance is calculated from RF residual vectors after subtracting the
leave-one-animal-out CCF-to-RF mean map. Full-session, random cell-half, and
interleaved physical-block estimates are compared with separately matched
templates. Training-session covariance is recomputed at the same cell-half or
physical-half density before animal-balanced averaging.

An exact-support null retains every RF center and permutes conditional-scatter
values over those locations.

## Results

| Session | Full shift (az, el) | Cell-half difference | Physical-half difference | Full basin points | Exact-support shuffle p | Phenotype |
|---|---:|---:|---:|---:|---:|---|
| 781842082 | (-6, -2) | 10.0 deg | 34.1 deg | 199 | 0.00 | Cell-stable, regionally inconsistent |
| 798911424 | (-28, -16) | 10.8 deg | 60.0 deg | 693 | 0.03 | Annular/boundary ambiguity |
| 778240327 | (8, -6) | 28.1 deg | 41.2 deg | 384 | 0.00 | Broad and unstable |
| 760345702 | (12, 0) | 7.2 deg | 2.0 deg | 150 | 0.00 | Only robust discovery candidate |
| 771990200 | (24, 24) | 57.2 deg | 33.9 deg | 205 | 0.00 | Multimodal and unstable |

All five contain covariance-to-location organization that fits better than the
exact-support shuffle. That is evidence for real spatial structure, but it is not
sufficient for a uniquely identifiable translation.

Session 798911424 is the clearest warning: its two random cell halves agree because
they choose nearby points on the same broad annular ridge, while interleaved
physical sections choose opposite sides of that ridge. Random cell-split agreement
alone would falsely label it successful.

Session 760345702 is the only case with a single bowl-like objective and agreement
across both random cells and physical sections. Even here, full versus subset
solutions differ by roughly 10 degrees, so the current practical uncertainty is
not negligible.

## Good-session phenotype

The current clean phenotype is therefore more restrictive than data quality or a
significant descriptor-location null:

1. a single compact, non-boundary basin rather than a ridge or ring;
2. stable random-cell halves using density-matched templates;
3. stable physically separated/interleaved blocks;
4. subset solutions near the full-session basin;
5. real descriptor placement outperforming exact-support shuffles.

Only one of five independently quality-selected sessions presently shows the full
phenotype. This is an exploratory discovery result, not yet a validated
registration rule.

Absolute RF size is displayed descriptively and was not used to select or accept
any session.

## Outputs

- `Figure_v1_covariance_discovery_set.png`
- `discovery_set_selection.csv`
- `discovery_quality_audit_all_sessions.csv`
- `discovery_covariance_translation_results.csv`
- `discovery_covariance_landscapes.csv.gz`
- `discovery_exact_support_shuffle_null.csv.gz`
- `run_manifest.json`

Reproduce from the repository root with:

```bash
python -m scripts.inspect_v1_covariance_discovery_set --overwrite
```
