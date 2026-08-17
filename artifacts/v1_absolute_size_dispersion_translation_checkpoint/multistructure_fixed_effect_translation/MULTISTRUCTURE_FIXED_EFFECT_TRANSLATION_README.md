# Decoupled shape/translation registration across V1+HVA and LGd

## Question

Can a per-session visual-field translation δ_s be estimated without coupling it to the
population retinotopic map's shape, the way the earlier alternating EM
(`fit_joint_multistructure_dispersion_em.py`) coupled them and produced unstable,
structure-disagreeing optima (median ~21° component-to-joint discrepancy across 16 sessions,
and one exact V1–LGd match that turned out to be a shared optimizer-boundary artifact)?

## Method

Shape and translation are estimated in two decoupled stages:

1. **Shape (translation-free).** A kernel-weighted local-linear regression of RF on 2D
   (AP, LR) CCF position is fit at each node of a coarse grid, pooling cells from *all*
   sessions simultaneously. Before the regression, both the CCF-offset and RF inputs are
   demeaned **within session, using the same kernel weights** (the local generalization of
   the within-session centering already used in `check_v1_cross_animal_mean_map_support.py`).
   This makes the fitted local slope exactly invariant to each session's own additive
   translation: a session contributing one point to a neighborhood demeans to an exact zero
   row and contributes nothing, so pooling across sessions can never leak a translation into
   the shape estimate. Huber-IRLS reweighting sits on top of the kernel weights (local-linear
   fits are far more leverage-sensitive than the existing order-0 Nadaraya-Watson averaging in
   `anatomical_residuals`).
2. **Translation (decoupled).** The pooled Jacobian field is path-integrated along each
   session's own sampled CCF positions (a minimum-spanning-tree walk, trapezoidal steps),
   giving a shape-only predicted RF field per session, up to one unknown additive constant.
   δ_s is then a Huber-robust location fit of `observed RF − shape-only prediction`, per
   session — a direct fit, not a grid search or damped alternation, because the shape field
   never moves once stage 1 is done.

V1+HVA are fit as **one connected domain** (a single shared translation, no per-area
intercept), because Allen targeted most probe penetrations to be retinotopically matched in
eccentricity across areas; a per-area offset would impose an artificial break at every area
boundary that the underlying map does not have. LGd is a separate modality/coordinate frame
and gets its own shape field and its own reliability-weighted contribution to the final
per-session δ_s, rather than being hand-excluded, per the user's stated (and here confirmed)
doubt that current LGd data would be very informative.

Cortex and LGd per-session estimates are combined by clip(shuffle_z, 0, 3)/3 reliability
weight (same convention already used in
`checkpoint_joint_multistructure_dispersion_likelihood.py`), then recentered to zero mean
across sessions.

## Results (52 sessions with cortex coverage; 19-24 with usable LGd)

| Check | Cortex | LGd |
|---|---:|---:|
| Sessions passing shuffle p<0.05 (shape beats chance) | 92% | 38% |
| Median split-half translation reproducibility | 3.4° | 8.6° |
| Median shuffle z (shape reliability) | ~10 | ~0 |

Cortex shape reliability and split-half translation reproducibility are both far stronger
than anything in the prior dispersion-trace pipeline (median 3.4° here vs. median 21°
component-to-joint discrepancy in the old EM, and vs. the released-LGd 12-15° split-half
discrepancy reported for the earlier thalamic corroboration attempt). LGd is usable but
markedly weaker, as expected, and correctly ends up downweighted rather than excluded: median
domain weight ≈0.19 for LGd vs. ≈1.0 for cortex.

Cross-checked (not fused) against two existing independent estimates:
- median 14.2° from the existing V1-anatomy translation prior (`all_session_anatomy_offsets.csv`)
- median 33.3° from the old dispersion-trace EM's best-initialization shifts

The large disagreement with the old EM is not itself evidence this method is wrong — the old
EM's own internal component agreement was already poor (median 21° V1/HVA/LGd disagreement),
so it is not a trustworthy reference point. This disagreement is reported here, not resolved;
resolving it (e.g. by checking which method's shape field better predicts genuinely held-out
sessions) is future work.

## A caught boundary artifact, fixed before these results

The first run reused `fit_joint_multistructure_dispersion_em.py::recenter`'s default 30°
clip bound unchanged. Because this method's translation fit is never constrained by a search
window (unlike the old grid search), several sessions were silently clipped onto an artificial
30° boundary (two sessions landed on the exact same clipped azimuth value, `-30.16°`) — the
same class of artifact the report's ±30°→±60° expanded-bound test caught previously for the
dispersion-trace EM. Fixed by widening the safety-net bound to 90° (comfortably beyond the
~60° stimulus screen extent); the exact-duplicate clipped values disappeared and the
resulting distribution of shifts looks unremarkable (max magnitude 45.9°, no boundary pileup).

## Limits

- 2D (AP, LR) CCF only; depth-dependent projection is not modeled (same limitation already
  flagged in `check_v1_cross_animal_mean_map_support.py`).
- The pooled Jacobian field is fit once on the full session pool, not with a per-session
  leave-one-session-out refit (that would multiply the grid-fit cost by the session count);
  split-half refitting is used as the primary reproducibility check instead. True per-session
  LOO is a possible stronger follow-up.
- `min_effective_n` / Huber-cutoff / bandwidth defaults (250 µm cortex, 400 µm LGd) are not
  yet swept; see the open decisions in the implementation plan.
- LGd's own low reliability is consistent with, not a resolution of, the report's
  recommendation to replace point-estimated LGd centers with a full raw-Gabor-response
  likelihood before treating LGd as a strong anchor.

## Outputs

- `session_translations.csv` — final recentered per-session δ_s
- `domain_translation_audit.csv`, `domain_weights.csv` — per-domain δ_s and combination weights
- `domain_shuffle_reliability.csv` — per-session, per-domain shuffle p/z
- `split_half_reproducibility.csv` — split-half distance per domain
- `cross_check_against_existing_estimates.csv` — non-fused comparison to existing estimates
- `Figure_multistructure_fixed_effect_translation.png`
- `run_manifest.json`

Reproduce from the repository root with:

```bash
env MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdgcache \
python -m scripts.fit_multistructure_fixed_effect_translation
```
