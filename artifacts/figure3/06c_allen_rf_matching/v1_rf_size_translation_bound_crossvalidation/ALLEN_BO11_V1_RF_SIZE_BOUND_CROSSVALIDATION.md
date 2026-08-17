# Allen BO 1.1 RF-size translation bound cross-validation

Translations are fitted using one random unit half and scored on the other half's RF-size surface; both directions are averaged within session.
All bounds use the same fixed per-degree regularization and the same held-out correlation-gain scale.
This comparison is unaffected by the smaller numerical offset span available to narrow bounds.

| Bound | Median held-out Δr | Positive sessions | Wilcoxon p |
| ---: | ---: | ---: | ---: |
| ±10° | +0.038 | 52% | 0.721 |
| ±15° | +0.022 | 58% | 0.29 |
| ±20° | +0.041 | 55% | 0.189 |
| ±30° | +0.068 | 58% | 0.163 |
