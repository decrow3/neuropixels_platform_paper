# Adding MouseV2 V1 to the coarse V1/HVA comparison

Exploratory sensitivity calculation, kept separate from the current figure.

Retain the existing matched Allen V1/HVA session pairs and add one V1 mean per MouseV2 session, averaging equally across the session's anatomically eligible V1 probe means. Each observed session/group mean has equal weight; the pooled V1 group therefore contains eight more observations than HVA. Allen pairs remain paired in resampling. Bootstrap Allen sessions and MouseV2 sessions independently within source, retaining source sample sizes (5,000 replicates; seed 44). The pooled denominator and group balance differ from the original matched comparison.

| Metric | Allen-only explained variance | With MouseV2 | Pooled 95% bootstrap interval |
|---|---:|---:|---:|
| TTFS | 4.7% | 6.9% | 0.06–19.1% |
| log10 F1/F0 | 5.1% | −1.3% | −1.6–3.2% |
| Response timescale | 25.5% | 26.3% | 16.5–38.0% |

The control retains 45, 28, and 53 Allen pairs for TTFS, F1/F0, and timescale, respectively, and adds eight MouseV2 V1 session means for each metric. The within-V1 location estimates and the Across HVAs vs within V1 effect difference in the existing figure already include MouseV2.

For log10 F1/F0, the average session means are −0.1454 for Allen V1, −0.1222 for Allen HVA, and −0.0655 for MouseV2 V1. Thus the two V1 sources lie on opposite sides of Allen HVA. This pooled comparison mixes anatomical grouping with cross-project differences; these results do not identify a cause for that source discrepancy.

No pooled shuffle p-values or significance stars are assigned. MouseV2 has no matched HVA group, so the existing pair-swap null cannot simply be extended to these added sessions. Keeping MouseV2 labels fixed would test a conditional null, while shuffling across projects would require cross-project exchangeability. The pooled result is currently a descriptive sensitivity estimate with source-stratified bootstrap uncertainty, not a replacement for the matched control's permutation test.

Reproduce with `python scripts/figure4_pooled_v1_sensitivity.py` after generating the current figure inputs. Exact pooled inputs and results are saved as `Figure4_pooled_v1_sensitivity_inputs.csv` and `Figure4_pooled_v1_sensitivity.csv`. Verified one row per metric/session/group, eight MouseV2 V1-only rows per metric, and complete retained Allen pairs.
