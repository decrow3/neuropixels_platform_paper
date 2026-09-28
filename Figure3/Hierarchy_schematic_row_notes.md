# Hierarchy schematic: score verification and spread brackets

Panel B uses the anatomical hierarchy scores from Siegle et al. (2021), not the CCG-derived functional hierarchy. All six cortical values were checked by area against the original repository's `Figure4/Figure4.py` (`regions` / `hierScore`) at upstream commit `4f8bcc26fa6fbec1ae687c650bacc9ea67a7036d` on 2026-09-21. The previous values matched after rounding to ten decimal places; the renderer now retains full source precision.

| Area | Anatomical hierarchy score |
|---|---:|
| V1 | -0.35733209934482374 |
| LM | -0.09388855125761343 |
| RL | -0.05987132463908328 |
| AL | 0.15221797920142832 |
| PM | 0.32766807486511995 |
| AM | 0.440986074378801 |

The original eight-area fit includes LGd and LP. This schematic displays the six cortical areas without refitting or renormalizing their scores. The paper's Fig. 2a and anatomical hierarchy methods identify these as anatomy-derived scores recomputed from the connectivity reference; the methods also distinguish them from CCG directionality-based functional scores.

Sources:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10399640/#F2
- https://github.com/AllenInstitute/neuropixels_platform_paper/blob/4f8bcc26fa6fbec1ae687c650bacc9ea67a7036d/Figure4/Figure4.py

Panel D's right-side brackets span the minimum and maximum illustrative node-center heights for the HVA and V1 groups. They are visual guides, not measured ranges or confidence intervals. The results quantify V = mean_g[(m_g - mean(m))^2] for standardized functional category means. They do not estimate within-V1 anatomical hierarchy scores or a hierarchy-score range. The original illustrative node coordinates are unchanged.

Rebuilt outputs: `Figure3/Hierarchy_schematic_row.{svg,pdf,png}` and `artifacts/figure4_rerun/v8_final/Figure4_CDE_rerun_with_schematic.{pdf,png}`. The original CDE PDF is preserved.
