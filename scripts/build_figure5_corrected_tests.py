#!/usr/bin/env python3
"""Figure 5: Bonferroni adjustment across the six panel-F component tests."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts import build_figure5_variance_contrast as previous
OUT=ROOT/'artifacts/figure5/v17_corrected_tests'
old_style=previous.style
SOURCE=previous.previous.prior.STATS

def adjusted_table():
    data=pd.read_csv(SOURCE)
    assert data.permutation_p_greater.notna().sum()==6
    data['permutation_p_bonferroni']=(6*data.permutation_p_greater).clip(upper=1)
    return data

def style(fig):
    old_style(fig)
    data=adjusted_table()
    for ax,metric in zip(fig.axes[-3:],data.metric.unique()):
        for t in ax.texts:
            x,y=t.get_position()
            if x not in (0,1):continue
            pop=['V1','HVA'][int(x)]
            p=data.loc[data.metric.eq(metric)&data.population.eq(pop),'permutation_p_bonferroni'].item()
            t.set_text('***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.')
    fig.canvas.draw()

def main():
    previous.OUT=OUT;previous.style=style;previous.main()
    adjusted_table().to_csv(OUT/'Figure5_F_estimates_and_tests.csv',index=False)
    p=OUT/'Figure5_CDE_rerun_with_schematic_legend.md';s=p.read_text()
    s=s.replace('95% percentile intervals from 5,000 whole-animal bootstrap replicates. Unlike E', 'individual 95% percentile intervals from 5,000 whole-animal bootstrap replicates (not simultaneous intervals). Unlike E')
    s=s.replace('5,000 permutations; uncorrected one-sided tests','5,000 permutations; one-sided tests with Bonferroni correction across the six component tests: three metrics × two populations')
    s=s.replace('Intra-V1 phase modulation is reported explicitly as p = 0.051 because it is close to the threshold.','Annotations use adjusted p-values, calculated as min(1, 6 × raw p). Adjusted inter-HVA p-values are 0.002400 for TTFS and 0.001200 for timescale; all remaining component tests have adjusted p ≥ .05.')
    p.write_text(s)
    (OUT/'Figure5_F_condensed_caption.md').write_text('''**F)** Location-associated variance (bias-corrected ω²) within V1, across HVAs, and their difference (inter-HVA − intra-V1; blue diamonds, in percentage points). Whiskers show individual 95% whole-animal bootstrap intervals. Asterisks indicate one-sided within-session location-block shuffle tests, Bonferroni-adjusted across six component tests (** indicates adjusted p < .01; n.s., adjusted p ≥ .05). Difference intervals include zero for all metrics; this does not establish equivalence.
''')
    p=OUT/'Figure5_CDE_rerun_with_schematic_layout_manifest.json';m=json.loads(p.read_text());m.update(renderer=str(Path(__file__).relative_to(ROOT)),description='Figure 5 v17: six panel-F component shuffle tests Bonferroni-adjusted; all estimates and individual bootstrap intervals unchanged.',multiplicity={'method':'Bonferroni','family':'3 metrics × 2 populations','n_tests':6,'formula':'min(1, 6 × raw p)','bootstrap_intervals':'individual 95%; not simultaneous'});p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('# Figure 5 v17\n\nBonferroni correction across six F component shuffle tests. Raw and adjusted p-values retained in the table. TTFS and timescale HVA effects receive two stars; other component effects are n.s. Direct contrast estimates and individual 95% bootstrap intervals unchanged. Earlier versions retain their historical uncorrected annotations.\n\nRebuild: `python scripts/build_figure5_corrected_tests.py`.\n')
if __name__=='__main__':main()
