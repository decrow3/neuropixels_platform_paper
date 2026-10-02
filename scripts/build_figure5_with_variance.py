#!/usr/bin/env python3
"""Figure 5 v14: right-hand location-associated variance panel."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from matplotlib.text import Text
from scripts import build_figure5_shift_hierarchy as previous
from scripts import build_figure5_original_vectors as original
OUT=ROOT/'artifacts/figure5/v14_variance_panel'
old_style=previous.style
WIDTH=1010.;HEIGHT=709.
STATS=ROOT/'artifacts/figure5/supplement_location_variance_v1/estimates_and_tests.csv'


def style(fig):
    previous.OUT=OUT
    old_style(fig)
    axes=list(fig.axes);positions=[ax.get_position().frozen() for ax in axes]
    texts=[(t,t.get_position()) for t in fig.texts]
    fig.set_size_inches(WIDTH/72,HEIGHT/72)
    ratio=852.648/WIDTH
    for ax,p in zip(axes,positions):ax.set_position([p.x0*ratio,p.y0,p.width*ratio,p.height])
    for t,(x,y) in texts:t.set_position((x*ratio,y))
    data=pd.read_csv(STATS)
    metrics=['TTFS (ms)','log10 F1/F0','Response timescale (ms)']
    # The last nine axes are the three split-E sections for each metric.
    control=axes[-9::3]
    assert len(control)==3
    faxes=[]
    for row,(metric,ref) in enumerate(zip(metrics,control)):
        p=ref.get_position();ax=fig.add_axes([865/WIDTH,p.y0,125/WIDTH,p.height]);faxes.append(ax)
        ax.set_xlim(-.55,1.55);ax.set_ylim(-.55,5.15);ax.set_yticks([0,1,2,3,4,5])
        for x,pop,color in [(0,'V1','#7c63a9'),(1,'HVA','#4b535c')]:
            r=data.loc[data.metric.eq(metric)&data.population.eq(pop)].iloc[0]
            ax.vlines(x,100*r.low,100*r.high,color=color,lw=1.54,zorder=3)
            ax.plot(x,100*r.omega_squared,'o',color=color,ms=4.5,zorder=4)
            prob=r.permutation_p_greater
            label=(f'p = {prob:.3f}' if pop=='V1' and metric==metrics[1] else '***' if prob<.001 else '**' if prob<.01 else '*' if prob<.05 else 'n.s.')
            ax.text(x,4.65,label,ha='center',va='center',fontsize=10.2,family='Liberation Sans',color=color)
        ax.set_xticks([0,1],['Intra-V1','Inter-HVA'],rotation=55,ha='right',rotation_mode='anchor')
        ax.tick_params(labelsize=10.85,pad=3,length=3,width=.7,colors='#555555')
        if row<2:ax.tick_params(labelbottom=False)
        ax.set_ylabel('ω² (%)',fontsize=11.16,labelpad=3,family='Liberation Sans')
        ax.set_axisbelow(True);ax.grid(axis='y',color='#e4e4e4',lw=.55)
        ax.axhline(0,color='#888888',lw=.693,ls='--')
        for key in ['top','right']:ax.spines[key].set_visible(False)
        for sp in ax.spines.values():sp.set_color('#999999');sp.set_linewidth(.7)
        for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontfamily('Liberation Sans')
        if row==0:ax.set_title('Location-associated\nvariance',fontsize=12,pad=7,family='Liberation Sans')
    # Match the baseline and physical font size of the existing panel letters.
    e=next(t for t in fig.texts if t.get_text()=='E')
    fig.text(829/WIDTH,e.get_position()[1],'F',fontsize=32,va='top',family='Liberation Sans')
    fig.canvas.draw();r=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(r)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside},indent=2)+'\n')


def main():
    original.OUT=OUT;original.style=style;original.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic';old=ROOT/'artifacts/figure5/v13_shift_hierarchy'
    caption=(old/(stem.name+'_legend.md')).read_text()
    f='''**F)** Location-associated variance, expressed as bias-corrected one-way omega-squared (ω², %), for intra-V1 recording location and inter-HVA area identity. Points and whiskers show estimates and 95% percentile intervals from 5,000 whole-animal bootstrap replicates. Unlike E’s balanced, session-adjusted category-mean SDs, F uses raw neuronal responses with equal cell weights and no session adjustment; negative estimates are retained. Significance annotations use matching within-session location-block shuffles (5,000 permutations; uncorrected one-sided tests): * p < .05, ** p < .01, *** p < .001; n.s., p ≥ .05. All neurons in a session/location move together; tests assume exchangeability of available labels within sessions. For MouseV2 timescale, the observed and shuffled ω² statistics average ten draw-specific values, using one coherent label permutation per session across draws. Intra-V1 phase modulation is reported explicitly as p = 0.051 because it is close to the threshold. These tests concern each location effect, not the difference between intra-V1 and inter-HVA effects; nonsignificance does not establish absence of an effect. Full null distributions and exact p-values are supplied in the location-variance supplement.

'''
    caption=caption.replace('Timescale estimation uses',f+'Timescale estimation uses')
    stem.with_name(stem.name+'_legend.md').write_text(caption)
    pd.read_csv(STATS).query("population != 'HVA_minus_V1'").to_csv(OUT/'Figure5_F_estimates_and_tests.csv',index=False)
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads((old/p.name).read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v14: panel F to the right of E, showing current-cohort omega² and matched shuffle annotations; A–E retain physical size and results.','artboard_points':[WIDTH,HEIGHT],'panel_F_source':str(STATS.relative_to(ROOT))})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v14 — location-variance panel F

Added F to the right of E, aligned to TTFS, phase modulation and timescale rows. Shows two current-cohort omega-squared estimates per row with whole-animal bootstrap intervals and matching block-shuffle significance. The near-threshold intra-V1 phase-modulation p-value is explicit. Null distributions and the omega-squared difference remain in the supplement. A–E retain their physical sizes and values; the artboard expands to 1010 × 709 pt.

Caption includes the distinction between E and F weighting/adjustment. Rebuild: `python scripts/build_figure5_with_variance.py`.
''')

if __name__=='__main__':main()
