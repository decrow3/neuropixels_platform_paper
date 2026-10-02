#!/usr/bin/env python3
"""Restore the direct omega-squared contrast in compact panel F."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from matplotlib.text import Text
from scripts import build_figure5_narrow_ef as previous
OUT=ROOT/'artifacts/figure5/v16_variance_contrast'
old_style=previous.style

def style(fig):
    old_style(fig)
    data=pd.read_csv(previous.prior.STATS)
    for row,(ax,metric) in enumerate(zip(fig.axes[-3:],data.metric.unique())):
        r=data.loc[data.metric.eq(metric)&data.population.eq('HVA_minus_V1')].iloc[0]
        ax.vlines(2,100*r.low,100*r.high,color='#315f9a',lw=1.7,zorder=3)
        ax.plot(2,100*r.omega_squared,'D',color='#315f9a',ms=5,zorder=4)
        ax.set_xlim(-.55,2.55);ax.set_ylim(-2.5,5.15);ax.set_yticks([-2,0,2,4])
        ax.set_xticks([0,1,2],['Intra-V1','Inter-HVA','Inter-HVA\n− intra-V1'],rotation=90,ha='center',va='top')
        if row<2:ax.tick_params(labelbottom=False)
        for t in ax.texts:t.set_fontsize(9)
        for t in ax.get_xticklabels():t.set_fontfamily('Liberation Sans');t.set_rotation_mode('default')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(renderer)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside,'E_width_ratio':.5,'F_width_ratio':.5,'F_contrast_present':True},indent=2)+'\n')

def main():
    previous.OUT=OUT;previous.style=style;previous.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    p=stem.with_name(stem.name+'_legend.md');s=p.read_text();s=s.replace('for intra-V1 recording location and inter-HVA area identity.','for intra-V1 recording location and inter-HVA area identity, followed by their signed inter-HVA minus intra-V1 difference (blue diamonds). The difference is expressed in percentage points and its 95% interval is calculated from independent-population whole-animal bootstrap differences. No shuffle-test significance symbol is assigned to this contrast. All three contrast intervals include zero; this does not establish equivalence.');p.write_text(s)
    pd.read_csv(previous.prior.STATS).to_csv(OUT/'Figure5_F_estimates_and_tests.csv',index=False)
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads(p.read_text());m.update(renderer=str(Path(__file__).relative_to(ROOT)),description='Figure 5 v16: compact E/F with direct inter-HVA minus intra-V1 omega-squared contrast restored in F. Blue diamonds indicate the contrast; no dagger.');p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('# Figure 5 v16\n\nDirect omega-squared contrast restored in F as blue diamonds with 95% bootstrap intervals; no dagger or shuffle stars on contrasts. Component tests unchanged. E/F retain half widths. F scale extended to include all contrast intervals.\n\nRebuild: `python scripts/build_figure5_variance_contrast.py`.\n')
if __name__=='__main__':main()
