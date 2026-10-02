#!/usr/bin/env python3
"""Figure 5 v5: more space below A/B and compact, segmented E on shared scales."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_taller as prior
from scripts import build_figure5_original_vectors as original
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v5_split_E'
HEIGHT=709.0
GAP=24.0
BOTTOM=35.0
old_style=prior.taller_style


def style(fig):
    prior.OUT=OUT
    old_style(fig)
    old_axes=list(fig.axes)
    positions=[a.get_position().frozen() for a in old_axes]
    fig.set_size_inches(852.648/72,HEIGHT/72)
    for i,(ax,p) in enumerate(zip(old_axes,positions)):
        shift=GAP+BOTTOM if i<4 else BOTTOM
        ax.set_position([p.x0,(p.y0*650+shift)/HEIGHT,p.width,p.height*650/HEIGHT])
    for t in fig.texts:
        x,y=t.get_position()
        shift=BOTTOM if t.get_text() in ['C','D','E'] else GAP+BOTTOM
        t.set_position((x,(y*650+shift)/HEIGHT))
    source=OUT/'Figure5_CDE_rerun_with_schematic_estimates.csv'
    estimates=pd.read_csv(source)
    metrics=['TTFS (ms)','log10 F1/F0','Response timescale (ms)']
    specs=[(['matched_hva_minus_v1_mean'],['HVAs\n− V1'],'Mean\ndifference',.041),
           (['v1_sd','hva_sd'],['Intra-V1','Inter-HVA'],'SD of\nmeans',.061),
           (['hva_minus_v1_sd'],['Inter-HVA\n− intra-V1'],'SD\ndifference',.047)]
    for row,index in enumerate([8,13,18]):
        old=old_axes[index];p=old.get_position().frozen()
        ylim=old.get_ylim();ticks=old.get_yticks();ylabel=old.get_ylabel()
        left=p.x0
        fig.delaxes(old)
        for col,(effects,labels,title,width) in enumerate(specs):
            ax=fig.add_axes([left,p.y0,width,p.height]);left+=width+.012
            ax.set_ylim(ylim);ax.set_yticks(ticks);ax.set_ylim(ylim)
            ax.set_xlim(-.6,len(effects)-.4)
            for j,effect in enumerate(effects):
                r=estimates.loc[estimates.metric.eq(metrics[row])&estimates.effect.eq(effect)].iloc[0]
                color='#a66a20' if col==0 else '#555555' if col==1 else '#333333'
                marker='s' if effect=='hva_sd' else 'o'
                ax.vlines(j,r.low,r.high,color=color,lw=1.54,zorder=3)
                ax.plot(j,r.estimate,marker=marker,color=color,mfc='white' if col==1 else color,ms=4.235,lw=0,zorder=4)
            ax.set_xticks(range(len(labels)),labels,rotation=90,ha='center',va='top')
            ax.tick_params(axis='both',labelsize=10.85,pad=3,length=3,width=.7,colors='#555555')
            if row<2:ax.tick_params(labelbottom=False)
            ax.set_axisbelow(True);ax.grid(axis='y',color='#e4e4e4',lw=.55)
            ax.axhline(0,color='#888888',lw=.693,ls='--')
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
            for sp in ax.spines.values():sp.set_color('#999999');sp.set_linewidth(.7)
            if col:
                ax.spines['left'].set_visible(False);ax.tick_params(axis='y',left=False,labelleft=False)
            else:ax.set_ylabel(ylabel,fontsize=11.16,labelpad=3,family='Liberation Sans')
            if row==0:ax.set_title(title,fontsize=10.85,pad=7,family='Liberation Sans')
            for label in ax.get_xticklabels()+ax.get_yticklabels():label.set_fontfamily('Liberation Sans')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(renderer)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside,'E_shared_y_scale_per_row':True},indent=2)+'\n')


def main():
    original.OUT=OUT;original.style=style;original.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    p=stem.with_name(stem.name+'_legend.md')
    s=p.read_text().replace('All four estimates share a response-unit axis;', 'Each row is split horizontally into three sections (mean difference, two component SDs, and SD difference), with one common y-scale and identical y-limits. The gaps separate categories; no numerical y-axis interval is removed. All four estimates share a response-unit scale;')
    p.write_text(s)
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads(p.read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v5: 24 pt extra A/B-to-C/D/E gap; E split horizontally into three sections with unchanged common row scales.', 'artboard_points':[852.648,HEIGHT], 'E_layout':{'section_widths_fraction':[.041,.061,.047],'gaps_fraction':.012,'labels_rotation_deg':90,'y_scale':'identical within each metric; no numerical axis break'},'added_top_results_gap_pt':GAP,'added_bottom_margin_pt':BOTTOM})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v5 — spaced top row and compact E

Artboard: 852.648 × 709 pt. Added 24 pt between A/B and C/D/E; retained the taller result rows. E now has three narrow sections per metric, with identical y-limits/ticks across sections, and vertical bottom labels. Section gaps do not omit numerical values. Added bottom margin accommodates labels. All estimates, intervals and sample sizes are unchanged.

Rebuild: `python scripts/build_figure5_split_e.py`.
''')

if __name__=='__main__':main()
