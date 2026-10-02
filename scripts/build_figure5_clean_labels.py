#!/usr/bin/env python3
"""Figure 5 v6: clearer SD labels and stronger C/D population headings."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_split_e as prior
from scripts import build_figure5_original_vectors as original
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v6_clean_labels'
old_style=prior.style


def style(fig):
    prior.OUT=OUT
    old_style(fig)
    # Identify pooled axes by the explicit SD-span annotations, independent of
    # the split E axes appended by the previous renderer.
    pooled=[a for a in fig.axes if any(hasattr(t,'_sd_span') for t in a.texts)]
    assert len(pooled)==6
    for ax in pooled:
        ax.set_title('')
        for t in list(ax.texts):
            if not hasattr(t,'_sd_span'):continue
            x,lo,hi=t._sd_span
            name='Cells SD' if t.get_text()=='cells' else 'Means SD'
            t.remove()
            # Put each label outside its bar at a fixed central height. Horizontal
            # displacement keeps it clear of the bar regardless of metric scale.
            ax.text(-.16 if name=='Cells SD' else 1.18,.5,name,
                    transform=ax.transAxes,rotation=90,ha='center',va='center',
                    fontsize=10.85,family='Liberation Sans',color='#333333',clip_on=False)
    for ax in fig.axes:
        if ax.get_title() in ['Inter-HVA','Intra-V1']:
            ax.set_title(ax.get_title(),fontsize=16,pad=10,family='Liberation Sans')
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
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    p=stem.with_name(stem.name+'_legend.md');s=p.read_text()
    s=s.replace('“Means” refers to area means in C and location means in D.', '“Means SD” refers to the SD of area means in C and location means in D; “Cells SD” refers to raw pooled cellular SD. Labels are placed consistently outside the corresponding bars.')
    s=s.replace('All four estimates share a response-unit axis;', 'E is separated into three horizontal sections with identical y-scales within each row; no numerical y-axis interval is removed. All four estimates share a response-unit scale;')
    p.write_text(s)
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads(p.read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v6: explicit Cells SD / Means SD labels consistently outside pooled bars; Pooled SD titles removed; C/D headings increased to 16 pt. A hierarchy unchanged.', 'artboard_points':[852.648,709.0]})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v6 — clear pooled SD labels

Removed the Pooled SD headings. Added consistent Cells SD and Means SD labels outside each pooled SD bar. Inter-HVA and Intra-V1 headings above C/D increased to 16 pt. Hierarchy labels and all statistical results are unchanged. Retains v5 split E and 852.648 × 709 pt artboard.

Rebuild: `python scripts/build_figure5_clean_labels.py`.
''')

if __name__=='__main__':main()
