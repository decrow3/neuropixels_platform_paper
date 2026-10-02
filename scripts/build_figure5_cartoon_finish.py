#!/usr/bin/env python3
"""Figure 5 v12: enlarge cartoons and tighten their label spacing."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_raise_schematics as previous
from scripts import build_figure5_original_vectors as original
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v12_cartoon_finish'
old_style=previous.style


def style(fig):
    previous.OUT=OUT
    old_style(fig)
    for i in (0,2):
        ax=fig.axes[i];p=ax.get_position().frozen()
        w=p.width*1.07;h=p.height*1.07
        # Hold the top of the cartoon frame fixed as it grows downwards.
        ax.set_position([p.x0-(w-p.width)/2,p.y1-h,w,h])
        for t in ax.texts:
            if t.get_text() in ['Inter-HVA','Intra-V1']:
                x,y=t.get_position();t.set_position((x,134))
    for t in fig.axes[1].texts:
        if t.get_text()=='Anatomical score':
            x,y=t.get_position();t.set_position((x-.30,y))
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
    old=ROOT/'artifacts/figure5/v11_raised_schematics'
    caption=stem.with_name(stem.name+'_legend.md');caption.write_text((old/caption.name).read_text())
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads((old/p.name).read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v12: cartoons uniformly enlarged 7%; Inter-HVA/Intra-V1 captions moved closer beneath them; Anatomical score label shifted left toward the cortical cartoon.', 'cartoon_scale_vs_v11':1.07})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v12 — cartoon finishing

Cartoons enlarged uniformly by another 7%, keeping the top fixed. Inter-HVA/Intra-V1 labels moved closer below the cartoons. Anatomical score label moved left toward A’s cartoon. Other content and all numerical results unchanged.

Rebuild: `python scripts/build_figure5_cartoon_finish.py`.
''')

if __name__=='__main__':main()
