#!/usr/bin/env python3
"""Figure 5 v13: move the entire anatomical hierarchy toward the cortical cartoon."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_cartoon_finish as previous
from scripts import build_figure5_original_vectors as original
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v13_shift_hierarchy'
old_style=previous.style


def style(fig):
    previous.OUT=OUT
    old_style(fig)
    # Undo v12's label-only offset, then translate the complete diagram.
    ax=fig.axes[1]
    for t in ax.texts:
        if t.get_text()=='Anatomical score':
            x,y=t.get_position();t.set_position((x+.30,y))
    p=ax.get_position().frozen()
    ax.set_position([p.x0-18/852.648,p.y0,p.width,p.height])
    geometry={'A_hierarchy':list(ax.get_position().bounds)}
    fig.canvas.draw();r=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(r)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'raised_schematic_layout.json').write_text(json.dumps({'axes':geometry,'horizontal_shift_pt':-18,'outside_artboard':outside},indent=2)+'\n')


def main():
    original.OUT=OUT;original.style=style;original.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads(p.read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v13: complete anatomical hierarchy moved 18 pt left; label-only offset from v12 reversed.', 'raised_schematics':json.loads((OUT/'raised_schematic_layout.json').read_text())})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v13 — shifted hierarchy

The entire anatomical hierarchy (nodes, curves, axis and label) moves 18 pt left toward the cartoon. The previous label-only offset is reversed. Other layout and all numerical results are unchanged.

Rebuild: `python scripts/build_figure5_shift_hierarchy.py`.
''')
    canonical=ROOT/'artifacts/figure5/captions/Figure5_caption.md'
    caption=stem.with_name(stem.name+'_legend.md')
    technical=stem.with_name(stem.name+'_technical_legend.md')
    if not technical.exists():
        technical.write_text(caption.read_text())
    caption.write_text(canonical.read_text())


if __name__=='__main__':main()
