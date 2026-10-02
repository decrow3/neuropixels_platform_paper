#!/usr/bin/env python3
"""Figure 5 v11: raise A/B schematics relative to the cortical cartoons."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_larger_ab as previous
from scripts import build_figure5_original_vectors as original
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v11_raised_schematics'
old_style=previous.style


def style(fig):
    previous.OUT=OUT
    old_style(fig)
    geometry={}
    for i in (1,3):
        ax=fig.axes[i];p=ax.get_position().frozen()
        ax.set_position([p.x0,p.y0+10/709,p.width,p.height])
        geometry[str(i)]=list(ax.get_position().bounds)
    fig.canvas.draw();r=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(r)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'raised_schematic_layout.json').write_text(json.dumps({'axes':geometry,'vertical_shift_pt':10,'outside_artboard':outside},indent=2)+'\n')


def main():
    original.OUT=OUT;original.style=style;original.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads(p.read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v11: hierarchy and functional-metric schematics shifted upward 10 pt; cortical cartoons and C–E unchanged.', 'raised_schematics':json.loads((OUT/'raised_schematic_layout.json').read_text())})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v11 — raised schematics

Both the anatomical-hierarchy and functional-metric schematics move upward by 10 pt relative to the unchanged cortical cartoons. Their mutual reference-level alignment is retained. All other layout and numerical results are unchanged.

Rebuild: `python scripts/build_figure5_raise_schematics.py`.
''')

if __name__=='__main__':main()
