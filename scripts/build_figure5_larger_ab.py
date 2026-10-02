#!/usr/bin/env python3
"""Figure 5 v10: enlarge A/B art and move B left for annotation space."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_art_spacing as previous
from scripts import build_figure5_original_vectors as original
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v10_larger_AB'
old_style=previous.style


def style(fig):
    previous.OUT=OUT
    old_style(fig)
    # Preserve the aligned anatomical/functional reference heights: both
    # schematic axes receive the same vertical expansion and center shift.
    # Expand downward into the available gap, keeping the top edge fixed.
    geometry={}
    for i in range(4):
        ax=fig.axes[i];p=ax.get_position().frozen()
        scale=1.13 if i in (0,2) else 1.18
        height=p.height*scale
        width=p.width*(scale if i in (0,2) else 1.08)
        shift={0:0,1:-4,2:-22,3:-27}[i]/852.648
        x=p.x0-(width-p.width)/2+shift
        y=p.y1-height
        ax.set_position([x,y,width,height])
        geometry[str(i)]=[x,y,width,height]
    # Match B's header to its shifted content.
    for t in fig.texts:
        if t.get_text() in ['B','Intra-V1 benchmark']:
            x,y=t.get_position();t.set_position((x-22/852.648,y))
    fig.canvas.draw();r=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(r)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'AB_layout.json').write_text(json.dumps({'axes':geometry,'outside_artboard':outside},indent=2)+'\n')


def main():
    original.OUT=OUT;original.style=style;original.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads(p.read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v10: cartoons enlarged 13%, schematic axes 18% taller and 8% wider; B shifted left for SD annotations.', 'AB_layout':json.loads((OUT/'AB_layout.json').read_text())})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v10 — larger A/B

Both cartoons are uniformly enlarged another 13%. Both quantitative/conceptual schematics are 18% taller and 8% wider, retaining aligned reference levels. B is shifted left (22 pt cartoon/header; 27 pt functional schematic) to reserve right-edge space for SD labels. C–E and numerical results are unchanged.

Rebuild: `python scripts/build_figure5_larger_ab.py`.
''')

if __name__=='__main__':main()
