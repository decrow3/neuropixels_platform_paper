#!/usr/bin/env python3
"""Figure 5 v4: taller C/D/E with top schematics at their existing physical size."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_original_vectors as original
from scripts import build_figure5_cartoons as cartoons
from scripts import build_figure5_revision as revision
from matplotlib.text import Text

OUT=ROOT/'artifacts/figure5/v4_taller_results'
OLD_HEIGHT=574.601
NEW_HEIGHT=650.0
DELTA=NEW_HEIGHT-OLD_HEIGHT
base_style=original.style


def taller_style(fig):
    # Recreate the accepted layout at its original size, then extend the three
    # result rows downward. This preserves fonts, map aspect and top alignment.
    fig.set_size_inches(852.648/72,OLD_HEIGHT/72)
    base_style(fig)
    positions=[ax.get_position().frozen() for ax in fig.axes]
    text_positions=[(t,t.get_position()) for t in fig.texts]
    fig.set_size_inches(852.648/72,NEW_HEIGHT/72)
    for i,(ax,pos) in enumerate(zip(fig.axes,positions)):
        if i<4:
            y=(pos.y0*OLD_HEIGHT+DELTA)/NEW_HEIGHT
            height=pos.height*OLD_HEIGHT/NEW_HEIGHT
        else:
            row=(i-4)//5
            # Each result row gains one third of the added artboard height.
            y=(pos.y0*OLD_HEIGHT+DELTA-(row+1)*DELTA/3)/NEW_HEIGHT
            height=(pos.height*OLD_HEIGHT+DELTA/3)/NEW_HEIGHT
        ax.set_position([pos.x0,y,pos.width,height])
    for t,(x,y) in text_positions:
        t.set_position((x,(y*OLD_HEIGHT+DELTA)/NEW_HEIGHT))
    for ax in (fig.axes[8],fig.axes[13],fig.axes[18]):
        ax.set_xticks([0,1.2,2.1,3.5], ['HVAs\n− V1','Intra-\nV1','Inter-\nHVA','Inter-HVA\n− intra-V1'])
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(renderer)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside},indent=2)+'\n')


def main():
    original.OUT=OUT
    original.style=taller_style
    original.main()
    manifest_path=OUT/'Figure5_CDE_rerun_with_schematic_layout_manifest.json'
    m=json.loads(manifest_path.read_text())
    m['renderer']=str(Path(__file__).relative_to(ROOT))
    m['description']='Figure 5 v4: each C/D/E result row gains 25.133 pt height; top schematics retain v3 physical size.'
    m['artboard_points']=[852.648,NEW_HEIGHT]
    for key,rect in m['schematic_layout'].items():
        x,y,w,h=rect
        m['schematic_layout'][key]=[x,(y*OLD_HEIGHT+DELTA)/NEW_HEIGHT,w,h*OLD_HEIGHT/NEW_HEIGHT]
    m['vertical_layout']={'previous_height_pt':OLD_HEIGHT,'height_pt':NEW_HEIGHT,'added_height_per_result_row_pt':DELTA/3,'top_schematic_physical_size':'unchanged','row_gaps_pt':'unchanged'}
    manifest_path.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v4 — taller result panels

Artboard: 852.648 × 650 pt, increased from 574.601 pt high. Each C/D/E result row gains 25.133 pt (about 27% more plot height). Top schematics retain their prior physical dimensions and distance from the top. Row gaps, font sizes, colors and all data are unchanged.

Rebuild: `python scripts/build_figure5_taller.py`.
''')

if __name__=='__main__':main()
