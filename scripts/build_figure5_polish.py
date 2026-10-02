#!/usr/bin/env python3
"""Figure 5 v7: remaining typography, annotation and distribution refinements."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_clean_labels as previous
from scripts import build_figure5_original_vectors as original
from matplotlib.collections import PolyCollection
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v7_visual_polish'
old_style=previous.style


def style(fig):
    previous.OUT=OUT
    old_style(fig)
    for t in fig.texts:
        if t.get_text()=='Visual areas hierarchy':t.set_text('Anatomical hierarchy')
    for t in fig.axes[3].texts:
        if 'SD of means' in t.get_text():
            x,y=t.get_position();t.set_position((x+.23,y))
    for ax in fig.axes[4:]:
        for c in ax.collections:
            if isinstance(c,PolyCollection):
                faces=c.get_facecolors().copy()
                if len(faces) and np.ptp(faces[0,:3])>.08:
                    edges=faces.copy();edges[:,3]=.65
                    c.set_alpha(None);c.set_facecolors(faces);c.set_edgecolors(edges);c.set_linewidth(.8)
    for ax in fig.axes:
        if not ax.axison:continue
        labels=ax.get_xticklabels()
        if any(t.get_rotation()==90 for t in labels):
            for t in labels:
                t.set_rotation(55);t.set_ha('right');t.set_va('top');t.set_rotation_mode('anchor')
            ax.tick_params(axis='x',pad=5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(renderer)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside},indent=2)+'\n')


def main():
    original.OUT=OUT;original.style=style;original.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    old=ROOT/'artifacts/figure5/v6_clean_labels'
    # Inherit the approved caption with its explicit SD-label terminology.
    caption=stem.with_name(stem.name+'_legend.md')
    caption.write_text((old/caption.name).read_text())
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads(p.read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)),'description':'Figure 5 v7: Anatomical hierarchy title, additional spacing before B SD annotations, stronger colored violin outlines, 55-degree E labels. Hierarchy labels and pooled SD labels retained.', 'artboard_points':[852.648,709.0], 'visual_refinements':{'E_label_rotation_deg':55,'violin_edge_alpha':.65,'violin_edge_width_pt':.8,'fill_and_scatter':'unchanged'}})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v7 — visual polish

Renamed A to Anatomical hierarchy. Added spacing between the schematic SD bars and their labels in B. Strengthened colored violin outlines without changing their fills or cell dots. E labels use a 55° angle instead of vertical text. Retains the approved hierarchy label positions and explicit pooled Cells SD / Means SD labels. All numerical data are unchanged.

Rebuild: `python scripts/build_figure5_polish.py`.
''')

if __name__=='__main__':main()
