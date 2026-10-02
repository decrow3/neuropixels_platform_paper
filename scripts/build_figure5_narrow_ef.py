#!/usr/bin/env python3
"""Figure 5 v15: halve E/F plotting widths, retaining all numerical results."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import build_figure5_with_variance as prior
from matplotlib.text import Text
OUT=ROOT/'artifacts/figure5/v15_narrow_EF'
WIDTH=852.648
old_style=prior.style

def style(fig):
    old_style(fig)
    axes=list(fig.axes);positions=[a.get_position().frozen() for a in axes]
    texts=[(t,t.get_position()) for t in fig.texts]
    fig.set_size_inches(WIDTH/72,709/72)
    for ax,p in zip(axes,positions):ax.set_position([p.x0*1010/WIDTH,p.y0,p.width*1010/WIDTH,p.height])
    for t,(x,y) in texts:t.set_position((x*1010/WIDTH,y))
    eaxes=axes[-12:-3];faxes=axes[-3:]
    origin=positions[-12].x0*1010
    for i,ax in enumerate(eaxes):
        p=ax.get_position().frozen()
        ax.set_position([(origin+(p.x0*WIDTH-origin)/2)/WIDTH,p.y0,p.width/2,p.height])
        for t in ax.get_xticklabels():t.set_rotation(90);t.set_ha('center');t.set_va('top');t.set_rotation_mode('default')
        if i<3:
            ax.set_title(['Mean\ndifference','SD of\nmeans','SD\ndifference'][i],fontsize=10.85,rotation=90,ha='center',va='bottom',pad=7)
    for ax in faxes:
        p=ax.get_position().frozen();ax.set_position([765/WIDTH,p.y0,p.width/2,p.height])
        for t in ax.get_xticklabels():t.set_rotation(90);t.set_ha('center');t.set_va('top');t.set_rotation_mode('default')
        for t in ax.texts:
            if t.get_text().startswith('p ='):t.set_text('p =\n0.051')
    faxes[0].set_title('Location-\nassociated\nvariance',fontsize=10.85,pad=7)
    next(t for t in fig.texts if t.get_text()=='F').set_x(729/WIDTH)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(renderer)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside,'E_width_ratio':.5,'F_width_ratio':.5},indent=2)+'\n')

def main():
    prior.OUT=OUT;prior.style=style;prior.main()
    p=OUT/'Figure5_CDE_rerun_with_schematic_layout_manifest.json';m=json.loads(p.read_text());m.update(renderer=str(Path(__file__).relative_to(ROOT)),artboard_points=[WIDTH,709],description='Figure 5 v15: E and F half plotting width; vertical labels; all numerical results unchanged.');p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('# Figure 5 v15\n\nE and F plotting widths halved. Vertical labels preserve readability. Artboard 852.648 × 709 pt. All data and statistical annotations unchanged.\n\nRebuild: `python scripts/build_figure5_narrow_ef.py`.\n')
if __name__=='__main__':main()
