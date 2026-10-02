#!/usr/bin/env python3
"""Figure 5 v8: larger fitted-mean diamonds with clearer white keylines."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_polish as previous
from scripts import build_figure5_original_vectors as original
from matplotlib.lines import Line2D
OUT=ROOT/'artifacts/figure5/v8_clear_diamonds'
old_style=previous.style


def style(fig):
    previous.OUT=OUT
    old_style(fig)
    count=0
    for line in fig.findobj(Line2D):
        if line.get_marker() in ['D','d']:
            line.set_markersize(line.get_markersize()*1.45)
            line.set_markeredgecolor('white')
            line.set_markeredgewidth(.85)
            line.set_zorder(max(line.get_zorder(),9))
            count+=1
    assert count==54, count
    fig.canvas.draw()


def main():
    original.OUT=OUT;original.style=style;original.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    old=ROOT/'artifacts/figure5/v7_visual_polish'
    caption=stem.with_name(stem.name+'_legend.md');caption.write_text((old/caption.name).read_text())
    p=stem.with_name(stem.name+'_layout_manifest.json');m=json.loads((old/p.name).read_text())
    m.update({'renderer':str(Path(__file__).relative_to(ROOT)), 'description':'Figure 5 v8: fitted-mean diamonds 45% larger in linear size, with 0.85 pt white outlines and raised drawing order.', 'mean_markers':{'number':54,'linear_size_multiplier_vs_v7':1.45,'edge_color':'white','edge_width_pt':.85}})
    p.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v8 — clearer mean diamonds

All 54 fitted-mean diamonds (27 individual-category and 27 repeated pooled markers) are 45% larger in linear size, with 0.85 pt white outlines and raised drawing order. Other styling and all numerical results are unchanged.

Rebuild: `python scripts/build_figure5_clear_diamonds.py`.
''')

if __name__=='__main__':main()
