#!/usr/bin/env python3
"""Use the actual handmade SVG paths in Figure 5, with intra/inter terminology."""
from pathlib import Path
import sys, copy, json, hashlib, re
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_figure5_cartoons as previous
from scripts import build_figure4_integrated as base
from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path
from matplotlib.path import Path as MP
from matplotlib.patches import PathPatch, Circle
from matplotlib.text import Text

OUT=ROOT/'artifacts/figure5/v3_original_vectors'
SOURCE=Path('/home/huklaban5/.codex/attachments/48d9e94f-7e31-4e0e-b77f-634b6ed3cee5/Artboard 11 copy.svg')
S='{http://www.w3.org/2000/svg}'
root=ET.parse(SOURCE).getroot()
# These source groups contain the handmade outlines, not the older data panels.
inter=root[331]; intra=root[333]
assert len(inter.findall('.//'+S+'path'))==6
assert len(intra.findall('.//'+S+'path'))==6
inter_dots=[root[i] for i in range(335,340)]
assert all(e.tag==S+'circle' for e in inter_dots)

class ExactPathPen(BasePen):
    def __init__(self): super().__init__(None); self.vertices=[]; self.codes=[]; self.start=None
    def _moveTo(self,p): self.start=p; self.vertices.append(p); self.codes.append(MP.MOVETO)
    def _lineTo(self,p): self.vertices.append(p); self.codes.append(MP.LINETO)
    def _curveToOne(self,a,b,c): self.vertices.extend([a,b,c]); self.codes.extend([MP.CURVE4]*3)
    def _qCurveToOne(self,a,b): self.vertices.extend([a,b]); self.codes.extend([MP.CURVE3]*2)
    def _closePath(self): self.vertices.append(self.start); self.codes.append(MP.CLOSEPOLY)
    def _endPath(self): pass


def draw_original(ax, within):
    group=intra if within else inter
    ax.clear(); ax.set_aspect('equal'); ax.axis('off')
    # Equal coordinate spans retain the relative scale of the two original drawings.
    ax.set_xlim((518,665) if within else (391,538)); ax.set_ylim(146,14)
    for e in group.findall('.//'+S+'path'):
        assert not e.get('transform'), 'Unexpected transform in source art'
        pen=ExactPathPen(); parse_path(e.attrib['d'],pen)
        # Filled outline paths: preserve every original Bezier, no smoothing or tracing.
        ax.add_patch(PathPatch(MP(pen.vertices,pen.codes),facecolor='#231f20',edgecolor='none'))
    dots=group.findall('.//'+S+'circle') if within else inter_dots
    cats=['A','E','B','C'] if within else ['LM','AL','RL','AM','PM']
    palette=base.results.V1_COLORS if within else base.results.HVA_COLORS
    for e,cat in zip(dots,cats):
        ax.add_patch(Circle((float(e.get('cx')),float(e.get('cy'))),float(e.get('r')),
                           facecolor=palette[cat],edgecolor='none'))
    for e in group.findall('.//'+S+'text'):
        label=''.join(e.itertext())
        x,y=map(float,re.search(r'translate\(([^)]+)\)',e.get('transform'))[1].split())
        ax.text(x,y,label,fontsize=11 if label=='V1' else 10,
                family='Liberation Sans',ha='left',va='baseline',
                color='#231f20' if label=='V1' else palette[label])
    ax.text(591.5 if within else 464.5,145,'Intra-V1' if within else 'Inter-HVA',
            fontsize=12,family='Liberation Sans',ha='center',va='top')

old_style=previous.style

def style(fig):
    old_style(fig)
    draw_original(fig.axes[0],False);draw_original(fig.axes[2],True)
    for t in fig.findobj(Text):
        label=t.get_text()
        label=label.replace('Within-area benchmark','Intra-V1 benchmark')
        label=label.replace('Across HVAs','Inter-HVA').replace('Within V1','Intra-V1').replace('within V1','intra-V1')
        label=label.replace('Within\nV1','Intra-V1').replace('Across\nHVAs','Inter-HVA')
        t.set_text(label)
    for ax in (fig.axes[8],fig.axes[13],fig.axes[18]):
        ax.set_xticks([0,1.2,2.1,3.5], ['HVAs\n− V1','Intra-V1','Inter-HVA','Inter-HVA\n− intra-V1'])
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        b=t.get_window_extent(renderer)
        if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
    assert not outside,outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside},indent=2)+'\n')


def main():
    previous.OUT=OUT;previous.style=style;previous.main()
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    caption=stem.with_name(stem.name+'_legend.md')
    text=caption.read_text().replace('within V1 and across higher visual areas','Intra-V1 and inter-HVA functional variation')
    # Correct the title separately, while keeping descriptive prose readable.
    text=text.replace('Functional variation Intra-V1 and inter-HVA functional variation','Intra-V1 and inter-HVA functional variation')
    text=text.replace('The paired cartoons in A and B use the same simplified outlines:',
        'The cartoons in A and B directly reuse the handmade vector outlines and marker coordinates from “Artboard 11 copy.svg”, with category colors and typography adapted to this figure:')
    text=text.replace('B uses the same cortical cartoon as A, retaining area outlines while placing all four colored markers within V1.',
        'B uses the original intra-V1 cartoon, retaining its handmade area outlines and four marker positions within V1.')
    text=text.replace('labelled Across HVAs and Within V1','labelled Inter-HVA and Intra-V1')
    text=text.replace('labelled “Across HVAs − within V1”','labelled “Inter-HVA − intra-V1”')
    text=text.replace('**A–B, anatomy and schematic.**',
        '**Terminology.** “Intra-V1” means variation among sampled locations within V1; “Inter-HVA” means variation among the five named higher visual areas. These labels describe the comparison, not neuronal connectivity. “HVAs − V1” remains the separate matched Allen mean difference.\n\n**A–B, anatomy and schematic.**')
    caption.write_text(text)
    # Keep an editable extracted SVG containing the untouched source elements/styles.
    ET.register_namespace('',S[1:-1])
    asset=ET.Element(S+'svg',{'width':'280','height':'125','viewBox':'390 15 280 125'})
    asset.append(copy.deepcopy(root[0]))
    for element in [inter,intra,*inter_dots]:asset.append(copy.deepcopy(element))
    ET.ElementTree(asset).write(OUT/'Original_handmade_cortical_vectors.svg',encoding='utf-8',xml_declaration=True)
    manifest_path=stem.with_name(stem.name+'_layout_manifest.json')
    m=json.loads(manifest_path.read_text())
    m['renderer']=str(Path(__file__).relative_to(ROOT))
    m['description']='Figure 5 v3: actual source SVG handmade vectors, with intra-V1/inter-HVA terminology. C–E data unchanged.'
    m['cartoons']={'source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'source_root_child_indices':[331,333,335,336,337,338,339],
      'geometry':'All 12 filled outline paths copied via exact SVG Bezier parsing; all 9 original circle coordinates/radii preserved.',
      'adaptations':['uniform layout scale/translation','current category colors','Liberation Sans labels'],'empirical_coordinates':False}
    manifest_path.write_text(json.dumps(m,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v3 — original handmade vectors

Directly copies the cortical art from Artboard 11 copy.svg: all 12 outline paths and all 9 marker positions/radii are preserved. Curves are parsed exactly, not approximated or redrawn. Original editable SVG elements are also provided in Original_handmade_cortical_vectors.svg. Figure colors and typography are applied consistently with the rest of Figure 5.

Intra-V1 means among sampled locations within V1; Inter-HVA means among the five named HVAs. The matched Allen mean contrast remains HVAs − V1. All plotted values, intervals and sample sizes are unchanged. Ridgelines remain deferred.

Rebuild: `python scripts/build_figure5_original_vectors.py`.
''')

if __name__=='__main__':main()
