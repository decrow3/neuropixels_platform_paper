#!/usr/bin/env python3
"""Figure 5 v2: paired conceptual cortical cartoons; retain C–E unchanged."""
from pathlib import Path
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_figure5_revision as revision
from scripts import build_figure4_integrated as base
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

OUT = ROOT / 'artifacts/figure5/v2_cortical_cartoons'
# Deliberately simplified spatial cartoons, not fitted or atlas coordinates.
SHAPES = {
    'V1': [(44,66),(53,62),(64,45),(72,22),(65,15),(45,13),(29,16),(31,35),(37,54)],
    'LM': [(25,44),(29,37),(25,20),(15,25),(14,36),(18,42)],
    'AL': [(27,63),(30,58),(27,48),(17,47),(14,54),(20,61)],
    'RL': [(42,79),(47,76),(40,66),(32,62),(26,67),(30,74)],
    'AM': [(58,88),(64,83),(64,74),(55,71),(48,76),(51,83)],
    'PM': [(68,72),(77,67),(79,55),(73,48),(67,53),(61,65)],
}
CENTERS = {'LM':(21,32),'AL':(22,55),'RL':(36,71),'AM':(57,80),'PM':(72,61)}
LABELS = {'LM':(6,30),'AL':(7,54),'RL':(24,84),'AM':(70,89),'PM':(88,62)}
PROBES = {'A':(45,56),'E':(35,37),'C':(47,23),'B':(62,35)}


def smooth_outline(points):
    points = np.asarray(points, float)
    vertices = [points[0]]; codes = [MplPath.MOVETO]
    for i in range(len(points)):
        a,b,c,d = [points[j % len(points)] for j in (i-1,i,i+1,i+2)]
        vertices.extend([b+(c-a)/6, c-(d-b)/6, c])
        codes.extend([MplPath.CURVE4]*3)
    vertices.append(points[0]); codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices,codes)


def cartoon(ax, within=False):
    ax.clear()
    ax.set_xlim(0,100); ax.set_ylim(5,100)
    ax.set_aspect('equal'); ax.axis('off')
    for name,points in SHAPES.items():
        ax.add_patch(PathPatch(smooth_outline(points), facecolor='white',
                              edgecolor='#383838', linewidth=1.05, zorder=1))
    ax.text(49,43,'V1', ha='center', va='center', fontsize=12,
            fontfamily='Liberation Sans', color='#202124')
    if within:
        for key,(x,y) in PROBES.items():
            ax.scatter(x,y,s=42,color=base.results.V1_COLORS[key],
                       edgecolor='white',linewidth=.7,zorder=3)
    else:
        for area,(x,y) in CENTERS.items():
            color=base.results.HVA_COLORS[area]
            ax.scatter(x,y,s=42,color=color,edgecolor='white',linewidth=.7,zorder=3)
            lx,ly=LABELS[area]
            ax.text(lx,ly,area,fontsize=11,ha='center',va='center',
                    fontfamily='Liberation Sans',color=color)
    ax.text(49,3,'Within V1' if within else 'Across HVAs',fontsize=12,
            fontfamily='Liberation Sans',ha='center',va='top',color='#202124')


def style(fig):
    # Apply the existing publication style before replacing just the maps.
    revision.OUT = OUT
    revision.revision_style(fig)
    fig.axes[0].set_position([.055,.725,.195,.22])
    fig.axes[2].set_position([.535,.725,.185,.22])
    cartoon(fig.axes[0])
    cartoon(fig.axes[2],within=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    from matplotlib.text import Text
    outside=[]
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text(): continue
        if text.axes is not None and not text.axes.axison and text not in text.axes.texts: continue
        box=text.get_window_extent(renderer)
        if box.width and box.height and (box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height): outside.append(text.get_text())
    assert not outside, outside
    (OUT/'text_bounds_audit.json').write_text(json.dumps({'outside_artboard':outside},indent=2)+'\n')


def main():
    base.main(OUT,'harris2019',figure_number=5,artboard_points=revision.ARTBOARD,style_callback=style)
    stem=OUT/'Figure5_CDE_rerun_with_schematic'
    caption=stem.with_name(stem.name+'_legend.md')
    text=caption.read_text()
    text=text.replace('A combines the Allen CCFv3 dorsal visual-area map with the anatomical-hierarchy schematic.',
        'A combines a conceptual cortical cartoon with the anatomical-hierarchy schematic. The paired cartoons in A and B use the same simplified outlines: colored markers occupy five named HVAs in A and four locations within V1 in B. Cartoon boundaries and marker positions are illustrative, not atlas contours or measured recording coordinates. Figure 4 documents the actual MouseV2 electrophysiology and optical-physiology sampling and collection methods.')
    text=text.replace('B uses the same atlas geometry with non-V1 masks merged.','B uses the same cortical cartoon as A, retaining area outlines while placing all four colored markers within V1.')
    text=text.replace('Probe markers show the mean AP/ML coordinates across eight animals of the most superficial VISp-labelled contact on each probe.','The colored within-V1 markers indicate sampling categories only; they do not encode probe entry points, mean contact projections, or the extent of the recorded tissue.')
    caption.write_text(text)
    manifest_path=stem.with_name(stem.name+'_layout_manifest.json')
    manifest=json.loads(manifest_path.read_text())
    manifest['renderer']=str(Path(__file__).relative_to(ROOT))
    manifest['description']='Figure 5 v2: paired conceptual cortical cartoons replace both atlas maps; Harris hierarchy and panels C–E retained from v1.'
    manifest['probe_location_overlay']={'shown':False,'note':'Legacy projection table exported by the shared renderer for provenance only; coordinates do not enter the cartoons.'}
    manifest['schematic_layout'].update({'A_map':[.055,.725,.195,.22],'B_map':[.535,.725,.185,.22]})
    manifest['cartoons']={'geometry':'Hand-authored schematic outlines, identical across A and B','empirical_coordinates':False,'sampling_methods_reference':'Figure 4','ridge_lines':'Deferred; current distributions retained'}
    manifest['numerical_validation']='All estimates, CIs, displayed means, counts, pooled SDs and directed anatomical connections identical to Figure 5 v1.'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Figure 5 v2 — paired cortical cartoons

A and B now use matched conceptual cortical cartoons with colored markers across five HVAs versus within V1. The Harris hierarchy and functional-metric schematic remain. Panels C–E, category palettes, estimates and confidence intervals are unchanged. Ridgelines are deferred.

Artboard: 852.648 × 574.601 pt. Caption points to Figure 4 for actual sampling and data collection. Cartoon geometry is illustrative, not atlas-derived or measured. The legacy probe-position CSV is provenance only and is not plotted.

Rebuild: `python scripts/build_figure5_cartoons.py`.
''')

if __name__=='__main__': main()
