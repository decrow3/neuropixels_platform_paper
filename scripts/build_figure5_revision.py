#!/usr/bin/env python3
"""Render current Figure 5 analysis on the eLife revision artboard; no refitting."""
from pathlib import Path
import sys
import json
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_figure4_integrated as base
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, LineCollection

OUT = ROOT / 'artifacts/figure5/v1_elife_artboard'
ARTBOARD = (852.648, 574.601)
REFERENCE = Path('/home/huklaban5/.codex/attachments/c44dff5a-3074-4761-b7e5-e44c86c238b9/2026_elife_revision (1).pdf')


def revision_style(fig):
    """Match the reference's regular sans typography and light plotting strokes."""
    for text in fig.findobj(Text):
        text.set_fontfamily('Liberation Sans')
        text.set_fontweight('normal')
        if text.get_gid() == 'panel-letter':
            text.set_fontsize(32)
        elif text.get_gid() == 'panel-heading':
            text.set_fontsize(16)
        else:
            text.set_fontsize(text.get_fontsize() * 1.55)
    for line in fig.findobj(Line2D):
        line.set_linewidth(line.get_linewidth() * 1.4)
        line.set_markersize(line.get_markersize() * 1.4)
        line.set_markeredgewidth(line.get_markeredgewidth() * 1.4)
    for collection in fig.findobj(PathCollection):
        collection.set_sizes(collection.get_sizes() * 1.4**2)
        collection.set_linewidths(collection.get_linewidths() * 1.4)
    for collection in fig.findobj(LineCollection):
        collection.set_linewidths(collection.get_linewidths() * 1.4)
    for ax in fig.axes:
        ax.tick_params(pad=2.5, length=3, width=.7)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
        ax.set_title(ax.get_title(), fontsize=14, pad=7, fontfamily='Liberation Sans')
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
    # Keep a reproducible audit of visible text bounds for this larger layout.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        if text.axes is not None and not text.axes.axison:
            # Texts explicitly added to schematic axes still render with axis off.
            if text not in text.axes.texts:
                continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height):
            outside.append(text.get_text())
    (OUT / 'text_bounds_audit.json').write_text(json.dumps({'outside_artboard': outside}, indent=2)+'\n')


def main():
    base.main(OUT, 'harris2019', figure_number=5,
              artboard_points=ARTBOARD, style_callback=revision_style)
    prefix = 'Figure5_CDE_rerun_with_schematic'
    old = ROOT / 'artifacts/figure4_rerun/v22_compact_E'
    checks = {}
    for suffix in ['estimates', 'display', 'mean_spreads', 'anatomical_connections']:
        original = pd.read_csv(old / f'Figure4_CDE_rerun_with_schematic_{suffix}.csv')
        current = pd.read_csv(OUT / f'{prefix}_{suffix}.csv')
        pd.testing.assert_frame_equal(original, current)
        checks[suffix] = 'identical to Figure 4 v22'
    manifest_path = OUT / f'{prefix}_layout_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        'figure_number': 5,
        'renderer': str(Path(__file__).relative_to(ROOT)),
        'style': {'reference': str(REFERENCE), 'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                  'font': 'Liberation Sans (Arial-compatible substitute)', 'panel_letters_pt': 32,
                  'panel_headings_pt': 16, 'plot_titles_pt': 14, 'axes_linewidth_pt': .7},
        'numerical_checks': checks,
        'description': 'Figure 5, large eLife revision artboard; OpenScope explainer reserved as Figure 4. Existing category colors and all current analysis results retained.'})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    (OUT / 'README.md').write_text('''# Figure 5 — eLife revision, version 1

Figure 5 now follows the separate OpenScope explainer (Figure 4).

- Artboard: 852.648 × 574.601 pt, matching the supplied revision PDF.
- Current A–E composition expanded across the artboard, without stretching the atlas geometry.
- Regular sans-serif typography, 32 pt panel letters, 14–16 pt headings, and thin axes loosely follow the reference figures.
- Liberation Sans substitutes for the reference’s Arial/Myriad fonts, which are not installed locally.
- Category colors and all estimates, intervals, displayed means and cell counts match Figure 4 v22.
- The reference PDF’s older statistical panels are not used as a numerical source.

PDF and SVG are vector exports (dense neuron dots remain rasterized); PNG is a 400 dpi preview. SVG text remains editable. The full caption and exact numerical tables accompany the exports.

Rebuild: `python scripts/build_figure5_revision.py`.
''')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
