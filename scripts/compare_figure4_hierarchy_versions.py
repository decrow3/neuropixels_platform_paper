#!/usr/bin/env python3
"""Render a common-scale comparison; full figures use build_figure4_integrated.py."""
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_hierarchy_schematic_row as schematic
from scripts import build_figure4_integrated as integrated

OUT = ROOT / 'artifacts/figure4_rerun/hierarchy_comparison_v2'
VERSIONS = {'harris2019': 'Harris 2019 · visual/medial', 'siegle2021': 'Siegle 2021 · recomputed'}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    schematic.AREA_COLORS.update(integrated.results.HVA_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    validations = {}
    for ax, (version, title) in zip(axes, VERSIONS.items()):
        schematic.draw_published_hierarchy(ax, version)
        ax.set_aspect('auto')
        ax.set_title(title, fontsize=13, pad=18)
        scores = schematic.hierarchy_scores(version)
        edges = [p for p in ax.patches if (p.get_gid() or '').startswith('anatomical-connection-')]
        assert len(edges) == 30
        for edge in edges:
            _, _, source, target = edge.get_gid().split('-')
            vertices = edge.get_path().vertices
            assert np.allclose(vertices[[0, 3], 1], [scores[source], scores[target]])
            side = -1 if scores[target] > scores[source] else 1
            assert np.all(side * (vertices[1:3, 0] - .1) > 0)
        validations[version] = {'directed_edges': len(edges), 'endpoint_scores_and_arc_sides': 'pass'}
    fig.subplots_adjust(left=.09, right=.97, bottom=.08, top=.88, wspace=.30)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'hierarchy_comparison.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    scores = pd.DataFrame({v: schematic.hierarchy_scores(v) for v in VERSIONS}).rename_axis('area')
    scores.to_csv(OUT / 'hierarchy_scores.csv')
    canonical = ROOT / 'artifacts/figure4_rerun/v8_final'
    prefix = 'Figure4_CDE_rerun_with_schematic'
    for version in VERSIONS:
        checked = []
        for suffix in ['estimates.csv', 'display.csv', 'mean_spreads.csv', 'probe_positions.csv', 'spread_bootstrap.csv.gz']:
            name = f'{prefix}_{suffix}'
            pd.testing.assert_frame_equal(pd.read_csv(canonical / name), pd.read_csv(OUT / version / name))
            checked.append(suffix)
        validations[version]['unchanged_analysis_tables'] = checked
    pd.testing.assert_frame_equal(*[pd.read_csv(OUT / v / f'{prefix}_anatomical_connections.csv') for v in VERSIONS])
    validations['identical_connection_densities_and_widths'] = True
    (OUT / 'validation.json').write_text(json.dumps(validations, indent=2) + '\n')
    print(scores.round(6).to_string())
    print('All checks passed.')

if __name__ == '__main__':
    main()
