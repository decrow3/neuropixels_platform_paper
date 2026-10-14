#!/usr/bin/env python3
"""Rebuild anatomical inputs from the historical electrode and unit-link snapshot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROBE_LABELS = {'ProbeA': 'Anterior', 'ProbeE': 'Lateral', 'ProbeC': 'Posterior', 'ProbeB': 'Medial'}


def read_snapshot(nwb_root):
    from nwb_io import read_column
    import h5py
    electrodes, links = [], []
    raw = json.loads((HERE / 'raw_sources.json').read_text())
    for record in raw['anatomy']['sessions']:
        path = nwb_root / record['path']
        if path.stat().st_size != record['bytes']:
            raise ValueError(f'Historical anatomy size mismatch: {path}')
        with h5py.File(path, 'r') as file:
            stem = '/general/extracellular_ephys/electrodes/'
            table = pd.DataFrame({name: read_column(file, stem + name)
                                  for name in ('group_name', 'channel_name', 'location', 'rel_y', 'x', 'y', 'z')})
            table = table.loc[table.channel_name.str.startswith('AP')].rename(columns={'group_name': 'probe'})
            table['extremum_channel_index'] = table.channel_name.str.removeprefix('AP').astype(int) - 1
            units = pd.DataFrame({name: read_column(file, '/units/' + name)
                                  for name in ('id', 'device_name', 'extremum_channel_index')})
            units = units.rename(columns={'device_name': 'probe'})
            units['unit_id'] = units.id.astype(int) + record['id_offset']
            units = units.drop(columns='id')
            for frame in (table, units):
                frame['subject_id'] = record['subject_id']
                frame['site_number'] = record['site_number']
                frame['site'] = record['site']
            electrodes.append(table); links.append(units)
    return pd.concat(electrodes, ignore_index=True), pd.concat(links, ignore_index=True)


def unit_locations(electrodes, links):
    keys = ['subject_id', 'site_number', 'site', 'probe', 'extremum_channel_index']
    merged = links.merge(electrodes[keys + ['location']], on=keys, how='left', validate='many_to_one')
    if merged.location.isna().any() or merged.unit_id.duplicated().any():
        raise ValueError('Invalid unit-to-electrode mapping')
    return merged[['unit_id', 'site_number', 'subject_id', 'probe', 'location']]


def derive_entry_points(points: pd.DataFrame, atlas: np.ndarray,
                        labels: dict[int, str]) -> pd.DataFrame:
    """Use the most superficial VISp-labelled AP contact as the entry proxy."""
    visp = points.loc[points["location"].astype(str).str.startswith("VISp")].copy()
    # Freeze the original tie choices rather than depending on NumPy quicksort.
    choices = pd.DataFrame(json.loads((HERE / 'raw_sources.json').read_text())['surface_anatomy']['tie_breaks'])
    entries = visp.merge(choices, on=['subject_id', 'probe', 'channel_name'], validate='one_to_one')
    maxima = visp.groupby(['subject_id', 'probe']).rel_y.max()
    observed = entries.set_index(['subject_id', 'probe']).rel_y
    if not observed.eq(maxima.reindex(observed.index)).all():
        raise ValueError('Frozen entry contact is no longer the most superficial VISp contact')
    if len(entries) != 32 or entries.groupby("subject_id")["probe"].nunique().ne(4).any():
        raise ValueError(f"Expected 32 entries (8 sessions x 4 probes), found {len(entries)}")
    entries["ccf_ap_um"] = entries["x"].astype(float)
    entries["ccf_dv_um"] = entries["y"].astype(float)
    entries["ccf_ml_um"] = entries["z"].astype(float)
    entries["surface_ap_voxel"] = np.floor(entries["ccf_ap_um"] / 10).astype(int)
    entries["surface_ml_voxel"] = np.floor(entries["ccf_ml_um"] / 10).astype(int)
    entries["surface_atlas_index"] = [
        int(atlas[ml, ap])
        for ml, ap in zip(entries.surface_ml_voxel, entries.surface_ap_voxel)
    ]
    entries["surface_atlas_acronym"] = entries.surface_atlas_index.map(labels).fillna("void")
    entries["surface_agrees_visp"] = entries.surface_atlas_acronym.eq("VISp")
    entries["probe_display"] = entries.probe.map(PROBE_LABELS)
    return entries[
        ["subject_id", "site", "probe", "probe_display", "channel_name",
         "rel_y", "location", "ccf_ap_um", "ccf_dv_um", "ccf_ml_um",
         "surface_ap_voxel", "surface_ml_voxel", "surface_atlas_index",
         "surface_atlas_acronym", "surface_agrees_visp"]
    ].sort_values(["subject_id", "probe"])


def regenerate(output, *, nwb_root=None):
    # Imported only here so NWB metadata extraction does not require Matplotlib.
    from schematics import read_nrrd, read_labels, ATLAS_PATH, ATLAS_LABELS_PATH
    if nwb_root:
        electrodes, links = read_snapshot(nwb_root)
    else:
        electrodes = pd.read_csv(HERE / 'sources/anatomy/electrodes.csv.gz')
        links = pd.read_csv(HERE / 'sources/anatomy/unit_electrodes.csv.gz')
    locations = unit_locations(electrodes, links)
    surface = pd.read_csv(HERE / 'sources/anatomy/surface_electrodes.csv.gz')
    entries = derive_entry_points(surface, read_nrrd(ATLAS_PATH), read_labels(ATLAS_LABELS_PATH))
    output.mkdir(parents=True, exist_ok=True)
    locations.to_csv(output / 'mousev2_analysis_units.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    entries.to_csv(output / 'mousev2_ccf_surface_entry_points.csv', index=False)
    for name, keys in [('mousev2_analysis_units.csv.gz', ['unit_id']),
                       ('mousev2_ccf_surface_entry_points.csv', ['subject_id', 'probe'])]:
        expected = pd.read_csv(HERE / 'sources' / name).sort_values(keys).reset_index(drop=True)
        actual = pd.read_csv(output / name).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        print(f'MATCH anatomy: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'work/anatomy')
    parser.add_argument('--nwb-root', type=Path, help='Optional exact historical anatomy NWBs, instead of bundled scalar tables')
    args = parser.parse_args()
    if args.output_dir.resolve() == (HERE / 'sources').resolve():
        raise ValueError('Refusing to overwrite canonical sources')
    regenerate(args.output_dir.resolve(), nwb_root=args.nwb_root)


if __name__ == '__main__':
    main()
