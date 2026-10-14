#!/usr/bin/env python3
"""Recompute Figure 5 response metrics from pinned public or original NWB inputs.

Outputs are separate from bundled reference metrics. Released Allen unit
metadata and timescales, and the historical anatomical design, are explicit
source inputs; they are not inferred from a changing DANDI draft.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import urllib.request
import shutil
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AREAS = {'VISp': 'V1', 'VISl': 'LM', 'VISrl': 'RL', 'VISal': 'AL',
         'VISpm': 'PM', 'VISam': 'AM', 'LP': 'LP'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def qc(table):
    isi = 'isi_violations_ratio' if 'isi_violations_ratio' in table else 'isi_violations'
    return (pd.to_numeric(table.amplitude_cutoff, errors='coerce').lt(.1)
            & pd.to_numeric(table.presence_ratio, errors='coerce').gt(.8)
            & pd.to_numeric(table[isi], errors='coerce').lt(.5))


def validate_raw(path, record, hash_raw):
    path = Path(path)
    variants = [('original', record)]
    if 'public_asset' in record:
        variants.append(('public', record['public_asset']))
    if not path.is_file():
        raise ValueError(f'Missing NWB: {path}')
    candidates = [(name, item) for name, item in variants if path.stat().st_size == item['bytes']]
    if not candidates:
        raise ValueError(f'NWB size does not match a declared source: {path}')
    digest = sha256(path) if hash_raw else None
    if digest:
        candidates = [(name, item) for name, item in candidates
                      if not item.get('sha256') or digest == item['sha256']]
        if not candidates:
            raise ValueError(f'NWB checksum mismatch: {path}')
    name, item = candidates[0]
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': digest, 'validation': 'sha256' if hash_raw else 'size only',
            'source_variant': name, 'asset_id': item.get('asset_id')}


def download_mouse(records, root):
    """Download pinned public assets anonymously; never read DANDI credentials."""
    for record in records:
        asset = record['public_asset']
        target = root / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            validate_raw(target, asset, True)
            continue
        temporary = target.with_suffix('.part')
        print(f"Downloading public MouseV2 {record['site']}", flush=True)
        try:
            with urllib.request.urlopen(asset['url'], timeout=120) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            validate_raw(temporary, asset, True)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def extract_mouse(record, path, metrics, support):
    from nwb_io import read_session
    from flash_metrics import prepare_flash_presentations, response_selected_ttfs, matched_timescales
    from gratings import compute_drifting_grating_metrics, prepare_grating_table
    flash_name = 'flash_field_block_presentations'
    grating_name = 'drifting_gratings_field_block_presentations'
    tables = ([flash_name] if set(metrics) & {'ttfs', 'timescale'} else [])
    if 'gratings' in metrics: tables.append(grating_name)
    units, spikes, intervals = read_session(path, tables)
    units['source_unit_id'] = units.id.astype(int)
    units['unit_id'] = units.source_unit_id + record['id_offset']
    units['probe_letter'] = units.device_name.str.removeprefix('Probe')
    units['area_coarse'] = 'V1'
    mask = qc(units)
    if 'default_qc' in units and not np.array_equal(mask.to_numpy(), units.default_qc.astype(bool).to_numpy()):
        raise ValueError('MouseV2 declared default_qc differs from the canonical waveform rule')
    selected = units.loc[mask].copy()
    result = {}
    if set(metrics) & {'ttfs', 'timescale'}:
        flashes = prepare_flash_presentations(intervals[flash_name])
        if len(flashes) != 300: raise ValueError('Expected 300 MouseV2 flashes')
    if 'ttfs' in metrics:
        table = response_selected_ttfs(spikes, selected, flashes)
        table = table.loc[table.selected & table.preferred_0_250_ttfs_ms.lt(100)].copy()
        table['cohort'] = 'MouseV2'; table['session_id'] = record['site_number']
        table['location'] = table.unit_id.map(units.set_index('unit_id').probe_letter)
        result['mouse_ttfs'] = table[['cohort', 'session_id', 'unit_id', 'location', 'selected', 'preferred_0_250_ttfs_ms']]
    if 'gratings' in metrics:
        table = prepare_grating_table(intervals[grating_name])
        table = table.loc[table.orientation.isin(support['orientation_deg'])
                          & table.temporal_frequency.isin(support['temporal_frequency_hz'])
                          & np.isclose(table.spatial_frequency, support['spatial_frequency_cpd'])
                          & np.isclose(table.contrast, support['contrast'])]
        counts = table.groupby(['orientation', 'temporal_frequency', 'spatial_frequency', 'contrast']).size()
        if len(counts) != 20 or not counts.eq(15).all():
            raise ValueError('Expected 20 common-support grating conditions with 15 trials each')
        computed = compute_drifting_grating_metrics(units.source_unit_id, spikes, table)
        computed.unit_id += record['id_offset']
        table = units[['unit_id', 'probe_letter']].merge(computed[['unit_id', 'f1_f0_dg']], on='unit_id', validate='one_to_one')
        table['site_number'] = record['site_number']; table['default_qc'] = mask.to_numpy()
        result['mouse_f1_f0'] = table.rename(columns={'f1_f0_dg': 'f1_f0_dg_common_support'})[
            ['unit_id', 'site_number', 'probe_letter', 'default_qc', 'f1_f0_dg_common_support']]
    if 'timescale' in metrics:
        result['mouse_timescale'] = matched_timescales(spikes, selected, flashes, record['site_number'])
    return result


def extract_allen(record, path, metrics, support, release):
    from nwb_io import read_session
    from flash_metrics import prepare_flash_presentations, response_selected_ttfs
    from gratings import condition_starts, prepare_unit, summarize_draw
    group = release.loc[release.ecephys_session_id.eq(record['session_id']) & qc(release)
                        & release.ecephys_structure_acronym.isin(AREAS)].copy()
    if group.empty: return {}
    group['unit_id'] = group.ecephys_unit_id.astype(int)
    group['source_unit_id'] = group.unit_id
    group['area_coarse'] = group.ecephys_structure_acronym.map(AREAS)
    do_gratings = 'gratings' in metrics and record['session_type'] == 'brain_observatory_1.1'
    tables = ['flashes_presentations'] if 'ttfs' in metrics else []
    if do_gratings: tables.append('drifting_gratings_presentations')
    if not tables: return {}
    _, spikes, intervals = read_session(path, tables)
    result = {}
    if 'ttfs' in metrics:
        flashes = prepare_flash_presentations(intervals['flashes_presentations'])
        table = response_selected_ttfs(spikes, group, flashes, area_adjustment=True)
        table['session_id'] = record['session_id']
        result['allen_ttfs'] = table.rename(columns={'selected': 'selected_positive_responder_area'})[
            ['session_id', 'unit_id', 'area_coarse', 'selected_positive_responder_area', 'preferred_0_250_ttfs_ms']]
    if do_gratings:
        conditions = condition_starts(intervals['drifting_gratings_presentations'], common_support=support)
        if len(conditions) == 20 and all(len(starts) == 15 for _, starts in conditions):
            rows = []
            for unit in group.itertuples():
                value = summarize_draw(prepare_unit(spikes[unit.source_unit_id], conditions),
                                       [np.arange(15) for _ in conditions])
                rows.append({'ecephys_session_id': record['session_id'], 'ecephys_unit_id': unit.unit_id,
                             'area_coarse': unit.area_coarse, 'f1_f0_dg_harmonized': value['f1_f0']})
            result['allen_f1_f0'] = pd.DataFrame(rows)
    return result


def compare_metric_table(table, reference, keys):
    table = table.sort_values(keys).reset_index(drop=True)
    reference = reference.sort_values(keys).reset_index(drop=True)
    measurements = [name for name in table.columns if name in {
        'preferred_0_250_ttfs_ms', 'f1_f0_dg_common_support',
        'f1_f0_dg_harmonized', 'timescale_ms'}]
    identity = [name for name in table.columns if name not in measurements]
    pd.testing.assert_frame_equal(table[identity], reference[identity],
                                  check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(table[measurements], reference[measurements],
                                  check_dtype=False, check_exact=False, rtol=1e-7, atol=1e-8)


def verify_metrics(output, *, require_complete=False):
    source_manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for item in source_manifest['sources']:
        if '/sources/metrics/' in item['path']:
            if sha256(HERE.parent / item['path']) != item['sha256']:
                raise ValueError(f"Bundled metric source checksum mismatch: {item['label']}")
    manifest = json.loads((output / 'extraction_manifest.json').read_text())
    if not manifest['outputs']:
        raise ValueError('Extraction has no metric outputs')
    if require_complete:
        required = {'mouse_ttfs', 'allen_ttfs', 'mouse_f1_f0', 'allen_f1_f0', 'mouse_timescale'}
        if set(manifest['outputs']) != required:
            raise ValueError('Full reproduction requires all five raw-extracted metric tables')
        raw = json.loads((HERE / 'raw_sources.json').read_text())
        for cohort in ('mousev2', 'allen'):
            expected = {r.get('site_number', r.get('session_id')) for r in raw[cohort]['sessions']}
            if set(manifest['sessions'][cohort]) != expected:
                raise ValueError(f'Incomplete {cohort} extraction; subset runs are audits only')
    failures = []
    for label in manifest['outputs']:
        if sha256(output / f'{label}.csv.gz') != manifest['outputs'][label]:
            raise ValueError(f'Extraction output checksum mismatch: {label}')
        table = pd.read_csv(output / f'{label}.csv.gz')
        reference = pd.read_csv(HERE / 'sources/metrics' / f'{label}.csv.gz')
        if list(table.columns) != list(reference.columns):
            raise ValueError(f'Extraction schema mismatch: {label}')
        if label == 'mouse_ttfs': reference = reference.loc[reference.cohort.eq('MouseV2')]
        session = 'ecephys_session_id' if label == 'allen_f1_f0' else ('site_number' if label == 'mouse_f1_f0' else 'session_id')
        cohort = 'mousev2' if label.startswith('mouse_') else 'allen'
        reference = reference.loc[reference[session].isin(manifest['sessions'][cohort])]
        keys = [session, 'ecephys_unit_id' if label == 'allen_f1_f0' else 'unit_id']
        if label == 'mouse_timescale': keys.append('subsample')
        try:
            compare_metric_table(table, reference[table.columns], keys)
        except AssertionError as exc:
            failures.append(f'{label}: {exc}')
        else:
            print(f'MATCH {label}: {len(table)} rows', flush=True)
    if failures: raise ValueError('\n'.join(failures))
    print('Extracted metrics match the bundled reference for the requested sessions and metrics.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'verify', 'download-allen', 'download-mouse', 'audit-allen-timescales'])
    parser.add_argument('--mouse-root', type=Path, help='Directory containing public or original sub-*/ MouseV2 NWBs')
    parser.add_argument('--allen-root', type=Path, help='Directory containing session_*/ Allen NWBs')
    parser.add_argument('--path-map', type=Path, help='Optional JSON mapping site/session IDs to local NWB paths')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'work/metrics')
    parser.add_argument('--cohorts', nargs='+', choices=['mousev2', 'allen'], default=['mousev2', 'allen'])
    parser.add_argument('--sessions', nargs='+', help='Optional site names or Allen session IDs; subset runs are audits')
    parser.add_argument('--metrics', nargs='+', choices=['ttfs', 'gratings', 'timescale'], default=['ttfs', 'gratings', 'timescale'])
    parser.add_argument('--hash-raw', action='store_true', help='Read each full NWB to verify/record SHA-256 in addition to file size')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.command == 'verify': verify_metrics(output); return
    raw = json.loads((HERE / 'raw_sources.json').read_text())
    mapping = json.loads(args.path_map.read_text()) if args.path_map else {}
    requested = set(args.sessions or [])
    records = [(cohort, record) for cohort in args.cohorts for record in raw[cohort]['sessions']
               if not requested or str(record.get('site', record.get('session_id'))) in requested]
    found = {str(r.get('site', r.get('session_id'))) for _, r in records}
    if requested - found: parser.error(f'Unknown sessions: {sorted(requested - found)}')
    if args.command == 'download-mouse':
        if args.mouse_root is None: parser.error('--mouse-root is required')
        selected = [record for cohort, record in records if cohort == 'mousev2']
        if not selected: parser.error('No MouseV2 sessions selected')
        download_mouse(selected, args.mouse_root)
        return
    if args.command == 'download-allen':
        if args.allen_root is None: parser.error('--allen-root is required')
        for cohort, record in records:
            if cohort != 'allen': continue
            target = args.allen_root / record['path']; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): validate_raw(target, record, args.hash_raw); continue
            temporary = target.with_suffix('.part')
            try:
                with urllib.request.urlopen(record['url']) as response, temporary.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
                validate_raw(temporary, record, args.hash_raw); temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return
    # Check the complete requested file inventory before doing any extraction.
    for cohort, record in records:
        key = str(record.get('site', record.get('session_id')))
        root = args.mouse_root if cohort == 'mousev2' else args.allen_root
        if key not in mapping and root is None: parser.error(f'Provide root or --path-map entry for {key}')
        path = Path(mapping[key]) if key in mapping else root / record['path']
        validate_raw(path, record, False)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    release = pd.read_csv(HERE / 'sources/metrics/allen_units.csv.gz')
    if args.command == 'audit-allen-timescales':
        from nwb_io import read_session, read_column
        from flash_metrics import prepare_flash_presentations, reproduce_unit, exclude_invalid_spikes
        import h5py
        rows = []
        for cohort, record in records:
            if cohort != 'allen': continue
            sid = record['session_id']; key = str(sid)
            if key not in mapping and args.allen_root is None: parser.error('--allen-root or --path-map is required')
            path = Path(mapping[key]) if key in mapping else args.allen_root / record['path']
            validate_raw(path, record, args.hash_raw)
            _, spikes, tables = read_session(path, ['flashes_presentations'])
            flashes = prepare_flash_presentations(tables['flashes_presentations'])
            with h5py.File(path, 'r') as file:
                intervals = (np.column_stack([read_column(file, 'intervals/invalid_times/' + name)
                                             for name in ('start_time', 'stop_time')])
                             if 'intervals/invalid_times' in file else np.empty((0, 2)))
            group = release.loc[release.ecephys_session_id.eq(sid) & release.ecephys_structure_acronym.isin(AREAS)
                                & release.timescale_ac.notna()]
            print(f'Auditing Allen timescales: {sid}', flush=True)
            for unit in group.itertuples():
                result = reproduce_unit(exclude_invalid_spikes(spikes[unit.ecephys_unit_id], intervals),
                                        flashes.start_time.to_numpy(float))
                rows.append({'session_id': sid, 'unit_id': unit.ecephys_unit_id,
                             'released_timescale_ac': unit.timescale_ac,
                             'released_err_ac': unit.err_ac, 'released_spike_count_ac': unit.spike_count_ac, **result})
        pd.DataFrame(rows).to_csv(output / 'allen_timescale_audit.csv', index=False)
        print('Audit written; published Allen fits remain the canonical source values.')
        return
    frames = {}; audit = {'sessions': {'mousev2': [], 'allen': []}, 'inputs': [], 'outputs': {},
                           'metrics': args.metrics, 'python': platform.python_version(),
                           'numpy': np.__version__, 'pandas': pd.__version__,
                           'source_boundary': 'Released Allen unit metadata/timescales and historical anatomical design remain source inputs.'}
    for cohort, record in records:
        key = str(record.get('site', record.get('session_id')))
        root = args.mouse_root if cohort == 'mousev2' else args.allen_root
        if key not in mapping and root is None: parser.error(f'Provide root or --path-map entry for {key}')
        path = Path(mapping[key]) if key in mapping else root / record['path']
        audit['inputs'].append(validate_raw(path, record, args.hash_raw))
        print(f'{cohort} {key}: extracting {args.metrics}', flush=True)
        result = (extract_mouse(record, path, args.metrics, raw['common_support']) if cohort == 'mousev2'
                  else extract_allen(record, path, args.metrics, raw['common_support'], release))
        audit['sessions'][cohort].append(record.get('site_number', record.get('session_id')))
        for label, table in result.items(): frames.setdefault(label, []).append(table)
    for label, parts in frames.items():
        path = output / f'{label}.csv.gz'
        pd.concat(parts, ignore_index=True).to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
        audit['outputs'][label] = sha256(path)
    import scipy
    import h5py
    audit['scipy'] = scipy.__version__
    audit['h5py'] = h5py.__version__
    audit['code_sha256'] = {name: sha256(HERE / name) for name in
                            ('extract.py', 'nwb_io.py', 'flash_metrics.py', 'gratings.py')}
    audit['raw_manifest_sha256'] = sha256(HERE / 'raw_sources.json')
    (output / 'extraction_manifest.json').write_text(json.dumps(audit, indent=2) + '\n')
    verify_metrics(output)


if __name__ == '__main__':
    main()
