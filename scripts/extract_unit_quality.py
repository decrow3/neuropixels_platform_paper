"""
scripts/extract_unit_quality.py

Read quality metrics from each session's NWB units table and save
unit_quality.csv to the corresponding site_processed directory.
No spike processing — fast.

unit_id in output = nwb_id + id_offset  (same scheme as generate_retinotopic_csvs.py)
"""

import h5py
import numpy as np
import pandas as pd
import os

def _read_col(fid, f, path):
    """Read a units column, handling 80-bit floats via low-level h5py API."""
    try:
        dset_id = h5py.h5d.open(fid, path.encode())
        t = dset_id.get_type()
        if isinstance(t, (h5py.h5t.TypeFloatID, h5py.h5t.TypeIntegerID)):
            n = dset_id.get_space().get_simple_extent_npoints()
            arr = np.empty(n, dtype='<f8')
            mem_type = h5py.h5t.py_create(np.dtype('<f8'), logical=True)
            dset_id.read(h5py.h5s.ALL, h5py.h5s.ALL, arr, mem_type)
            return arr
        else:
            raw = f[path][:]
            return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw])
    except Exception as e:
        print(f'    read failed {path}: {e}')
        return None

NWB_BASE = '/media/huklaban5/Data/MouseV2/001568'

# site_num → (sub_id, id_offset)
SITE_MAP = {
    2: ('816305', 2_000_000),
    3: ('810531', 3_000_000),
    4: ('810532', 4_000_000),
    5: ('813810', 5_000_000),
    6: ('815152', 6_000_000),
    7: ('816308', 7_000_000),
    8: ('817334', 8_000_000),
    9: ('817335', 9_000_000),
}

QUALITY_COLS = ['snr', 'firing_rate', 'amplitude_cutoff',
                'presence_ratio', 'isi_violations_ratio', 'default_qc']

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for site_num, (sub_id, offset) in sorted(SITE_MAP.items()):
    sub_dir = os.path.join(NWB_BASE, f'sub-{sub_id}')
    nwb_files = [f for f in os.listdir(sub_dir) if f.endswith('.nwb')]
    if not nwb_files:
        print(f'site{site_num}: no NWB found in {sub_dir}')
        continue
    nwb_path = os.path.join(sub_dir, nwb_files[0])
    out_dir  = os.path.join(code_dir, 'data', f'site{site_num}_processed')

    fid = h5py.h5f.open(nwb_path.encode())
    with h5py.File(nwb_path, 'r') as f:
        units = f['units']
        n_units_idx = _read_col(fid, f, '/units/spike_times_index')
        n_units = len(n_units_idx)
        rows = {'unit_id': np.arange(n_units, dtype=np.int64) + offset}
        for col in QUALITY_COLS:
            if col not in units:
                print(f'  site{site_num}: {col!r} not in NWB, skipping')
                continue
            data = _read_col(fid, f, f'/units/{col}')
            if data is None or len(data) != n_units:
                print(f'  site{site_num}: {col!r} length mismatch ({len(data) if data is not None else "None"} vs {n_units}), skipping')
                continue
            rows[col] = data

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'unit_quality.csv')
    df.to_csv(out_path, index=False)
    n_good = (df['default_qc'] == 'good').sum() if 'default_qc' in df.columns else '?'
    print(f'site{site_num} (sub-{sub_id}): {len(df)} units, '
          f'{n_good} default_qc=good  -> {out_path}')
