"""Small read-only NWB reader; no AllenSDK or research-workspace dependencies."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def read_column(file, path):
    dataset = file[path]
    if isinstance(dataset.id.get_type(), (h5py.h5t.TypeFloatID, h5py.h5t.TypeIntegerID)):
        # Explicit conversion also handles the MouseV2 extended-precision floats.
        result = np.empty(dataset.shape, dtype='<f8')
        dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, result,
                        h5py.h5t.py_create(result.dtype, logical=True))
        return result
    if dataset.dtype.kind == 'b':
        return dataset[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]])


def read_session(path: Path, tables: list[str]):
    """Return scalar unit metadata, sorted spikes keyed by NWB ID, and intervals."""
    with h5py.File(path, 'r') as file:
        ids = read_column(file, '/units/id').astype(np.int64)
        columns = {'id': ids}
        for name in ('device_name', 'amplitude_cutoff', 'presence_ratio',
                     'isi_violations_ratio', 'isi_violations', 'default_qc'):
            if name in file['units']:
                columns[name] = read_column(file, '/units/' + name)
        flat = read_column(file, '/units/spike_times')
        stops = read_column(file, '/units/spike_times_index').astype(np.int64)
        starts = np.r_[0, stops[:-1]]
        spikes = {int(uid): np.sort(flat[start:stop])
                  for uid, start, stop in zip(ids, starts, stops)}
        intervals = {}
        for table in tables:
            group = file['intervals/' + table]
            count = len(group['start_time'])
            intervals[table] = pd.DataFrame({
                name: read_column(file, group[name].name)
                for name in ('start_time', 'stop_time', 'contrast', 'color', 'orientation',
                             'temporal_frequency', 'spatial_frequency', 'phase')
                if name in group and group[name].shape == (count,)
            })
    return pd.DataFrame(columns), spikes, intervals
