"""Contract and scientific edge-case tests for the standalone Figure 5 pipeline."""
from pathlib import Path
import hashlib
import contextlib
import io
import json
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import prepare
import flash_metrics as flashes
import gratings
from extract import verify_metrics, validate_raw, download_mouse, compare_metric_table
from unittest.mock import patch


class ReproductionTests(unittest.TestCase):
    def test_metric_comparison_requires_exact_identity(self):
        reference = pd.DataFrame({'unit_id': [1000000000], 'session_id': [900000000],
                                  'selected': [True], 'preferred_0_250_ttfs_ms': [50.]})
        rounded = reference.copy()
        rounded['preferred_0_250_ttfs_ms'] += 1e-8
        compare_metric_table(rounded, reference, ['session_id', 'unit_id'])
        for field in ('unit_id', 'session_id', 'selected'):
            changed = reference.copy()
            changed[field] = False if field == 'selected' else changed[field] + 1
            with self.assertRaises(AssertionError):
                compare_metric_table(changed, reference, ['session_id', 'unit_id'])

    def test_resume_preserves_generation_provenance(self):
        import reproduce
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = '{"commands": [["prepare.py", "regenerate", "--metrics-dir", "raw"]]}'
            (root / 'reproduction.json').write_text(original)
            with patch.object(sys, 'argv', ['reproduce.py', '--resume', '--output-dir', directory]), \
                 patch('reproduce.subprocess.run'), patch('reproduce.version', return_value='test'), \
                 contextlib.redirect_stdout(io.StringIO()):
                reproduce.main()
                reproduce.main()
            self.assertEqual((root / 'reproduction.json').read_text(), original)
            events = [json.loads(line) for line in (root / 'resume_events.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertTrue(all('no statistical regeneration' in e['operation'] for e in events))

    def test_public_download_is_anonymous_and_hash_verified(self):
        content = b'public NWB fixture'
        asset = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                 'url': 'https://api.dandiarchive.org/api/assets/test/download/'}
        record = {'site': 'site2', 'path': 'sub-test/test.nwb', 'bytes': 1,
                  'sha256': hashlib.sha256(b'x').hexdigest(), 'public_asset': asset}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('extract.urllib.request.urlopen', return_value=io.BytesIO(content)) as request:
                download_mouse([record], root)
                request.assert_called_once_with(asset['url'], timeout=120)
            self.assertEqual(validate_raw(root / record['path'], record, True)['source_variant'], 'public')
            with patch('extract.urllib.request.urlopen') as request:
                download_mouse([record], root)
                request.assert_not_called()
            (root / record['path']).write_bytes(b'x')
            self.assertEqual(validate_raw(root / record['path'], record, True)['source_variant'], 'original')
            with self.assertRaises(ValueError):
                download_mouse([record], root)

    def test_public_download_rejects_corruption_and_cleans_partial_file(self):
        content = b'wrong'
        asset = {'bytes': len(content), 'sha256': hashlib.sha256(b'right').hexdigest(),
                 'url': 'https://api.dandiarchive.org/api/assets/test/download/'}
        record = {'site': 'site2', 'path': 'sub-test/test.nwb', 'public_asset': asset}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('extract.urllib.request.urlopen', return_value=io.BytesIO(content)):
                with self.assertRaises(ValueError): download_mouse([record], root)
            self.assertFalse((root / record['path']).exists())
            self.assertFalse((root / record['path']).with_suffix('.part').exists())

    def test_bundled_sources_are_complete_and_hash_locked(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(prepare.check_sources())
        manifest = json.loads((HERE / 'dependencies.json').read_text())
        for name in manifest['data_inputs'] + manifest['production_code']:
            path = HERE.parent / name
            self.assertTrue(path.is_file(), name)
            self.assertTrue(path.resolve().is_relative_to(HERE), name)

    def test_registry_reconstructed_from_metrics_and_anatomy(self):
        actual = prepare.apply_cell_floor(prepare.build_response_registry()).reset_index(drop=True)
        expected = pd.read_csv(HERE / 'data/response_registry_floor5.csv.gz',
                               dtype={k: str for k in ['unit_id', 'session_id', 'animal_id', 'physical_probe_id']})
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-12)

    def test_missing_derived_outputs_fail_even_without_registry(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(prepare.verify_outputs(Path(directory), full=False))

    def test_partial_raw_extraction_cannot_be_full_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'extraction_manifest.json').write_text(json.dumps({
                'outputs': {'mouse_ttfs': 'placeholder'}, 'sessions': {'mousev2': [2], 'allen': []}}))
            with self.assertRaisesRegex(ValueError, 'all five'):
                verify_metrics(path, require_complete=True)

    def test_ttfs_excludes_empty_trials_and_keeps_200ms_bin(self):
        spikes = np.array([.029, .0305, 1.2009])
        self.assertEqual(flashes.first_spike_ms(spikes, np.array([0., 1., 2.])), 115.)
        self.assertTrue(np.isnan(flashes.first_spike_ms(np.array([]), np.array([0.]))))

    def test_f1_f0_averages_trials_and_excludes_zero_f0(self):
        counts = np.zeros((3, 1000))
        counts[0, 0] = 1
        counts[1, [0, 250]] = 1
        self.assertAlmostEqual(gratings.f1_f0_from_trial_counts(counts, 2., 1.), 1.)

    def test_balanced_draws_preserve_both_flash_polarities(self):
        table = pd.DataFrame({'flash_polarity': ['bright'] * 150 + ['dark'] * 150})
        first = flashes.balanced_subsample_masks(table, n_subsamples=10, seed=20260807)
        second = flashes.balanced_subsample_masks(table, n_subsamples=10, seed=20260807)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a[:150].sum(), 75)
            self.assertEqual(a[150:].sum(), 75)

    def test_anatomical_inputs_regenerate(self):
        import anatomy
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            anatomy.regenerate(Path(directory))

    def test_cached_resampling_preserves_original_random_stream(self):
        reference = json.loads((HERE / 'tests/resampling_reference.json').read_text())
        registry = pd.read_csv(HERE / 'data/response_registry_floor5.csv.gz',
                               dtype={k: str for k in ['unit_id', 'session_id', 'animal_id', 'physical_probe_id']})
        generators = {kind: np.random.default_rng(reference['seed']) for kind in ('primary', 'control')}
        for fixture in reference['fixtures']:
            metric = fixture['metric']; rng = generators[fixture['kind']]
            if fixture['kind'] == 'primary':
                population = fixture['population']
                source = registry.loc[registry.metric.eq(metric) & registry.population.eq(population)]
                plan = prepare._resampling_plan(source, per_session=True)
                table = prepare._resample_population(registry, metric, population, rng, prepared=plan)
            else:
                source = prepare._control_population(registry, metric)
                plan = prepare._resampling_plan(source, per_session=False)
                table = prepare._resample_animals(source, rng, prepared=plan)
            digest = hashlib.sha256(table[reference['columns']].to_csv(index=False).encode()).hexdigest()
            self.assertEqual(digest, fixture['sha256'], fixture)

    def test_nwb_spikes_use_ids_not_row_numbers(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('Run this test in requirements-extraction.txt environment')
        from nwb_io import read_session
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.nwb'
            with h5py.File(path, 'w') as file:
                group = file.require_group('units')
                # Explicit HDF5 integer storage also works with wheels built
                # against a newer NumPy than the extraction environment.
                for name, values in [('id', [42, 9]), ('spike_times_index', [2, 3])]:
                    data = np.array(values, dtype='<i8')
                    dataset = h5py.h5d.create(group.id, name.encode(), h5py.h5t.STD_I64LE,
                                             h5py.h5s.create_simple(data.shape))
                    dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, data, mtype=h5py.h5t.STD_I64LE)
                file['units/spike_times'] = [.2, .1, .3]
                file['intervals/flashes/start_time'] = [0.]
                file['intervals/flashes/stop_time'] = [.25]
            _, spikes, _ = read_session(path, ['flashes'])
            self.assertEqual(set(spikes), {42, 9})
            np.testing.assert_array_equal(spikes[42], [.1, .2])
            np.testing.assert_array_equal(spikes[9], [.3])


if __name__ == '__main__':
    unittest.main()
