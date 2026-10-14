#!/usr/bin/env python3
"""Rebuild anatomy, registry, inference, and Figure 5 from bundled source tables."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'work/reproduction')
    parser.add_argument('--metrics-dir', type=Path, help='Optional complete extract.py output, verified against bundled metric sources')
    parser.add_argument('--resume', action='store_true', help='Verify existing regenerated tables and continue to rendering')
    args = parser.parse_args()
    if args.resume and args.metrics_dir:
        parser.error('--metrics-dir applies to a new run, not --resume')
    output = args.output_dir.resolve()
    if output == HERE or output in HERE.parents:
        parser.error('Use a separate output directory, not the figure or repository root')
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output directory is nonempty; use a new directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started = time.monotonic()
    commands = [
        ['prepare.py', 'check'],
        ['anatomy.py', '--output-dir', str(output / 'anatomy')],
    ]
    if args.resume:
        commands.append(['prepare.py', 'verify', '--output-root', str(output)])
    else:
        command = ['prepare.py', 'regenerate', '--output-root', str(output)]
        if args.metrics_dir: command += ['--metrics-dir', str(args.metrics_dir.resolve())]
        commands.append(command)
    commands.append(['Figure5.py', '--input-root', str(output), '--output-dir', str(output / 'figure')])
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        subprocess.run([sys.executable, str(HERE / command[0]), *command[1:]], check=True, env=env)
    report = {'status': 'verified', 'python': platform.python_version(),
              'elapsed_seconds': round(time.monotonic() - started, 2),
              'commands': commands, 'figure': str(output / 'figure/Figure5.pdf'),
              'statistics': '5000 resamples; original seeds and population definitions',
              'source_boundary': 'Bundled metric-level data and historical anatomical scalar tables; see METHODS.md for raw-source provenance.'}
    report['packages'] = {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'matplotlib', 'fonttools')}
    report['source_manifest_sha256'] = hashlib.sha256((HERE / 'source_manifest.json').read_bytes()).hexdigest()
    report['code_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(HERE.glob('*.py'))}
    if args.resume:
        report['operation'] = 'verify existing tables and render; no statistical regeneration'
        report['timestamp_unix'] = time.time()
        with (output / 'resume_events.jsonl').open('a') as stream:
            stream.write(json.dumps(report) + '\n')
    else:
        (output / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Verified reproduction: {output / "figure/Figure5.pdf"}', flush=True)


if __name__ == '__main__':
    main()
