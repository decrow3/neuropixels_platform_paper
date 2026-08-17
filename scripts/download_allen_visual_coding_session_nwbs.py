#!/usr/bin/env python3
"""Download all Allen Visual Coding Neuropixels session NWBs, resumably.

The download is intentionally limited to the 58 session NWBs. Probe LFP NWBs,
stimulus templates, and raw acquisition files are not requested. Existing NWBs
in the configured pilot/bridge caches are reused in place and never modified.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
UNIT_TABLE = ROOT / "data" / "unit_table.csv"
DEFAULT_CACHE = Path(
    "/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions"
)
EXISTING_ROOTS = (
    Path("/media/huklaban5/Data/MouseV2/allen_bo11_rf_pilot_cache"),
    Path("/media/huklaban5/Data/MouseV2/allen_v1_bridge"),
    Path("/mnt/nvme0/ecephys_cache_dir_2"),
)
API_ROOT = "http://api.brain-map.org"
TARGET_AREAS = {"VISp", "VISal", "VISrl", "VISam", "VISl", "VISpm"}
MINIMUM_FREE_GIB = 250


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def session_priorities():
    sessions = {}
    with UNIT_TABLE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            session_id = int(row["ecephys_session_id"])
            record = sessions.setdefault(
                session_id,
                {
                    "ecephys_session_id": session_id,
                    "session_type": row["session_type"],
                    "target_units": 0,
                    "qc_target_units": 0,
                },
            )
            if row["ecephys_structure_acronym"] not in TARGET_AREAS:
                continue
            record["target_units"] += 1
            try:
                passes = (
                    float(row["amplitude_cutoff"]) <= 0.1
                    and float(row["presence_ratio"]) >= 0.95
                    and float(row["isi_violations"]) <= 0.5
                )
            except (TypeError, ValueError):
                passes = False
            record["qc_target_units"] += int(passes)
    if len(sessions) != 58:
        raise RuntimeError(f"Expected 58 sessions in {UNIT_TABLE}, found {len(sessions)}")
    return sessions


def official_file_records():
    criteria = (
        "model::WellKnownFile,rma::criteria,"
        "well_known_file_type[name$eq'EcephysNwb'],"
        "rma::options[num_rows$eqall]"
    )
    url = f"{API_ROOT}/api/v2/data/query.json?criteria={urllib.parse.quote(criteria)}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    if not payload.get("success") or len(payload.get("msg", [])) != 58:
        raise RuntimeError("Allen API did not return the expected 58 EcephysNwb records")
    return {
        int(row["attachable_id"]): {
            "well_known_file_id": int(row["id"]),
            "url": f"{API_ROOT}{row['download_link']}",
        }
        for row in payload["msg"]
    }


def content_length(url: str):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def discover_existing_nwbs():
    found = {}
    for root in EXISTING_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.nwb"):
            text = str(path)
            for token in path.stem.replace("-", "_").split("_"):
                if token.isdigit() and len(token) == 9:
                    found.setdefault(int(token), path.resolve())
    return found


def inspect_nwb(path: Path):
    with h5py.File(path, "r") as nwb:
        return {
            "bytes": path.stat().st_size,
            "has_units": "units" in nwb,
            "has_gabor_presentations": "intervals/gabors_presentations" in nwb,
            "has_raw_gaze_mapping": "processing/raw_gaze_mapping" in nwb,
            "has_ccf_unit_columns": all(
                f"units/{name}" in nwb
                for name in (
                    "anterior_posterior_ccf_coordinate",
                    "dorsal_ventral_ccf_coordinate",
                    "left_right_ccf_coordinate",
                )
            ),
        }


def main():
    args = parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    lock_stream = (cache / "download.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another downloader already holds {cache / 'download.lock'}")
    lock_stream.write(f"pid={os.getpid()} started={utc_now()}\n")
    lock_stream.flush()

    events_path = cache / "download_events.jsonl"
    inventory_path = cache / "session_inventory.json"
    priorities = session_priorities()
    files = official_file_records()
    existing = discover_existing_nwbs()
    queue = sorted(
        priorities.values(),
        key=lambda row: (-row["qc_target_units"], -row["target_units"], row["ecephys_session_id"]),
    )
    append_jsonl(
        events_path,
        {
            "event": "run_start",
            "time_utc": utc_now(),
            "pid": os.getpid(),
            "cache": str(cache),
            "dry_run": args.dry_run,
            "free_bytes": shutil.disk_usage(cache).free,
        },
    )
    inventory = []
    for index, record in enumerate(queue, start=1):
        session_id = record["ecephys_session_id"]
        remote = files.get(session_id)
        if remote is None:
            raise RuntimeError(f"No official EcephysNwb record for session {session_id}")
        expected_bytes = content_length(remote["url"])
        destination = cache / f"session_{session_id}" / f"session_{session_id}.nwb"
        if destination.is_file() and destination.stat().st_size == expected_bytes:
            source = destination
            status = "existing_in_bulk_cache"
        elif session_id in existing and existing[session_id].stat().st_size == expected_bytes:
            source = existing[session_id]
            status = "reused_external"
        else:
            source = destination
            status = "planned" if args.dry_run else "downloaded"
            if not args.dry_run:
                free = shutil.disk_usage(cache).free
                required = expected_bytes + MINIMUM_FREE_GIB * 1024**3
                if free < required:
                    raise RuntimeError(
                        f"Insufficient space before session {session_id}: "
                        f"{free / 1024**3:.1f} GiB free, {required / 1024**3:.1f} GiB required"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(".nwb.partial")
                print(
                    f"[{index:02d}/58] downloading {session_id} "
                    f"({expected_bytes / 1024**3:.2f} GiB; "
                    f"QC V1/HVA units={record['qc_target_units']})",
                    flush=True,
                )
                append_jsonl(
                    events_path,
                    {
                        "event": "session_start",
                        "time_utc": utc_now(),
                        **record,
                        "expected_bytes": expected_bytes,
                        "destination": str(destination),
                    },
                )
                started = time.time()
                subprocess.run(
                    [
                        "curl", "--fail", "--location", "--continue-at", "-",
                        "--retry", "8", "--retry-delay", "10", "--retry-all-errors",
                        "--output", str(partial), remote["url"],
                    ],
                    check=True,
                )
                if partial.stat().st_size != expected_bytes:
                    raise RuntimeError(
                        f"Size mismatch for {session_id}: {partial.stat().st_size} != {expected_bytes}"
                    )
                partial.replace(destination)
                elapsed = time.time() - started
                append_jsonl(
                    events_path,
                    {
                        "event": "session_downloaded",
                        "time_utc": utc_now(),
                        **record,
                        "bytes": expected_bytes,
                        "elapsed_seconds": elapsed,
                        "destination": str(destination),
                    },
                )

        validation = None if args.dry_run and status == "planned" else inspect_nwb(source)
        if validation and (
            validation["bytes"] != expected_bytes
            or not validation["has_units"]
            or not validation["has_gabor_presentations"]
        ):
            raise RuntimeError(f"NWB validation failed for {session_id}: {validation}")
        item = {
            **record,
            **remote,
            "expected_bytes": expected_bytes,
            "status": status,
            "nwb_path": str(source),
            "validation": validation,
        }
        inventory.append(item)
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        if status.startswith("existing") or status == "reused_external":
            print(f"[{index:02d}/58] {status}: {session_id} -> {source}", flush=True)

    append_jsonl(
        events_path,
        {
            "event": "run_finish",
            "time_utc": utc_now(),
            "sessions": len(inventory),
            "downloaded": sum(row["status"] == "downloaded" for row in inventory),
            "reused": sum(row["status"] != "downloaded" for row in inventory),
            "free_bytes": shutil.disk_usage(cache).free,
        },
    )
    print(f"Complete: {len(inventory)} sessions recorded in {inventory_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
