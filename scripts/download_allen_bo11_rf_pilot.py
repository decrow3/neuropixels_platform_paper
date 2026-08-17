#!/usr/bin/env python3
"""Download and validate a conservative Allen BO 1.1 RF/gaze pilot cache.

Only the session NWBs are requested; probe LFP files and stimulus templates are
not downloaded. Existing nonempty session files are left in place and checked,
never removed by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
    EcephysProjectCache,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "config" / "allen_bo11_rf_pilot_download.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_nwb(path: Path, compute_hash: bool):
    with h5py.File(path, "r") as nwb:
        processing = nwb.get("processing")
        processing_names = [] if processing is None else list(processing.keys())
        # Allen's historical ecephys files store one DynamicTable per stimulus
        # under /intervals rather than one combined stimulus_presentations table.
        has_gabor_presentations = "intervals/gabors_presentations" in nwb
        result = {
            "bytes": path.stat().st_size,
            "processing_modules": processing_names,
            "has_units": "units" in nwb,
            "has_gabor_presentations": has_gabor_presentations,
            "has_raw_gaze_mapping": "processing/raw_gaze_mapping" in nwb,
            "has_filtered_gaze_mapping": "processing/filtered_gaze_mapping" in nwb,
            "has_eye_tracking": "processing/eye_tracking" in nwb,
        }
    if compute_hash:
        result["sha256"] = sha256(path)
    return result


def append_event(path: Path, event: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def main():
    args = parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cache_dir = Path(plan["cache_directory"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(cache_dir).free
    minimum_free = int(plan["minimum_free_space_before_download_gib"] * 1024 ** 3)
    if free_before < minimum_free:
        raise RuntimeError(
            f"Only {free_before / 1024**3:.1f} GiB free at {cache_dir}; "
            f"the plan requires at least {minimum_free / 1024**3:.1f} GiB."
        )

    events_path = cache_dir / "download_events.jsonl"
    append_event(events_path, {
        "event": "run_start",
        "time_utc": utc_now(),
        "plan": str(plan_path),
        "free_bytes": free_before,
    })
    print(f"Cache: {cache_dir}", flush=True)
    print(f"Free before download: {free_before / 1024**3:.1f} GiB", flush=True)

    cache = EcephysProjectCache.from_warehouse(
        manifest=cache_dir / "manifest.json", fetch_tries=3, timeout=1200
    )
    results = []
    for record in plan["download_sessions"]:
        session_id = int(record["ecephys_session_id"])
        destination = cache_dir / f"session_{session_id}" / f"session_{session_id}.nwb"
        started = time.time()
        status = "existing" if destination.is_file() and destination.stat().st_size else "downloaded"
        event = {"event": "session_start", "time_utc": utc_now(),
                 "ecephys_session_id": session_id, "destination": str(destination)}
        append_event(events_path, event)
        print(f"[{session_id}] {status}: {destination}", flush=True)
        try:
            session = cache.get_session_data(session_id)
            del session
            gc.collect()
            validation = inspect_nwb(destination, compute_hash=not args.skip_sha256)
            if not validation["has_units"] or not validation["has_gabor_presentations"]:
                raise ValueError("NWB lacks required units or Gabor-presentations tables")
            if not validation["has_raw_gaze_mapping"]:
                raise ValueError("NWB lacks the expected raw gaze mapping")
            result = {
                **record,
                "status": status,
                "destination": str(destination),
                "elapsed_seconds": time.time() - started,
                **validation,
            }
            print(
                f"[{session_id}] validated {validation['bytes'] / 1024**3:.2f} GiB; "
                f"raw gaze={validation['has_raw_gaze_mapping']}", flush=True
            )
        except Exception as exc:
            result = {
                **record,
                "status": "failed",
                "destination": str(destination),
                "elapsed_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"[{session_id}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        results.append(result)
        append_event(events_path, {"event": "session_finish", "time_utc": utc_now(), **result})

    free_after = shutil.disk_usage(cache_dir).free
    summary = {
        "run_finished_utc": utc_now(),
        "plan": str(plan_path),
        "cache_directory": str(cache_dir),
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "disk_bytes_consumed_this_run": free_before - free_after,
        "validated_session_nwb_bytes_total": sum(
            int(row.get("bytes", 0)) for row in results if row["status"] != "failed"
        ),
        "successful_sessions": sum(row["status"] != "failed" for row in results),
        "failed_sessions": sum(row["status"] == "failed" for row in results),
        "sessions": results,
    }
    summary_path = cache_dir / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    append_event(events_path, {"event": "run_finish", **summary})
    print(f"Wrote {summary_path}", flush=True)
    if summary["failed_sessions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
