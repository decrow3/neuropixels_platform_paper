#!/usr/bin/env python3
"""Run resumable full-population Allen V1/HVA RF extraction and fitting.

The production workflow has two deliberately separate resource regimes:

1. Load at most two historical NWBs concurrently and write compact Gabor
   trial/spike-count caches without reading gaze data.
2. Fit up to six compact caches concurrently. Every published-like unit gets
   point and analytic-aperture fits in both axis-aligned and freely rotated
   forms, using all Gabor trials and five rotation starts.

No LFP/raw files are accessed and no existing files are deleted.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = Path(
    "/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions/"
    "session_inventory.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_full_rf_production_v1"
DEFAULT_PYTHON = Path("/home/huklaban5/anaconda3/envs/allensdk/bin/python")
UNIT_TABLE = ROOT / "data" / "unit_table.csv"
EXTRACT_SCRIPT = ROOT / "scripts" / "test_allen_population_gaze_rf.py"
FIT_SCRIPT = ROOT / "scripts" / "fit_allen_multisession_rf_geometry.py"
TARGET_AREAS = {"VISp", "VISal", "VISrl", "VISam", "VISl", "VISpm"}
EVENT_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "extract", "fit", "aggregate", "status"),
        default="all",
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--unit-table", type=Path, default=UNIT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sessions", nargs="+", type=int)
    parser.add_argument("--extract-workers", type=int, default=2)
    parser.add_argument("--fit-workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, record: dict):
    payload = {"time_utc": utc_now(), **record}
    with EVENT_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def make_layout(output: Path):
    layout = {
        "root": output,
        "manifest": output / "00_manifest",
        "cache": output / "01_compact_cache",
        "fits": output / "02_session_fits",
        "aggregate": output / "03_aggregate",
        "validation": output / "04_validation",
        "logs": output / "logs",
        "runtime_cache": output / ".runtime_cache",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    layout["events"] = output / "production_events.jsonl"
    return layout


def build_plan(inventory_path: Path, unit_table_path: Path, requested_sessions=None):
    inventory = pd.DataFrame(json.loads(inventory_path.read_text(encoding="utf-8")))
    required_inventory = {"ecephys_session_id", "session_type", "nwb_path"}
    if not required_inventory.issubset(inventory):
        raise ValueError(f"Inventory lacks {sorted(required_inventory - set(inventory))}")
    units = pd.read_csv(unit_table_path, low_memory=False)
    target = units.loc[units["ecephys_structure_acronym"].isin(TARGET_AREAS)].copy()
    target["published_like"] = (
        target["p_value_rf"].lt(0.01)
        & target["area_rf"].lt(2500)
        & target["snr"].gt(1)
        & target["firing_rate_dg"].gt(0.1)
    )
    counts = target.groupby("ecephys_session_id").agg(
        visual_units=("ecephys_unit_id", "size"),
        fit_units=("published_like", "sum"),
    ).reset_index()
    plan = inventory.merge(counts, on="ecephys_session_id", how="inner", validate="one_to_one")
    if requested_sessions:
        requested = set(map(int, requested_sessions))
        missing = requested - set(plan["ecephys_session_id"].astype(int))
        if missing:
            raise ValueError(f"Requested sessions absent from inventory/unit table: {sorted(missing)}")
        plan = plan.loc[plan["ecephys_session_id"].isin(requested)]
    plan["nwb_exists"] = plan["nwb_path"].map(lambda value: Path(value).is_file())
    if not plan["nwb_exists"].all():
        missing = plan.loc[~plan["nwb_exists"], "ecephys_session_id"].tolist()
        raise FileNotFoundError(f"Missing NWBs for sessions {missing}")
    return plan.sort_values(
        ["fit_units", "visual_units", "ecephys_session_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def validate_extraction(cache_dir: Path, expected_visual_units: int):
    errors = []
    population_path = cache_dir / "visual_unit_population.csv"
    trials_path = cache_dir / "gabor_trial_gaze_table.csv"
    spikes_path = cache_dir / "gabor_spike_counts.npz"
    summary_path = cache_dir / "extraction_summary.json"
    required = (population_path, trials_path, spikes_path, summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"complete": False, "errors": [f"missing {path}" for path in missing]}
    try:
        population = pd.read_csv(population_path, low_memory=False)
        trials = pd.read_csv(trials_path, low_memory=False)
        with np.load(spikes_path) as payload:
            unit_ids = payload["unit_ids"].astype(int)
            counts_shape = payload["counts"].shape
        if len(population) != int(expected_visual_units):
            errors.append(f"population rows {len(population)} != {expected_visual_units}")
        condition_columns = ["x_position", "y_position", "orientation"]
        if not set(condition_columns).issubset(trials):
            errors.append("Gabor trial table lacks position/orientation columns")
        else:
            condition_counts = trials.groupby(condition_columns, observed=True).size()
            if len(condition_counts) != 243:
                errors.append(f"RF conditions {len(condition_counts)} != 243")
            split_counts = trials.groupby(
                condition_columns + ["trial_split"], observed=True
            ).size().unstack("trial_split", fill_value=0)
            for split in ("train", "test"):
                if split not in split_counts or int(split_counts[split].min()) < 1:
                    errors.append(f"one or more RF conditions lack {split} presentations")
        if counts_shape != (len(population), len(trials)):
            errors.append(f"count matrix shape {counts_shape} is inconsistent")
        if not np.array_equal(unit_ids, population["ecephys_unit_id"].to_numpy(int)):
            errors.append("spike-count unit order mismatch")
        if not trials["trial_split"].isin(["train", "test"]).all():
            errors.append("invalid trial split labels")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"complete": not errors, "errors": errors}


def validate_fit(session_dir: Path, expected_fit_units: int):
    errors = []
    table_path = session_dir / "unit_geometry_fits.csv"
    figures = (
        session_dir / "Figure_geometry_population.png",
        session_dir / "Figure_geometry_examples.png",
    )
    if not table_path.is_file():
        return {"complete": False, "errors": [f"missing {table_path}"]}
    try:
        table = pd.read_csv(table_path, low_memory=False)
        required_columns = {
            "ecephys_unit_id", "spatial_model", "rotation_test_gain",
            "axis_area_deg2", "rotation_area_deg2", "axis_optimizer_nfev",
            "rotation_optimizer_nfev", "model_elapsed_seconds",
        }
        if not required_columns.issubset(table):
            errors.append(f"missing columns {sorted(required_columns - set(table))}")
        else:
            if table["ecephys_unit_id"].nunique() != int(expected_fit_units):
                errors.append(
                    f"fit units {table['ecephys_unit_id'].nunique()} != {expected_fit_units}"
                )
            if len(table) != 2 * int(expected_fit_units):
                errors.append(f"model rows {len(table)} != {2 * int(expected_fit_units)}")
            pairs = table.groupby("ecephys_unit_id")["spatial_model"].agg(set)
            if len(pairs) != int(expected_fit_units) or not pairs.map(
                lambda value: value == {"point", "aperture"}
            ).all():
                errors.append("each unit does not have exactly point and aperture rows")
            numeric = [
                "rotation_test_gain", "axis_area_deg2", "rotation_area_deg2",
                "axis_optimizer_nfev", "rotation_optimizer_nfev",
                "model_elapsed_seconds",
            ]
            if not table[numeric].notna().all().all():
                errors.append("full rotation or optimizer telemetry is incomplete")
        for figure in figures:
            if not figure.is_file() or figure.stat().st_size == 0:
                errors.append(f"missing or empty figure {figure}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"complete": not errors, "errors": errors}


def child_environment(layout, session_id):
    environment = os.environ.copy()
    mpl = layout["runtime_cache"] / f"session_{session_id}" / "matplotlib"
    xdg = layout["runtime_cache"] / f"session_{session_id}" / "xdg"
    mpl.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)
    environment.update({
        "MPLCONFIGDIR": str(mpl),
        "XDG_CACHE_HOME": str(xdg),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return environment


def execute_job(stage, session, command, log_path, events_path, environment):
    session_id = int(session.ecephys_session_id)
    started = time.time()
    append_event(events_path, {
        "event": f"{stage}_start", "session_id": session_id,
        "command": command, "log": str(log_path),
    })
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    record = {
        "event": f"{stage}_finish", "session_id": session_id,
        "returncode": result.returncode, "elapsed_seconds": time.time() - started,
        "log": str(log_path),
    }
    append_event(events_path, record)
    return record


def run_parallel(stage, plan, workers, command_builder, validator, layout, dry_run):
    pending = []
    completed = []
    for session in plan.itertuples(index=False):
        validation = validator(session)
        if validation["complete"]:
            completed.append(int(session.ecephys_session_id))
        else:
            pending.append(session)
    print(
        f"{stage}: {len(completed)} complete, {len(pending)} pending, "
        f"workers={workers}", flush=True,
    )
    if dry_run:
        for session in pending:
            print(f"  would run {stage} session {session.ecephys_session_id}")
        return []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for session in pending:
            session_id = int(session.ecephys_session_id)
            command = command_builder(session)
            log_path = layout["logs"] / f"{stage}_session_{session_id}.log"
            future = executor.submit(
                execute_job, stage, session, command, log_path, layout["events"],
                child_environment(layout, session_id),
            )
            futures[future] = session
        for future in as_completed(futures):
            session = futures[future]
            session_id = int(session.ecephys_session_id)
            try:
                record = future.result()
                validation = validator(session)
                if record["returncode"] or not validation["complete"]:
                    failures.append({
                        "session_id": session_id,
                        "returncode": record["returncode"],
                        "errors": validation["errors"],
                    })
                    print(f"{stage} FAILED {session_id}: {failures[-1]}", flush=True)
                else:
                    print(
                        f"{stage} complete {session_id} in "
                        f"{record['elapsed_seconds'] / 60:.1f} min", flush=True,
                    )
            except Exception as exc:
                failures.append({
                    "session_id": session_id,
                    "returncode": None,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                })
                print(f"{stage} FAILED {session_id}: {failures[-1]}", flush=True)
    return failures


def extraction_validator(layout):
    def validate(session):
        return validate_extraction(
            layout["cache"] / f"session_{int(session.ecephys_session_id)}",
            int(session.visual_units),
        )
    return validate


def fit_validator(layout):
    def validate(session):
        return validate_fit(
            layout["fits"] / f"session_{int(session.ecephys_session_id)}",
            int(session.fit_units),
        )
    return validate


def extraction_command(args, layout, session):
    session_id = int(session.ecephys_session_id)
    return [
        str(args.python), str(EXTRACT_SCRIPT),
        "--nwb", str(session.nwb_path),
        "--unit-table", str(args.unit_table.resolve()),
        "--session-id", str(session_id),
        "--output-dir", str(layout["cache"] / f"session_{session_id}"),
        "--extract-only", "--skip-gaze", "--overwrite",
    ]


def fit_command(args, layout, session):
    return [
        str(args.python), str(FIT_SCRIPT),
        "--sessions", str(int(session.ecephys_session_id)),
        "--cache-root", str(layout["cache"]),
        "--output-dir", str(layout["fits"]),
        "--rotation-limit", "100000",
        "--rotation-all-units", "--all-gabor-trials",
        "--per-session-only", "--resume",
    ]


def validation_table(plan, layout):
    rows = []
    extract_validate = extraction_validator(layout)
    fit_validate = fit_validator(layout)
    for session in plan.itertuples(index=False):
        extraction = extract_validate(session)
        fit = fit_validate(session)
        rows.append({
            "ecephys_session_id": int(session.ecephys_session_id),
            "session_type": session.session_type,
            "visual_units": int(session.visual_units),
            "fit_units": int(session.fit_units),
            "extraction_complete": extraction["complete"],
            "fit_complete": fit["complete"],
            "extraction_errors": json.dumps(extraction["errors"]),
            "fit_errors": json.dumps(fit["errors"]),
        })
    return pd.DataFrame(rows)


def aggregate(plan, layout):
    validation = validation_table(plan, layout)
    validation.to_csv(layout["validation"] / "session_validation.csv", index=False)
    incomplete = validation.loc[~validation["fit_complete"]]
    if len(incomplete):
        raise RuntimeError(
            f"Cannot aggregate: {len(incomplete)} sessions are incomplete: "
            f"{incomplete['ecephys_session_id'].tolist()}"
        )
    tables = []
    for session_id in plan["ecephys_session_id"].astype(int):
        path = layout["fits"] / f"session_{session_id}" / "unit_geometry_fits.csv"
        tables.append(pd.read_csv(path, low_memory=False))
    fits = pd.concat(tables, ignore_index=True)
    fits.to_csv(
        layout["aggregate"] / "all_session_unit_geometry_fits.csv",
        index=False, float_format="%.9g",
    )
    rows = []
    for keys, local in fits.groupby(
        ["session_id", "group", "spatial_model"], observed=True
    ):
        area_ratio = local["rotation_area_deg2"] / local["axis_area_deg2"]
        rows.append({
            "session_id": int(keys[0]), "group": keys[1], "spatial_model": keys[2],
            "units": len(local),
            "median_rotation_test_gain": local["rotation_test_gain"].median(),
            "fraction_rotation_gain_positive": local["rotation_test_gain"].gt(0).mean(),
            "median_rotation_area_ratio": area_ratio.median(),
            "axis_censored_fraction": local["axis_censored"].mean(),
            "rotation_bound_fraction": (
                local["rotation_sigma_lower_bound"] | local["rotation_sigma_upper_bound"]
            ).mean(),
            "fit_model_seconds": local["model_elapsed_seconds"].sum(),
            "maximum_model_seconds": local["model_elapsed_seconds"].max(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(
        layout["aggregate"] / "session_group_model_summary.csv",
        index=False, float_format="%.9g",
    )
    render_summary(summary, fits, layout["aggregate"] / "Figure_production_rotation_summary.png")
    payload = {
        "completed_utc": utc_now(),
        "sessions": int(plan["ecephys_session_id"].nunique()),
        "fit_units": int(fits["ecephys_unit_id"].nunique()),
        "model_rows": len(fits),
        "total_model_cpu_hours": float(fits["model_elapsed_seconds"].sum() / 3600),
        "median_model_seconds": float(fits["model_elapsed_seconds"].median()),
        "p99_model_seconds": float(fits["model_elapsed_seconds"].quantile(0.99)),
        "maximum_model_seconds": float(fits["model_elapsed_seconds"].max()),
    }
    (layout["aggregate"] / "production_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def render_summary(summary, fits, path):
    aperture = summary.loc[summary["spatial_model"].eq("aperture")]
    colors = {"V1": "#3366aa", "HVA": "#d97736"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for group in ("V1", "HVA"):
        local = aperture.loc[aperture["group"].eq(group)]
        axes[0, 0].hist(
            local["median_rotation_test_gain"], bins=24, histtype="step",
            linewidth=2, color=colors[group], label=group,
        )
        axes[0, 1].hist(
            local["fraction_rotation_gain_positive"], bins=np.linspace(0, 1, 21),
            histtype="step", linewidth=2, color=colors[group], label=group,
        )
    axes[0, 0].axvline(0, color="#555555", ls="--")
    axes[0, 0].set(title="Session median held-out rotation gain", xlabel="Gain", ylabel="Sessions")
    axes[0, 1].axvline(0.5, color="#555555", ls="--")
    axes[0, 1].set(title="Fraction of units helped", xlabel="Fraction", ylabel="Sessions")
    session_timing = fits.groupby("session_id").agg(
        units=("ecephys_unit_id", "nunique"),
        model_seconds=("model_elapsed_seconds", "sum"),
        slowest_seconds=("model_elapsed_seconds", "max"),
    )
    axes[1, 0].scatter(session_timing["units"], session_timing["model_seconds"] / 60, s=24)
    axes[1, 0].set(xlabel="Fit units", ylabel="Model CPU minutes", title="Per-session computation")
    axes[1, 1].hist(
        fits["model_elapsed_seconds"], bins=np.geomspace(0.03, max(1.0, fits["model_elapsed_seconds"].max()), 40),
        histtype="stepfilled", alpha=.6, color="#756bb1",
    )
    axes[1, 1].set(xscale="log", xlabel="Seconds per model", ylabel="Model rows", title="Optimizer runtime tail")
    for axis in axes.ravel():
        axis.grid(alpha=.15)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False)
    figure.suptitle("Allen full-population RF production audit")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_status(plan, layout):
    validation = validation_table(plan, layout)
    print(
        validation[["extraction_complete", "fit_complete"]].sum().rename("sessions").to_string()
    )
    print(f"total sessions: {len(validation)}")
    incomplete = validation.loc[~validation["extraction_complete"] | ~validation["fit_complete"]]
    if len(incomplete):
        print("incomplete sessions:")
        print(
            incomplete[["ecephys_session_id", "extraction_complete", "fit_complete"]]
            .to_string(index=False)
        )


def main():
    args = parse_args()
    if args.extract_workers < 1 or args.fit_workers < 1:
        raise ValueError("Worker counts must be positive")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    output = args.output_dir.resolve()
    layout = make_layout(output)
    lock_stream = None
    if not args.dry_run and args.mode != "status":
        lock_path = output / "production.lock"
        lock_stream = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another production runner holds {lock_path}")
        lock_stream.seek(0)
        lock_stream.truncate()
        lock_stream.write(json.dumps({
            "pid": os.getpid(), "mode": args.mode, "started_utc": utc_now()
        }) + "\n")
        lock_stream.flush()
    plan = build_plan(
        args.inventory.resolve(), args.unit_table.resolve(), args.sessions
    )
    plan.to_csv(layout["manifest"] / "session_plan.csv", index=False)
    configuration = {
        "created_utc": utc_now(), "mode": args.mode,
        "inventory": str(args.inventory.resolve()),
        "unit_table": str(args.unit_table.resolve()),
        "output_dir": str(output), "python": str(args.python),
        "sessions": plan["ecephys_session_id"].astype(int).tolist(),
        "extract_workers": args.extract_workers, "fit_workers": args.fit_workers,
        "rotation_all_units": True, "gaze_loaded": False,
        "all_gabor_trials": True,
    }
    (layout["manifest"] / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Plan: {len(plan)} sessions, {int(plan['visual_units'].sum())} visual units, "
        f"{int(plan['fit_units'].sum())} fit units", flush=True,
    )
    if args.mode == "status":
        print_status(plan, layout)
        return

    failures = []
    append_event(layout["events"], {
        "event": "runner_start", "pid": os.getpid(), "mode": args.mode,
        "sessions": len(plan), "dry_run": args.dry_run,
    })
    if args.mode in ("all", "extract"):
        failures.extend(run_parallel(
            "extract", plan, args.extract_workers,
            lambda session: extraction_command(args, layout, session),
            extraction_validator(layout), layout, args.dry_run,
        ))
    if args.mode in ("all", "fit"):
        extract_validate = extraction_validator(layout)
        ready = plan.loc[
            [extract_validate(session)["complete"] for session in plan.itertuples(index=False)]
        ]
        if len(ready) != len(plan):
            print(f"fit: {len(plan) - len(ready)} sessions lack valid compact caches", flush=True)
        failures.extend(run_parallel(
            "fit", ready, args.fit_workers,
            lambda session: fit_command(args, layout, session),
            fit_validator(layout), layout, args.dry_run,
        ))
    if args.mode in ("all", "aggregate") and not args.dry_run and not failures:
        payload = aggregate(plan, layout)
        print(json.dumps(payload, indent=2), flush=True)
    print_status(plan, layout)
    append_event(layout["events"], {
        "event": "runner_finish", "pid": os.getpid(), "mode": args.mode,
        "failures": len(failures),
    })
    if failures:
        failure_path = layout["validation"] / "run_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
