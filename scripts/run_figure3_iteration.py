#!/usr/bin/env python3
"""Run and provenance a Figure 3 analysis iteration in one command."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
STIMULUS_MANIFEST = ROOT / "config" / "mousev2_stimulus_manifest.json"
DEFAULT_BASELINE = ROOT / "artifacts" / "figure3" / "00_pipeline_baseline" / "rerun"

PRODUCTS = (
    "Figure3_with_V1sites.png",
    "Figure3_probe_zoom.png",
    "Figure3_split_comparison.png",
    "Figure3_stats.md",
)

ENTRY_POINTS = (
    ROOT / "Figure3" / "Figure3_with_V1sites.py",
    ROOT / "Figure3" / "Figure3_probe_zoom.py",
    ROOT / "Figure3" / "Figure3_split_comparison.py",
    ROOT / "scripts" / "eta_squared_comparison.py",
)

RF_ENTRY_POINT = ROOT / "Figure3" / "Figure3_rf_position.py"
RF_IMPORT_FILES = (
    "rf_unit_peaks.csv",
    "rf_probe_summary.csv",
    "rf_probe_ordering.csv",
    "import_manifest.json",
)
RF_PRODUCTS = (
    "Figure3_rf_position.png",
    "rf_metric_session_probe.csv",
    "rf_metric_correlations.csv",
    "rf_position_report.md",
)
POPULATION_ENTRY_POINT = ROOT / "scripts" / "population_flow.py"
POPULATION_PRODUCTS = (
    "population_flow.csv",
    "population_by_group.csv",
    "population_flow.png",
    "population_profile_report.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return hashlib.sha256(rgba.tobytes()).hexdigest()


def normalized_markdown_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"_Generated \d{4}-\d{2}-\d{2}", "_Generated DATE", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_record(path: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.rstrip()

    status = run("status", "--short").splitlines()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_clean": not status,
        "working_tree_status": status,
    }


def compare_product(product: Path, baseline: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "product": file_record(product),
        "baseline": file_record(baseline),
        "file_bytes_identical": sha256(product) == sha256(baseline),
    }
    if product.suffix == ".png":
        product_pixels = pixel_sha256(product)
        baseline_pixels = pixel_sha256(baseline)
        record.update(
            {
                "product_pixel_sha256": product_pixels,
                "baseline_pixel_sha256": baseline_pixels,
                "scientifically_equivalent": product_pixels == baseline_pixels,
            }
        )
    else:
        product_normalized = normalized_markdown_sha256(product)
        baseline_normalized = normalized_markdown_sha256(baseline)
        record.update(
            {
                "product_normalized_sha256": product_normalized,
                "baseline_normalized_sha256": baseline_normalized,
                "scientifically_equivalent": product_normalized == baseline_normalized,
            }
        )
    return record


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> dict[str, object]:
    started = datetime.now().astimezone()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    finished = datetime.now().astimezone()
    log_path.write_text(
        f"command: {' '.join(command)}\n"
        f"started: {started.isoformat()}\n"
        f"finished: {finished.isoformat()}\n"
        f"exit_code: {result.returncode}\n\n"
        f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}",
        encoding="utf-8",
    )
    record = {
        "command": command,
        "exit_code": result.returncode,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "log": relative(log_path),
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="01_reproducible_baseline")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "figure3",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--rf-import-dir",
        type=Path,
        default=None,
        help="Validated Pilot RF import snapshot; enables the measured-retinotopy output.",
    )
    parser.add_argument(
        "--grating-metrics-dir",
        type=Path,
        default=None,
        help="Validated full-condition grating import snapshot.",
    )
    parser.add_argument(
        "--grating-metric",
        choices=("f1_f0_dg", "mod_idx_dg"),
        default="f1_f0_dg",
    )
    parser.add_argument(
        "--flash-metrics-dir",
        type=Path,
        default=None,
        help="Validated pooled and polarity-specific MouseV2 flash import snapshot.",
    )
    parser.add_argument(
        "--flash-variant",
        choices=("pooled", "bright", "dark"),
        default="pooled",
    )
    parser.add_argument(
        "--ttfs-display",
        choices=("mean_matched", "raw_nwb"),
        default="mean_matched",
    )
    parser.add_argument(
        "--within-v1-x-mode",
        choices=("legacy_pseudo_hierarchy", "display_only"),
        default="legacy_pseudo_hierarchy",
        help=(
            "Use historical pseudo-hierarchy positions or non-metric display offsets "
            "centered on VISp for MouseV2 probes/sessions."
        ),
    )
    parser.add_argument(
        "--population-profile",
        choices=(
            "pipeline_baseline",
            "common_qc",
            "published_like",
            "intersection",
            "pilot_rf_qc_diagnostic",
        ),
        default=None,
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip schema/coverage tests (not recommended for reviewed runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, dash, underscore")

    output_root = args.output_root.resolve()
    run_dir = output_root / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty iteration directory: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()

    config_path = args.config.resolve()
    baseline_dir = args.baseline_dir.resolve()
    rf_import_dir = args.rf_import_dir.resolve() if args.rf_import_dir else None
    grating_metrics_dir = (
        args.grating_metrics_dir.resolve() if args.grating_metrics_dir else None
    )
    flash_metrics_dir = (
        args.flash_metrics_dir.resolve() if args.flash_metrics_dir else None
    )
    for required in (config_path, STIMULUS_MANIFEST):
        if not required.is_file():
            raise FileNotFoundError(required)
    for name in PRODUCTS:
        if not (baseline_dir / name).is_file():
            raise FileNotFoundError(baseline_dir / name)
    if rf_import_dir is not None:
        for name in RF_IMPORT_FILES:
            if not (rf_import_dir / name).is_file():
                raise FileNotFoundError(rf_import_dir / name)
    if grating_metrics_dir is not None:
        if not (grating_metrics_dir / "import_manifest.json").is_file():
            raise FileNotFoundError(grating_metrics_dir / "import_manifest.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for session in config["sessions"]:
            path = grating_metrics_dir / session["site"] / "grating_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
    if flash_metrics_dir is not None:
        if not (flash_metrics_dir / "import_manifest.json").is_file():
            raise FileNotFoundError(flash_metrics_dir / "import_manifest.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for session in config["sessions"]:
            path = flash_metrics_dir / session["site"] / "flash_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)

    started = datetime.now().astimezone()
    environment = os.environ.copy()
    cache_root = Path("/tmp") / f"neuropixels-figure3-{args.run_id}-{os.getpid()}"
    environment["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(cache_root / "cache")
    if rf_import_dir is not None:
        environment["FIGURE3_RF_IMPORT_DIR"] = str(rf_import_dir)

    command_records = []
    if not args.skip_tests:
        test_paths = ["tests/test_figure3_mousev2.py"]
        if grating_metrics_dir is not None:
            test_paths.append("tests/test_drifting_gratings.py")
        if flash_metrics_dir is not None:
            test_paths.append("tests/test_flashes.py")
        if args.population_profile is not None:
            test_paths.append("tests/test_population_masks.py")
        if rf_import_dir is not None:
            test_paths.append("tests/test_rf_import.py")
        test_command = [sys.executable, "-m", "pytest", "-q", *test_paths]
        command_records.append(
            run_logged(test_command, logs_dir / "schema_tests.log", environment)
        )

    for entry_point in ENTRY_POINTS:
        command = [
            sys.executable,
            relative(entry_point),
            "--output-dir",
            str(run_dir),
            "--config",
            str(config_path),
            "--within-v1-x-mode",
            args.within_v1_x_mode,
        ]
        if grating_metrics_dir is not None:
            command.extend(
                [
                    "--grating-metrics-dir",
                    str(grating_metrics_dir),
                    "--grating-metric",
                    args.grating_metric,
                ]
            )
        if flash_metrics_dir is not None:
            command.extend(
                [
                    "--flash-metrics-dir",
                    str(flash_metrics_dir),
                    "--flash-variant",
                    args.flash_variant,
                    "--ttfs-display",
                    args.ttfs_display,
                ]
            )
        if args.population_profile is not None:
            command.extend(["--population-profile", args.population_profile])
        command_records.append(
            run_logged(command, logs_dir / f"{entry_point.stem}.log", environment)
        )

    if rf_import_dir is not None:
        rf_command = [
            sys.executable,
            relative(RF_ENTRY_POINT),
            "--output-dir",
            str(run_dir),
            "--rf-import-dir",
            str(rf_import_dir),
            "--config",
            str(config_path),
        ]
        if grating_metrics_dir is not None:
            rf_command.extend(
                [
                    "--grating-metrics-dir",
                    str(grating_metrics_dir),
                    "--grating-metric",
                    args.grating_metric,
                ]
            )
        if flash_metrics_dir is not None:
            rf_command.extend(
                [
                    "--flash-metrics-dir",
                    str(flash_metrics_dir),
                    "--flash-variant",
                    args.flash_variant,
                ]
            )
        if args.population_profile is not None:
            rf_command.extend(["--population-profile", args.population_profile])
        command_records.append(
            run_logged(rf_command, logs_dir / f"{RF_ENTRY_POINT.stem}.log", environment)
        )

    if args.population_profile is not None:
        population_command = [
            sys.executable,
            relative(POPULATION_ENTRY_POINT),
            "--output-dir",
            str(run_dir),
            "--config",
            str(config_path),
            "--population-profile",
            args.population_profile,
        ]
        if grating_metrics_dir is not None:
            population_command.extend(
                ["--grating-metrics-dir", str(grating_metrics_dir)]
            )
        if rf_import_dir is not None:
            population_command.extend(["--rf-import-dir", str(rf_import_dir)])
        command_records.append(
            run_logged(
                population_command,
                logs_dir / f"{POPULATION_ENTRY_POINT.stem}.log",
                environment,
            )
        )

    comparisons = {}
    for name in PRODUCTS:
        product = run_dir / name
        if not product.is_file():
            raise FileNotFoundError(product)
        comparisons[name] = compare_product(product, baseline_dir / name)

    baseline_bundle = baseline_dir.parent
    for name in ("metric_counts.csv", "exclusions.csv"):
        source = baseline_bundle / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)

    rf_import_records = []
    additional_output_records = {}
    if rf_import_dir is not None:
        for name in RF_IMPORT_FILES:
            source = rf_import_dir / name
            destination = run_dir / name
            shutil.copy2(source, destination)
            rf_import_records.append(file_record(destination))
        for name in RF_PRODUCTS:
            product = run_dir / name
            if not product.is_file():
                raise FileNotFoundError(product)
            additional_output_records[name] = file_record(product)
    if args.population_profile is not None:
        for name in POPULATION_PRODUCTS:
            product = run_dir / name
            if not product.is_file():
                raise FileNotFoundError(product)
            additional_output_records[name] = file_record(product)

    package_versions = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "statsmodels", "Pillow", "pytest"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    code_paths = [
        *ENTRY_POINTS,
        ROOT / "common" / "figure3_mousev2.py",
        Path(__file__).resolve(),
        ROOT / "tests" / "test_figure3_mousev2.py",
    ]
    scientific_changes = []
    purpose = "Reproduce Iteration 0 after the non-scientific reproducibility refactor."
    if rf_import_dir is not None:
        code_paths.extend(
            [
                RF_ENTRY_POINT,
                ROOT / "common" / "figure3_rf.py",
                ROOT / "scripts" / "import_pilot_rf_peaks.py",
                ROOT / "tests" / "test_rf_import.py",
            ]
        )
        rf_metric_note = (
            "baseline response metrics are unchanged."
            if grating_metrics_dir is None
            else "the response panels use the selected versioned grating metric."
        )
        scientific_changes.append(
            "Added a provisional measured-retinotopy view using Pilot-QC median "
            f"per-unit RF grid argmax coordinates; {rf_metric_note}"
        )
        purpose = (
            "Add measured RF coordinates while retaining and regression-checking "
            "the baseline response-property outputs."
        )
    if grating_metrics_dir is not None:
        code_paths.extend(
            [
                ROOT / "common" / "drifting_gratings.py",
                ROOT / "generate_retinotopic_csvs.py",
                ROOT / "scripts" / "extract_mousev2_grating_metrics.py",
                ROOT / "tests" / "test_drifting_gratings.py",
            ]
        )
        selected_name = (
            "full-condition F1/F0"
            if args.grating_metric == "f1_f0_dg"
            else "Allen Welch-spectrum modulation index"
        )
        scientific_changes.append(
            f"Replaced the pooled-SF MouseV2 grating overlay with {selected_name}; "
            "preferred conditions include orientation x temporal frequency x spatial frequency."
        )
        purpose = (
            f"Regenerate Figure 3 using {selected_name} while retaining the pooled-SF "
            "values as a named legacy diagnostic."
        )
    if flash_metrics_dir is not None:
        code_paths.extend(
            [
                ROOT / "common" / "flashes.py",
                ROOT / "scripts" / "extract_mousev2_flash_metrics.py",
                ROOT / "tests" / "test_flashes.py",
            ]
        )
        scientific_changes.append(
            f"Recomputed MouseV2 TTFS and response timescale from the "
            f"'{args.flash_variant}' flash presentation set; TTFS is raw relative "
            "to NWB start_time and timescale bins are selected by AllenSDK centers."
        )
        purpose = (
            f"Regenerate Figure 3 using versioned {args.flash_variant} MouseV2 flash "
            "metrics without an inferred display-latency calibration."
        )
    if args.population_profile is not None:
        code_paths.extend(
            [
                ROOT / "common" / "population_masks.py",
                ROOT / "tests" / "test_population_masks.py",
                POPULATION_ENTRY_POINT,
            ]
        )
        scientific_changes.append(
            f"Applied the named population profile '{args.population_profile}' "
            "to both Allen and MouseV2 tables before metric-specific filters."
        )
        purpose = (
            f"Regenerate Figure 3 with the cross-dataset population profile "
            f"'{args.population_profile}' and the selected grating metric."
        )
    if args.within_v1_x_mode == "display_only":
        scientific_changes.append(
            "Replaced pseudo-hierarchy positions for known within-V1 recordings with "
            "small display-only offsets centered on VISp and removed the invalid "
            "within-V1 hierarchy regression."
        )
        purpose = (
            "Represent known within-V1 recording locations without assigning an "
            "unsupported anatomical hierarchy score."
        )
    manifest = {
        "run_id": args.run_id,
        "purpose": purpose,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "scientific_changes": scientific_changes,
        "all_products_scientifically_equivalent": all(
            record["scientifically_equivalent"] for record in comparisons.values()
        ),
        "all_baseline_products_scientifically_equivalent": all(
            record["scientifically_equivalent"] for record in comparisons.values()
        ),
        "commands": command_records,
        "configuration": file_record(config_path),
        "stimulus_manifest": file_record(STIMULUS_MANIFEST),
        "comparison_baseline": relative(baseline_dir),
        "outputs": comparisons,
        "additional_outputs": additional_output_records,
        "rf_import_snapshot": rf_import_records,
        "grating_metric": args.grating_metric if grating_metrics_dir is not None else None,
        "flash_variant": args.flash_variant if flash_metrics_dir is not None else None,
        "ttfs_display": args.ttfs_display,
        "within_v1_x_mode": args.within_v1_x_mode,
        "population_profile": args.population_profile,
        "grating_import_manifest": (
            file_record(grating_metrics_dir / "import_manifest.json")
            if grating_metrics_dir is not None
            else None
        ),
        "flash_import_manifest": (
            file_record(flash_metrics_dir / "import_manifest.json")
            if flash_metrics_dir is not None
            else None
        ),
        "copied_diagnostics": [
            name for name in ("metric_counts.csv", "exclusions.csv") if (run_dir / name).is_file()
        ],
        "code": [file_record(path) for path in code_paths],
        "repositories": {
            "neuropixels_platform_paper": git_record(ROOT),
            "openscope_v2species": git_record(ROOT.parent.parent / "openscope_v2species"),
            "PilotAnalysis": git_record(ROOT.parent.parent / "PilotAnalysis" / "PilotAnalysis"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions,
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    expected_change = (
        grating_metrics_dir is not None
        or flash_metrics_dir is not None
        or args.population_profile is not None
        or args.within_v1_x_mode != "legacy_pseudo_hierarchy"
    )
    status = (
        "PASS"
        if manifest["all_products_scientifically_equivalent"]
        else ("EXPECTED CHANGE" if expected_change else "FAIL")
    )
    comparison_lines = [
        f"- `{name}`: {'equivalent' if record['scientifically_equivalent'] else 'CHANGED'}"
        for name, record in comparisons.items()
    ]
    if grating_metrics_dir is not None:
        selected_name = (
            "full-condition F1/F0"
            if args.grating_metric == "f1_f0_dg"
            else "Allen Welch-spectrum modulation index"
        )
        delta_lines = [
            f"# {args.run_id} delta from baseline",
            "",
            f"**Baseline comparison: {status}.**",
            "",
            f"Scientific change: MouseV2 and Allen units are compared using {selected_name}.",
            "MouseV2 preferred conditions include orientation, temporal frequency, and",
            "spatial frequency. The existing pooled-SF values remain available as",
            "`f1_f0_dg_pooled_sf_legacy` and are not overwritten.",
            "",
            *comparison_lines,
            "",
        ]
        if rf_import_dir is not None:
            delta_lines.extend(
                [
                    "The measured-retinotopy diagnostic was also regenerated with the selected metric.",
                    "",
                    *[f"- `{name}`" for name in RF_PRODUCTS],
                    "",
                ]
            )
        if args.population_profile is not None:
            delta_lines.extend(
                [
                    f"Population profile: `{args.population_profile}` was applied to both datasets",
                    "before the metric-specific validity filters.",
                    "",
                ]
            )
        if flash_metrics_dir is not None:
            delta_lines.extend(
                [
                    f"MouseV2 flash variant: `{args.flash_variant}`; TTFS display: `{args.ttfs_display}`.",
                    "TTFS is aligned to NWB interval start_time without cross-dataset mean matching.",
                    "Response timescale uses AllenSDK bin-center selection (45–285 ms centers).",
                    "",
                ]
            )
        if args.within_v1_x_mode == "display_only":
            delta_lines.extend(
                [
                    "Within-V1 x positions are categorical display offsets centered on VISp,",
                    "not anatomical hierarchy scores; the probe-mean hierarchy fit is removed.",
                    "Measured RF azimuth/elevation remains a separate companion view.",
                    "",
                ]
            )
    elif args.population_profile is not None:
        delta_lines = [
            f"# {args.run_id} delta from baseline",
            "",
            f"**Baseline comparison: {status}.**",
            "",
            f"Scientific change: applied `{args.population_profile}` to both Allen and MouseV2 units.",
            "",
            *comparison_lines,
            "",
        ]
    elif rf_import_dir is not None:
        delta_lines = [
            f"# {args.run_id} delta from baseline",
            "",
            f"**Baseline-product regression check: {status}.**",
            "",
            "Scientific change: added a provisional measured-retinotopy figure using",
            "Pilot-QC median per-unit RF grid argmax coordinates and unit-bootstrap",
            "uncertainty. The baseline response metrics, filters, and statistics did not",
            "change, and the original categorical probe figure is retained.",
            "",
            *comparison_lines,
            "",
            "Additional review products:",
            "",
            *[f"- `{name}`" for name in RF_PRODUCTS],
            "",
            "See `rf_position_report.md` for mapping coverage, ordering consistency,",
            "descriptive response-coordinate associations, and limitations.",
            "",
        ]
    else:
        delta_lines = [
            "# Iteration 1 delta from Iteration 0",
            "",
            f"**Equivalence check: {status}.**",
            "",
            "No scientific definition, filtering rule, or statistical method changed.",
            "This iteration centralizes the eight-session configuration, validates",
            "schemas and coverage, adds explicit output directories, freezes the",
            "stimulus protocol, and runs all products with one command.",
            "",
            *comparison_lines,
            "",
            "The metric-count and exclusion tables are copied from Iteration 0 because",
            "the configured inputs and populations are unchanged.",
            "",
        ]
    delta = "\n".join(
        delta_lines
    )
    (run_dir / "delta_from_previous.md").write_text(delta, encoding="utf-8")

    print(f"Iteration written to {run_dir}")
    print(f"Baseline comparison: {status}")
    for line in comparison_lines:
        print(line)
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
