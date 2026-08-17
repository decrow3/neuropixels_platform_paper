#!/usr/bin/env python3
"""Freeze and inventory the unchanged Figure 3 MouseV2 baseline.

This utility is intentionally diagnostic: it does not recompute metrics or
figures. Run the four existing Figure 3 entry points first, then run this file
from anywhere to snapshot their products and describe the exact inputs,
software, filters, and unit counts used by the current pipeline.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "00_pipeline_baseline"
RUN_DIR = ROOT / "artifacts" / "figure3" / RUN_ID
PRE_DIR = RUN_DIR / "pre_rerun"
RERUN_DIR = RUN_DIR / "rerun"

FIGURE_PRODUCTS = (
    "Figure3_with_V1sites.png",
    "Figure3_probe_zoom.png",
    "Figure3_split_comparison.png",
    "Figure3_stats.md",
)

ENTRY_POINTS = (
    "python Figure3/Figure3_with_V1sites.py",
    "python Figure3/Figure3_probe_zoom.py",
    "python Figure3/Figure3_split_comparison.py",
    "python scripts/eta_squared_comparison.py",
)

SITE_SUBJECTS = {
    2: 816305,
    3: 810531,
    4: 810532,
    5: 813810,
    6: 815152,
    7: 816308,
    8: 817334,
    9: 817335,
}

KNOWN_LIMITATIONS = (
    "The current F1/F0 preferred condition groups orientation and temporal "
    "frequency while pooling the five spatial frequencies.",
    "The paper reference uses mod_idx_dg, while the MouseV2 overlay currently "
    "uses F1/F0 under a renamed modulation_index column.",
    "Published receptive-field significance and area filters are absent.",
    "default_qc is used by the split comparison and statistics, but not by "
    "the hierarchy overlay or probe zoom.",
    "Probe positions are placeholders ordered B, C, A, E rather than measured "
    "receptive-field centers.",
    "All targeted probe units are labeled V1 and cortical layer is unavailable.",
    "Some plots show a TTFS mean-matching shift to Allen V1 rather than an "
    "independent display-timing calibration.",
    "The statistics bootstrap probes independently rather than preserving the "
    "matched within-session probe vector.",
    "LP is included with the post-V1 cortical areas despite being thalamic.",
    "The CCG output is a single-session smoke test and is not part of this "
    "baseline figure bundle.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pixel_record(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return {
            "width": rgba.width,
            "height": rgba.height,
            "mode": rgba.mode,
            "pixel_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
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


def normalized_markdown(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"_Generated \d{4}-\d{2}-\d{2}", "_Generated DATE", text)
    return text.encode("utf-8")


def load_site(site: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_dir = ROOT / "data" / f"site{site}_processed"
    layer = pd.read_csv(site_dir / "layer_info.csv")
    modulation = pd.read_csv(site_dir / "change_modulation_data.csv")
    timescale = pd.read_csv(site_dir / "timescale_metrics.csv")
    ttfs = pd.read_csv(site_dir / "time_to_first_spike.csv")
    quality = pd.read_csv(site_dir / "unit_quality.csv")

    for name, table in (
        ("layer_info", layer),
        ("change_modulation_data", modulation),
        ("timescale_metrics", timescale),
        ("time_to_first_spike", ttfs),
        ("unit_quality", quality),
    ):
        if table["unit_id"].duplicated().any():
            raise ValueError(f"Duplicate unit_id values in site{site} {name}")

    joined = (
        layer.merge(modulation, on="unit_id", validate="one_to_one")
        .merge(timescale, on="unit_id", validate="one_to_one")
        .merge(ttfs, on="unit_id", validate="one_to_one")
        .merge(
            quality[["unit_id", "default_qc"]],
            on="unit_id",
            how="left",
            validate="one_to_one",
        )
    )
    expected_prefix = f"V1_site{site}_"
    if not joined["ecephys_structure_acronym"].str.startswith(expected_prefix).all():
        raise ValueError(f"Unexpected structure acronym in site{site}")
    joined["site"] = f"site{site}"
    joined["subject_id"] = SITE_SUBJECTS[site]
    joined["probe"] = joined["ecephys_structure_acronym"].str.rsplit("_", n=1).str[-1]
    layer = layer.copy()
    layer["probe"] = layer["ecephys_structure_acronym"].str.rsplit("_", n=1).str[-1]
    return layer, joined


def metric_masks(table: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    ttfs_finite = np.isfinite(table["time_to_first_spike"].astype(float))
    f1_f0_finite = np.isfinite(table["modulation_index"].astype(float))
    tau_finite = np.isfinite(table["autocorr_tau"].astype(float))
    timescale_aux_finite = (
        np.isfinite(table["spike_count_ac"].astype(float))
        & np.isfinite(table["err_ac"].astype(float))
    )
    return {
        "ttfs": (
            ttfs_finite,
            ttfs_finite & (table["time_to_first_spike"].astype(float) < 0.1),
        ),
        "f1_f0": (
            f1_f0_finite,
            f1_f0_finite & (table["modulation_index"].astype(float) > 0),
        ),
        "timescale": (
            tau_finite & timescale_aux_finite,
            tau_finite
            & timescale_aux_finite
            & table["autocorr_tau"].astype(float).between(1, 300)
            & (table["spike_count_ac"].astype(float) > 50)
            & (table["err_ac"].astype(float) < 20),
        ),
    }


def build_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []

    for site in SITE_SUBJECTS:
        layer, joined = load_site(site)
        for probe in ("A", "B", "C", "E"):
            raw_probe = layer[layer["probe"] == probe]
            probe_table = joined[joined["probe"] == probe]
            raw_n = len(raw_probe)
            joined_n = len(probe_table)
            qc = probe_table["default_qc"].fillna(False).astype(bool)

            exclusion_rows.extend(
                [
                    {
                        "site": f"site{site}",
                        "subject_id": SITE_SUBJECTS[site],
                        "probe": probe,
                        "metric": "all",
                        "reason": "missing_from_four_way_metric_join",
                        "eligible_n": raw_n,
                        "excluded_n": raw_n - joined_n,
                    },
                    {
                        "site": f"site{site}",
                        "subject_id": SITE_SUBJECTS[site],
                        "probe": probe,
                        "metric": "all",
                        "reason": "failed_or_missing_default_qc",
                        "eligible_n": joined_n,
                        "excluded_n": int((~qc).sum()),
                    },
                ]
            )

            for metric, (finite, valid) in metric_masks(probe_table).items():
                count_rows.append(
                    {
                        "site": f"site{site}",
                        "subject_id": SITE_SUBJECTS[site],
                        "probe": probe,
                        "metric": metric,
                        "raw_units": raw_n,
                        "joined_units": joined_n,
                        "default_qc_units": int(qc.sum()),
                        "metric_finite_units": int(finite.sum()),
                        "current_unfiltered_valid_units": int(valid.sum()),
                        "current_qc_valid_units": int((valid & qc).sum()),
                    }
                )
                exclusion_rows.extend(
                    [
                        {
                            "site": f"site{site}",
                            "subject_id": SITE_SUBJECTS[site],
                            "probe": probe,
                            "metric": metric,
                            "reason": "nonfinite_metric_or_required_auxiliary",
                            "eligible_n": joined_n,
                            "excluded_n": int((~finite).sum()),
                        },
                        {
                            "site": f"site{site}",
                            "subject_id": SITE_SUBJECTS[site],
                            "probe": probe,
                            "metric": metric,
                            "reason": "failed_current_metric_threshold",
                            "eligible_n": int(finite.sum()),
                            "excluded_n": int((finite & ~valid).sum()),
                        },
                    ]
                )

    return pd.DataFrame(count_rows), pd.DataFrame(exclusion_rows)


def main() -> None:
    if not PRE_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected the preserved pre-rerun products at {PRE_DIR}"
        )

    RERUN_DIR.mkdir(parents=True, exist_ok=True)
    for name in FIGURE_PRODUCTS:
        source = ROOT / "Figure3" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, RERUN_DIR / name)

    counts, exclusions = build_counts()
    counts.to_csv(RUN_DIR / "metric_counts.csv", index=False)
    exclusions.to_csv(RUN_DIR / "exclusions.csv", index=False)

    output_comparison: dict[str, dict[str, object]] = {}
    for name in FIGURE_PRODUCTS:
        previous = PRE_DIR / name
        rerun = RERUN_DIR / name
        record: dict[str, object] = {
            "pre_rerun": file_record(previous),
            "rerun": file_record(rerun),
            "file_bytes_identical": sha256(previous) == sha256(rerun),
        }
        if name.endswith(".png"):
            record["pre_rerun_pixels"] = pixel_record(previous)
            record["rerun_pixels"] = pixel_record(rerun)
            record["pixels_identical"] = (
                record["pre_rerun_pixels"]["pixel_sha256"]
                == record["rerun_pixels"]["pixel_sha256"]
            )
        else:
            record["identical_ignoring_generated_date"] = (
                normalized_markdown(previous) == normalized_markdown(rerun)
            )
        output_comparison[name] = record

    code_paths = [
        ROOT / "Figure3" / "Figure3_with_V1sites.py",
        ROOT / "Figure3" / "Figure3_probe_zoom.py",
        ROOT / "Figure3" / "Figure3_split_comparison.py",
        ROOT / "scripts" / "eta_squared_comparison.py",
        Path(__file__).resolve(),
    ]
    input_paths = [ROOT / "data" / "unit_table.csv"]
    for site in SITE_SUBJECTS:
        site_dir = ROOT / "data" / f"site{site}_processed"
        input_paths.extend(
            site_dir / name
            for name in (
                "layer_info.csv",
                "change_modulation_data.csv",
                "timescale_metrics.csv",
                "time_to_first_spike.csv",
                "unit_quality.csv",
            )
        )

    packages = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "statsmodels", "Pillow"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    manifest = {
        "run_id": RUN_ID,
        "captured_at": datetime.now().astimezone().isoformat(),
        "purpose": "Freeze the unchanged first-pass MouseV2 Figure 3 pipeline.",
        "scientific_change_from_pre_rerun": False,
        "commands": list(ENTRY_POINTS),
        "command_exit_codes": [0, 0, 0, 0],
        "session_subject_map": {f"site{k}": v for k, v in SITE_SUBJECTS.items()},
        "probe_labels": ["A", "B", "C", "E"],
        "random_seeds": {
            "Figure3_with_V1sites.py": 10,
            "Figure3_probe_zoom.py": 42,
            "Figure3_split_comparison.py": 42,
            "eta_squared_comparison.py": 42,
        },
        "analysis_profiles": {
            "Figure3_with_V1sites.py": {"default_qc": False},
            "Figure3_probe_zoom.py": {"default_qc": False},
            "Figure3_split_comparison.py": {"default_qc": True},
            "eta_squared_comparison.py": {"default_qc": True},
            "metric_validity": {
                "ttfs": "finite time_to_first_spike < 0.1 seconds",
                "f1_f0": "finite modulation_index > 0",
                "timescale": (
                    "finite 1 <= autocorr_tau <= 300 ms, spike_count_ac > 50, "
                    "and err_ac < 20"
                ),
            },
        },
        "repositories": {
            "neuropixels_platform_paper": git_record(ROOT),
            "openscope_v2species": git_record(ROOT.parent.parent / "openscope_v2species"),
            "PilotAnalysis": git_record(ROOT.parent.parent / "PilotAnalysis" / "PilotAnalysis"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "declared_environment": file_record(ROOT / "environment.yml"),
        },
        "code": [file_record(path) for path in code_paths],
        "inputs": [file_record(path) for path in input_paths],
        "outputs": output_comparison,
        "counts": {
            "raw_mousev2_units": int(counts.groupby(["site", "probe"])["raw_units"].first().sum()),
            "default_qc_mousev2_units": int(
                counts.groupby(["site", "probe"])["default_qc_units"].first().sum()
            ),
            "metric_count_table": "metric_counts.csv",
            "exclusion_table": "exclusions.csv",
        },
        "known_limitations": list(KNOWN_LIMITATIONS),
    }
    with (RUN_DIR / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(f"Wrote baseline bundle to {RUN_DIR}")
    print(
        "MouseV2 units: "
        f"{manifest['counts']['raw_mousev2_units']} raw, "
        f"{manifest['counts']['default_qc_mousev2_units']} default_qc"
    )
    for name, record in output_comparison.items():
        comparison = (
            record.get("pixels_identical")
            if name.endswith(".png")
            else record.get("identical_ignoring_generated_date")
        )
        print(f"{name}: scientifically equivalent={comparison}")


if __name__ == "__main__":
    main()
