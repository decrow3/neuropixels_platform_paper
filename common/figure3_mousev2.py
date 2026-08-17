"""Validated loading helpers for the Figure 3 Allen and MouseV2 tables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from common.population_masks import apply_population_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "figure3_mousev2.json"

FINE_TO_COARSE = {
    "VISp": "V1",
    "VISl": "LM",
    "VISrl": "RL",
    "VISal": "AL",
    "VISpm": "PM",
    "VISam": "AM",
}

METRIC_RENAMES = {
    "time_to_first_spike": "time_to_first_spike_fl",
    "modulation_index": "f1_f0_dg_pooled_sf_legacy",
    "autocorr_tau": "timescale_ac",
}

WITHIN_V1_X_MODES = ("legacy_pseudo_hierarchy", "display_only")


def within_v1_x_positions(
    labels: list[str] | tuple[str, ...],
    mode: str,
    *,
    visp_score: float,
    legacy_bounds: tuple[float, float],
    display_half_span: float,
) -> dict[str, float]:
    """Return plotting positions without treating display offsets as scores.

    ``legacy_pseudo_hierarchy`` preserves the historical figure geometry for
    regression checks. ``display_only`` places labels symmetrically around the
    published VISp score; callers must label these offsets as non-metric and
    must not fit a hierarchy trend through them.
    """
    if mode not in WITHIN_V1_X_MODES:
        raise ValueError(f"Unknown within-V1 x mode: {mode}")
    labels = list(labels)
    if not labels:
        return {}
    if mode == "display_only":
        left = float(visp_score) - float(display_half_span)
        right = float(visp_score) + float(display_half_span)
    else:
        left, right = map(float, legacy_bounds)
    if len(labels) == 1:
        return {labels[0]: (left + right) / 2.0}
    step = (right - left) / (len(labels) - 1)
    return {label: left + index * step for index, label in enumerate(labels)}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Read and minimally validate the versioned MouseV2 configuration."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"data_directory", "allen_unit_table", "probe_labels", "sessions"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing Figure 3 configuration keys: {sorted(missing)}")
    if not config["sessions"]:
        raise ValueError("Figure 3 configuration contains no MouseV2 sessions")
    return config


def _resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_unique(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    table = pd.read_csv(path)
    if "unit_id" not in table.columns:
        raise ValueError(f"{label} has no unit_id column: {path}")
    duplicates = table["unit_id"].duplicated()
    if duplicates.any():
        examples = table.loc[duplicates, "unit_id"].head().tolist()
        raise ValueError(f"Duplicate unit_id values in {label}: {examples}")
    return table


def load_allen_units(
    config_path: str | Path | None = None,
    *,
    population_profile: str | None = None,
) -> pd.DataFrame:
    """Load the published Allen unit table with the shared coarse-area label."""
    config = load_config(config_path)
    path = _resolve_from_root(config["allen_unit_table"])
    table = pd.read_csv(path, low_memory=False)
    if "ecephys_structure_acronym" not in table.columns:
        raise ValueError(f"Allen unit table lacks ecephys_structure_acronym: {path}")
    table["area_coarse"] = table["ecephys_structure_acronym"].map(
        lambda value: FINE_TO_COARSE.get(value, value)
    )
    if population_profile is not None:
        table = apply_population_profile(
            table, profile=population_profile, dataset="allen"
        )
    return table


def load_mousev2_units(
    *,
    apply_qc: bool,
    config_path: str | Path | None = None,
    grating_metrics_dir: str | Path | None = None,
    flash_metrics_dir: str | Path | None = None,
    flash_variant: str = "pooled",
    population_profile: str | None = None,
) -> pd.DataFrame:
    """Load all configured MouseV2 metric tables in stable session order.

    The joins and optional QC filter intentionally reproduce the current Figure
    3 scripts. Validation is stricter: missing files, duplicate unit IDs,
    unexpected labels, row-count drift, and probe-coverage drift fail loudly.
    """
    if apply_qc and population_profile is not None:
        raise ValueError(
            "Use either legacy apply_qc=True or a named population_profile, not both"
        )
    config = load_config(config_path)
    data_dir = _resolve_from_root(config["data_directory"])
    grating_dir = (
        _resolve_from_root(grating_metrics_dir)
        if grating_metrics_dir is not None
        else None
    )
    flash_dir = (
        _resolve_from_root(flash_metrics_dir)
        if flash_metrics_dir is not None
        else None
    )
    if flash_variant not in {"pooled", "bright", "dark"}:
        raise ValueError("flash_variant must be 'pooled', 'bright', or 'dark'")
    expected_probes = set(config["probe_labels"])
    frames = []

    for session in config["sessions"]:
        site = session["site"]
        site_number = int(session["site_number"])
        site_dir = data_dir / f"{site}_processed"

        layer = _read_unique(site_dir / "layer_info.csv", f"{site} layer_info")
        modulation = _read_unique(
            site_dir / "change_modulation_data.csv", f"{site} change_modulation_data"
        )
        timescale = _read_unique(
            site_dir / "timescale_metrics.csv", f"{site} timescale_metrics"
        )
        ttfs = _read_unique(
            site_dir / "time_to_first_spike.csv", f"{site} time_to_first_spike"
        )

        table = (
            layer.merge(modulation, on="unit_id", validate="one_to_one")
            .merge(timescale, on="unit_id", validate="one_to_one")
            .merge(ttfs, on="unit_id", validate="one_to_one")
            .rename(columns=METRIC_RENAMES)
        )

        if grating_dir is None:
            # Preserve Iteration 0 behavior while naming its pooled-SF origin.
            table["f1_f0_dg"] = table["f1_f0_dg_pooled_sf_legacy"]
        else:
            grating = _read_unique(
                grating_dir / site / "grating_metrics.csv",
                f"{site} full-condition grating_metrics",
            )
            overlap = set(table.columns).intersection(grating.columns).difference(
                {"unit_id"}
            )
            if overlap:
                raise ValueError(
                    f"Unexpected {site} grating column collisions: {sorted(overlap)}"
                )
            table = table.merge(grating, on="unit_id", validate="one_to_one")
            # Iteration 3 froze preferred-condition mean spike counts over the
            # validated nominal analysis duration.  Preserve compatibility
            # with that import while exposing the published filter's rate name.
            if (
                "firing_rate_dg" not in table.columns
                and {"preferred_mean_spikes_dg", "analysis_duration_s_dg"}.issubset(
                    table.columns
                )
            ):
                table["firing_rate_dg"] = (
                    pd.to_numeric(table["preferred_mean_spikes_dg"], errors="coerce")
                    / pd.to_numeric(table["analysis_duration_s_dg"], errors="coerce")
                )

        if flash_dir is not None:
            flash = _read_unique(
                flash_dir / site / "flash_metrics.csv",
                f"{site} versioned flash_metrics",
            )
            overlap = set(table.columns).intersection(flash.columns).difference(
                {"unit_id"}
            )
            if overlap:
                raise ValueError(
                    f"Unexpected {site} flash column collisions: {sorted(overlap)}"
                )
            table = table.merge(flash, on="unit_id", validate="one_to_one")
            table["time_to_first_spike_fl_pooled_legacy"] = table[
                "time_to_first_spike_fl"
            ]
            table["timescale_ac_pooled_legacy"] = table["timescale_ac"]
            table["err_ac_pooled_legacy"] = table["err_ac"]
            table["spike_count_ac_pooled_legacy"] = table["spike_count_ac"]
            table["time_to_first_spike_fl"] = table[
                f"time_to_first_spike_{flash_variant}"
            ]
            table["timescale_ac"] = table[f"autocorr_tau_{flash_variant}"]
            table["err_ac"] = table[f"err_ac_{flash_variant}"]
            table["spike_count_ac"] = table[f"spike_count_ac_{flash_variant}"]
            table["flash_variant"] = flash_variant

        expected_units = int(session["expected_units"])
        if len(layer) != expected_units or len(table) != expected_units:
            raise ValueError(
                f"{site} row-count drift: expected {expected_units}, "
                f"layer={len(layer)}, joined={len(table)}"
            )

        pattern = rf"V1_site{site_number}_([A-Z])"
        extracted = table["ecephys_structure_acronym"].astype(str).str.extract(
            pattern, expand=False
        )
        if extracted.isna().any():
            bad = table.loc[extracted.isna(), "ecephys_structure_acronym"].head().tolist()
            raise ValueError(f"Unexpected {site} structure labels: {bad}")
        observed_probes = set(extracted.unique())
        if observed_probes != expected_probes:
            raise ValueError(
                f"{site} probe coverage drift: expected {sorted(expected_probes)}, "
                f"observed {sorted(observed_probes)}"
            )

        table["site"] = site
        table["session_num"] = site_number
        table["subject_id"] = int(session["subject_id"])
        table["probe_letter"] = extracted.values

        if apply_qc or population_profile is not None:
            quality = _read_unique(site_dir / "unit_quality.csv", f"{site} unit_quality")
            if "default_qc" not in quality.columns:
                raise ValueError(f"{site} unit_quality has no default_qc column")
            quality_columns = (
                list(quality.columns)
                if population_profile is not None
                else ["unit_id", "default_qc"]
            )
            table = table.merge(
                quality[quality_columns],
                on="unit_id",
                how="left",
                validate="one_to_one",
            )
        if apply_qc:
            table = table[table["default_qc"] == True].drop(columns=["default_qc"])
        elif population_profile is not None:
            table = apply_population_profile(
                table, profile=population_profile, dataset="mousev2"
            )

        frames.append(table)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined["unit_id"].duplicated().any():
        raise ValueError("MouseV2 unit_id values collide across configured sessions")
    return combined


def site_coarse_label(structure_acronym: object) -> str | None:
    """Map V1_siteN_probe labels to the baseline V1_sN display labels."""
    match = re.match(r"V1_site(\d+)_?", str(structure_acronym))
    return f"V1_s{match.group(1)}" if match else None
