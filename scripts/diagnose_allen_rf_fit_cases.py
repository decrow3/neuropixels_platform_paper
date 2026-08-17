#!/usr/bin/env python3
"""Compare Allen's RF Gaussian with baseline-aware fits on selected native maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import (
    ReceptiveFieldMapping,
    fit_2d_gaussian,
    threshold_rf,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NWB = Path(
    "/media/huklaban5/Data/MouseV2/allen_v1_bridge/000021/"
    "sub-718643564/sub-718643564_ses-737581020.nwb"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_rf_improved_fit_diagnostic" / "checkpoint1"
DEFAULT_CASES = (951867908, 951868026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--unit-ids", type=int, nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gaussian_no_baseline(shape: tuple[int, int], params: np.ndarray) -> np.ndarray:
    amplitude, center_y, center_x, sigma_y, sigma_x = params
    y, x = np.indices(shape)
    return amplitude * np.exp(
        -0.5 * (((y - center_y) / sigma_y) ** 2 + ((x - center_x) / sigma_x) ** 2)
    )


def gaussian_with_baseline(shape: tuple[int, int], params: np.ndarray) -> np.ndarray:
    baseline, amplitude, center_y, center_x, sigma_y, sigma_x = params
    return baseline + gaussian_no_baseline(
        shape, np.array([amplitude, center_y, center_x, sigma_y, sigma_x])
    )


def initial_parameters(matrix: np.ndarray) -> np.ndarray:
    baseline = max(float(np.quantile(matrix, 0.20)), 0.0)
    peak_y, peak_x = np.unravel_index(np.argmax(matrix), matrix.shape)
    amplitude = max(float(matrix.max() - baseline), 1e-3)
    return np.array([baseline, amplitude, peak_y, peak_x, 1.5, 1.5], dtype=float)


def fit_baseline_model(matrix: np.ndarray, bounded: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    rows, columns = matrix.shape
    start = initial_parameters(matrix)
    if bounded:
        lower = np.array([0.0, 0.0, 0.0, 0.0, 0.35, 0.35])
        upper = np.array([np.inf, np.inf, rows - 1.0, columns - 1.0, 4.0, 4.0])
        name = "baseline + screen-bounded Gaussian"
    else:
        lower = np.array([0.0, 0.0, -16.0, -16.0, 0.20, 0.20])
        upper = np.array([np.inf, np.inf, 24.0, 24.0, 50.0, 50.0])
        name = "baseline + wide-bound Gaussian"
    result = least_squares(
        lambda p: (gaussian_with_baseline(matrix.shape, p) - matrix).ravel(),
        start,
        bounds=(lower, upper),
        method="trf",
        max_nfev=20000,
    )
    prediction = gaussian_with_baseline(matrix.shape, result.x)
    return result.x, prediction, {
        "model": name,
        "success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "sse": float(np.square(prediction - matrix).sum()),
        "rmse_per_pixel": float(np.sqrt(np.square(prediction - matrix).mean())),
        "parameters_at_bound": bool(
            np.any(np.isclose(result.x, lower, rtol=0, atol=1e-5))
            or np.any(np.isclose(result.x, upper, rtol=0, atol=1e-5))
        ),
    }


def analyze_case(unit_id: int, matrix: np.ndarray) -> tuple[list[dict], dict[str, np.ndarray]]:
    outputs: dict[str, np.ndarray] = {"observed": matrix}
    rows = []

    allen_params, allen_success = fit_2d_gaussian(matrix)
    allen_params = np.asarray(allen_params, dtype=float)
    allen_prediction = gaussian_no_baseline(matrix.shape, allen_params)
    outputs["Allen unbounded"] = allen_prediction
    rows.append(
        {
            "unit_id": unit_id,
            "model": "Allen unbounded, no baseline",
            "success": bool(allen_success),
            "baseline": 0.0,
            "amplitude": allen_params[0],
            "center_y_px": allen_params[1],
            "center_x_px": allen_params[2],
            "sigma_y_px": allen_params[3],
            "sigma_x_px": allen_params[4],
            "height_rf_deg": allen_params[3] * 10.0,
            "width_rf_deg": allen_params[4] * 10.0,
            "sse": float(np.square(allen_prediction - matrix).sum()),
            "rmse_per_pixel": float(np.sqrt(np.square(allen_prediction - matrix).mean())),
            "parameters_at_bound": False,
        }
    )

    for bounded, short_name in ((False, "Baseline + wide bounds"), (True, "Baseline + screen bounds")):
        params, prediction, audit = fit_baseline_model(matrix, bounded=bounded)
        outputs[short_name] = prediction
        rows.append(
            {
                "unit_id": unit_id,
                "model": audit["model"],
                "success": audit["success"],
                "baseline": params[0],
                "amplitude": params[1],
                "center_y_px": params[2],
                "center_x_px": params[3],
                "sigma_y_px": params[4],
                "sigma_x_px": params[5],
                "height_rf_deg": params[4] * 10.0,
                "width_rf_deg": params[5] * 10.0,
                "sse": audit["sse"],
                "rmse_per_pixel": audit["rmse_per_pixel"],
                "parameters_at_bound": audit["parameters_at_bound"],
            }
        )

    mask, center_x, center_y, area_pixels = threshold_rf(matrix, 1.0)
    outputs["Allen threshold mask"] = mask.astype(float)
    for row in rows:
        row.update(
            {
                "threshold_center_x_px": center_x,
                "threshold_center_y_px": center_y,
                "observed_area_pixels": area_pixels,
                "observed_area_deg2": area_pixels * 100.0,
                "observed_mean": float(matrix.mean()),
                "observed_min": float(matrix.min()),
                "observed_max": float(matrix.max()),
            }
        )
    return rows, outputs


def render_figure(case_outputs: dict[int, dict[str, np.ndarray]], metrics: pd.DataFrame, path: Path) -> None:
    columns = ("observed", "Allen unbounded", "Baseline + wide bounds", "Baseline + screen bounds")
    fig, axes = plt.subplots(len(case_outputs), len(columns), figsize=(14.2, 3.6 * len(case_outputs)), squeeze=False)
    for row_index, (unit_id, outputs) in enumerate(case_outputs.items()):
        limit = max(float(outputs["observed"].max()), 1.0)
        for column_index, label in enumerate(columns):
            ax = axes[row_index, column_index]
            artist = ax.imshow(outputs[label], cmap="viridis", vmin=0, vmax=limit, origin="upper")
            if label == "observed":
                title = f"Unit {unit_id}\nobserved spike-count map"
            else:
                selected = metrics.loc[(metrics["unit_id"].eq(unit_id)) & metrics["model"].str.startswith(label.split(" +")[0] if label.startswith("Allen") else "baseline")]
                if label == "Baseline + wide bounds":
                    selected = metrics.loc[(metrics["unit_id"].eq(unit_id)) & metrics["model"].str.contains("wide-bound")]
                elif label == "Baseline + screen bounds":
                    selected = metrics.loc[(metrics["unit_id"].eq(unit_id)) & metrics["model"].str.contains("screen-bounded")]
                else:
                    selected = metrics.loc[(metrics["unit_id"].eq(unit_id)) & metrics["model"].str.startswith("Allen")]
                record = selected.iloc[0]
                title = (
                    f"{label}\nσx={record.sigma_x_px:.2g}, σy={record.sigma_y_px:.2g} px; "
                    f"RMSE={record.rmse_per_pixel:.2f}"
                )
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(9))
            ax.set_yticks(range(9))
            ax.tick_params(labelsize=7)
            fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        "Concrete RF-fit diagnostic: positive baseline versus unconstrained Gaussian scale",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    session = EcephysSession.from_nwb_path(
        args.nwb.resolve(),
        api_kwargs={
            "amplitude_cutoff_maximum": np.inf,
            "presence_ratio_minimum": -np.inf,
            "isi_violations_maximum": np.inf,
            "filter_by_validity": False,
        },
    )
    unit_ids = list(args.unit_ids)
    analysis = ReceptiveFieldMapping(session, filter=unit_ids, mask_threshold=1.0)
    metric_rows = []
    case_outputs = {}
    for unit_id in unit_ids:
        matrix = analysis.get_receptive_field(unit_id).astype(float)
        rows, outputs = analyze_case(unit_id, matrix)
        metric_rows.extend(rows)
        case_outputs[unit_id] = outputs
        np.savetxt(output_dir / f"unit_{unit_id}_observed_map.csv", matrix, delimiter=",", fmt="%.8g")
        np.savetxt(
            output_dir / f"unit_{unit_id}_threshold_mask.csv",
            outputs["Allen threshold mask"], delimiter=",", fmt="%d",
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "fit_comparison.csv", index=False, float_format="%.8g")
    figure_path = output_dir / "Figure_concrete_rf_fit_diagnostic.png"
    render_figure(case_outputs, metrics, figure_path)

    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "checkpoint": "initial concrete evidence",
        "nwb": {"path": str(args.nwb.resolve()), "sha256": sha256(args.nwb.resolve())},
        "unit_ids": unit_ids,
        "models": {
            "allen": "five-parameter unbounded least-squares Gaussian without baseline",
            "baseline_wide": "nonnegative baseline/amplitude; center [-16,24], sigma [0.2,50] pixels",
            "baseline_screen_bounded": "nonnegative baseline/amplitude; center [0,8], sigma [0.35,4] pixels",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote initial RF-fit diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
