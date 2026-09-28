#!/usr/bin/env python3
"""Render MouseV2 probe tracks from the refreshed NWB electrode CCF coordinates.

The refreshed DANDI:001568 draft adds x/y/z and atlas location to every row of
the NWB electrodes table. This script uses AP-band electrodes only (excluding
the duplicate LFP subset), fits one total-least-squares line per session/probe,
and retains non-void coordinates for track fitting and display.

CCF y is the dorsal-ventral coordinate: increasing probe-relative position
(tip to surface) consistently moves toward smaller y. The NWBs call the other
two dimensions only x and z, so figures preserve those names rather than
silently assigning AP and ML without external pipeline provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "figure3_mousev2.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "figure3" / "06r_mousev2_probe_tracks_from_ccf"
)
PROBE_ORDER = ("ProbeA", "ProbeB", "ProbeC", "ProbeE")
PROBE_COLORS = {
    "ProbeA": "#d73027",
    "ProbeB": "#fc8d59",
    "ProbeC": "#66bd63",
    "ProbeE": "#4575b4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode(errors="replace") if isinstance(value, bytes) else str(value) for value in values]
    )


def read_float64(fid: object, path: str, shape: tuple[int, ...]) -> np.ndarray:
    """Read NWB long-double-like columns through an explicit float64 memory type."""
    import h5py

    dataset = h5py.h5d.open(fid, path.encode())
    values = np.empty(shape, dtype="<f8")
    memory_type = h5py.h5t.py_create(values.dtype, logical=True)
    dataset.read(h5py.h5s.ALL, h5py.h5s.ALL, values, memory_type)
    return values


def read_int64(fid: object, path: str, shape: tuple[int, ...]) -> np.ndarray:
    """Read NWB integer columns through an explicit standard memory type."""
    import h5py

    dataset = h5py.h5d.open(fid, path.encode())
    values = np.empty(shape, dtype=np.int64)
    dataset.read(h5py.h5s.ALL, h5py.h5s.ALL, values, h5py.h5t.NATIVE_INT64)
    return values


def fit_track(points_um: np.ndarray, rel_y_um: np.ndarray) -> dict[str, object]:
    """Fit a straight 3-D track and orient it from probe tip toward surface."""
    centroid = points_um.mean(axis=0)
    centered = points_um - centroid
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projection = centered @ direction
    if np.corrcoef(projection, rel_y_um)[0, 1] < 0:
        direction = -direction
        projection = -projection
    r2 = float(singular_values[0] ** 2 / np.sum(singular_values**2))
    return {
        "centroid": centroid,
        "direction": direction,
        "projection": projection,
        "r2_colinearity": r2,
        "angle_from_ccf_y_deg": float(
            np.degrees(np.arccos(np.clip(abs(direction[1]), -1.0, 1.0)))
        ),
        "ccf_path_length_um": float(np.ptp(projection)),
    }


def extract_session(
    nwb_path: Path, session: dict[str, object]
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    import h5py

    fid = h5py.h5f.open(str(nwb_path).encode(), flags=h5py.h5f.ACC_RDONLY)
    with h5py.File(nwb_path, "r") as nwb:
        table = nwb["/general/extracellular_ephys/electrodes"]
        n_rows = len(table["location"])
        columns = {
            name: read_float64(
                fid, f"/general/extracellular_ephys/electrodes/{name}", (n_rows,)
            )
            for name in ("x", "y", "z", "rel_y")
        }
        frame = pd.DataFrame(columns)
        frame["electrode_row"] = np.arange(n_rows, dtype=int)
        frame["channel_name"] = decode(table["channel_name"][:])
        frame["probe"] = decode(table["group_name"][:])
        frame["location"] = decode(table["location"][:])

        units = nwb["/units"]
        n_units = len(units["id"])
        unit_locations = pd.DataFrame({
            "nwb_unit_id": read_int64(fid, "/units/id", (n_units,)),
            "probe": decode(units["device_name"][:]),
            "extremum_channel_index": read_int64(
                fid, "/units/extremum_channel_index", (n_units,)
            ),
            "unit_depth_um": read_float64(fid, "/units/depth", (n_units,)),
        })
    fid.close()

    frame = frame.loc[frame.channel_name.str.startswith("AP")].copy()
    frame["extremum_channel_index"] = (
        frame["channel_name"].str.removeprefix("AP").astype(int) - 1
    )
    frame.insert(0, "site", str(session["site"]))
    frame.insert(0, "subject_id", int(session["subject_id"]))

    rows: list[dict[str, object]] = []
    for probe in PROBE_ORDER:
        full = frame.loc[frame.probe.eq(probe)].sort_values("rel_y")
        inside = full.loc[full.location.ne("void")].copy()
        if len(inside) < 20:
            raise ValueError(
                f"{session['subject_id']} {probe}: only {len(inside)} non-void AP electrodes"
            )
        points = inside[["x", "y", "z"]].to_numpy(float)
        fit = fit_track(points, inside.rel_y.to_numpy(float))
        direction = np.asarray(fit["direction"])
        centroid = np.asarray(fit["centroid"])
        projection = np.asarray(fit["projection"])
        deep = centroid + projection.min() * direction
        surface = centroid + projection.max() * direction
        rows.append(
            {
                "site": session["site"],
                "subject_id": int(session["subject_id"]),
                "probe": probe,
                "n_ap_electrodes": len(full),
                "n_nonvoid_electrodes": len(inside),
                "n_visp_electrodes": int(inside.location.str.startswith("VISp").sum()),
                "n_structures": int(inside.location.nunique()),
                "r2_colinearity": fit["r2_colinearity"],
                "angle_from_ccf_y_deg": fit["angle_from_ccf_y_deg"],
                "ccf_path_length_um": fit["ccf_path_length_um"],
                "rel_y_span_um": float(inside.rel_y.max() - inside.rel_y.min()),
                "direction_x": direction[0],
                "direction_y": direction[1],
                "direction_z": direction[2],
                "deep_x_um": deep[0],
                "deep_y_um": deep[1],
                "deep_z_um": deep[2],
                "surface_x_um": surface[0],
                "surface_y_um": surface[1],
                "surface_z_um": surface[2],
            }
        )
    electrode_lookup = frame[
        [
            "probe", "extremum_channel_index", "electrode_row", "channel_name",
            "rel_y", "x", "y", "z", "location",
        ]
    ]
    unit_locations = unit_locations.merge(
        electrode_lookup,
        on=["probe", "extremum_channel_index"],
        how="left",
        validate="many_to_one",
    )
    if unit_locations["location"].isna().any():
        raise ValueError(
            f"{session['subject_id']}: some units do not map to an AP extremum channel"
        )
    unit_locations.insert(0, "site", str(session["site"]))
    unit_locations.insert(1, "site_number", int(session["site_number"]))
    unit_locations.insert(2, "subject_id", int(session["subject_id"]))
    return frame, rows, unit_locations


def set_equal_3d_aspect(ax, points: np.ndarray) -> None:
    spans = np.ptp(np.asarray(points, dtype=np.float64), axis=0)
    spans = np.where(spans > 1e-6, spans, 1e-6)
    ax.set_box_aspect(tuple(float(value) for value in spans))


def draw_track(ax, row: pd.Series, *, alpha: float = 1.0, label: str | None = None) -> None:
    color = PROBE_COLORS[row.probe]
    deep = np.asarray(
        [row.deep_x_um, row.deep_z_um, row.deep_y_um], dtype=np.float64
    ) / 1000.0
    surface = np.asarray(
        [row.surface_x_um, row.surface_z_um, row.surface_y_um], dtype=np.float64
    ) / 1000.0
    ax.plot(
        [float(deep[0]), float(surface[0])],
        [float(deep[1]), float(surface[1])],
        [float(deep[2]), float(surface[2])],
        color=color,
        linewidth=2.0,
        alpha=alpha,
        label=label,
    )


def style_ccf_axis(ax) -> None:
    ax.invert_zaxis()
    ax.set(
        xlabel="CCF x (mm)",
        ylabel="CCF z (mm)",
        zlabel="CCF y (mm; + ventral)",
    )


def make_3d_figure(table: pd.DataFrame, points: pd.DataFrame, output_path: Path) -> int:
    session_quality = table.groupby("subject_id").r2_colinearity.min()
    representative = int(session_quality.idxmax())
    rep_table = table.loc[table.subject_id.eq(representative)]
    rep_points = points.loc[
        points.subject_id.eq(representative) & points.location.ne("void")
    ]

    fig = plt.figure(figsize=(18, 6.5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for probe in PROBE_ORDER:
        row = rep_table.loc[rep_table.probe.eq(probe)].iloc[0]
        sample = rep_points.loc[rep_points.probe.eq(probe)]
        sample_xyz = sample[["x", "z", "y"]].to_numpy(dtype=np.float64) / 1000.0
        ax.scatter(
            sample_xyz[:, 0],
            sample_xyz[:, 1],
            sample_xyz[:, 2],
            s=5,
            alpha=0.35,
            color=PROBE_COLORS[probe],
            edgecolor="none",
        )
        draw_track(ax, row, label=probe.replace("Probe", ""))
        ax.scatter(
            [float(row.surface_x_um) / 1000],
            [float(row.surface_z_um) / 1000],
            [float(row.surface_y_um) / 1000],
            color=PROBE_COLORS[probe],
            marker="^",
            s=45,
        )
    set_equal_3d_aspect(ax, rep_points[["x", "z", "y"]].to_numpy() / 1000)
    style_ccf_axis(ax)
    ax.set_title(
        f"Example subject {representative}\n"
        "highest minimum track r²\n(▲ = dorsal non-void endpoint)"
    )
    ax.legend(frameon=False, title="probe")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    for _, row in table.iterrows():
        draw_track(ax, row, alpha=0.42)
    endpoints = table[["deep_x_um", "deep_z_um", "deep_y_um"]].to_numpy() / 1000
    endpoints2 = table[["surface_x_um", "surface_z_um", "surface_y_um"]].to_numpy() / 1000
    set_equal_3d_aspect(ax, np.vstack([endpoints, endpoints2]))
    style_ccf_axis(ax)
    ax.set_title("All 32 localized V1 probe tracks\nin absolute CCF coordinates")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    length_mm = 4.0
    for _, row in table.iterrows():
        tip = np.asarray(
            [row.direction_x, row.direction_z, row.direction_y], dtype=np.float64
        ) * length_mm
        ax.plot(
            [0, tip[0]],
            [0, tip[1]],
            [0, tip[2]],
            color=PROBE_COLORS[row.probe],
            linewidth=1.2,
            alpha=0.45,
        )
    ax.plot([0, 0], [0, 0], [0, -length_mm], "k--", linewidth=1.4)
    ax.set_box_aspect((1, 1, 1))
    ax.set(
        xlim=(-length_mm, length_mm),
        ylim=(-length_mm, length_mm),
        zlim=(-length_mm, length_mm),
    )
    style_ccf_axis(ax)
    ax.set_title("All insertion directions\ncommon origin (tip → surface)")
    handles = [
        plt.Line2D([0], [0], color=PROBE_COLORS[p], lw=2, label=p.replace("Probe", ""))
        for p in PROBE_ORDER
    ]
    ax.legend(handles=handles, frameon=False, title="probe")

    fig.suptitle("MouseV2 V1 probe trajectories in 3-D CCF space", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return representative


def make_session_grid(table: pd.DataFrame, output_path: Path) -> None:
    subjects = sorted(table.subject_id.unique())
    fig = plt.figure(figsize=(17, 10.5))
    for index, subject in enumerate(subjects, start=1):
        ax = fig.add_subplot(2, 4, index, projection="3d")
        sub = table.loc[table.subject_id.eq(subject)]
        all_endpoints = []
        for probe in PROBE_ORDER:
            row = sub.loc[sub.probe.eq(probe)].iloc[0]
            draw_track(ax, row)
            all_endpoints.extend(
                [
                    [row.deep_x_um, row.deep_z_um, row.deep_y_um],
                    [row.surface_x_um, row.surface_z_um, row.surface_y_um],
                ]
            )
        set_equal_3d_aspect(ax, np.asarray(all_endpoints) / 1000)
        style_ccf_axis(ax)
        ax.set(xlabel="", ylabel="", zlabel="")
        ax.set_title(
            f"subject {subject}\nmin r²={sub.r2_colinearity.min():.3f}",
            pad=8,
        )
        ax.tick_params(labelsize=7)
    handles = [
        plt.Line2D([0], [0], color=PROBE_COLORS[p], lw=2, label=p.replace("Probe", ""))
        for p in PROBE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        title="probe",
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle("MouseV2 V1 CCF probe tracks by subject", fontsize=15, y=0.99)
    fig.text(
        0.5,
        0.015,
        "Displayed axes: CCF x, CCF z, CCF y (+ventral downward)",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.86, wspace=0.12, hspace=0.30)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_summary_figure(table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    rng = np.random.default_rng(0)
    for index, probe in enumerate(PROBE_ORDER):
        sub = table.loc[table.probe.eq(probe)]
        jitter = rng.uniform(-0.14, 0.14, len(sub))
        axes[0].scatter(
            index + jitter,
            sub.angle_from_ccf_y_deg,
            color=PROBE_COLORS[probe],
            s=30,
            alpha=0.75,
        )
        axes[0].hlines(
            sub.angle_from_ccf_y_deg.median(), index - 0.25, index + 0.25, color="black", lw=2
        )
    axes[0].set_xticks(range(len(PROBE_ORDER)), [p.replace("Probe", "") for p in PROBE_ORDER])
    axes[0].set(
        xlabel="probe",
        ylabel="angle from CCF y axis (deg)",
        title="Insertion angle",
    )

    axes[1].scatter(
        table.rel_y_span_um / 1000,
        table.ccf_path_length_um / 1000,
        c=table.r2_colinearity,
        cmap="viridis",
        vmin=0.95,
        vmax=1,
        s=35,
    )
    limit = max(table.rel_y_span_um.max(), table.ccf_path_length_um.max()) / 1000 * 1.05
    axes[1].plot([0, limit], [0, limit], "k--", lw=1)
    axes[1].set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="probe-relative non-void span (mm)",
        ylabel="CCF fitted path length (mm)",
        title="Path-length sanity check",
    )

    axes[2].hist(table.r2_colinearity, bins=np.linspace(0.9, 1, 21), color="#6f6f6f")
    axes[2].axvline(table.r2_colinearity.median(), color="black", lw=2)
    axes[2].set(
        xlabel="track r² colinearity",
        ylabel="session–probe tracks",
        title=f"Fit quality (median={table.r2_colinearity.median():.3f})",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("MouseV2 V1 CCF track validation (n=32 tracks)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(table: pd.DataFrame, representative: int, output_path: Path) -> None:
    by_probe = table.groupby("probe").agg(
        tracks=("subject_id", "size"),
        median_angle_deg=("angle_from_ccf_y_deg", "median"),
        median_r2=("r2_colinearity", "median"),
        min_r2=("r2_colinearity", "min"),
    ).reindex(PROBE_ORDER)
    lines = [
        "# MouseV2 V1 probe tracks from refreshed CCF electrodes",
        "",
        "All 32 session–probe tracks were fit to non-void AP-electrode CCF coordinates from "
        "the refreshed DANDI:001568 draft. LFP rows were excluded because they duplicate an AP subset.",
        "",
        f"- Representative subject: {representative}, selected algorithmically by the highest "
        "minimum track r² across its four probes.",
        f"- Median track r²: {table.r2_colinearity.median():.4f}; minimum: "
        f"{table.r2_colinearity.min():.4f}.",
        f"- Median angle from CCF y: {table.angle_from_ccf_y_deg.median():.1f} deg "
        f"(range {table.angle_from_ccf_y_deg.min():.1f}–{table.angle_from_ccf_y_deg.max():.1f}).",
        "- CCF y is treated as dorsal–ventral because every track moves toward smaller y from tip "
        "to surface. The NWBs do not name the anatomical meanings of x and z, so figures retain "
        "the raw CCF axis names.",
        "",
        "## By probe",
        "",
        "```text",
        by_probe.to_string(float_format=lambda value: f"{value:.3f}"),
        "```",
        "",
        "## Outputs",
        "",
        "- `Figure_mousev2_probe_tracks_3d.png`: representative, absolute pooled, and common-origin views.",
        "- `Figure_mousev2_probe_tracks_by_session.png`: all eight subjects shown separately.",
        "- `Figure_mousev2_probe_track_summary.png`: angle, span, and fit-quality diagnostics.",
        "- `mousev2_probe_track_fits.csv`: one row per session–probe track.",
        "- `mousev2_probe_track_points.csv`: source non-void AP-electrode points used for fitting.",
        "- `mousev2_unit_ccf_locations.csv`: every analysis unit mapped through its NWB "
        "extremum AP channel to CCF coordinates and atlas location.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = json.loads(config_path.read_text())
    nwb_root = Path(config["nwb_input"]["default_root"])
    output.mkdir(parents=True, exist_ok=True)

    point_frames: list[pd.DataFrame] = []
    unit_location_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    inputs = []
    for session in config["sessions"]:
        path = nwb_root / str(session["nwb_relative_path"])
        if path.stat().st_size != int(session["expected_nwb_bytes"]):
            raise ValueError(f"NWB size drift: {path}")
        points, rows, unit_locations = extract_session(path, session)
        unit_locations["unit_id"] = (
            int(session["site_number"]) * 1_000_000 + unit_locations["nwb_unit_id"]
        )
        layer_path = ROOT / config["data_directory"] / f"{session['site']}_processed/layer_info.csv"
        analysis_unit_ids = pd.read_csv(layer_path, usecols=["unit_id"])["unit_id"]
        if set(unit_locations["unit_id"]) != set(analysis_unit_ids):
            raise ValueError(f"NWB/analysis unit-ID reconciliation failed: {session['site']}")
        point_frames.append(points.loc[points.location.ne("void")])
        unit_location_frames.append(unit_locations)
        fit_rows.extend(rows)
        inputs.append({"path": str(path), "bytes": path.stat().st_size})

    points = pd.concat(point_frames, ignore_index=True)
    unit_locations = pd.concat(unit_location_frames, ignore_index=True)
    table = pd.DataFrame(fit_rows).sort_values(["subject_id", "probe"]).reset_index(drop=True)
    if len(table) != 32:
        raise ValueError(f"Expected 32 tracks, found {len(table)}")

    table.to_csv(output / "mousev2_probe_track_fits.csv", index=False, float_format="%.6g")
    points.to_csv(output / "mousev2_probe_track_points.csv", index=False, float_format="%.6g")
    unit_locations.to_csv(
        output / "mousev2_unit_ccf_locations.csv", index=False, float_format="%.6g"
    )
    representative = make_3d_figure(
        table, points, output / "Figure_mousev2_probe_tracks_3d.png"
    )
    make_session_grid(table, output / "Figure_mousev2_probe_tracks_by_session.png")
    make_summary_figure(table, output / "Figure_mousev2_probe_track_summary.png")
    write_report(table, representative, output / "MOUSEV2_PROBE_TRACKS_FROM_CCF.md")

    manifest = {
        "schema_version": 1,
        "status": "32 refreshed-NWB CCF electrode tracks rendered",
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "nwb_root": str(nwb_root),
        "inputs": inputs,
        "selection": {
            "representative_subject": representative,
            "criterion": "highest minimum r2_colinearity across four probes",
        },
        "methods": {
            "electrode_rows": "AP channels only; LFP duplicates excluded",
            "fit_rows": "location != void",
            "fit": "total least squares / PCA",
            "orientation": "increasing rel_y, probe tip to surface",
            "coordinate_labels": "raw NWB CCF x/y/z; y supported as DV; x/z not renamed",
        },
        "summary": {
            "tracks": len(table),
            "subjects": int(table.subject_id.nunique()),
            "median_r2": float(table.r2_colinearity.median()),
            "minimum_r2": float(table.r2_colinearity.min()),
            "median_angle_from_ccf_y_deg": float(table.angle_from_ccf_y_deg.median()),
            "units": len(unit_locations),
            "units_on_visp_channels": int(unit_locations.location.str.startswith("VISp").sum()),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(table.groupby("probe")[["angle_from_ccf_y_deg", "r2_colinearity"]].median())
    print(f"representative subject: {representative}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
