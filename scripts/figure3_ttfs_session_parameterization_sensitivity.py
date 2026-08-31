"""TTFS checkpoint: non-centered session effects and no-V1-session sensitivity."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure3_hierarchical_identity_model import (  # noqa: E402
    DEFAULT_INPUT,
    build_design,
    log_interaction_density,
    log_scale_density,
    slice_sample,
    summarize_chains,
)


DEFAULT_OUTPUT = ROOT / "Figure3" / "hierarchical_error_model" / "ttfs_session_sensitivity"
matplotlib.rcParams["pdf.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--draws", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def sample_coefficients(
    y: np.ndarray,
    known_se: np.ndarray,
    design: np.ndarray,
    prior_precision: np.ndarray,
    sigma_interaction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = 1.0 / (np.square(known_se) + sigma_interaction**2)
    precision = design.T @ (weights[:, None] * design)
    precision.flat[:: precision.shape[0] + 1] += prior_precision
    rhs = design.T @ (weights * y)
    chol = np.linalg.cholesky(precision)
    mean = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    return mean + np.linalg.solve(chol.T, rng.normal(size=len(mean)))


def log_noncentered_session_density(
    log_sigma: float,
    y_minus_fixed: np.ndarray,
    session_signal: np.ndarray,
    known_se: np.ndarray,
    sigma_interaction: float,
) -> float:
    sigma = np.exp(log_sigma)
    variance = np.square(known_se) + sigma_interaction**2
    residual = y_minus_fixed - sigma * session_signal
    return float(
        -0.5 * np.sum(np.log(variance) + np.square(residual) / variance)
        -0.5 * sigma**2
        + log_sigma
    )


def run_noncentered_chain(
    table: pd.DataFrame, *, scale: float, tune: int, draws: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (table["mean"].to_numpy(float) - table["mean"].mean()) / scale
    known_se = table["bootstrap_se"].to_numpy(float) / scale
    base_design, group_columns, _, sessions = build_design(table)
    session_start = int(group_columns[-1] + 1)
    session_columns = np.arange(session_start, base_design.shape[1])
    sigmas = np.array([0.35, 0.50, 0.35], dtype=float)
    beta = np.zeros(base_design.shape[1])
    kept = {"sigma_identity": [], "sigma_session": [], "sigma_interaction": []}
    for iteration in range(tune + draws):
        design = base_design.copy()
        design[:, session_columns] *= sigmas[1]
        prior = np.zeros(design.shape[1])
        prior[0] = 1 / 25
        prior[group_columns] = 1 / sigmas[0] ** 2
        prior[session_columns] = 1.0
        beta = sample_coefficients(y, known_se, design, prior, sigmas[2], rng)
        group_effects = beta[group_columns]
        z_session = beta[session_columns]
        fixed = base_design[:, :session_start] @ beta[:session_start]
        session_signal = base_design[:, session_columns] @ z_session
        sigmas[0] = np.exp(slice_sample(
            np.log(sigmas[0]), lambda value: log_scale_density(value, group_effects), rng
        ))
        sigmas[1] = np.exp(slice_sample(
            np.log(sigmas[1]),
            lambda value: log_noncentered_session_density(
                value, y - fixed, session_signal, known_se, sigmas[2]
            ), rng,
        ))
        residual = y - fixed - sigmas[1] * session_signal
        sigmas[2] = np.exp(slice_sample(
            np.log(sigmas[2]),
            lambda value: log_interaction_density(value, residual, known_se), rng,
        ))
        if iteration >= tune:
            kept["sigma_identity"].append(sigmas[0] * scale)
            kept["sigma_session"].append(sigmas[1] * scale)
            kept["sigma_interaction"].append(sigmas[2] * scale)
    return {key: np.asarray(value) for key, value in kept.items()}


def run_no_session_chain(
    table: pd.DataFrame, *, scale: float, tune: int, draws: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (table["mean"].to_numpy(float) - table["mean"].mean()) / scale
    known_se = table["bootstrap_se"].to_numpy(float) / scale
    full_design, group_columns, _, _ = build_design(table)
    design = full_design[:, : int(group_columns[-1] + 1)]
    sigmas = np.array([0.35, 0.35], dtype=float)
    beta = np.zeros(design.shape[1])
    kept = {"sigma_identity": [], "sigma_interaction": []}
    for iteration in range(tune + draws):
        prior = np.zeros(design.shape[1])
        prior[0] = 1 / 25
        prior[group_columns] = 1 / sigmas[0] ** 2
        beta = sample_coefficients(y, known_se, design, prior, sigmas[1], rng)
        sigmas[0] = np.exp(slice_sample(
            np.log(sigmas[0]), lambda value: log_scale_density(value, beta[group_columns]), rng
        ))
        residual = y - design @ beta
        sigmas[1] = np.exp(slice_sample(
            np.log(sigmas[1]),
            lambda value: log_interaction_density(value, residual, known_se), rng,
        ))
        if iteration >= tune:
            kept["sigma_identity"].append(sigmas[0] * scale)
            kept["sigma_interaction"].append(sigmas[1] * scale)
    return {key: np.asarray(value) for key, value in kept.items()}


def stack_chains(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([result[key] for result in results]) for key in results[0]}


def identity_contrast(
    v1: np.ndarray, hva: np.ndarray, *, rng: np.random.Generator, specification: str
) -> dict[str, float | str]:
    v1_flat = v1.reshape(-1)
    hva_flat = hva.reshape(-1)
    count = min(len(v1_flat), len(hva_flat))
    difference = rng.choice(hva_flat, count, replace=False) - rng.choice(v1_flat, count, replace=False)
    return {
        "specification": specification,
        "v1_identity_median": float(np.median(v1_flat)),
        "hva_identity_median": float(np.median(hva_flat)),
        "delta_mean": float(np.mean(difference)),
        "delta_median": float(np.median(difference)),
        "delta_ci_low": float(np.percentile(difference, 2.5)),
        "delta_ci_high": float(np.percentile(difference, 97.5)),
        "probability_hva_gt_v1": float(np.mean(difference > 0)),
    }


def render_comparison(contrasts: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(contrasts))[::-1]
    for position, row in zip(y, contrasts.itertuples(index=False)):
        ax.errorbar(
            row.delta_median, position,
            xerr=[[row.delta_median - row.delta_ci_low], [row.delta_ci_high - row.delta_median]],
            fmt="o", color="#2f5597", capsize=4, lw=1.6,
        )
        ax.text(row.delta_ci_high + 0.12, position, f"P(HVA>V1)={row.probability_hva_gt_v1:.3f}",
                va="center", fontsize=8)
    ax.axvline(0, color="#777777", lw=1, ls="--")
    ax.set_yticks(y, contrasts["specification"])
    ax.set_xlabel("HVA − V1 stable identity SD (ms)")
    ax.set_title("TTFS session-component sensitivity", loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.input, dtype={"session_id": str})
    table = table.loc[(table["metric"] == "TTFS (ms)") & (table["n_units"] >= args.min_units)]
    scale = float(table["centered_mean"].std(ddof=1))
    seeds = np.random.SeedSequence(args.seed).spawn(3 * args.chains)
    v1 = table.loc[table["dataset"] == "Within-V1"].copy()
    hva = table.loc[table["dataset"] == "Post-V1"].copy()
    v1_nc = stack_chains([
        run_noncentered_chain(v1, scale=scale, tune=args.tune, draws=args.draws,
                              seed=int(seed.generate_state(1)[0]))
        for seed in seeds[:args.chains]
    ])
    hva_nc = stack_chains([
        run_noncentered_chain(hva, scale=scale, tune=args.tune, draws=args.draws,
                              seed=int(seed.generate_state(1)[0]))
        for seed in seeds[args.chains:2 * args.chains]
    ])
    v1_no = stack_chains([
        run_no_session_chain(v1, scale=scale, tune=args.tune, draws=args.draws,
                             seed=int(seed.generate_state(1)[0]))
        for seed in seeds[2 * args.chains:]
    ])
    summaries = []
    for specification, dataset, chains in [
        ("non-centered session", "Within-V1", v1_nc),
        ("non-centered session", "Post-V1", hva_nc),
        ("V1 session SD fixed to zero", "Within-V1", v1_no),
    ]:
        local = summarize_chains(chains)
        local.insert(0, "dataset", dataset)
        local.insert(0, "specification", specification)
        summaries.append(local)
    summary = pd.concat(summaries, ignore_index=True)
    rng = np.random.default_rng(args.seed + 1)
    contrasts = pd.DataFrame([
        identity_contrast(v1_nc["sigma_identity"], hva_nc["sigma_identity"], rng=rng,
                          specification="Non-centered session effects"),
        identity_contrast(v1_no["sigma_identity"], hva_nc["sigma_identity"], rng=rng,
                          specification="V1 session SD fixed to zero"),
    ])
    summary.to_csv(output / "parameterization_summary.csv", index=False)
    contrasts.to_csv(output / "identity_contrast_sensitivity.csv", index=False)
    np.savez_compressed(
        output / "sensitivity_chains.npz",
        **{f"v1_noncentered_{k}": v for k, v in v1_nc.items()},
        **{f"hva_noncentered_{k}": v for k, v in hva_nc.items()},
        **{f"v1_no_session_{k}": v for k, v in v1_no.items()},
    )
    render_comparison(contrasts, output / "identity_contrast_sensitivity.png")
    max_rhat = float(summary["split_rhat"].max())
    min_ess = float(summary["ess"].min())
    lines = [
        "# TTFS session-variance parameterization checkpoint", "",
        f"_Generated {date.today().isoformat()}; {args.chains} chains × {args.draws} draws after {args.tune} tuning._", "",
        "The non-centered fit is the same scientific model as the centered pilot. The second",
        "specification fixes only the V1 session SD to zero while retaining the non-centered HVA fit.", "",
        "| Specification | V1 identity median | HVA identity median | Median HVA−V1 | 95% interval | P(HVA>V1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.specification} | {row.v1_identity_median:.3f} | {row.hva_identity_median:.3f} | "
            f"{row.delta_median:+.3f} | [{row.delta_ci_low:+.3f}, {row.delta_ci_high:+.3f}] | "
            f"{row.probability_hva_gt_v1:.3f} |"
        )
    lines += [
        "", "## Convergence gate", "",
        f"Maximum split-Rhat = {max_rhat:.4f}; minimum ESS = {min_ess:.0f}.",
        "Production extension requires split-Rhat < 1.01 for every sampled component and",
        "adequate ESS, plus a stable identity contrast across the two specifications.",
    ]
    (output / "SESSION_PARAMETERIZATION_CHECKPOINT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
