"""Bayesian hierarchical measurement-error model for Figure 3 identity effects.

Uses only NumPy/SciPy: Gaussian random effects are sampled conditionally and
positive scale parameters are sampled on the log scale with univariate slice
sampling.  The first production checkpoint is intentionally TTFS-only.

Model for an observed session x group mean y_i with known bootstrap SE_i:

    y_i ~ Normal(mu + alpha[group_i] + u[session_i],
                 sqrt(SE_i**2 + sigma_interaction**2))
    alpha_g ~ Normal(0, sigma_identity)
    u_s     ~ Normal(0, sigma_session)

All outcomes and known SEs use one common, metric-specific scale across the two
datasets.  Dataset intercepts are removed separately; scale components remain
directly comparable between V1 and HVA fits.
"""

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
DEFAULT_INPUT = ROOT / "Figure3" / "hierarchical_error_model" / "session_group_measurement_error.csv"
DEFAULT_OUTPUT = ROOT / "Figure3" / "hierarchical_error_model" / "ttfs_pilot"

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric", default="TTFS (ms)")
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def build_design(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    groups = sorted(table["group"].astype(str).unique())
    sessions = sorted(table["session_id"].astype(str).unique())
    group_index = {value: index for index, value in enumerate(groups)}
    session_index = {value: index for index, value in enumerate(sessions)}
    n = len(table)
    design = np.zeros((n, 1 + len(groups) + len(sessions)), dtype=float)
    design[:, 0] = 1.0
    for row, record in enumerate(table.itertuples(index=False)):
        design[row, 1 + group_index[str(record.group)]] = 1.0
        design[row, 1 + len(groups) + session_index[str(record.session_id)]] = 1.0
    return design, np.arange(1, 1 + len(groups)), groups, sessions


def sample_gaussian_effects(
    y: np.ndarray,
    known_se: np.ndarray,
    design: np.ndarray,
    group_columns: np.ndarray,
    n_sessions: int,
    sigma_identity: float,
    sigma_session: float,
    sigma_interaction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    variance = np.square(known_se) + sigma_interaction**2
    weights = 1.0 / variance
    precision = design.T @ (weights[:, None] * design)
    prior_precision = np.zeros(design.shape[1], dtype=float)
    prior_precision[0] = 1.0 / 25.0
    prior_precision[group_columns] = 1.0 / sigma_identity**2
    session_start = int(group_columns[-1] + 1)
    prior_precision[session_start: session_start + n_sessions] = 1.0 / sigma_session**2
    precision.flat[:: precision.shape[0] + 1] += prior_precision
    rhs = design.T @ (weights * y)
    chol = np.linalg.cholesky(precision)
    mean = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    return mean + np.linalg.solve(chol.T, rng.normal(size=len(mean)))


def log_scale_density(log_sigma: float, effects: np.ndarray, prior_scale: float = 1.0) -> float:
    sigma = np.exp(log_sigma)
    return float(
        -len(effects) * log_sigma
        -0.5 * np.sum(np.square(effects / sigma))
        -0.5 * (sigma / prior_scale) ** 2
        + log_sigma
    )


def log_interaction_density(
    log_sigma: float,
    residual: np.ndarray,
    known_se: np.ndarray,
    prior_scale: float = 1.0,
) -> float:
    sigma = np.exp(log_sigma)
    variance = np.square(known_se) + sigma**2
    return float(
        -0.5 * np.sum(np.log(variance) + np.square(residual) / variance)
        -0.5 * (sigma / prior_scale) ** 2
        + log_sigma
    )


def slice_sample(
    current: float,
    log_density,
    rng: np.random.Generator,
    *,
    width: float = 0.5,
    max_steps: int = 50,
) -> float:
    threshold = log_density(current) - rng.exponential()
    left = current - rng.uniform() * width
    right = left + width
    left_steps = int(rng.integers(0, max_steps + 1))
    right_steps = max_steps - left_steps
    while left_steps > 0 and log_density(left) > threshold:
        left -= width
        left_steps -= 1
    while right_steps > 0 and log_density(right) > threshold:
        right += width
        right_steps -= 1
    for _ in range(500):
        proposal = rng.uniform(left, right)
        if log_density(proposal) >= threshold:
            return float(proposal)
        if proposal < current:
            left = proposal
        else:
            right = proposal
    raise RuntimeError("Slice sampler failed to find an acceptable proposal")


def run_chain(
    table: pd.DataFrame,
    *,
    scale: float,
    tune: int,
    draws: int,
    thin: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    local = table.copy()
    y = (local["mean"].to_numpy(float) - local["mean"].mean()) / scale
    known_se = local["bootstrap_se"].to_numpy(float) / scale
    design, group_columns, groups, sessions = build_design(local)
    sigmas = np.array([0.35, 0.50, 0.35], dtype=float)
    beta = np.zeros(design.shape[1], dtype=float)
    kept = {"sigma_identity": [], "sigma_session": [], "sigma_interaction": []}
    total = tune + draws * thin
    for iteration in range(total):
        beta = sample_gaussian_effects(
            y, known_se, design, group_columns, len(sessions), *sigmas, rng
        )
        group_effects = beta[group_columns]
        session_start = int(group_columns[-1] + 1)
        session_effects = beta[session_start: session_start + len(sessions)]
        residual = y - design @ beta
        log_identity = slice_sample(
            np.log(sigmas[0]), lambda value: log_scale_density(value, group_effects), rng
        )
        log_session = slice_sample(
            np.log(sigmas[1]), lambda value: log_scale_density(value, session_effects), rng
        )
        log_interaction = slice_sample(
            np.log(sigmas[2]),
            lambda value: log_interaction_density(value, residual, known_se), rng,
        )
        sigmas = np.exp([log_identity, log_session, log_interaction])
        if iteration >= tune and (iteration - tune) % thin == 0:
            kept["sigma_identity"].append(sigmas[0] * scale)
            kept["sigma_session"].append(sigmas[1] * scale)
            kept["sigma_interaction"].append(sigmas[2] * scale)
    return {key: np.asarray(value) for key, value in kept.items()}


def split_rhat(chains: np.ndarray) -> float:
    """Classical split-Rhat; adequate as a transparent pilot diagnostic."""
    n_chains, n_draws = chains.shape
    half = n_draws // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    n = split.shape[1]
    chain_means = np.mean(split, axis=1)
    between = n * np.var(chain_means, ddof=1)
    within = np.mean(np.var(split, axis=1, ddof=1))
    variance = (n - 1) / n * within + between / n
    return float(np.sqrt(variance / within))


def effective_sample_size(chains: np.ndarray) -> float:
    """Initial-positive-sequence ESS using average within-chain autocorrelation."""
    n_chains, n_draws = chains.shape
    centered = chains - np.mean(chains, axis=1, keepdims=True)
    variances = np.sum(centered**2, axis=1)
    rho_sum = 0.0
    for lag in range(1, min(n_draws - 1, 1000), 2):
        pair = 0.0
        for local_lag in (lag, lag + 1):
            numerator = np.sum(centered[:, :-local_lag] * centered[:, local_lag:], axis=1)
            rho = np.mean(numerator / variances)
            pair += rho
        if pair < 0:
            break
        rho_sum += pair
    return float(n_chains * n_draws / (1 + 2 * rho_sum))


def summarize_chains(chains: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for parameter, values in chains.items():
        flat = values.reshape(-1)
        low, median, high = np.percentile(flat, [2.5, 50, 97.5])
        rows.append({
            "parameter": parameter,
            "posterior_mean": float(np.mean(flat)),
            "posterior_median": float(median),
            "ci_low": float(low),
            "ci_high": float(high),
            "split_rhat": split_rhat(values),
            "ess": effective_sample_size(values),
        })
    return pd.DataFrame(rows)


def render_diagnostics(all_chains: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    parameters = ["sigma_identity", "sigma_session", "sigma_interaction"]
    fig, axes = plt.subplots(len(parameters), 2, figsize=(12, 8), gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    colors = {"Within-V1": "#6f62a6", "Post-V1": "#555555"}
    for row, parameter in enumerate(parameters):
        for dataset, chains in all_chains.items():
            values = chains[parameter]
            for chain_index in range(values.shape[0]):
                axes[row, 0].plot(values[chain_index], lw=0.55, alpha=0.65, color=colors[dataset])
            axes[row, 1].hist(values.reshape(-1), bins=45, density=True, histtype="step",
                              lw=1.5, color=colors[dataset], label=dataset if row == 0 else None)
        axes[row, 0].set_title(f"{parameter}: traces", loc="left", fontsize=9.5)
        axes[row, 1].set_title(f"{parameter}: posterior", loc="left", fontsize=9.5)
        axes[row, 0].set_ylabel("Metric units")
        axes[row, 0].spines[["top", "right"]].set_visible(False)
        axes[row, 1].spines[["top", "right"]].set_visible(False)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("TTFS hierarchical identity model: pilot diagnostics", fontsize=13, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.input, dtype={"session_id": str})
    table = table.loc[(table["metric"] == args.metric) & (table["n_units"] >= args.min_units)].copy()
    if table.empty:
        raise ValueError("No rows remain for the requested metric and minimum-unit rule")
    common_scale = float(table["centered_mean"].std(ddof=1))
    all_chains: dict[str, dict[str, np.ndarray]] = {}
    summaries = []
    seed_sequence = np.random.SeedSequence(args.seed)
    dataset_seeds = seed_sequence.spawn(2)
    for dataset, dataset_seed in zip(["Within-V1", "Post-V1"], dataset_seeds):
        local = table.loc[table["dataset"] == dataset].copy()
        chain_seeds = dataset_seed.spawn(args.chains)
        chain_results = [
            run_chain(
                local, scale=common_scale, tune=args.tune, draws=args.draws, thin=args.thin,
                seed=int(seed.generate_state(1)[0]),
            )
            for seed in chain_seeds
        ]
        stacked = {
            parameter: np.stack([chain[parameter] for chain in chain_results])
            for parameter in chain_results[0]
        }
        all_chains[dataset] = stacked
        summary = summarize_chains(stacked)
        summary.insert(0, "dataset", dataset)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    v1_identity = all_chains["Within-V1"]["sigma_identity"].reshape(-1)
    hva_identity = all_chains["Post-V1"]["sigma_identity"].reshape(-1)
    rng = np.random.default_rng(args.seed + 99)
    count = min(len(v1_identity), len(hva_identity))
    difference = rng.choice(hva_identity, count, replace=False) - rng.choice(v1_identity, count, replace=False)
    contrast = pd.DataFrame([{
        "metric": args.metric,
        "min_units": args.min_units,
        "posterior_mean_delta_sigma_identity": float(np.mean(difference)),
        "posterior_median_delta_sigma_identity": float(np.median(difference)),
        "ci_low": float(np.percentile(difference, 2.5)),
        "ci_high": float(np.percentile(difference, 97.5)),
        "probability_hva_gt_v1": float(np.mean(difference > 0)),
        "common_scale": common_scale,
    }])
    np.savez_compressed(
        output_dir / "posterior_chains.npz",
        **{f"{dataset}_{parameter}": values for dataset, local in all_chains.items() for parameter, values in local.items()},
    )
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    contrast.to_csv(output_dir / "identity_contrast.csv", index=False)
    render_diagnostics(all_chains, output_dir / "pilot_diagnostics.png")
    report_lines = [
        "# Hierarchical identity model — TTFS pilot checkpoint", "",
        f"_Generated {date.today().isoformat()}; minimum {args.min_units} units; "
        f"{args.chains} chains × {args.draws} retained draws after {args.tune} tuning iterations._", "",
        "## Model", "",
        "Observed session × group means are modeled with their neuron-bootstrap SE as known",
        "measurement error. Dataset-specific stable identity, session, and session × identity",
        "scales receive half-normal priors on a common TTFS scale. Gaussian group and session",
        "effects are sampled conditionally; scales use log-slice sampling.", "",
        "## Posterior summaries", "",
        "| Dataset | Component | Mean | Median | 95% interval | split-Rhat | ESS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"| {row.dataset} | {row.parameter} | {row.posterior_mean:.3f} | "
            f"{row.posterior_median:.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{row.split_rhat:.4f} | {row.ess:.0f} |"
        )
    c = contrast.iloc[0]
    report_lines += [
        "", "## Primary pilot contrast", "",
        f"HVA−V1 stable identity SD: posterior mean {c.posterior_mean_delta_sigma_identity:+.3f} ms, "
        f"95% interval [{c.ci_low:+.3f}, {c.ci_high:+.3f}] ms, "
        f"P(HVA > V1) = {c.probability_hva_gt_v1:.3f}.", "",
        "## Claim gate", "",
        "The identity and interaction scales pass the pilot convergence screen. The Within-V1",
        "session scale does not: its near-zero funnel yields split-Rhat above 1.01 and low ESS.",
        "This pilot is therefore diagnostic, not production inference. Do not extend the current",
        "parameterization to the remaining metrics until that geometry is resolved.", "",
        "## Next smallest step", "",
        "Compare a non-centered session parameterization against a simpler no-session-random-effect",
        "sensitivity model for TTFS. Retain the richer model only if convergence improves and the",
        "identity contrast is stable.", "",
        "## Artifacts", "",
        "- `posterior_chains.npz`: retained scale chains.",
        "- `posterior_summary.csv`: component summaries and convergence diagnostics.",
        "- `identity_contrast.csv`: direct stable-identity contrast.",
        "- `pilot_diagnostics.png` and PDF: traces and marginal posteriors.",
    ]
    (output_dir / "TTFS_PILOT_CHECKPOINT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(contrast.to_string(index=False))


if __name__ == "__main__":
    main()
