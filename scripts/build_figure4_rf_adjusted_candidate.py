"""RF-adjusted single-cell C/D/E including legacy Central V1.

Nested linear models use a fixed quadratic RF surface and categorical group.
Whole-session bootstrap and label shuffles refit both models from sufficient
statistics, preserving shared Allen sessions across comparisons.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_figure4_cell_location_controls as base
from scripts import build_figure4_central_v1_candidate as central
from scripts.mousev2_frequency_preference_surfaces import to_allen_display_coordinates

PREFIX = base.OUT / "Figure4_CDE_single_cell_central_RF_adjusted_candidate"
BASELINE = base.OUT / "Figure4_CDE_single_cell_central_RF_subset_candidate"
MOUSE_RF = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
ALLEN_RF = ROOT / "artifacts/figure3/06c_allen_rf_matching/rf_unit_common_support.csv"
FEATURES = ["az", "el", "az2", "az_el", "el2"]


def attach_rf(table):
    mouse = pd.read_csv(MOUSE_RF, dtype={"unit_id": str})
    mouse = mouse.loc[mouse.rf_model_supported].copy()
    mouse["azimuth"], mouse["elevation"] = to_allen_display_coordinates(
        mouse.supported_rf_center_x_deg, mouse.supported_rf_center_y_deg)
    mouse["session_id"] = mouse.site_number.astype(str)
    mouse["source"] = "MouseV2"
    allen = pd.read_csv(ALLEN_RF, dtype={"ecephys_unit_id": str, "ecephys_session_id": str})
    allen = allen.rename(columns={"ecephys_unit_id": "unit_id", "ecephys_session_id": "session_id",
                                  "azimuth_rf": "azimuth", "elevation_rf": "elevation"})
    allen["source"] = "Allen"
    columns = ["source", "session_id", "unit_id", "azimuth", "elevation"]
    rf = pd.concat([mouse[columns], allen[columns]], ignore_index=True)
    assert not rf.duplicated(columns[:3]).any()
    result = table.merge(rf, on=columns[:3], how="left", validate="many_to_one")
    result["has_rf"] = np.isfinite(result[["azimuth", "elevation"]]).all(axis=1)
    result["raw_value"] = result.value
    result["retained"] = False
    valid = result.loc[result.has_rf].copy()
    # Keep the original population-specific support floors after RF selection.
    counts = valid.groupby(["comparison", "metric", "source", "session_id", "group"]).value.transform("size")
    floor = np.where(valid.metric.eq("TTFS (ms)") & valid.comparison.ne("v1"), 10, 5)
    valid = valid.loc[counts.ge(floor)].copy()
    # Control HVA neurons must still belong to an eligible individual HVA group.
    join_keys = ["metric", "source", "session_id", "unit_id"]
    eligible_hva = pd.MultiIndex.from_frame(valid.loc[valid.comparison.eq("hva"), join_keys])
    is_control_hva = valid.comparison.eq("control") & valid.group.eq("hva")
    valid = valid.loc[~is_control_hva | pd.MultiIndex.from_frame(valid[join_keys]).isin(eligible_hva)].copy()
    counts = valid.groupby(["comparison", "metric", "source", "session_id", "group"]).value.transform("size")
    floor = np.where(valid.metric.eq("TTFS (ms)") & valid.comparison.ne("v1"), 10, 5)
    valid = valid.loc[counts.ge(floor)].copy()
    paired = valid.groupby(["comparison", "metric", "source", "session_id"]).group.transform("nunique")
    valid = valid.loc[valid.comparison.ne("control") | paired.eq(2)].copy()
    result.loc[valid.index, "retained"] = True
    audit = result.groupby(["comparison", "metric", "source", "group"], as_index=False).agg(
        original_cells=("value", "size"), rf_cells=("has_rf", "sum"), retained_cells=("retained", "sum"))
    for metric in base.style.METRICS:
        v1 = valid.loc[valid.metric.eq(metric) & valid.comparison.eq("v1")]
        assert set(v1.group) == set(base.style.PROBE_ORDER + ["Central"])
    return valid, audit, result


def rf_features(table):
    # Fixed scaling improves numerical conditioning without estimating an anchor.
    az = (table.azimuth.to_numpy(float) - 50.) / 25.
    el = (table.elevation.to_numpy(float) - 10.) / 25.
    return np.column_stack([az, el, az**2, az*el, el**2])


def regression_cube(table, sessions, groups):
    """Moments of [1, outcome, five RF predictors] for session × group."""
    cube = np.zeros((len(sessions), len(groups), 7, 7))
    sx = {s: i for i, s in enumerate(sessions)}
    gx = {g: i for i, g in enumerate(groups)}
    for (session, group), part in table.groupby(["session_id", "group"], sort=True):
        z = np.column_stack([np.ones(len(part)), part.value.to_numpy(float), rf_features(part)])
        cube[sx[session], gx[group]] = z.T @ z
    return cube


def fit_moments(moments):
    """Partial omega² for added group labels, conditional on the RF surface.

    (SSE_RF - SSE_full - q*MSE_full)/(SSE_RF + MSE_full),
    where q is the difference in model ranks. Full model has group intercepts
    plus RF predictors; reduced model has one intercept plus the same RF terms.
    """
    m = np.asarray(moments)
    k = m.shape[-3]
    pooled = m.sum(axis=-3)
    p = k + 5
    xtx = np.zeros((*m.shape[:-3], p, p))
    xty = np.zeros((*m.shape[:-3], p))
    indices = np.arange(k)
    xtx[..., indices, indices] = m[..., :, 0, 0]
    xtx[..., :k, k:] = m[..., :, 0, 2:]
    xtx[..., k:, :k] = np.swapaxes(m[..., :, 0, 2:], -1, -2)
    xtx[..., k:, k:] = pooled[..., 2:, 2:]
    xty[..., :k] = m[..., :, 0, 1]
    xty[..., k:] = pooled[..., 2:, 1]
    keep = [0, 2, 3, 4, 5, 6]
    reduced_x = pooled[..., keep, :][..., :, keep]
    reduced_y = pooled[..., keep, 1]
    beta = np.einsum("...ij,...j->...i", np.linalg.pinv(xtx, rcond=1e-10), xty)
    beta_reduced = np.einsum("...ij,...j->...i", np.linalg.pinv(reduced_x, rcond=1e-10), reduced_y)
    # Use the same rank threshold as the pseudoinverse.
    singular = np.linalg.svd(xtx, compute_uv=False)
    singular_reduced = np.linalg.svd(reduced_x, compute_uv=False)
    rank = np.sum(singular > singular[..., :1] * 1e-10, axis=-1)
    rank_reduced = np.sum(singular_reduced > singular_reduced[..., :1] * 1e-10, axis=-1)
    n = pooled[..., 0, 0]
    sse_full = np.maximum(0, pooled[..., 1, 1] - np.sum(beta * xty, axis=-1))
    sse_rf = np.maximum(0, pooled[..., 1, 1] - np.sum(beta_reduced * reduced_y, axis=-1))
    q = rank - rank_reduced
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = sse_full / (n - rank)
        omega = (sse_rf - sse_full - q*mse) / (sse_rf + mse)
    omega = np.where((q > 0) & (n > rank) & (sse_rf > 0), omega, np.nan)
    return dict(omega=omega, beta=beta, sse_rf=sse_rf, sse_full=sse_full,
                rank=rank, rank_reduced=rank_reduced, n=n)


def analyze_adjusted(table, n_draws, seed):
    rng = np.random.default_rng(seed)
    records, resamples, displays, diagnostics, coefficients = [], [], [], [], []
    for metric in base.style.METRICS:
        local = table.loc[table.metric.eq(metric)]
        groups = {key: sorted(local.loc[local.comparison.eq(key), "group"].unique()) for key in ["v1", "hva", "control"]}
        moments = {key: np.zeros((n_draws, len(groups[key]), 7, 7)) for key in groups}
        for source, part in local.groupby("source", sort=True):
            sessions = sorted(part.session_id.unique())
            multiplicities = central.session_multiplicities(len(sessions), n_draws, rng)
            for key in groups:
                cube = regression_cube(part.loc[part.comparison.eq(key)], sessions, groups[key])
                moments[key] += np.einsum("bs,sgij->bgij", multiplicities, cube)
        observed, boots, nulls = {}, {}, {}
        for key in groups:
            part = local.loc[local.comparison.eq(key)].copy()
            total = np.zeros((len(groups[key]), 7, 7))
            shuffled = np.zeros((n_draws, len(groups[key]), 7, 7))
            for source, source_part in part.groupby("source", sort=True):
                cube = regression_cube(source_part, sorted(source_part.session_id.unique()), groups[key])
                total += cube.sum(axis=0)
                for block in cube:
                    present = np.flatnonzero(block[:, 0, 0])
                    permutations = np.argsort(rng.random((n_draws, len(present))), axis=1)
                    shuffled[:, present] += block[present][permutations]
            fit = fit_moments(total)
            if fit["rank"] != len(groups[key]) + 5 or fit["rank_reduced"] != 6:
                raise ValueError(f"Rank-deficient observed RF model: {metric}, {key}")
            boots[key] = fit_moments(moments[key])["omega"]
            nulls[key] = fit_moments(shuffled)["omega"]
            if not np.isfinite(boots[key]).all() or not np.isfinite(nulls[key]).all():
                raise ValueError(f"Degenerate resample: {metric}, {key}")
            observed[key] = float(fit["omega"])
            r = rf_features(part)
            rf_beta = fit["beta"][-5:]
            # Adjust to the same pooled RF-feature mean for all groups in a
            # comparison, retaining the raw population mean and outcome units.
            part["rf_adjustment"] = (r - r.mean(axis=0)) @ rf_beta
            part["value"] = part.raw_value - part.rf_adjustment
            np.testing.assert_allclose(part.value.mean(), part.raw_value.mean(), atol=1e-10)
            displays.append(part)
            for term, coefficient in zip([*groups[key], *FEATURES], fit["beta"]):
                coefficients.append(dict(metric=metric, comparison=key, term=term, coefficient=coefficient))
            diagnostics.append(dict(metric=metric, comparison=key, n=int(fit["n"]),
                                    full_rank=int(fit["rank"]), rf_rank=int(fit["rank_reduced"]),
                                    sse_rf=float(fit["sse_rf"]), sse_full=float(fit["sse_full"]),
                                    raw_variance=float(np.var(part.raw_value)), adjusted_variance=float(np.var(part.value))))
            records.append(dict(metric=metric, comparison=key, observed=observed[key],
                                bootstrap_low=np.quantile(boots[key], .025), bootstrap_high=np.quantile(boots[key], .975),
                                shuffle_median=np.median(nulls[key]), shuffle_low=np.quantile(nulls[key], .025),
                                shuffle_high=np.quantile(nulls[key], .975),
                                permutation_p_greater=(1 + (nulls[key] >= observed[key]).sum()) / (n_draws + 1),
                                conditional_shuffle=key == "v1", n_observations=len(part),
                                n_sessions=part[["source", "session_id"]].drop_duplicates().shape[0]))
        boots["delta"] = boots["hva"] - boots["v1"]
        nulls["delta"] = nulls["hva"] - nulls["v1"]
        records.append(dict(metric=metric, comparison="delta", observed=observed["hva"]-observed["v1"],
                            bootstrap_low=np.quantile(boots["delta"], .025), bootstrap_high=np.quantile(boots["delta"], .975),
                            shuffle_median=np.median(nulls["delta"]), shuffle_low=np.quantile(nulls["delta"], .025),
                            shuffle_high=np.quantile(nulls["delta"], .975), permutation_p_greater=np.nan,
                            conditional_shuffle=False, n_observations=np.nan, n_sessions=np.nan))
        for key in boots:
            resamples.append(pd.DataFrame(dict(metric=metric, comparison=key, replicate=np.arange(n_draws),
                                               bootstrap=boots[key], shuffle=nulls[key])))
    return pd.DataFrame(records), pd.concat(resamples, ignore_index=True), pd.concat(displays, ignore_index=True), pd.DataFrame(diagnostics), pd.DataFrame(coefficients)


def render_rf_support(table, output):
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11), sharex=True, sharey=True)
    for row, metric in enumerate(base.style.METRICS):
        for col, key in enumerate(["v1", "hva"]):
            ax = axes[row, col]
            part = table.loc[table.metric.eq(metric) & table.comparison.eq(key)]
            order = [*base.style.PROBE_ORDER[:2], "Central", *base.style.PROBE_ORDER[2:]] if key == "v1" else base.style.AREA_ORDER
            colors = {**base.style.PROBE_COLORS, **base.style.AREA_COLORS, "Central": "#555555"}
            for group in order:
                p = part.loc[part.group.eq(group)]
                ax.scatter(p.azimuth, p.elevation, s=5, alpha=.22, color=colors[group], rasterized=True)
                ax.scatter(p.azimuth.mean(), p.elevation.mean(), s=55, color=colors[group], edgecolor="black",
                           label=base.style.PROBE_DISPLAY_LABELS.get(group, group))
            if row == 0:
                ax.legend(frameon=False, fontsize=8, ncol=2)
            ax.set_title(f"{metric}: {'V1 + Central' if key == 'v1' else 'HVAs'}", fontsize=11)
            ax.set_xlabel("Display RF azimuth (deg)")
            ax.set_ylabel("Display RF elevation (deg)")
            base.clean_axis(ax)
    fig.suptitle("RF coverage of the neurons retained for adjustment", fontsize=15)
    fig.tight_layout(rect=[0, .025, 1, .97])
    fig.text(.07, .015, "Points = cells; outlined circles = group means. Display coordinates are not corrected for eye position.", fontsize=9)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    input_path = central.PREFIX.with_name(central.PREFIX.name + "_inputs.csv")
    original = pd.read_csv(input_path, dtype={"session_id": str, "unit_id": str})
    table, audit, eligibility = attach_rf(original)
    print(audit.to_string(index=False), flush=True)
    table.to_csv(PREFIX.with_name(PREFIX.name + "_raw_inputs.csv"), index=False)
    audit.to_csv(PREFIX.with_name(PREFIX.name + "_coverage.csv"), index=False)
    eligibility.to_csv(PREFIX.with_name(PREFIX.name + "_eligibility.csv.gz"), index=False)
    subset_stats, subset_draws = central.analyze(table, args.draws, args.seed)
    subset_stats.to_csv(BASELINE.with_name(BASELINE.name + "_statistics.csv"), index=False)
    subset_draws.to_csv(BASELINE.with_name(BASELINE.name + "_resamples.csv.gz"), index=False)
    base.render(table, subset_stats, BASELINE, True, central_mode="include", rf_subset=True)
    stats, draws, display, diagnostic, coefficients = analyze_adjusted(table, args.draws, args.seed)
    for suffix, frame in [("statistics.csv", stats), ("resamples.csv.gz", draws), ("adjusted_inputs.csv", display),
                          ("model_diagnostics.csv", diagnostic), ("coefficients.csv", coefficients)]:
        frame.to_csv(PREFIX.with_name(PREFIX.name + "_" + suffix), index=False)
    base.render(display, stats, PREFIX, True, central_mode="include", rf_adjusted=True)
    render_rf_support(table, PREFIX.with_name(PREFIX.name + "_RF_coverage"))
    compare = pd.concat([pd.read_csv(central.PREFIX.with_name(central.PREFIX.name + "_statistics.csv")).assign(version="All original cells"),
                         subset_stats.assign(version="RF subset, unadjusted"), stats.assign(version="RF adjusted")], ignore_index=True)
    compare.to_csv(PREFIX.with_name(PREFIX.name + "_comparison.csv"), index=False)
    config = dict(seed=args.seed, draws=args.draws, central_included=True,
                  RF_model="group intercepts + azimuth + elevation + azimuth^2 + azimuth*elevation + elevation^2",
                  coordinates="Display axes: MouseV2 x+50, y+10; native Allen azimuth/elevation; no eye-position correction",
                  effect="Partial omega-squared: (SSE_RF-SSE_full-q*MSE_full)/(SSE_RF+MSE_full)",
                  displayed_outcome="raw outcome minus (RF features minus pooled-comparison RF-feature mean) times fitted RF coefficients",
                  bootstrap="Both models refitted in every whole-session draw; stratified by source, shared Allen session draws across comparisons",
                  shuffle="Whole session/group labels permuted, RF covariates retained and both models refitted; Central fixed",
                  sources=[str(p.relative_to(ROOT)) for p in [input_path, MOUSE_RF, ALLEN_RF]])
    PREFIX.with_name(PREFIX.name + "_method.json").write_text(json.dumps(config, indent=2) + "\n")
    lines = ["# RF-adjusted single-cell C/D/E with Central", "",
             "This is a separate exploratory candidate. The original and Central-inclusive figures are preserved. Central remains in the middle of C and enters C's V1 effect and E's difference.", "",
             "## RF selection and coordinates", "",
             "MouseV2 RFs use supported parametric center estimates (`rf_model_supported`). Allen RFs use the existing published-like RF support table (finite centers, p<.01, RF area<2500 deg², SNR>1, drifting-grating firing rate>0.1 Hz, and an available V1 session anchor). These RF-quality rules differ between sources. RF columns are joined by source, session, and neuron with many-to-one validation. The full eligibility table records each exclusion; the coverage table separates missing RFs from subsequent session/group support exclusions.", "",
             "The existing fixed display-coordinate conversion maps MouseV2 x/y to Allen azimuth/elevation using +50/+10 degrees. Native Allen coordinates are used, not V1-centered relative coordinates. These are screen coordinates, not verified gaze-corrected retinal positions. No shared-support restriction or RF-location matching is imposed; the model can extrapolate where group coverage differs. The RF coverage figure shows that coverage explicitly.", "",
             "After RF selection, V1 groups (including Central) require at least 5 cells per session. HVA and control groups require at least 10 TTFS cells or 5 other-metric cells. Control HVA cells must belong to retained individual HVA groups, and only sessions with both control populations remain. The same RF-qualified cells are used in the adjusted and unadjusted companion.", "",
             "## Adjustment and effect definition", "",
             "For each metric and comparison separately, fit y = group intercepts + a common quadratic RF surface (azimuth, elevation, azimuth², azimuth×elevation, elevation²). Display axes are centered at (50°,10°) and scaled by 25° for numerical stability only. No RF-by-group interactions are fit. Group terms are included when estimating RF coefficients, so an RF-only fit does not absorb all group-associated variation indiscriminately. This remains a model-dependent descriptive adjustment, not a causal separation of area and retinotopy.", "",
             "C/D show each neuron's raw outcome minus its fitted RF contribution relative to the pooled RF-feature mean for that comparison. Thus response units and each comparison's grand mean are preserved. Adjusted cell values can fall outside the raw selection range. C and D use independently estimated RF surfaces and reference feature distributions; their absolute adjusted means should not be used to infer a V1–HVA shift. The matched E control models that contrast directly.", "",
             "Effects are partial omega-squared for adding group labels to the RF-only model: (SSE_RF − SSE_full − q×MSE_full)/(SSE_RF + MSE_full), with q the difference in model ranks and MSE_full using all fitted degrees of freedom. This is the bias-corrected fraction of residual variation after RF adjustment associated with group identity. It is not the original omega-squared denominator and not simply omega-squared evaluated on the adjusted dots. Negative estimates are retained. Unadjusted-subset and adjusted values therefore differ in both conditioning and denominator.", "",
             f"Both models are refitted in each of {args.draws:,} bootstrap replicates (seed {args.seed}). Whole sessions are resampled within MouseV2 and Allen separately, with the same Allen multiplicities used for Central, HVAs, and the control. The bootstrap difference is computed within replicate. The procedure propagates RF coefficient uncertainty conditional on measured RF centers; uncertainty in the RF-center fits is not modeled. No out-of-sample predictive claim is made.", "",
             "Location-block shuffles retain each cell's outcome/RF pair, reassign available location labels within source/session, and refit the full model. Central remains fixed, so V1 labels carry ‡: they test the four MouseV2 labels conditional on Central, not an omnibus five-group or Central-versus-periphery null. Conditional exchangeability of those label blocks remains an assumption; RF adjustment does not guarantee it. p-values are one-sided, uncorrected. Difference labels † use whether the 95% session-bootstrap interval includes zero, not a shuffle p-value or equivalence test.", "",
             "Central still uses legacy response metrics and comes from a different cohort. RF regression does not remove that dataset/processing confounding. The control retains HVA-matched response processing and is restricted to RF-qualified matched sessions.", "",
             "## Direct comparison", "", "| Metric | Version | V1 effect (%) | HVA effect (%) | HVA − V1 (pp), 95% interval |", "|---|---|---:|---:|---:|"]
    for metric in base.style.METRICS:
        for version in compare.version.unique():
            s = compare.loc[compare.metric.eq(metric) & compare.version.eq(version)].set_index("comparison")
            d = s.loc["delta"]
            lines.append(f"| {metric} | {version} | {100*s.loc['v1','observed']:.2f} | {100*s.loc['hva','observed']:.2f} | {100*d.observed:+.2f} [{100*d.bootstrap_low:+.2f}, {100*d.bootstrap_high:+.2f}] |")
    lines += ["", "Reproduce with `OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-figure4-cells MPLBACKEND=Agg python scripts/build_figure4_rf_adjusted_candidate.py`. Outputs include adjusted and same-cell unadjusted PDFs, PNGs and SVGs, RF-coverage PDF/PNG, inputs, eligibility/coverage audits, model coefficients/diagnostics, effects, and all resamples."]
    PREFIX.with_name(PREFIX.name + "_notes.md").write_text("\n".join(lines) + "\n")
    print(stats[["metric", "comparison", "observed", "bootstrap_low", "bootstrap_high", "permutation_p_greater"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
