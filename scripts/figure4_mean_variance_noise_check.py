"""Exploratory animal-cluster covariance correction for category-mean variance."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scripts.figure4_rerun_core import METRICS, category_recording_weights, fit_standardized_dispersion
from scripts.run_figure4_primary_dispersion import point_table

OUT = ROOT / 'artifacts/figure4_rerun/mean_variance_noise_check'


def calculate(table):
    cats = sorted(table.category.unique())
    sessions = sorted(table.session_id.unique())
    x = np.column_stack([*[table.category.eq(c) for c in cats],
                         *[table.session_id.eq(s) for s in sessions[1:]]]).astype(float)
    y = table.value.to_numpy(float)
    w = category_recording_weights(table).to_numpy()
    fit = sm.WLS(y, x, weights=w).fit()
    assert np.linalg.matrix_rank(x) == x.shape[1]
    k = len(cats)
    p = np.eye(k) - np.ones((k,k))/k
    beta = fit.params[:k]
    raw = beta @ p @ beta / k
    cov = fit.get_robustcov_results(cov_type='cluster', groups=table.animal_id.to_numpy(),
                                  use_correction=True).cov_params()[:k,:k]
    noise = np.trace(p @ cov)/k
    # Verify the weighted cluster sandwich independently of statsmodels.
    bread = np.linalg.inv(x.T @ (w[:,None]*x))
    scores = pd.DataFrame(x * (w*fit.resid)[:,None]).groupby(table.animal_id.to_numpy()).sum().to_numpy()
    g = table.animal_id.nunique()
    correction = g/(g-1) * (len(y)-1)/(len(y)-x.shape[1])
    check = correction * bread @ (scores.T @ scores) @ bread
    np.testing.assert_allclose(cov,check[:k,:k],atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(raw,fit_standardized_dispersion(table).dispersion,atol=1e-9)
    return dict(plugin_variance=raw, estimated_noise_variance=noise,
                corrected_variance=raw-noise,n_animals=g,n_cells=len(y),k=k)


def main():
    OUT.mkdir(exist_ok=True)
    registry=pd.read_csv(ROOT/'artifacts/figure4_rerun/v1/response_registry_floor5.csv.gz',
                         dtype={'unit_id':str,'session_id':str})
    rows=[]
    for metric in METRICS:
        for pop in ['V1','HVA']:
            draws=range(10) if metric==METRICS[2] and pop=='V1' else [0]
            for draw in draws:
                rows.append(dict(metric=metric,population=pop,draw=draw,
                                 **calculate(point_table(registry,metric,pop,draw))))
    points=pd.DataFrame(rows)
    points.to_csv(OUT/'draw_estimates.csv',index=False)
    summary=points.groupby(['metric','population'],sort=False).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'estimates.csv',index=False)
    contrasts=[]
    for metric in METRICS:
        s=summary.loc[summary.metric.eq(metric)].set_index('population')
        h,v=s.loc['HVA','corrected_variance'],s.loc['V1','corrected_variance']
        contrasts.append(dict(metric=metric,corrected_variance_difference=h-v,
                              corrected_sd_difference=np.sqrt(h)-np.sqrt(v) if min(h,v)>=0 else np.nan,
                              sd_difference_defined=bool(min(h,v)>=0)))
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(6.5,2.5))
    for ax,metric in zip(axes,METRICS):
        s=summary.loc[summary.metric.eq(metric)].set_index('population').loc[['V1','HVA']]
        ax.axhline(0,color='.6',lw=.7,ls='--')
        ax.plot([0,1],s.plugin_variance,'o',color='.6',label='Plug-in')
        ax.plot([0,1],s.corrected_variance,'D',color='#222222',label='Noise-corrected')
        for i,(_,r) in enumerate(s.iterrows()):
            ax.plot([i,i],[r.corrected_variance,r.plugin_variance],color='.6',lw=1)
        ax.set_xticks([0,1],['Within\nV1','Across\nHVA'])
        ax.set_xlim(-.5,1.5)
        ax.set_title(metric.replace('Response timescale','Timescale'))
        ax.set_ylabel('Variance of means\n'+('ms²' if metric!=METRICS[1] else '(log10 F1/F0)²'))
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=7,loc='best')
    fig.tight_layout()
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/f'mean_variance_noise_check.{ext}',dpi=300)
    (OUT/'methods.md').write_text('''# Exploratory estimation-noise sensitivity

The unchanged weighted category-plus-session model is fitted on the current eligible cells. Let m be its K category means, P = I − 11′/K, and C their animal-clustered covariance. The plug-in variance is m′Pm/K. The estimated noise contribution is tr(PC)/K; subtracting it gives the corrected variance. This identity includes covariances between estimated means and uses the actual K and sampling design. Covariance uses the CR1 small-sample factor G/(G−1) × (N−1)/(N−p), as implemented in statsmodels WLS. The weighted sandwich is independently verified against direct animal score sums.

This is an approximate correction, not a proven unbiased estimate: it assumes independent animals and a suitable mean model, and CR1 may be unreliable with only eight V1 animals. It does not remove recording confounding or between-dataset effects. Negative corrected variances are retained, indicating that the estimated noise contribution exceeds observed mean spread; no SD is assigned to a negative variance. These are point-estimate sensitivity checks, without new confidence intervals or p-values, and do not replace G's original bootstrap intervals. Connectors show the amount subtracted, not uncertainty.

For V1 timescale, corrections are calculated separately for all ten accepted draws and averaged in variance units, matching the primary point estimand. No zero clipping or averaging of square roots is used. The supplemental contrast reports corrected HVA-minus-V1 variance and reports a corrected SD difference only when both corrected variances are nonnegative. Each metric retains the original cohorts and nested fitting weights.

Covariance implementation: https://www.statsmodels.org/dev/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html
''')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
