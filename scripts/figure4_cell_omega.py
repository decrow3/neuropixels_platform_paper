"""Raw cell-weighted omega squared on current cohorts; whole-animal bootstrap."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_figure4_primary_dispersion import point_table
from scripts.figure4_rerun_core import METRICS, V1_CATEGORIES, HVA_CATEGORIES

OUT = ROOT / 'artifacts/figure4_rerun/cell_omega'


def moments(t, cats):
    return np.array([[len(v := t.loc[t.category.eq(c), 'value'].to_numpy()),
                      v.sum(), v @ v] for c in cats], float)


def effect(m):
    n, s, q = m.T
    if (n == 0).any():
        return np.nan, np.nan, np.nan
    total = n.sum()
    sst = q.sum() - s.sum() ** 2 / total
    ssb = (s*s/n).sum() - s.sum() ** 2 / total
    mse = (sst-ssb)/(total-len(n))
    return (ssb-(len(n)-1)*mse)/(sst+mse), ssb/sst, np.sqrt(sst/total)


def main():
    OUT.mkdir(exist_ok=True)
    registry = pd.read_csv(ROOT/'artifacts/figure4_rerun/v1/response_registry_floor5.csv.gz',
                           dtype={'unit_id': str, 'session_id': str})
    rng = np.random.default_rng(20260921)
    summaries, bootrows, pointrows = [], [], []
    for metric in METRICS:
        for pop, cats in [('V1', V1_CATEGORIES), ('HVA', HVA_CATEGORIES)]:
            ndraw = 10 if metric == METRICS[2] and pop == 'V1' else 1
            tables = [point_table(registry, metric, pop, d) for d in range(ndraw)]
            pts = []
            for d, t in enumerate(tables):
                assert set(t.category) == set(cats)
                m = moments(t, cats)
                omega, eta, sd = effect(m)
                # Independent expanded-row sums verify the moment calculation.
                grand = t.value.mean()
                sst = ((t.value-grand)**2).sum()
                ssb = sum(len(g)*(g.value.mean()-grand)**2 for _,g in t.groupby('category'))
                mse = (sst-ssb)/(len(t)-len(cats))
                np.testing.assert_allclose(omega, (ssb-(len(cats)-1)*mse)/(sst+mse), atol=1e-12)
                pts.append(omega)
                pointrows.append(dict(metric=metric,population=pop,draw=d,n_cells=len(t),
                                      omega_squared=omega,eta_squared=eta,pooled_cell_sd=sd))
            source = registry.loc[registry.metric.eq(metric)&registry.population.eq(pop)]
            animals = sorted(source.animal_id.unique())
            # Cache sufficient statistics per animal/session/draw, preserving coherent draws.
            blocks = []
            for animal in animals:
                sessions = []
                for session in source.loc[source.animal_id.eq(animal),'session_id'].unique():
                    sessions.append(np.stack([moments(t.loc[t.animal_id.eq(animal)&t.session_id.eq(session)],cats)
                                              for t in tables]))
                blocks.append(sessions)
            draws = []
            for b in range(5000):
                m = np.zeros((len(cats),3))
                for a in rng.integers(0,len(animals),len(animals)):
                    for session in blocks[a]:
                        m += session[rng.integers(ndraw)]
                value = effect(m)[0]
                draws.append(value)
                bootrows.append(dict(metric=metric,population=pop,replicate=b,omega_squared=value))
            valid = np.asarray(draws)[np.isfinite(draws)]
            low, high = np.quantile(valid,[.025,.975])
            summaries.append(dict(metric=metric,population=pop,omega_squared=np.mean(pts),
                                  low=low,high=high,n_animals=len(animals),
                                  valid_bootstraps=len(valid),failed_bootstraps=5000-len(valid)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'estimates.csv',index=False)
    pd.DataFrame(pointrows).to_csv(OUT/'point_draws.csv',index=False)
    pd.DataFrame(bootrows).to_csv(OUT/'bootstrap.csv.gz',index=False)
    note = ('Cell-weighted one-way omega squared on the current eligible cohorts. '
            'ω² = [SS_between − (K−1) MS_within] / [SS_total + MS_within]. '
            'SS_total = N × pooled_cell_SD² (population SD, ddof=0), exactly matching the '
            'temporary TTFS all-cell bracket. Category means in this calculation are raw '
            'cell means, not the adjusted diamonds. Equal cell weighting; no session adjustment. '
            'Negative estimates are retained. The algebraic correction uses cell counts and '
            'is not a multilevel bias correction. Intervals are percentile intervals from '
            '5,000 whole-animal bootstrap replicates, seed 20260921, retaining sessions and '
            'cells together. V1 timescale point estimates average ten draw-specific omega values; '
            'bootstrap replicates choose a coherent draw per sampled session occurrence and '
            'use its five-cell floor. Other estimates use one value per cell. Replicates missing '
            'a category are marked invalid and counted. No independent-cell p-values are used.\n')
    (OUT/'methods.md').write_text(note)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
