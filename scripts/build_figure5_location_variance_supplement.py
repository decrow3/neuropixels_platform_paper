#!/usr/bin/env python3
"""Current-cohort omega² supplement and coherent within-session block shuffles."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.figure4_cell_omega import moments,effect
from scripts.run_figure4_primary_dispersion import point_table
from scripts.figure4_rerun_core import METRICS,V1_CATEGORIES,HVA_CATEGORIES
OUT=ROOT/'artifacts/figure5/supplement_location_variance_v1';OUT.mkdir(parents=True,exist_ok=True)
STEM=OUT/'Figure5_figure_supplement_location_variance'
N=5000;SEED=20260929
REG=ROOT/'artifacts/figure4_rerun/v1/response_registry_floor5.csv.gz'
EST=ROOT/'artifacts/figure4_rerun/cell_omega/estimates.csv'
BOOT=ROOT/'artifacts/figure4_rerun/cell_omega/bootstrap.csv.gz'


def omega(m):
    n,s,q=np.moveaxis(m,-1,0)
    assert (n>0).all(), 'Shuffle lost a category'
    total=n.sum(-1);k=n.shape[-1]
    grand=s.sum(-1)**2/total
    sst=q.sum(-1)-grand;ssb=(s*s/n).sum(-1)-grand
    mse=(sst-ssb)/(total-k)
    return (ssb-(k-1)*mse)/(sst+mse)


def main():
    reg=pd.read_csv(REG,dtype={'session_id':str,'unit_id':str})
    estimates=pd.read_csv(EST);boots=pd.read_csv(BOOT)
    rng=np.random.default_rng(SEED);rows=[];nullrows=[]
    for metric in METRICS:
        for pop,cats in [('V1',V1_CATEGORIES),('HVA',HVA_CATEGORIES)]:
            nd=10 if metric==METRICS[2] and pop=='V1' else 1
            tables=[point_table(reg,metric,pop,d) for d in range(nd)]
            observed=np.mean([effect(moments(t,cats))[0] for t in tables])
            saved=estimates.loc[estimates.metric.eq(metric)&estimates.population.eq(pop)].iloc[0]
            np.testing.assert_allclose(observed,saved.omega_squared,rtol=1e-10,atol=1e-12)
            null=np.zeros((N,nd,len(cats),3));checks=np.zeros((2,nd,len(cats),3));nsessions=0;fixed=0
            allrows=pd.concat(tables)
            for (animal,session),g in allrows.groupby(['animal_id','session_id'],sort=True):
                nsessions+=1
                local=[t.loc[t.animal_id.eq(animal)&t.session_id.eq(session)] for t in tables]
                block=np.stack([moments(t,cats) for t in local])
                available=np.flatnonzero(block[:,:,0].sum(0)>0)
                fixed+=len(available)==1
                # One permutation of the union of eligible session/category blocks
                # is shared across all ten draws. Missing-draw blocks move with labels.
                perm=np.argsort(rng.random((N,len(available))),axis=1)
                for d in range(nd):
                    null[:,d,available,:]+=block[d,available][perm]
                    # Expanded-cell check of first two permutations, independent of moments shuffle.
                    for rep in range(2):
                        mapping={cats[available[perm[rep,j]]]:cats[available[j]] for j in range(len(available))}
                        t=local[d].copy();t['category']=t.category.map(mapping)
                        checks[rep,d]+=moments(t,cats)
            np.testing.assert_allclose(null[:2],checks,rtol=1e-10,atol=1e-7)
            values=omega(null).mean(1)
            assert np.isfinite(values).all()
            count=int((values>=observed).sum());p=(1+count)/(N+1)
            rows.append(dict(metric=metric,population=pop,omega_squared=observed,low=saved.low,high=saved.high,
                             shuffle_median=np.median(values),shuffle_low=np.quantile(values,.025),shuffle_high=np.quantile(values,.975),
                             permutation_p_greater=p,exceedance_count=count,n_shuffles=N,n_sessions=nsessions,
                             fixed_single_category_sessions=fixed,n_point_draws=nd))
            nullrows.extend(dict(metric=metric,population=pop,replicate=i,omega_squared=v) for i,v in enumerate(values))
            print(metric,pop,'p=',p,flush=True)
        paired=boots.loc[boots.metric.eq(metric)].pivot(index='replicate',columns='population',values='omega_squared')
        assert len(paired)==N and np.isfinite(paired.to_numpy()).all()
        v=paired.HVA-paired.V1
        subset=estimates.loc[estimates.metric.eq(metric)].set_index('population')
        rows.append(dict(metric=metric,population='HVA_minus_V1',omega_squared=subset.loc['HVA','omega_squared']-subset.loc['V1','omega_squared'],low=v.quantile(.025),high=v.quantile(.975)))
    stats=pd.DataFrame(rows);stats.to_csv(OUT/'estimates_and_tests.csv',index=False)
    pd.DataFrame(nullrows).to_csv(OUT/'shuffle_draws.csv.gz',index=False)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(11.84,4.0),sharey=True)
    fig.subplots_adjust(left=.17,right=.96,bottom=.24,top=.73,wspace=.24)
    titles=['TTFS','Phase modulation','Response timescale'];colors={'V1':'#7c63a9','HVA':'#4b535c','HVA_minus_V1':'#315f9a'}
    for ax,metric,title,letter in zip(axes,METRICS,titles,'ABC'):
        local=stats.loc[stats.metric.eq(metric)].set_index('population')
        for y,pop in zip([2,1,0],['V1','HVA','HVA_minus_V1']):
            r=local.loc[pop];color=colors[pop]
            ax.errorbar(r.omega_squared*100,y+.09,xerr=[[100*(r.omega_squared-r.low)],[100*(r.high-r.omega_squared)]],fmt='o',color=color,ms=5,lw=1.5,capsize=0,zorder=3)
            if pop!='HVA_minus_V1':
                ax.errorbar(r.shuffle_median*100,y-.12,xerr=[[100*(r.shuffle_median-r.shuffle_low)],[100*(r.shuffle_high-r.shuffle_median)]],fmt='s',color='#999999',mfc='white',ms=4,lw=1.1,capsize=0,zorder=2)
                p=r.permutation_p_greater;label='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
                ax.text(1.01,y+.09,label,transform=ax.get_yaxis_transform(),va='center',fontsize=11,color=color)
        ax.axvline(0,color='#888888',lw=.8,ls='--',zorder=0)
        ax.axhline(.5,color='#e2e2e2',lw=.7)
        ax.set_ylim(-.5,2.5);ax.set_xlim(-2.2,4.8)
        ax.set_yticks([2,1,0],['Intra-V1','Inter-HVA','Inter-HVA − intra-V1†'])
        ax.set_xlabel('Variance explained (ω², %)')
        ax.set_title(title,pad=13,fontsize=13)
        ax.text(-.12,1.04,letter,transform=ax.transAxes,fontsize=25,va='bottom')
        for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#888888');ax.tick_params(axis='y',length=0)
    fig.suptitle('Variation attributable to location',fontsize=17,y=.98)
    fig.legend(handles=[Line2D([],[],marker='o',color='#4b535c',label='Observed: 95% bootstrap CI'),Line2D([],[],marker='s',mfc='white',color='#999999',label='Shuffled: median and central 95%')],loc='upper center',bbox_to_anchor=(.56,.92),ncol=2,frameon=False,fontsize=10)
    fig.text(.17,.045,'† Difference in percentage points; no shuffle-test stars on the contrast.  * p < .05; ** p < .01; *** p < .001; n.s. p ≥ .05.',fontsize=9)
    for ext in ['pdf','svg','png']:fig.savefig(STEM.with_suffix('.'+ext),dpi=300)
    plt.close(fig)
    caption='''**Figure 5—figure supplement (number to be assigned). Variation attributable to location.**

**A–C)** Bias-corrected explained variance (one-way omega-squared, ω²) for TTFS, phase modulation (log10 F1/F0) and response timescale. Rows show variation associated with recording location within V1 (intra-V1), area identity among five HVAs (inter-HVA), and their signed inter-HVA minus intra-V1 difference (†). Analyses use the current Figure 5 eligible cell populations, with equal weight per neuron and no session adjustment. This differs from the balanced, session-adjusted category-mean SDs in main Figure 5. Negative omega-squared estimates are retained; the algebraic bias correction uses neuron counts and is not a multilevel variance-component correction.

Colored circles and whiskers show observed estimates and 95% percentile intervals from 5,000 saved whole-animal bootstrap replicates. Open gray squares and whiskers show the median and central 95% range of 5,000 newly calculated within-session location-block label shuffles. All cells in a session/category block move together. For MouseV2 timescale, observed and shuffled statistics average ten draw-specific omega-squared estimates; one permutation of the session’s available category labels is shared across those draws. Draw-specific eligibility and cell counts remain attached to their blocks. Allen contributes one accepted estimate per neuron. Bootstrap intervals retain the existing coherent-draw-per-session resampling scheme.

Component-row annotations are uncorrected, one-sided permutation tests, p = (1 + number of shuffled statistics ≥ observed)/(5,001): * p < .05, ** p < .01, *** p < .001; n.s., p ≥ .05. The tests assume exchangeability of available location labels within sessions; single-category sessions cannot change. Shuffled ranges are null reference ranges, not confidence intervals on effects. The difference row uses paired-index independent-population bootstrap differences, expressed in percentage points, with no shuffle significance annotation. Neither nonsignificance nor an interval containing zero establishes equivalence. No matched Allen V1/HVA mean-control result is included in this location-effect supplement.
'''
    caption+='\nExact one-sided shuffle p-values:\n\n| Metric | Intra-V1 | Inter-HVA |\n|---|---:|---:|\n'
    for metric in METRICS:
        sub=stats.loc[stats.metric.eq(metric)].set_index('population')
        caption+=f"| {metric} | {sub.loc['V1','permutation_p_greater']:.6f} | {sub.loc['HVA','permutation_p_greater']:.6f} |\n"
    caption+='\nThe intra-V1 phase-modulation p-value is close to 0.05 and subject to Monte Carlo variation at 5,000 permutations; the n.s. label should not be interpreted as evidence of no location effect.\n'
    (OUT/'caption.md').write_text(caption)
    (OUT/'methods.md').write_text(caption+'\nSeed for new shuffles: '+str(SEED)+'. Source bootstraps: artifacts/figure4_rerun/cell_omega/bootstrap.csv.gz. No bootstrap reruns. The ten-draw observed statistic and its null use the same averaging operation; the inherited bootstrap intervals use one coherent selected draw per sampled session occurrence.\n')
    sources=[REG,EST,BOOT,Path(__file__),ROOT/'scripts/figure4_cell_omega.py',ROOT/'scripts/run_figure4_primary_dispersion.py',ROOT/'scripts/figure4_rerun_core.py']
    (OUT/'manifest.json').write_text(json.dumps({'shuffle_seed':SEED,'shuffles_per_component':N,'n_tests':6,'multiplicity':'uncorrected','observed_estimates_match_current_omega':True,'expanded_cell_checks':'First two shuffled moment arrays per population/metric checked against relabelled cell rows across all draws','sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
    print(STEM.with_suffix('.pdf'))

if __name__=='__main__':main()
