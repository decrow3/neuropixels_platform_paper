#!/usr/bin/env python3
"""Build a source-backed, paginated PDF companion to Figure 5 v13; no refitting."""
from pathlib import Path
import sys, json, hashlib, html, re
import numpy as np
import pandas as pd
sys.path.append('/usr/lib/python3/dist-packages')
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/figure4_rerun'
OUT=ROOT/'artifacts/figure5/methods_companion_v3'; OUT.mkdir(exist_ok=True)
FIG=ROOT/'artifacts/figure5/v13_shift_hierarchy'; STEM='Figure5_CDE_rerun_with_schematic'
used={}; md=[]; story=[]; sections=[]
def source(path):
 p=Path(path); p=p if p.is_absolute() else ROOT/p
 used[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return p
def read(path): return pd.read_csv(source(path),keep_default_na=False)
def r(path): return read(BASE/path)
def num(x):
 x=float(x)
 if x==0:return '0'
 return f'{x:.4g}'
def interval(e,l,h):return f'{num(e)} [{num(l)}, {num(h)}]'
font='/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('DV',font+'DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DV-Bold',font+'DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
styles={
 'body':ParagraphStyle('body',fontName='DV',fontSize=9.4,leading=14,spaceAfter=9,textColor=colors.HexColor('#253342')),
 'small':ParagraphStyle('small',fontName='DV',fontSize=7.9,leading=11,spaceAfter=7,textColor=colors.HexColor('#526070')),
 'h1':ParagraphStyle('h1',fontName='DV-Bold',fontSize=22,leading=27,spaceAfter=16,textColor=colors.HexColor('#173b51')),
 'h2':ParagraphStyle('h2',fontName='DV-Bold',fontSize=11.5,leading=15,spaceBefore=7,spaceAfter=8,textColor=colors.HexColor('#173b51')),
 'cell':ParagraphStyle('cell',fontName='DV',fontSize=8,leading=11,textColor=colors.HexColor('#253342')),
 'head':ParagraphStyle('head',fontName='DV-Bold',fontSize=8,leading=11,textColor=colors.white),
}
def p(s,style='body'):
 story.append(Paragraph(s,styles[style])); md.append(re.sub('<[^>]+>','',s)+'\n')
def page(title):
 if story: story.append(PageBreak())
 sections.append(title); md.append('\n# '+title+'\n');p(title,'h1')
def h(s):p(s,'h2')
def table(headers,rows,widths=None,size=8,padding=5):
 st=ParagraphStyle('tablecell',parent=styles['cell'],fontSize=size,leading=size+3)
 data=[[Paragraph(html.escape(str(v)),styles['head']) for v in headers]]
 data += [[Paragraph(html.escape(str(v)),st) for v in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#244d63')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4f6'),colors.white]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cbd5dc'))]))
 story.extend([t,Spacer(1,10)])
 md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
W=507
METRICS=['TTFS (ms)','log10 F1/F0','Response timescale (ms)']
short={'TTFS (ms)':'TTFS','log10 F1/F0':'log10 F1/F0','Response timescale (ms)':'Timescale'}
e=read(FIG/f'{STEM}_estimates.csv'); display=read(FIG/f'{STEM}_display.csv'); spreads=read(FIG/f'{STEM}_mean_spreads.csv')
primary=r('v4_primary_dispersion/primary_effects.csv'); boot=r('v4_primary_dispersion/bootstrap_effects.csv.gz')
control=r('v6_matched_control/matched_control.csv'); cb=r('v6_matched_control/matched_control_bootstrap.csv.gz')
counts=r('significance/figure4_counts.csv'); tails=r('significance/bootstrap_tail_counts.csv')
omega=r('cell_omega/estimates.csv'); noise=r('mean_variance_noise_check/estimates.csv'); nc=r('mean_variance_noise_check/contrasts.csv')
norm=r('v4_primary_dispersion/normalized_effects.csv'); sens=r('v4_primary_dispersion/sensitivities/point_sensitivities.csv'); loo=r('v4_primary_dispersion/sensitivities/leave_one_animal_out.csv')
sim=r('v3_model_validation/simulation_summary.csv'); targeted=r('v3_model_validation/ttfs_v1_targeted/simulation_summary.csv')
slopes=r('v5_hva_hierarchy/hierarchy_slopes.csv'); ladder=r('v5_hva_hierarchy/rf_effect_ladder.csv'); central=r('v7_central_companion/central_inclusive_effects.csv')
drawcounts=r('significance/timescale_draw_counts.csv'); harris=read('data/reference/allen_harris_2019_connectivity/harris_visualmedial_scores.csv')

# Verify the presentation revisions retained every original statistical table.
for suffix in ['estimates','display','mean_spreads','anatomical_connections']:
 current=read(FIG/f'{STEM}_{suffix}.csv')
 original=read(BASE/f'v22_compact_E/Figure4_CDE_rerun_with_schematic_{suffix}.csv')
 pd.testing.assert_frame_equal(current,original)
source(FIG/f'{STEM}_layout_manifest.json')
for renderer in sorted((ROOT/'scripts').glob('build_figure5_*.py')):
 source(renderer)
# Independent numerical reproduction of all 12 plotted summaries.
for row in e.itertuples():
 part=primary.loc[primary.metric.eq(row.metric)].set_index('effect')
 pair=boot.loc[boot.metric.eq(row.metric)&boot.effect.isin(['v1','hva'])].pivot(index='replicate',columns='effect',values='estimate')
 if row.effect in ['v1_sd','hva_sd']:
  k=row.effect[:-3]; point=np.sqrt(part.loc[k,'estimate']); vals=np.sqrt(pair[k])
 elif row.effect=='hva_minus_v1_sd':
  point=np.sqrt(part.loc['hva','estimate'])-np.sqrt(part.loc['v1','estimate']); vals=np.sqrt(pair.hva)-np.sqrt(pair.v1)
 else:
  point=control.set_index('metric').loc[row.metric,'hva_minus_v1'];vals=cb.loc[cb.metric.eq(row.metric),'hva_minus_v1']
 assert len(vals)==5000 and np.isfinite(vals).all()
 np.testing.assert_allclose([point,*np.quantile(vals,[.025,.975])],[row.estimate,row.low,row.high],rtol=1e-11,atol=1e-13)

page('Figure 5\nMethods, model and findings')
p('Scientific companion • 29 September 2026 • Figure version v13_shift_hierarchy','small')
p('<b>Question.</b> How large is variation among the mean responses of five higher visual areas compared with variation among four sampled locations within V1? A separate matched Allen comparison measures the pooled HVA–V1 mean difference.')
img=Image(str(source(FIG/f'{STEM}.png')),width=470,height=470*709/852.648);story.append(img);story.append(Spacer(1,12))
p('<b>Main result.</b> Estimated HVA spread is larger for TTFS and response timescale, but smaller for phase modulation. All three 95% bootstrap intervals for the SD difference contain zero. This does not establish equality or practical equivalence.')
p('This companion documents the saved analysis, not a new fit. It includes all 12 plotted estimates and intervals, omitted supporting results, validation limitations and decisions that remain open. Numerical tables are generated from the saved CSVs; Figure 5 v13 has the same statistical tables as Figure 4 v22. All 12 estimates and intervals were checked against the saved bootstrap draws; this update changes documentation, not the analysis.','small')

p('<b>Reading guide:</b> methods and model, pp. 2–5; figure numbers, pp. 6–8; omitted statistics and validation, pp. 9–14; historical choices and conclusions, pp. 15–16; supporting shuffle tests, p. 17; sources, p. 18.','small')

page('1  What each panel measures')
table(['Panel','Content','Interpretation'],[
 ['A','Handmade inter-HVA cartoon and Harris anatomical hierarchy','Anatomical context. Numerical hierarchy coordinates do not enter the category-dispersion model.'],
 ['B','Handmade intra-V1 cartoon and illustrative mean/spread schematic','A conceptual explanation; named-area marker heights on the right are manually arranged, not observed responses.'],
 ['C','Five HVA cellular distributions and fitted category means','LM, RL, AL, PM and AM in Allen recordings.'],
 ['D','Four V1-location distributions and fitted category means','Anterior, lateral, posterior and medial locations in MouseV2.'],
 ['E','Matched Allen mean difference; two component SDs; SD difference','Point estimates and pointwise 95% whole-animal bootstrap percentile intervals.']], [47,194,266])
p('<b>Terminology:</b> Inter-HVA means among the five named higher visual areas; Intra-V1 means among four sampled locations within V1. These describe variation, not connectivity. Figure 4 now covers MouseV2 data-collection methods.','small')
h('Three distinctions that must remain visible')
p('<b>Mean versus spread.</b> A pooled HVA–V1 shift can be present even when variation among the HVAs is small. The mean difference does not demonstrate an ordered hierarchy among HVAs.')
p('<b>Cells versus category means.</b> The pooled cellular SD describes diversity among individual neurons. The SD in E describes the spread of estimated category means. A broad cell distribution does not by itself imply a large area effect.')
p('<b>Within-area benchmark versus equivalence.</b> Comparing against sampled locations in a bona fide single area gives the HVA differences a useful empirical scale. It does not establish that anatomical areas are interchangeable or that all V1 locations have been sampled.')
h('The four estimates in E')
p('Left to right: (1) pooled Allen HVAs minus matched Allen V1 mean; (2) SD of four MouseV2 V1-location means; (3) SD of five Allen HVA means; (4) HVA SD minus V1 SD. All share response units. Only estimates 1 and 4 are signed differences. E has three separated horizontal sections with identical y-scales within each metric; the gaps omit no numerical interval.')
p('The matched control and the category-dispersion analysis need not include identical Allen sessions. Matching requires both V1 and an eligible HVA population in the same session. C/D use the larger eligible response populations. [S1–S4]','small')

page('2  Cohorts, inclusion and sample sizes')
p('MouseV2 supplies four within-V1 sampling locations; Allen supplies five cortical HVAs. LP is not an HVA category in this analysis. Common waveform-QC conventions are amplitude cutoff &lt;0.1, presence ratio &gt;0.8 and ISI-violation ratio &lt;0.5, retained through the accepted metric sources. Unit-level anatomical VISp restriction is applied to MouseV2.')
table(['Metric','Population','Mice / sessions','Recordings','Cells'],[[short[x.metric],{'MouseV2 V1':'MouseV2 V1','Allen HVA':'Allen HVAs','Allen matched V1 + HVA':'Allen control: both groups'}[x.population],f'{x.n_mice} / {x.n_sessions}',str(x.n_penetrations),str(x.n_cells)] for x in counts.itertuples()], [82,158,90,75,102])
p('“Recordings” here counts contributing physical probe penetrations. Control cell totals combine V1 and HVA cells. For TTFS the control has 473 V1 / 1,428 HVA cells; for F1/F0 1,953 / 6,142; for timescale 866 / 2,604. These are not matched neuron pairs.','small')
h('Metric-specific gates')
p('<b>TTFS:</b> accepted positive-response-selected, preferred-polarity latency below 100 ms. <b>Phase modulation:</b> positive harmonized conventional F1/F0 followed by log10 transformation. <b>Timescale:</b> accepted fits with 1 ≤ τ ≤ 300 ms, &gt;50 fitting-window spikes and fitted error &lt;20 ms. Each recording/category requires at least five eligible cells; for MouseV2 timescale this is reapplied per trial draw.')
p('The displayed MouseV2 timescale draw contains 1,503 cells. The union across ten draws contains 2,113 distinct cells; it must not be reported as the displayed sample size. The eight animals remain the independent resampling units.')
h('Probe entry is not cell location')
p('No eligible VISp cells are excluded because a probe’s superficial projection lies outside V1. A sloped probe can record deeper V1 cells after entering elsewhere. Two of 32 full-ledger superficial-contact projections lie outside dorsal VISp. Current cartoons do not plot these coordinates; Figure 4 covers sampling details. The historical projection audit is retained as inclusion provenance. [S2, S5]')

page('3  The model in plain language')
p('For each metric and each population separately, the analysis fits a weighted linear model to the eligible single-neuron responses. It estimates a mean for each named category while allowing each recording session to have a common additive offset.')
p('<b>Model:</b> yᵢ = β(categoryᵢ) + γ(sessionᵢ) + εᵢ. All category indicators enter the design. All but the first sorted session indicator enter as nuisance terms; there is no additional intercept.')
p('A positive session effect raises all categories recorded in that session. Category coefficients describe systematic differences associated with the named locations or areas, rather than a session’s overall response level. This is a fixed-effects weighted least-squares model, not a hierarchical random-effects variance-component model.')
h('How the fitted means become an SD')
p('For K categories, let mₖ = βₖ and m̄ = (1/K)Σmₖ. The category variance is V = (1/K)Σ(mₖ − m̄)², and the category SD is S = √V. K=4 for V1 and K=5 for HVAs. The denominator is K, not K−1: these are the specified finite sets of categories, not an unbiased sample variance over hypothetical randomly sampled areas.')
p('The contrast is D = S<sub>HVA</sub> − S<sub>V1</sub>. It is invariant to shifting every category mean in either population by the same constant. The arbitrary reference session therefore does not affect within-population dispersion. However, separately fitted β centers should not be subtracted as an absolute cross-dataset mean difference.')
h('Equal weighting, explicitly')
p('Each cell receives wᵢ = 1/(K × Aₖ × Rₖₐ × Nₖₐᵣ), where Aₖ is the number of animals contributing category k, Rₖₐ is the number of physical recordings for that animal/category, and Nₖₐᵣ is the eligible cell count in that recording/category. Each category has equal total weight; animals, recordings and cells share weight equally within their respective nesting levels.')
p('Consequently, a densely sampled probe does not dominate simply because it contains more neurons. This target differs from a cell-weighted pooled mean. Fits use numpy.linalg.lstsq with rcond=1e−10 and require a full-rank design. No receptive-field covariates enter the main C–E model. [S2, S3]')
h('What it does not estimate')
p('The model does not isolate noise-free biological variance, a causal effect of area identity, or every individual HVA pairwise difference. Category-by-session interactions are not separately modeled; the target is the fitted average named-category pattern.')

page('4  Harmonization and timescale draws')
p('Two different kinds of matching should not be conflated. <b>Measurement matching</b> aligns trial support and metric definitions between datasets. <b>Matched Allen sessions</b> identify the mean difference within a common recording context. The former does not require pairing MouseV2 animals with Allen animals.')
h('75 bright + 75 dark flashes in each dataset')
p('MouseV2 contains 150 bright and 150 dark flashes. Each of ten subsamples selects 75 of each polarity without replacement within that draw. The draws can overlap. A shared session-wide mask is used for all cells, preserving simultaneous-recording dependence. Seeds are 20260805 + site number. Allen contributes its accepted 150-flash estimate per neuron.')
p('The bridge held binning, exponential fitting and validity rules fixed. Reducing MouseV2 from 300 to 150 flashes changed the historical equal-session mean from 47.53 to 45.92 ms, and mean validity from 0.207 to 0.168. The matched MouseV2 mean remained 2.04 ms above Allen BO and 2.71 ms above Allen FC. These historical populations are not the present Figure 5 cohort. [S6]')
h('Why repeat the subsampling?')
p('Repeated draws reduce dependence on one arbitrary trial selection and expose changes in fit eligibility. They are repeated estimates of the same recordings, not independent neurons or animals. Ten is the implemented computational choice; this companion does not establish that ten draws give converged Monte Carlo precision.')
h('Point estimate and uncertainty are different operations')
p('For timescale, the current point estimate is S = √[(1/10)ΣV<sub>d</sub>]. The Allen V is identical in the ten fits. The contrast subtracts the two resulting SDs. This is not mean(√V<sub>d</sub>) and not the SD of category means averaged across draws.')
p('Each bootstrap resamples whole animals and selects one coherent draw per sampled MouseV2 session occurrence. It refits the model and recomputes the effect. This includes sensitivity to trial selection, but it does not average all ten trial draws within every bootstrap replicate. Allen does not have analogous repeated trial estimates in the saved analysis; measurement-uncertainty treatment remains asymmetric.')
h('The single-draw display is a plotting choice')
p('C/D use draw_id=0, fixed by index, with one accepted estimate per cell and that draw’s category means. E uses the ten-draw variance summary. Thus the displayed V1 timescale SD is 1.585 ms, while E shows 2.072 ms. This discrepancy is disclosed; changing the display or estimator is a future decision, not silently implemented here. [S1, S3]')

page('5  All estimates displayed in E')
p('Each entry is point estimate [95% whole-animal bootstrap percentile interval]. All intervals use 5,000 saved replicates. The SD intervals come from square-root-transformed replicate variances; the difference interval comes from subtracting SDs within each replicate, not subtracting interval endpoints.')
labels={'matched_hva_minus_v1_mean':'Allen mean difference: HVAs − V1','v1_sd':'SD of means: within V1','hva_sd':'SD of means: across HVAs','hva_minus_v1_sd':'SD difference: HVAs − V1'}
for metric in METRICS:
 h(metric)
 rows=[]
 for effect in labels:
  x=e.loc[e.metric.eq(metric)&e.effect.eq(effect)].iloc[0]
  rows.append([labels[effect],interval(x.estimate,x.low,x.high)])
 table(['Quantity','Estimate [95% interval]'],rows,[266,241])
p('The mean difference uses Allen recordings with both eligible V1 and HVA groups. The two-group model includes session fixed effects; each group receives total weight 1/2, with equal HVA-area weights and nested animal/recording/cell weights. The SD comparison uses MouseV2 V1 and the eligible Allen HVA population. [S1, S3, S4]','small')
p('<b>Reading the result:</b> all matched mean-difference intervals exclude zero; all SD-difference intervals contain zero. Positive TTFS/timescale mean shifts and negative phase-modulation shifts are consistent with the expected coarse V1-to-HVA direction. They do not establish a graded sequence among HVAs. No multiplicity correction or equivalence margin is applied.')

page('6  Category means and cell counts in C/D')
p('These are the actual plotted means and eligible cell counts, not the number of dots shown. Dots are deterministic display subsamples. Timescale is the fixed display draw. Units follow each metric. No category-specific confidence intervals are drawn in C/D.')
rows=[]
for metric in METRICS:
 for x in display.loc[display.metric.eq(metric)].itertuples():
  rows.append([short[metric],x.panel, ({'A':'Anterior (A)','E':'Lateral (E)','C':'Posterior (C)','B':'Medial (B)'}.get(x.category,x.category) if x.panel=='D' else x.category), str(x.n_cells),num(x.category_mean)])
table(['Metric','Panel','Category','Cells','Displayed mean'],rows,[100,45,112,90,160],7.8,padding=3)
p('Display centering: m<sub>shown,k</sub> = m<sub>fitted,k</sub> − mean(m<sub>fitted</sub>) + mean(m<sub>raw categories</sub>). The raw category means are cell means; their center gives equal weight to categories. The same translation is applied to every fitted mean within a population/metric, preserving all fitted pairwise differences and their SD. This is a plotting anchor, not a cross-dataset calibration. [S1]','small')

page('7  Pooled distributions and visual choices')
rows=[]
for x in spreads.itertuples():
 rows.append([short[x.metric],x.population,num(x.pooled_cell_sd),num(x.display_sd),num(x.inference_sd)])
table(['Metric','Population','Pooled cell SD','Displayed mean SD','Inferential mean SD'],rows,[95,75,108,114,115])
p('The pooled distributions use eligible cells with equal cell weights. They answer a descriptive cellular-distribution question. The fitted mean SD uses balanced category/animal/recording weights. These distinct weightings are intentional and should not be described as one common variance decomposition.')
p('Response axes show the central 99.6% with padding for readability; all eligible response values enter the estimates. Half-violins show density shapes, enlarged colored diamonds show translated fitted category means, and pooled bars labelled Cells SD / Means SD distinguish cell spread from mean spread. The redundant Pooled SD headings have been removed. The colored category outlines were intentionally omitted from the narrow pooled plots to reduce clutter.')
h('A: anatomical values and connections')
table(['Area','Harris score'],[[x.area,num(x.hierarchy_score)] for x in harris.itertuples()],[120,387])
p('Harris coordinates are the six-area subset of the visual/medial CC+TC+CT iterated hierarchy, without recentering. All 30 directed non-self curves use source-area colors and widths linear in the released ipsilateral normalized connection density, maximum 2.4 pt. Ascending projections arc left and descending projections right. Width denotes anatomical density, not physiological synaptic efficacy. [S7]','small')
h('B: what is illustrative')
p('The HVA order matches C. Marker heights on the right are hand-arranged and carry no area-specific measured meaning. Dashed centers align on the page with anatomical V1 and the mean HVA height in A, but this is not a calibration from anatomical score to functional response. Both cortical cartoons directly reuse the original handmade vector paths and marker coordinates from Artboard 11 copy.svg, with current category colors and typography. Their outlines and marker locations are illustrative, not atlas contours or measured samples. Sampling details belong to Figure 4.','small')

page('8  Supporting variance and normalized results')
p('The original primary calculation was in variance units. The figure now uses square roots for readability. Both are retained here; they are not interchangeable contrasts. Timescale point variances average ten draw-specific variances.')
rows=[]
for x in primary.itertuples():
 rows.append([short[x.metric],{'v1':'Within V1','hva':'Across HVAs','hva_minus_v1':'HVA − V1'}[x.effect],interval(x.estimate,x.interval_low,x.interval_high)])
table(['Metric','Variance quantity','Estimate [95% interval]'],rows,[105,125,277])
p('Variance units are ms² or squared log10 F1/F0 units. All three variance-difference intervals contain zero. Nonnegative component-variance intervals excluding zero are not valid tests for the presence of a category effect; see validation in Section 10.','small')
h('Normalized dispersion: 100 × V / total weighted variance')
rows=[[short[x.metric],{'v1':'Within V1 (%)','hva':'Across HVAs (%)','hva_minus_v1':'Difference (pp)'}[x.effect],interval(x.estimate,x.interval_low,x.interval_high)] for x in norm.itertuples()]
table(['Metric','Quantity','Estimate [95% interval]'],rows,[105,125,277])
p('This fraction is not cross-validated predictive variance explained and is not classical omega-squared. For timescale, saved normalized summaries average draw-specific fractions; they should not be reconstructed by dividing separately averaged quantities. [S3, S8]','small')

page('9  Cell-level association and estimation noise')
h('One-way cell-weighted omega-squared')
table(['Metric','Population','ω² [95% interval]'],[[short[x.metric],x.population,interval(x.omega_squared,x.low,x.high)] for x in omega.itertuples()],[112,95,300])
p('ω² = [SS<sub>between</sub> − (K−1)MS<sub>within</sub>] / [SS<sub>total</sub> + MS<sub>within</sub>]. It uses raw cell means, equal cell weights and no session adjustment. SS<sub>total</sub> = N × pooled-cell SD². Negative estimates are retained. All six summaries have 5,000 valid whole-animal bootstrap replicates and zero failed replicates. Timescale averages draw-specific ω² values. This is an association summary, not the E estimator or a multilevel variance decomposition. [S9]','small')
h('Exploratory correction for noise in estimated means')
table(['Metric','Pop.','Plug-in V','Estimated noise','Corrected V'],[[short[x.metric],x.population,num(x.plugin_variance),num(x.estimated_noise_variance),num(x.corrected_variance)] for x in noise.itertuples()],[110,45,110,120,122])
p('Let P = I − 11′/K and C be the animal-clustered covariance of fitted category means. The correction subtracts tr(PC)/K from m′Pm/K. Covariance uses a CR1 small-sample correction, checked against direct animal score sums. This is approximate and particularly uncertain with eight V1 animals; it has no new confidence intervals.')
p('Corrected variance is negative for V1 TTFS and HVA F1/F0, so their corrected SD differences are undefined—not clipped to zero. The timescale corrected SD difference is +2.295 ms, compared with +1.317 ms uncorrected. Corrected variance differences are +4.648 ms², −0.0003021 squared-log units and +9.304 ms². The sensitivity shows that estimation precision can matter; it does not supply a replacement inferential result. [S10]')

page('10  What the bootstrap has been validated to do')
p('The analysis uses 5,000 whole-animal resamples. Cells and recordings remain with the sampled animal, and duplicated animal occurrences are relabeled. The model is refitted, eligibility floors reapplied where relevant, and effects recalculated. MouseV2 and Allen populations are resampled separately; their effects are combined by bootstrap replicate index. This is not a biological pairing of cohorts.')
h('Observed-design simulation results')
p('Synthetic outcomes retained observed cluster sizes and missingness and included session shifts, recording shifts and cell noise. Known category variance was 0.25 in standardized squared units. Initial screening used eight simulations per condition; the targeted V1 TTFS follow-up used 30 simulations with 100 bootstraps each. These small simulations are diagnostics, not precise coverage estimates.')
rows=[]
for x in sim.loc[sim.condition.eq('known')].itertuples():
 rows.append([short[x.metric],x.population,str(x.simulations),num(x.mean_estimate),f'{100*x.interval_coverage:.1f}%'])
x=targeted.loc[targeted.condition.eq('known')].iloc[0]
rows.append(['TTFS follow-up','V1',str(int(x.simulations)),num(x.mean_estimate),f'{100*x.interval_coverage:.1f}%'])
table(['Design','Pop.','Simulations','Mean V (truth .25)','95% coverage'],rows,[110,45,85,145,122])
p('<b>Finite-sample warning:</b> targeted TTFS coverage was 83.3% (25/30), not 95%. The initial eight-simulation designs also showed low coverage. This supports computational recoverability, not uniform nominal interval calibration. These studies assessed component dispersion, not a full validation of every displayed SD-difference interval.')
h('Why component SD intervals are not significance tests')
p('Under a true zero category variance, fitted category means still differ because of estimation noise. Every initial null design had positive plug-in bias and zero coverage of the boundary by ordinary percentile intervals. The targeted V1 TTFS null mean V was 0.01490 despite truth zero. Thus a lower SD/variance confidence limit above zero does not establish a real category effect.')
p('No category-loss or rank failures occurred in these model simulations. The archived significance audit verified all 5,000 saved draws for each displayed difference. Separate HVA RF/hierarchy bootstraps had three TTFS category-loss failures among 15,000 attempted fits, recorded rather than replaced. [S5, S11, S13]')

page('11  Tail summaries and interpreting zero')
p('The following omitted summaries were recovered from the saved bootstrap distributions. They are <b>bootstrap sign-tail summaries</b>, not p-values from a separately fitted null distribution or randomization test. They should not override the intervals or the validation limitations.')
rows=[]
for x in tails.itertuples():
 b=min(x.n_le_zero,x.n_ge_zero)
 val=f'{2*b/5000:.4f}' if b else '0/5,000 in opposite tail'
 rows.append([short[x.metric], 'SD difference' if x.effect=='hva_minus_v1_sd' else 'Allen mean difference',str(x.n_le_zero),str(x.n_ge_zero),val])
table(['Metric','Effect','≤ 0 draws','≥ 0 draws','Two-sided tail summary'],rows,[95,117,75,75,145])
p('For nonzero tails the reported calculation is min[1, 2 × min{Pr*(effect ≤ 0), Pr*(effect ≥ 0)}]. With no opposite-sign draws, an earlier audit printed “p &lt; 1/5000” by a requested reporting convention. That is not a confidence bound on the unknown tail probability; this report retains the actual count instead. No multiplicity adjustment was performed. [S5]','small')
h('What the current numbers support')
p('HVA/V1 SD point ratios are approximately 3.32 for TTFS, 0.46 for phase modulation and 1.64 for timescale. These are descriptive ratios, not interval-estimated fold changes. They show why “all intervals cross zero” must not be translated into “all spreads are equal.”')
p('The signed SD difference is the correct direct comparison of the two component spreads. Overlap between their separate intervals is not the comparison test. Equal spread is compatible with the saved intervals, but substantial differences also remain compatible—up to +2.13 ms for TTFS and +2.93 ms for timescale at their upper endpoints.')
p('A claim of practical equivalence would require an independently justified margin and an interval or test evaluated against it. No such margin was frozen. “Comparable scale” should be tied to response units or a clearly stated reference, not to nonsignificance alone. [S16]')
h('Suggested interpretation')
p('“Variation among HVA means was evaluated against variation among sampled V1 locations. Across-HVA spread was larger for TTFS and timescale but smaller for phase modulation; intervals for all three spread differences included zero. The uncertainty does not establish practical equivalence.”')

page('12  Sensitivity to weighting and inclusion')
p('These are saved point-estimate sensitivities, not new confidence intervals. SD differences below are freshly derived as √V<sub>HVA</sub> − √V<sub>V1</sub> from the archived sensitivity variances, preserving the figure’s units.')
rows=[]
for x in sens.itertuples():
 if x.analysis=='timescale_population' and x.scenario=='primary':continue
 label=f'Floor {x.cell_floor}; '+('balanced' if x.weighting=='balanced' else 'equal cells')
 if x.analysis=='timescale_population':label={'ten_complete_draws':'Valid in all 10 draws','fit_error_lt_10_ms':'Fit error <10 ms'}[x.scenario]
 rows.append([short[x.metric],label,num(np.sqrt(x.v1_dispersion)),num(np.sqrt(x.hva_dispersion)),num(np.sqrt(x.hva_dispersion)-np.sqrt(x.v1_dispersion))])
table(['Metric','Scenario','V1 SD','HVA SD','Difference'],rows,[94,160,78,78,97],7.6)
p('A tighter fit-error threshold changes which phenotypes are retained; it is not automatically a cleaner estimate of the same population. Requiring validity in every draw selects for estimator stability. The older August timescale sensitivity had different RF/population restrictions and is not substituted for these current results. [S12]','small')
h('Leave-one-animal-out component SD ranges')
rows=[]
for (metric,pop),g in loo.groupby(['metric','population'],sort=False):
 vals=pd.to_numeric(g.leave_one_out_dispersion,errors='coerce')
 rows.append([short[metric],pop,f'{np.sqrt(vals.min()):.4g}–{np.sqrt(vals.max()):.4g}',str(g.failure.ne('').sum())])
table(['Metric','Population','Leave-one-out SD range','Failures'],rows,[125,85,202,95],7.6)
p('These ranges quantify influence, not confidence intervals. V1 TTFS and timescale are sensitive to individual animals. All failed fits, if any, are counted rather than interpreted as zero. [S12]','small')

page('13  HVA hierarchy and receptive-field analyses')
p('<b>Separate supporting analysis.</b> These fits ask whether response changes with hierarchy score within HVAs; they do not estimate the main HVA-versus-V1 spread comparison. The saved regressions use rounded <b>Siegle</b> scores (LM −0.093, RL −0.059, AL 0.152, PM 0.327, AM 0.441), even though the current schematic uses Harris. No regression was refitted when the schematic changed.')
rows=[]
for x in slopes.itertuples():
 for label,base in [('Unadjusted','unadjusted_slope'),('RF adjusted','adjusted_slope'),('Adjusted − unadjusted','adjusted_minus_unadjusted')]:
  rows.append([short[x.metric],label,interval(getattr(x,base),getattr(x,base+'_low'),getattr(x,base+'_high'))])
table(['Metric','Same-cell slope comparison','Estimate [95% interval]'],rows,[98,160,249],7.8)
p('Units are response units per hierarchy-score unit. Both fits use identical locally shared-support cells, session nuisance effects and balanced weights. Adjustment adds azimuth, elevation, their squares and interaction. All paired slope-change intervals contain zero. A significance transition between separate slopes is not evidence of attenuation. [S13]','small')
h('Selection and adjustment must be separated')
rows=[]
for metric in METRICS:
 g=ladder.loc[ladder.metric.eq(metric)]
 # Column names are read from the released table rather than inferred from plots.
 for _,x in g.iterrows():
  rows.append([short[metric],str(x['stage']).split('_',1)[0],str(int(x['n_cells'])),num(x['hierarchy_slope'])])
table(['Metric','Stage','Cells','Slope'],rows,[125,65,115,202],7.2)
p('Stages: 1 response eligible; 2 RF qualified; 3 local shared support; 4 same cells, RF adjusted. Stages 1→2→3 change the population. Only 3→4 isolates the specified adjustment. Cross-population RF support failed the frozen gate, so no RF-adjusted V1-versus-HVA spread difference is claimed. RF adjustment is conditional modeling, not causal mediation.','small')

page('14  Historical choices and alternatives')
h('Harmonization was measurement alignment, not forced equality')
p('The F1/F0 bridge aligned a one-second window, 15 trials per condition, four shared orientations, common temporal-frequency support, SF 0.04 cycles/degree and contrast 0.8. The full-cohort comparison found MouseV2 − Allen BO = −0.0283 log10 [−0.0554, −0.0007], and MouseV2 − Allen FC at TF=2 = −0.0365 [−0.0617, −0.0087]. The later figure-specific four-probe center difference was −0.03796. These are different support/weighting summaries, not contradictory repetitions of one estimand. [S14]')
p('The historical timescale bridge reduced the MouseV2–Allen BO offset from +3.65 to +2.04 ms after matching flash counts. Harmonization therefore materially improved comparability. It does not prove residual cohort offsets are zero or independently calibrate physical light onset for TTFS. [S6]')
h('Why the mean difference remains Allen-only')
p('The current choice retains the matched Allen mean difference as an independent coarse V1-to-HVA benchmark. Subtracting the populations in C/D is possible descriptively, but would combine area and cohort differences. That alternative was discussed and not adopted; the current PDF does not silently replace the control.')
h('Why “Central” Allen V1 is not a fifth adjusted V1 location')
table(['Metric','Descriptive five-location V [95% interval]','Session-model rank'],[[short[x.metric],interval(x.dispersion,x.dispersion_low,x.dispersion_high),f'{x.fixed_session_rank}/{x.fixed_session_parameters}'] for x in central.itertuples()],[110,292,105])
p('Allen Central is nested in a separate cohort/session set. Category and session terms are rank deficient by one parameter. These archived five-location values are descriptive mixed-cohort variances in squared response units, not an identified anatomical-location effect. [S15]')
h('What has changed across versions')
p('Older figures used session-mean or cell-level omega-squared, sometimes including LP or different RF gates. The current main result is balanced, session-adjusted finite-category dispersion. The current figure changes variance to SD for readability, retains the Allen control, and adds the two component SDs. Earlier percentages, hierarchy slopes and sensitivity ratios must not be relabeled as the current effect.')

page('15  Conclusions, limitations and open decisions')
h('Findings that are directly supported')
p('1. The matched Allen mean differences are +5.45 ms for TTFS, −0.0898 log10 for phase modulation and +4.77 ms for timescale; their saved intervals exclude zero.')
p('2. Estimated across-HVA spread exceeds within-V1 spread for TTFS and timescale, while phase-modulation spread is smaller across HVAs. The signed differences remain uncertain; all three saved intervals include zero.')
p('3. Cell-level heterogeneity is much broader than category-mean spread. That is a descriptive scale comparison, not proof that area identity is biologically irrelevant.')
p('4. The RF-qualified HVA hierarchy analyses show positive TTFS and timescale slopes on their selected populations. Their existence is not contradicted by an uncertain HVA-versus-V1 spread difference: ordering, total spread and comparison with a benchmark are different questions.')
h('Limitations that change interpretation')
p('Eight MouseV2 animals constrain precision. Validation showed plug-in boundary bias and imperfect component-variance interval coverage. Category-mean noise, cohort composition, RF location, layer and behavioral state are not fully separated. A bootstrap cannot remove these systematic differences. The figure does not establish equal spreads, absence of HVA differences, or a universal lack of hierarchy.')
h('Decisions still open')
p('<b>Timescale display:</b> whether to keep a fixed coherent trial draw or adopt an explicitly draw-aggregated representation. <b>Timescale inferential target:</b> whether √mean(V) remains the desired summary and whether more trial draws are needed. <b>Claim strength:</b> whether “comparable scale” needs a prespecified practical margin. These were discussed but no replacement analysis is implemented in this companion.')
p('<b>Do not average away eligibility silently.</b> Neurons can pass timescale validity in some draws but not others. Averaging available fits per neuron changes weighting and selection. Stacking all fits as independent cells is also invalid. Any revision must state its unit of analysis and missing-fit rule.')
h('Suggested paper framing')
p('“We compared variation among the means of five HVAs with variation among four sampled locations within V1. The component estimates and their uncertainty place between-area differences in the context of within-area heterogeneity. These comparisons do not establish practical equivalence and do not negate ordered differences on selected HVA populations.”')
p('The current confidence intervals should be described as pointwise whole-animal bootstrap percentile intervals, with finite-sample and plug-in-estimator limitations documented. Avoid interpreting component SD bounds or bootstrap sign-tail summaries as independent confirmation of an area effect.')

# Provenance references: stable source IDs cited in the prose.
refs=[
 ('S1','Current figure, estimates, displayed means and caption','artifacts/figure5/v13_shift_hierarchy/'+STEM+'_legend.md'),
 ('S2','Eligibility, balanced weights and category/session model','scripts/figure4_rerun_core.py'),
 ('S3','Primary variance and whole-animal bootstrap implementation','scripts/run_figure4_primary_dispersion.py'),
 ('S4','Matched Allen control and weighting','scripts/run_figure4_matched_control.py'),
 ('S5','Counts, significance audit and provenance qualifications','artifacts/figure4_rerun/significance/REPORT.md'),
 ('S6','Timescale trial-count bridge','data/imports/mousev2_timescale_trial_bridge_v1/README.md'),
 ('S7','Harris/Siegle score and connection-density provenance','artifacts/figure4_rerun/hierarchy_comparison_v2/REPORT.md'),
 ('S8','Normalized-dispersion definition','artifacts/figure4_rerun/v4_primary_dispersion/NORMALIZED_DISPERSION.md'),
 ('S9','Cell-level omega-squared method','artifacts/figure4_rerun/cell_omega/methods.md'),
 ('S10','Exploratory mean-estimation-noise correction','artifacts/figure4_rerun/mean_variance_noise_check/methods.md'),
 ('S11','Model-validation decision and targeted follow-up','artifacts/figure4_rerun/v3_model_validation/VALIDATION_DECISION.md'),
 ('S12','Weighting, population and animal-influence sensitivities','scripts/audit_figure4_primary_sensitivities.py'),
 ('S13','Same-cell RF-conditional hierarchy and selection ladder','artifacts/figure4_rerun/v5_hva_hierarchy/HVA_HIERARCHY.md'),
 ('S14','Full-cohort harmonized V1 F1/F0 bridge','artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/04_allen_full_cohort_harmonized/FULL_COHORT_CHECKPOINT.md'),
 ('S15','Final analysis report and historical reconciliation','artifacts/figure4_rerun/v8_final/Figure4_rerun_final_report.md'),
]
page('16  Supporting within-V1 shuffle tests')
p('These calculated supporting results can be reported in the main text even though they are not plotted in Figure 5. They test location-associated variation within V1, a different question from the inter-HVA minus intra-V1 spread contrast in E. They are retained with their original analysis populations and estimators, rather than presented as new tests of the current SD model.')
h('Recovered session-mean and cell-level results')
shuffle_rows=[]
for version,path in [('Session means','Figure3/Figure4_location_controls_statistics.csv'),('Cell level','Figure3/Figure4_CDE_single_cell_candidate_statistics.csv')]:
 frame=read(path)
 for x in frame.loc[frame.comparison.eq('v1')].itertuples():
  shuffle_rows.append([short[x.metric],version,interval(x.observed,x.bootstrap_low,x.bootstrap_high),f'{float(x.permutation_p_greater):.5f}'])
table(['Metric','Analysis','ω² [95% bootstrap interval]','Shuffle p'],shuffle_rows,[90,90,247,80],7.6)
p('All six saved V1 tests used eight sessions, 5,000 permutations and uncorrected one-sided p = (1 + number of null statistics ≥ observed)/(5,001). Session means: labels are permuted among available location means within each session. Cell level: all cells in each session/location block move together when the available location labels are permuted. The statistic is bias-corrected one-way omega-squared; negative estimates are retained. Exchangeability of location labels within sessions is an assumption.','small')
h('Suggested main-text result')
p('“In a supporting analysis of session-level location means, within-session label-shuffle tests did not detect location-associated variation within V1 for TTFS, phase modulation or response timescale (one-sided, 5,000 shuffles; uncorrected p = 0.888, 0.127 and 0.541, respectively). These tests address detectable within-V1 location effects, separately from the comparison of within-V1 and across-HVA spread in Figure 5.”')
h('Scope and reconciliation')
p('The later cell-level block-shuffle tests likewise did not cross the 0.05 threshold (p = 0.892, 0.053 and 0.539). Phase modulation is close to that threshold. Its positive bootstrap interval and p = 0.053 are different inferential constructions, not a transcription discrepancy. Neither a nonsignificant result nor an interval crossing zero establishes absence of variation.')
p('The older cell-level timescale analysis averaged available valid fits per neuron without fit-draw resampling and included 773 V1 neurons, versus 1,503 in the current display draw and 2,113 across draws. Its result must not be assigned to the current ten-draw estimator. A separate saved session-layout candidate reports V1 p = 0.846, 0.115 and 0.591, with differing point estimates as well as a different random stream; do not mix those values into the selected original session-mean result.')
p('Sources: Figure3/Figure4_location_controls_statistics.csv; Figure3/Figure4_CDE_single_cell_candidate_statistics.csv; Figure3/Figure4_CDE_session_candidate_statistics.csv. Implementation: scripts/build_figure4_location_controls.py and scripts/build_figure4_cell_location_controls.py. No new shuffle runs were performed.','small')
for path in ['Figure3/Figure4_CDE_session_candidate_statistics.csv','Figure3/Figure4_CDE_single_cell_method.json','Figure3/Figure4_CDE_single_cell_notes.md','scripts/build_figure4_location_controls.py','scripts/build_figure4_cell_location_controls.py','scripts/figure3_robust_spread_comparison.py']:
 source(path)

page('17  Sources and reproducibility')
p('Repository root: /home/huklaban5/Documents/MouseV2/Hierarchy_analysis/neuropixels_platform_paper. Source IDs refer to local files below. This PDF summarizes saved outputs; it does not rerun raw-spike extraction, model fitting or bootstrapping.','small')
for ident,title,path in refs:
 source(path)
 p(f'<b>{ident} · {title}</b><br/>'+html.escape(path).replace('/','/\u200b'),'small')
p('<b>S16 · Statistical interpretation:</b> Lakens (2017), Equivalence Tests: A Practical Primer. <link href="https://doi.org/10.1177/1948550617697177" color="#176887">doi:10.1177/1948550617697177</link>. Difference intervals crossing zero do not establish practical equivalence.','small')
p('Anatomical papers: <link href="https://doi.org/10.1038/s41586-019-1716-z" color="#176887">Harris et al. 2019</link>; <link href="https://doi.org/10.1038/s41586-020-03171-x" color="#176887">Siegle et al. 2021</link>; <link href="https://doi.org/10.1016/j.neuron.2022.01.027" color="#176887">Jia et al. 2022</link>. Jia replots Harris; it is not an independent numerical hierarchy.','small')
p('Reproduce with: python scripts/build_figure5_methods_companion.py. Companion files include a Markdown transcript, source SHA-256 manifest and validation JSON. The historical analysis had a dirty working tree; file hashes provide source identification without claiming a fully committed original snapshot.','small')

# Capture implementation sources and relevant manifests in addition to data tables.
for path in ['scripts/build_figure5_methods_companion.py','scripts/build_figure4_integrated.py','scripts/build_figure4_rerun_outputs.py','scripts/build_hierarchy_schematic_row.py','scripts/mousev2_timescale_trial_bridge.py','artifacts/figure4_rerun/v8_final/final_analysis_manifest.json','artifacts/figure4_rerun/v3_model_validation/ttfs_v1_targeted/MODEL_VALIDATION.md','artifacts/figure4_rerun/v5_hva_hierarchy/RF_EFFECT_LADDER.md','artifacts/v1_systemic_gap_audit_v1/04_metric_replacement/07_figure3_harmonized_f1_f0/FIGURE3_HARMONIZED_F1_F0_CHECKPOINT.md']:
 source(path)

def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#d0dbe1'));c.line(44,39,551,39)
 c.setFont('DV',7);c.setFillColor(colors.HexColor('#667787'))
 c.drawString(44,26,'Figure 5 · Methods and findings · v13 analysis companion · 29 Sep 2026')
 c.drawRightString(551,26,str(doc.page))
 c.setTitle('Figure 5 — Methods, model, choices and findings')
 c.setAuthor('MouseV2 / Allen visual hierarchy analysis')
pdf=OUT/'Figure5_methods_model_and_findings.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),leftMargin=44,rightMargin=44,topMargin=40,bottomMargin=52)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'Figure5_methods_model_and_findings.md').write_text('\n'.join(md))
(OUT/'source_manifest.json').write_text(json.dumps({'figure_version':'v13_shift_hierarchy','sources':used},indent=2)+'\n')
(OUT/'validation.json').write_text(json.dumps({'plotted_estimates_reproduced':12,'bootstrap_replicates_per_estimate':5000,'new_model_fits':0,'new_bootstrap_runs':0,'planned_sections':sections,'figure_unchanged':True,'numerical_tables_identical_to_Figure4_v22':True,'current_figure_version':'v13_shift_hierarchy'},indent=2)+'\n')
print(pdf)
