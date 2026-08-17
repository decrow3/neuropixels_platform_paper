# V1 / screen-offset signal stocktake

## Core identifiability point

RF gradients, field sign, map reversals, and CCF-to-RF geometry locate sampled
anatomy within a retinotopic map, but they are unchanged by adding the same visual
translation to every cell in a session. They constrain animal-specific map shape;
they do not by themselves define the absolute screen origin.

An absolute or cross-session offset therefore needs both:

1. a geometry term that relates CCF anatomy to the retinotopic map; and
2. an anchor tied to screen, gaze, or a published retinal-coordinate landmark.

If the population template is learned only from the same uncorrected sessions,
only relative session offsets are identifiable; the population mean offset is
absorbed into the template.

## Current evidence ranking

### 1. CCF anatomy plus the complete V1/HVA RF topology — strongest existing map signal

Use fixed anatomical cell positions and allow the population retinotopic template
to warp per animal. The informative features are RF progression along probes,
two-dimensional CCF-to-RF Jacobians, map reversals across V1/HVA boundaries, and
the V1-LM-RL junction. The HVA reversals add constraints rather than being treated
as unrelated maps.

This is the most biologically direct use of the existing RF centers. It remains a
geometry/localization signal, however, and requires an absolute anchor. Previous
held-out staged geometry was not yet production-ready: median held-probe error was
22.5 deg with staged translation, versus 15.3 deg for the conventional no-
translation comparator.

Garrett et al. explicitly corrected head/eye alignment by assigning visual origin
to the V1-LM-RL intersection, which coincides with the horizontal/vertical-meridian
intersection. This makes that junction the most defensible literature-derived
absolute landmark.

The original per-animal ISI maps would be even better because they directly contain
the targeting map and its screen coordinates. The current local/released NWB audit
has not found those raw maps; only achieved CCF tracks, area labels, RFs, and the
fact that ISI guided targeting are presently available. If Allen can supply the
per-animal ISI products, they should supersede reconstruction from unit RFs.

### 2. Mean screen-gaze direction plus rig geometry — best underused direct anchor

Allen NWBs can contain spherical screen-gaze coordinates, pupil/eye ellipse fits,
and rig geometry. Our earlier gaze correction centered gaze within each Gabor block,
so it tested trial-to-trial motion but removed the session mean that could explain
a static V1/screen translation.

Availability is substantial but incomplete: 25/31 audited BO1.1 sessions have
valid eye tracking. Four extracted examples have mean spherical gaze centers that
differ by roughly 10 deg across sessions. Dynamic gaze correction was weak in the
four-session pilot, but that does not test the static mean-gaze hypothesis.

Important caveat: Allen's documentation states that absolute screen-gaze estimates
have no accuracy guarantee because of rig degrees of freedom. Mean gaze should be
used as a calibrated prior or covariate, with equipment/date effects and held-out
validation—not accepted as ground truth.

### 3. RF detection and screen censoring — strongest unused screen-boundary signal

The current analyses mainly use cells with fitted RF centers. Yet whether a cell's
RF is significant/on-screen is directly informative about where the finite stimulus
screen intersects the anatomical retinotopic map.

The unit table contains 8,603 V1 units across 56 sessions. `on_screen_rf` is present
for every unit: 5,732 are true and 2,871 false. `p_value_rf` is available for 7,494
units, while improved parametric fits cover 5,529. A censored likelihood could use
all V1 units, the known 81-position Gabor support, RF-fit likelihood/censor flags,
and expected RF size rather than discarding cells near or beyond the screen.

This is directly tied to screen offset, but response quality, layer, area, and RF
size must be nuisance variables. It should be evaluated with held-out cells and
sessions to prevent the screen edge from manufacturing a shift.

### 4. Simultaneous subcortical retinotopy — strong independent biological anchor

The same sessions contain RF measurements and CCF anatomy in LGd and LP, sometimes
superior colliculus. The released table contains 2,582 LGd units across 33 sessions
and 4,849 LP units across 42 sessions; 2,357 and 4,395 respectively have released RF
centers. These structures share retinal and eye-position offsets with cortex but
have independent anatomical maps.

Literature supports orderly dLGN retinotopy and retinotopic order in LP. Agreement
between independently fitted cortical and subcortical offsets would be much more
convincing than agreement among multiple V1-derived statistics. Improved RF models
would need to be extended to these structures, with nucleus-specific RF-size and
response models.

### 5. Cortical magnification / RF-gradient magnitude — useful translation-invariant locator

Garrett reports greater magnification near the V1-LM-RL central representation.
The local CCF-to-RF Jacobian, gradient magnitude, and anisotropic magnification can
therefore help determine where a probe lies in the template without using absolute
RF position. This is better motivated than RF-center scatter, but still needs one
of the absolute anchors above to determine screen translation.

### 6. Spatial-frequency organization — plausible weak auxiliary

Mouse V1 has a published elevation-dependent cutoff-SF gradient, with higher cutoff
SF in upper visual field/posterior V1. The local unit table has `pref_sf_sg` for
4,796 V1 units, but it is quantized to five stimulus values and is preferred rather
than cutoff SF. It may provide a weak elevation cue after controlling for layer,
RF quality, running/arousal, and session.

Previous SF/TF map comparisons did not consistently validate RF-derived
translations. SF should therefore be evaluated independently and given low weight;
TF has still weaker literature justification as an absolute-position anchor.

## Signals that have not earned registration weight

- Absolute RF size: spatial structure is weak and heterogeneous; held-out size did
  not corroborate covariance translations.
- Covariance trace: contains real spatial organization, but only one of five
  independently selected high-quality sessions showed a compact, physically stable
  localization basin. Nested uncertainty for that case was about 14 deg.
- Covariance anisotropy and pooled pairwise anisotropy: not split-half reliable and
  often outlier/boundary dominated.
- Trial-to-trial gaze correction: negligible held-out RF-fit gains in three of four
  pilot sessions and only a tiny gain in the fourth. This does not rule out static
  mean gaze.
- Joint SF/TF-driven transforms: capable of optimizing maps, but not yet independent
  evidence for the true screen offset.

## Other available variables worth using as nuisance or precision information

- RF-fit test deviance, train/test gain, censoring, edge distance, parameter bounds,
  rotation/axis ratio, and optimizer diagnostics;
- released `p_value_rf`, `on_screen_rf`, response amplitude/firing rate, and RF
  latency;
- cortical layer/depth and probe identity;
- running modulation, pupil area, within-trial gaze variability, and session state;
- acquisition date/equipment and NWB monitor/eye-camera rig geometry;
- area-specific HVA coverage priors and simultaneous probe targeting pattern.

These should mainly control reliability or selection, not be allowed to create an
offset unless a held-out prediction supports that role.

## Recommended model

Use a hierarchical, multimodal model with one latent visual translation per session:

- fixed CCF cell anatomy;
- animal-specific smooth warp of the Zhuang/Garrett cortical retinotopy template;
- RF-center likelihood weighted by fit uncertainty;
- field-sign, gradient, and V1/HVA boundary/junction constraints;
- screen-detection/censoring likelihood for fitted and unfitted V1 cells;
- calibrated mean-gaze/rig prior where eye tracking is valid;
- independently fitted LGd/LP offset as validation or an additional likelihood;
- SF elevation only as a weak held-out auxiliary.

The first concrete next test should be mean gaze plus screen censoring, because both
are directly tied to the screen offset and neither has been tested by the existing
geometry/size/covariance work. Estimate each without RF-size or SF/TF, then ask
whether they predict the direction and magnitude of the CCF/template RF residual in
held-out animals.

## Primary references

- Garrett et al. 2014, *Topography and Areal Organization of Mouse Visual Cortex*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4160785/
- Zhuang et al. 2017, *An extended retinotopic map of mouse cortex*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5218535/
- Siegle et al. 2021, *Survey of spiking in the mouse visual system reveals
  functional hierarchy*: https://doi.org/10.1038/s41586-020-03171-x
- AllenSDK ecephys session data and gaze documentation:
  https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_session.html
- Zhang et al. 2015, *The Topographical Arrangement of Cutoff Spatial Frequencies
  across Lower and Upper Visual Fields in Mouse V1*:
  https://www.nature.com/articles/srep07734
- Allen et al. 2016, *Visual input to the mouse lateral posterior and posterior
  thalamic nuclei*: https://doi.org/10.1113/JP271707
