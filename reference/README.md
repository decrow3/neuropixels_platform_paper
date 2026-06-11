# OpenScope Pilot Analysis

## Project Overview
This project is part of the OpenScope initiative by the Allen Institute. It aims to analyze pilot data and provide insights into experimental outcomes.

## Objectives
- Clarify project requirements.
- Scaffold the project structure.
- Customize the codebase to meet analysis needs.
- Install necessary extensions and dependencies.
- Compile and run the project.
- Ensure comprehensive documentation.

## Next Steps
1. Define the analysis plan.
2. Identify required tools and frameworks.
3. Set up the project environment.
4. Begin data analysis and visualization.

Feel free to update this document as the project progresses.

## Pilot Analysis Steps
To prepare for the analysis pipeline:

1. **Prepare Analysis Pipeline**: Develop a robust pipeline for analyzing existing test data.
2. **Test Data Analysis**: Validate the pipeline using existing test data to ensure readiness.
3. **V1 Data Integration**: Ensure the pipeline is adaptable for V1 data when it becomes available.

Project Proposal:
A critical re-examination of the mouse visual system
MOTIVATION: Beyond V1, the mouse visual cortex has been parcellated into several higher order visual areas (Fig 1A). These area delineations are largely based on anatomical (Fig 1B, Wang & Burkhalter, 2007) and functional (Garrett et al., 2014; Kalatsky & Stryker, 2003) signatures of a reversal in the progression of the visual field. Such reversals indicate a new map of the visual field, and therefore a new visual area, under the assumption of simple and linear retinotopic mapping. However, recent work (Sedigh-Sarvestani et al., 2020; Yu, Rowley et al., 2020) has shown that visual field reversals are a hallmark of non-linear retinotopic mapping within single higher order visual areas of primates and tree shrews (Fig 1B). This prompted us to ask whether the higher order areas of the mouse, delineated by reversals, may in fact be a single area V2. This would be consistent with the partial visual field coverage of higher order areas in the mouse (Zhuang et al., 2017), which only when combined provide near-full coverage of the visual field. In addition, the existence of a single area V2 in mice would make the visual cortex of this species consistent with other rodents, and mammals, who exhibit a single area V2 beyond V1(Rosa & Krubitzer, 1999). As it stands, the current definition of visual areas in the mouse makes the organization of visual cortex in this species distinct from nearly all other studied mammals, questioning the generalizability and translational potential of findings in the mouse.
If this hypothesis is true, how can we explain the observed anatomical and functional differences reported between the higher order visual areas? These include differences in spectral sensitivity (Denman et al., 2018; Rhim et al., 2017), temporal and spatial frequency preferences (Marshel et al., 2011), receptive field size, latency, binocular disparity and several other features. The simplest explanation is that these functional differences are reflective of the different visual field bias of each area (Sedigh-Sarvestani & Fitzpatrick, 2022). For instance, we would expect differences in spatiotemporal frequency and receptive field size between area RL and P, simply due to their bias towards lower central and upper peripheral parts of the visual field. In fact, a survey of the published literature suggests that nearly all functional differences reported across the higher order areas of the mouse are consistent with continuous changes across the retinotopic map – similar to continuous spectrum of functional properties across the retinotopic map of V1 and other single visual areas. Similarly, we suggest retinotopic bias can also explain observed anatomical differences used to delineate a cortical hierarchy (D’Souza et al., 2022; Harris et al., 2019; Wang et al., 2011). For instance, it has been reported that the laminar density of feedback projections from higher order areas back to V1 critically depends on the precise retinotopic match between the target and source regions (Morimoto et al., 2021). This suggests that failing to account for retinotopic bias of higher order visual areas can produce a cortical hierarchy that largely follows retinotopy (Fig 1A), bearing striking resemblance to the hierarchies reported in the literature (D’Souza et al., 2022; Harris et al., 2019).
Hypothesis: Higher order areas of the mouse visual system (except POR) are parts of a single area V2. Observed differences between higher order areas can be attributed to biased visual field representation.
We find that many differences attributed to distinct higher order visual areas can be better explained by a single area V2 if one considers two simple facts: visual field reversals within an area are possible under non-linear retinotopic maps and 2) anatomical and functional difference exist along the gradient of retinotopy within
Figure 1: Higher order visual areas in the mouse are sub-parts of a single area V2. A critical re-examination of functional and anatomical data supporting the current view of multiple higher order visual areas (A), coupled with new evidence of visual field reversals within single areas in the tree shrew (B), and macaque (not shown) supports a new view (C) wherein a single area V2 borders V1 in the mouse, similar to most other mammals.
the same visual area. However, we lack a particular dataset that could either critically strengthen or refute this hypothesis: Neural activity recorded from different retinotopic regions of V1. This data is needed to determine the degree to which functional properties change across the retinotopic map in mouse V1. This measure will allow us to determine whether functional property differences observed across the higher order visual areas go beyond what would be expected from retinotopic bias. We expect that once retinotopic biases are accounted for, the hierarchy of mouse higher order visual areas will collapse into two levels: V1, and a second level consisting of all higher order areas except POR (Fig 1C).
Aim 1: Record 6 single-plane sites across V1 using calcium imaging.
Aim 2: Record multiple sites across the retinotopic map of V1 (at least 4) with Neuropixels.
EXPERIMENTAL DESIGN: The experimental design is identical to the existing Observatory passive viewing datasets, with one major difference. We propose to record from multiple sites across the retinotopic map of V1 (Fig 2). We would need separate cohorts of mice for single-plane calcium imaging across 6 sites, and Neuropixels recordings across 4 sites, in V1. We propose imaging since most publications on functional properties of the mouse visual cortex rely on calcium imaging. We propose electrophysiology to obtain high-temporal precision responses needed to calculate latency, F1/F0, RF size etc to reproduce the hierarchical analysis in (D’Souza et al., 2022; Siegle et al., 2021)
The visual stimulus will be a subset of that used in the Observatory (Brain Observatory 1.1), and will include drifting gratings, natural scenes, and locally sparse noise. We will also need to begin each session with Gabor patches and full-field flashes used to map receptive fields (Siegle et al., 2021).
Stim
Depth
Cre-line
Area
# Mice
Cell-matching?
Drifting Gratings (DG)
Sparse Noise
Natural images
Gabor Patches Full-field Flash
Layer 2/3
Cux2-CreERT2;Camk2a-tTA; Ai93(TITL-GCaMP6f)
VISp: 1
3
Parent
VISp: 2
3
To DG
VISp: 3
3
To DG
VISp: 4
3
To DG
VISp: 5
3
To DG
VISp: 6
3
To DG
Stim
Mouse-line
Probe
Area
# Mice
Priority
Drifting Gratings (DG)
Sparse Noise
Natural images
Gabor Patches Full-field Flash
C57BL/6J
1
VISp: 1
3
Essential
2
VISp: 2
3
Essential
3
VISp: 4
3
Essential
4
VISp: 5
3
Secondary
ANALYSIS PLAN: Armed with data on the degree of functional difference arising from retinotopic location within a single visual area, we will determine: 1) If reported functional differences across the higher order visual areas go beyond what can be explained by retinotopic bias. 2) Whether a hierarchy emerges among higher order visual areas once retinotopic biases are accounted for and 3) Whether functional differences across the two axes of the visual field are similar or different in magnitude. The last point will help tie functional differences across the retinotopic map to ethological needs and demands of the animal. If awarded, lab members will handle analysis, under the advisement of the authors, and our Allen Institute partners. PhD student #1: 50% effort, PhD student #2: 50% effort.
In closing, we want to emphasize that area delineations are not merely semantics. They influence experimental design, bias the interpretation of experimental data, and influence cross-species understanding. Our hypothesis that the mouse has a single area V2 would bring the mouse visual system into alignment with that of other mammals. In addition, it generates a common set of rules for delineating visual areas across species - namely coverage of visual field and distinct anatomical and functional differences between areas. We believe the Allen Institute is uniquely positioned to help test our hypothesis, given the role they have played in establishing the organization of the mouse visual system. In our view, the OpenScope offers a perfect opportunity to test this timely hypothesis.
Figure 2: Design includes several sites in V1
Denman, D. J., Luviano, J. A., Ollerenshaw, D. R., Cross, S., Williams, D., Buice, M. A., Olsen, S. R., & Reid, R. C. (2018). Mouse color and wavelength-specific luminance contrast sensitivity are non-uniform across visual space. eLife, 7. https://doi.org/10.7554/ELIFE.31209
D’Souza, R. D., Wang, Q., Ji, W., Meier, A. M., Kennedy, H., Knoblauch, K., & Burkhalter, A. (2022). Hierarchical and nonhierarchical features of the mouse visual cortical network. Nature Communications, 13(1), 503. https://doi.org/10.1038/s41467-022-28035-y
Garrett, M. E., Nauhaus, I., Marshel, J. H., & Callaway, E. M. (2014). Topography and Areal Organization of Mouse Visual Cortex. Journal of Neuroscience, 34(37), 12587–12600. https://doi.org/10.1523/JNEUROSCI.1124-14.2014
Harris, J. A., Mihalas, S., Hirokawa, K. E., Whitesell, J. D., Choi, H., Bernard, A., Bohn, P., Caldejon, S., Casal, L., Cho, A., Feiner, A., Feng, D., Gaudreault, N., Gerfen, C. R., Graddis, N., Groblewski, P. A., Henry, A. M., Ho, A., Howard, R., … Zeng, H. (2019). Hierarchical organization of cortical and thalamic connectivity. Nature, 575(7781), 195–202. https://doi.org/10.1038/s41586-019-1716-z
Kalatsky, V. A., & Stryker, M. P. (2003). New paradigm for optical imaging: Temporally encoded maps of intrinsic signal. Neuron, 38(4), 529–545. https://doi.org/10.1016/S0896-6273(03)00286-1
Marshel, J. H., Garrett, M. E., Nauhaus, I., & Callaway, E. M. (2011). Functional specialization of seven mouse visual cortical areas. Neuron, 72(6), 1040–1054. https://doi.org/10.1016/j.neuron.2011.12.004
Morimoto, M. M., Uchishiba, E., & Saleem, A. B. (2021). Organization of feedback projections to mouse primary visual cortex. iScience, 24(5), 102450. https://doi.org/10.1016/j.isci.2021.102450
Rhim, I., Coello-Reyes, G., Ko, H. K., & Nauhaus, I. (2017). Maps of cone opsin input to mouse V1 and higher visual areas. Journal of Neurophysiology, 117(4), 1674–1682. https://doi.org/10.1152/jn.00849.2016
Rosa, M. G. P., & Krubitzer, L. A. (1999). The evolution of visual cortex: Where is V2? Trends in Neurosciences, 22(6), 242–248. https://doi.org/10.1016/S0166-2236(99)01398-3
Sedigh-Sarvestani, M., & Fitzpatrick, D. (2022). What and Where: Location-Dependent Feature Sensitivity as a Canonical Organizing Principle of the Visual System. Frontiers in Neural Circuits, 16, 18. https://doi.org/10.3389/FNCIR.2022.834876/BIBTEX
Sedigh-Sarvestani, M., Lee, K. S., Satterfield, R., Shultz, N., & Fitzpatrick, D. (2020). A sinusoidal transform of the visual field in cortical area V2. bioRxiv, 2. https://doi.org/10.1101/2020.12.08.416651
Siegle, J. H., Jia, X., Durand, S., Gale, S., Bennett, C., Graddis, N., Heller, G., Ramirez, T. K., Choi, H., Luviano, J. A., Groblewski, P. A., Ahmed, R., Arkhipov, A., Bernard, A., Billeh, Y. N., Brown, D., Buice, M. A., Cain, N., Caldejon, S., … Koch, C. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. Nature, 592(7852), 86–92. https://doi.org/10.1038/s41586-020-03171-x
Wang, Q., & Burkhalter, A. (2007). Area map of mouse visual cortex. Journal of Comparative Neurology, 502(3), 339–357. https://doi.org/10.1002/cne.21286
Wang, Q., Gao, E., & Burkhalter, A. (2011). Gateways of ventral and dorsal streams in mouse visual cortex. Journal of Neuroscience, 31(5), 1905–1918. https://doi.org/10.1523/JNEUROSCI.3488-10.2011
Yu, H.-H., Rowley, D., Price, N., Rosa, M., & Zavitz, E. (2020). A twisted visual field map in the primate cortex predicted by topographic continuity. Science Advances, 6(6), eaaz8763. https://doi.org/10.1101/682187
Zhuang, J., Ng, L., Williams, D., Valley, M., Li, Y., Garrett, M., & Waters, J. (2017). An extended retinotopic map of mouse cortex. eLife, 6, 1–29. https://doi.org/10.7554/elife.18372