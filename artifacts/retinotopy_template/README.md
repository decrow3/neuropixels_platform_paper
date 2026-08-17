# Retinotopy reference templates

## Selected reference

`zhuang2017_figure9/` is the **primary population template**. Figure 9C/D
provides the pooled V1-and-HVA altitude and azimuth geometry after the authors
centered animals on V1, aligned the major azimuth-gradient direction, and
corrected retinal position at the V1/LM/RL junction.

`garrett2014_figure5/` remains a **sensitivity reference**. It is useful for
testing whether conclusions depend on the chosen published population atlas,
but it is not the default registration target.

Neither template is an Allen CCF transform. The intended next model is a
per-animal mapping between cortical CCF coordinates and retinal coordinates,
regularized toward the Zhuang population geometry. V1 and the HVAs remain one
joint cortical/retinotopic surface; HVA reversals and distortions are useful
constraints on that mapping.

The current artifacts are source-extraction checkpoints. No continuous
interpolation, Han/Bonin common-map warp, or Allen CCF warp has yet been fitted.
