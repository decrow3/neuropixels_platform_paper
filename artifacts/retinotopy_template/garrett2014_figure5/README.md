# Garrett et al. 2014 Figure 5 template — initial evidence checkpoint

This directory is a source-derived, exploratory template from Figure 5 of
Garrett et al. (2014), *Topography and Areal Organization of Mouse Visual
Cortex*. It is **not** the unpublished Allen 35-experiment canonical atlas.

## Source and extraction

- Article: https://doi.org/10.1523/JNEUROSCI.1124-14.2014
- PDF used: https://cseweb.ucsd.edu/~gary/cs200/f14/Mouse%20Maps%20Published.pdf
- PDF SHA-256: `a7db966d128949f8b05a23830194eee71c09a81df7ae19a11fbe9d7ae2f9917b`
- Source location: PDF page 8, article page 12594, Figure 5
- Published population: 14 mice; maps were aligned on V1 center of mass and
  the mean V1 horizontal-retinotopy-gradient direction.

The field-sign surface is decoded from the embedded 435×435 raster using the
published `jet` color scale. Azimuth and altitude isolines and internal black
boundaries are extracted from vector paths, not re-traced by hand. Stroke
colors are decoded against `jet` and rounded to the published 5-degree contour
spacing.

## Coordinate frame

The origin is the center of mass of the largest connected component with
decoded field sign below -0.5, provisionally identified as V1. One unit equals
one source-panel width. Positive x points right in the published figure and
positive y points up. No additional rotation is applied because the published
population map was already rotation-normalized.

This is a figure coordinate frame, not yet AP/ML CCF space. The anatomical
axis orientation and absolute millimetre scale remain to be established from
the Allen/MouseV2 observations.

## Files

- `source_figure5_panels.png`: cropped primary evidence.
- `field_sign_grid.npz`: decoded sign field, color-decoding error, V1 mask,
  and normalized grid coordinates.
- `retinotopy_contours.csv.gz`: vector-derived azimuth/altitude line segments.
- `area_boundaries.csv.gz`: black internal boundary segments from both maps.
- `Figure_template_extraction_QA.png`: source-to-extraction visual audit.
- `run_manifest.json`: source hashes, parameters, counts, and chart contract.

## Current limitations

1. The V1 mask and centroid are algorithmic provisional choices and require
   visual approval.
2. Area names/centroids have not yet been manually transcribed; this avoids
   silently treating typography as anatomical data.
3. Contour values inherit raster/vector publication resolution and should be
   used as an initialization prior, not as ground truth.
4. Redistribution should preserve the article citation and source rights.
