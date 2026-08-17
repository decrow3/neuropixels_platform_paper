# Zhuang et al. 2017 Figure 9 template — initial evidence checkpoint

This is the primary population retinotopy template for the cross-animal
registration project. It encodes Figure 9C (altitude) and Figure 9D (azimuth)
from Zhuang et al. (2017), *An extended retinotopic map of mouse cortex*.

## Why this reference

The paper pooled maps across mice after centering each map on the V1 centroid,
rotating it to the major azimuth-gradient axis, and correcting retinal position
at the V1/LM/RL junction. It therefore captures the shared V1-and-HVA map
geometry that we want as a population prior. The repeated distortions and map
reversals across HVAs are retained as useful registration structure; they are
not split into unrelated area-specific maps.

## Source and evidence type

- Article: https://elifesciences.org/articles/18372
- Official eLife IIIF TIFF: https://iiif.elifesciences.org/lax/18372%2Felife-18372-fig9-v2.tif/full/full/0/default.tif
- TIFF SHA-256: `55b3957e0d17233c3c778d9ebeebda43949b6cf6813064f412f749adad70bc9e`
- Source dimensions: 1001 × 1452 pixels

Figure 9C/D publish **5-degree isolines**, not continuous numerical map
rasters: altitude −25° to 30° and azimuth 0° to 90°. This extraction stores
the labeled contour evidence sparsely and does not interpolate unobserved
pixels. The black mean field-sign borders are stored as a separate mask.

## Coordinate frame

Panel D is the common source-figure frame. Panel C is translated onto it using
the duplicated borders (dx=-1,
dy=0 pixels).
Coordinates are available as pixels and panel fractions; image y points down,
while `y_common_fraction_up` in the point table points up.

This is still a publication-figure frame—not CCF AP/ML and not the historical
Han/Bonin `retino_map.png` pixel frame. The previously retained MATLAB affine
must not be applied blindly to this v2 eLife image; its landmarks should be
re-fitted if the Han common map is recovered.

## Files

- `source_figure9_v2.tif`: exact lossless source used.
- `source_figure9_panels_CD.png`: cropped source evidence.
- `retinotopy_contour_grid.npz`: sparse altitude/azimuth degree grids,
  color-decoding errors, borders, coordinates, and base colors.
- `retinotopy_contour_points.csv.gz`: one row per decoded contour pixel.
- `Figure_template_extraction_QA.png`: source-to-extraction visual audit.
- `run_manifest.json`: provenance, parameters, counts, QA, and chart contract.

## Current limitations

1. Raster anti-aliasing is decoded to the nearest exact publication stroke
   color; the retained per-pixel color error makes that choice auditable.
2. Area names and landmark identities are not inferred from pixels.
3. No continuous retinal-coordinate surface is created yet. That interpolation
   belongs in the registration model and should be validated against cells with
   measured RFs.
4. No Han or Allen CCF transform has been fitted at this checkpoint.
