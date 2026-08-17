# Artificial RF-edge cropping: initial concrete checkpoint

The native 9×9 spike-count map for stable interior V1 unit 951867908 was cropped from each edge one row or column at a time. Original visual-grid coordinates were preserved. The uncropped map is the within-unit reference.

The top-edge trajectory is the most direct encroachment contrast because this RF response occupies rows 3–5. Removing three top rows places the sampled boundary directly against the response; removing four deletes the original peak row.

At three top rows removed, Allen's no-baseline major sigma changes from **10.86°** to **11.91°**, the baseline-aware major sigma changes from **9.51°** to **9.50°**, and threshold area changes from **300** to **400 deg²**.

Although Allen's sigma changes only modestly at that three-row crop, its fitted center moves **129.7°** from the full-map estimate. The baseline-aware center moves only **0.29°**. Once four top rows are removed and the original peak row is deleted, Allen's major sigma expands **4.85×** and its center moves **269.2°**; the baseline-aware sigma changes **1.04×**, its center moves **4.0°**, and its bound flag turns on.

The thresholded area is non-monotonic: it grows by 33–67% as smoothing and thresholding are recomputed on the cropped array, then falls after the peak is removed. A component-touching-edge flag is therefore necessary, but area change cannot be inferred from lost pixels alone.

Allen's Gaussian helper also swaps its returned row/column center labels. This is hidden on square 9×9 maps but was exposed by rectangular crops; the saved trajectory corrects the labels back to the original visual axes.

This checkpoint measures estimator response to controlled loss of spatial support. It does not yet compare the proposed DC-ring penalty or establish population-level bias.
