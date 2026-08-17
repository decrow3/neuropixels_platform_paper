# Allen session 746083955: population gaze correction

The analysis used 555 canonical visual-area units and 3645 Gabor presentations. The gain sweep was restricted to 318 RF/QC units; the chosen transform and nominal control were then fit for the full population.

The gaze trace is the per-presentation median filtered spherical screen-gaze coordinate, centered to zero over the Gabor block. Corrected stimulus coordinates equal nominal position minus the scaled gaze deviation. No per-neuron gaze transform is fitted.

The population-calibration subset selected `gain_x_0_gain_y_0.5`. On held-out neurons, the median test Poisson-deviance improvement was +0.000175, and 55.6% of units improved.

RFs use a nonnegative baseline plus three orientation amplitudes and a shared axis-aligned Gaussian. Centers may extend 20 degrees beyond the sampled grid; sigma is limited to 40 degrees in V1 and 50 degrees in HVAs. Bound-reaching fits are labeled censored. No enclosing border is used.

This is an exploratory single-session checkpoint. Candidate selection and final evaluation use disjoint neuron sets, and every fit uses separate training and test repeats within each position/orientation condition.
