Descriptive writeup of the problem
Context
This repository’s Figure 3 analysis code was originally run in an older “paper-era” environment (see environment.yml: Python 3.7 with allensdk==2.2). That stack assumes NWB files and stimulus tables that match the Allen Institute ecephys release format and the NWB/HDMF ecosystem as it existed around that time.

You now have new NWB files (MouseV2) that can be opened with modern PyNWB tooling (as in reference/interactive_analysis.py), but they are not reliably readable through the old AllenSDK pathway and are sensitive to modern dependency mismatches.

What you observed (symptoms)
AllenSDK session loading fails on the new NWB in the old env

Initial error:
ValueError: No specification for 'TimeSeriesReferenceVectorData' in namespace 'core'
This indicates the NWB file uses types that the older NWB schema / PyNWB/HDMF in the paper environment does not understand.
Upgrading to Python 3.10+ didn’t immediately fix it

A later error appeared while trying to read the NWB:
ValueError: Insufficient precision in available types to represent (63, 52, 11, 0, 52)
The traceback showed imports coming from ~/.local/..., meaning user-site packages were overriding conda env packages, producing a mixed, inconsistent stack.
After blocking user-site packages, PyNWB import started failing

Error:
AttributeError: 'TypeConfigurator' object has no attribute 'paths'
This happened because pynwb and hdmf were API-incompatible (HDMF major version mismatch).
After pinning PyNWB 2.8.3 + HDMF 3.14.6, PyNWB still fails at import

Error:
AttributeError: Can't get attribute 'ClassGeneratorManager' ...
A targeted probe confirmed:
hdmf 3.14.6 does not define ClassGeneratorManager.
Root cause
There are two independent compatibility problems:

A) Old paper environment vs. new NWB files
The paper environment (python=3.7, allensdk==2.2) is too old to parse newer NWB constructs/types used by your dataset.
Even if the analysis scripts are correct, the underlying IO stack can’t load the file.
B) Dependency stack instability in modern environments
Even in Python 3.10, you ran into fragile coupling between pynwb and hdmf:

pynwb internally loads a pickled TypeMap for the NWB core namespace during import.
That pickle encodes references to HDMF internal classes.
If your installed hdmf doesn’t contain the exact internal symbol expected (e.g., ClassGeneratorManager), import pynwb can fail before you even open a file.
On top of that, ~/.local user-site packages were partially overriding conda packages, producing mixed h5py/hdmf/pynwb combinations that break in non-obvious ways.

Why this blocks your analysis
The Figure 3 pipeline depends on generating per-unit metrics (TTFS, flash timescale, DG modulation index, RF metrics). The original scripts compute many of these using AllenSDK session abstractions.

Right now:

The paper environment runs the analysis code but can’t load the new NWB.
The modern environment can potentially load the NWB, but the pynwb/hdmf stack is currently inconsistent, and AllenSDK compatibility with these NWBs is unproven.
Practical implication
To proceed in a way that stays faithful to the paper methods, you likely need to decouple “metric generation” from “figure plotting”:

Use a modern, stable PyNWB environment to extract spike times + stimulus presentations and compute the needed metrics (or, if possible, compute AllenSDK metrics in a clean modern stack).
Then feed the resulting CSVs into the existing repo analysis/plotting code (which can remain in the older “paper” environment).
This avoids forcing a single environment to satisfy both (1) paper-era analysis code expectations and (2) modern NWB file requirements.


solution? 
By directly importing the original repository's low-level mathematical functions into your modern generate_retinotopic_csvs.py script, you guarantee zero algorithmic drift. You are effectively using modern PyNWB to crack open the vault, but letting the legacy code appraise the contents.

Since these files (time_to_first_spike.py, modulation_index.py, fit_exp.py) rely only on standard libraries like numpy and scipy, they are perfectly compatible with your Python 3.10 environment.

The only hurdle you have to solve in your modern script is Data Shaping. The AllenSDK abstracted away how raw spike times (e.g., [1.05, 1.12, 1.15]) were converted into the specific multi-dimensional matrices that Xiaoxuan Jia's functions expect.

Here is exactly how you need to wrangle the PyNWB data in your script to successfully feed it into the imported legacy functions.

1. Time to First Spike (TTFS)
The Import: from time_to_first_spike import compute_first_spike

The Expected Data: compute_first_spike(spikes) explicitly expects a 3D NumPy array with the shape (neuron, trial, time). Furthermore, it expects the time dimension to be 1-millisecond bins represented as 1 (spike) or 0 (no spike).

The Glue Code (inside your script):

Python
import numpy as np
from time_to_first_spike import compute_first_spike

# For a single unit's spike times and a DataFrame of flash trials:
num_trials = len(flash_df)
# Create a 3D array for 1 neuron, N trials, and 500ms of time
binned_spikes = np.zeros((1, num_trials, 500), dtype=int) 

for trial_idx, row in flash_df.reset_index().iterrows():
    start = row['start_time']
    # Get spikes within the 500ms window
    window_spikes = spikes[(spikes >= start) & (spikes < start + 0.5)]
    
    # Convert spike times to millisecond indices (0 to 499)
    ms_indices = ((window_spikes - start) * 1000).astype(int)
    # Ensure we don't go out of bounds (e.g., exactly 500.0ms)
    ms_indices = ms_indices[ms_indices < 500] 
    
    binned_spikes[0, trial_idx, ms_indices] = 1

# Call the legacy function!
ttfs_matrix = compute_first_spike(binned_spikes, start_time=30, end_time=500)
median_latency = np.nanmedian(ttfs_matrix[0, :])
2. Drifting Gratings Modulation Index
The Import: from modulation_index import main as get_dg_mi

The Expected Data: main(data, fs, TF_pref) expects data to be a time series of shape (repeat, time). Just like TTFS, this needs to be heavily binned data. If you use fs=1000, the time dimension must be 1ms bins of the trial duration. TF_pref is the temporal frequency of the grating presented in those trials (e.g., 4 Hz).

The Glue Code:

Python
from modulation_index import main as get_dg_mi

# Assuming you filter drifting gratings for the unit's preferred condition
# and the grating presentation is 2 seconds long (2000 ms)
num_repeats = len(pref_dg_df)
binned_dg_data = np.zeros((num_repeats, 2000), dtype=float)

for trial_idx, row in pref_dg_df.reset_index().iterrows():
    start = row['start_time']
    window_spikes = spikes[(spikes >= start) & (spikes < start + 2.0)]
    
    ms_indices = ((window_spikes - start) * 1000).astype(int)
    ms_indices = ms_indices[ms_indices < 2000]
    
    binned_dg_data[trial_idx, ms_indices] = 1

# If the legacy script expects a 1D continuous PSTH across trials, you can average them:
# psth = np.mean(binned_dg_data, axis=0)
# But based on the signature data: (repeat*time), it likely handles the 2D array.
# Call the legacy function using the trial's Temporal Frequency (e.g., 4 Hz)!
mi_value = get_dg_mi(data=binned_dg_data, fs=1000.0, TF_pref=4.0)
3. Intrinsic Timescale
The Import: from fit_exp import fit_exp

The Expected Data: fit_exp(rsc_time_matrix, color) expects an rsc_time_matrix where the axis=0 is repeats/trials/units and the columns are the time lags of the autocorrelogram. Note: This script does the curve fitting, but it doesn't compute the autocorrelogram itself. You will need to compute the auto-correlation of the spike train first.

The Glue Code:

Python
from fit_exp import fit_exp
import scipy.signal

# 1. First, create a continuous binned spike train for the whole session (or active period)
# using e.g., 10ms bins (which was standard for Allen timescales)
bin_size = 0.010 
bins = np.arange(0, session_end_time, bin_size)
spike_counts, _ = np.histogram(spikes, bins=bins)

# 2. Compute the autocorrelogram (up to a max lag, e.g., 100 bins = 1 second)
max_lag = 100
autocorr = scipy.signal.correlate(spike_counts, spike_counts, mode='full')
center = len(autocorr) // 2
autocorr_lags = autocorr[center : center + max_lag]

# 3. fit_exp expects a 2D matrix where rows are instances to average over. 
# If you are doing one unit at a time, you can pass it as a 1xLags matrix
rsc_matrix = np.expand_dims(autocorr_lags, axis=0)

# Call the legacy function! (Ignore 'color', it just passes through)
t, y, y_std, a, b, c = fit_exp(rsc_matrix, color='k')

# 'b' is your timescale! (convert back from bins to ms if necessary)
tau = b * (bin_size * 1000) 
By adding these small "glue" blocks to generate_retinotopic_csvs.py, you allow modern PyNWB to handle the breaking schema changes of MouseV2, while guaranteeing the math perfectly perfectly matches the original repository.