import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
# Keep warnings visible for debugging except noisy RuntimeWarnings from empty slices
warnings.filterwarnings('ignore', category=RuntimeWarning)
def get_color_palette(area):
    # Coarse area palette (mapped from fine VIS* colors)
    colors = [[217,141,194],   # LGd
              [129,116,177],   # V1
              [78,115,174],    # LM
              [101,178,201],   # RL
              [88,167,106],    # LP
              [202,183,120],   # AL
              [219,132,87],    # PM
              [194,79,84]]     # AM
    to_float = lambda c: tuple(v/255 for v in c)
    palette = {
        'LGd': to_float(colors[0]),
        'V1': to_float(colors[1]),
        'LM': to_float(colors[2]),
        'RL': to_float(colors[3]),
        'LP': to_float(colors[4]),
        'AL': to_float(colors[5]),
        'PM': to_float(colors[6]),
        'AM': to_float(colors[7])
    }
    return palette.get(area, '#C9C9C9')

### PATH VARIABLES ##############################
cache_directory = ''  # not used after refactor
code_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
###################################################

# Build a unified per-unit dataframe from available precomputed CSVs in ../data
data_dir = os.path.join(code_directory, 'data')

# Change modulation / latency data
change_path = os.path.join(data_dir, 'change_modulation_data.csv')
if os.path.isfile(change_path):
    df_change = pd.read_csv(change_path, low_memory=False)
    # Rename columns to match expected metric names
    df_change = df_change.rename(columns={
        'Region': 'ecephys_structure_acronym',
        'Time To First Spike': 'time_to_first_spike_fl',
        'Change Modulation Active': 'mod_idx_dg'
    })
    # Convert latency to seconds (original script expects seconds then converts to ms)
    if 'time_to_first_spike_fl' in df_change.columns:
        df_change['time_to_first_spike_fl'] = df_change['time_to_first_spike_fl'] / 1000.0
else:
    df_change = pd.DataFrame()

# Timescale metrics
timescale_path = os.path.join(data_dir, 'timescale_metrics.csv')
if os.path.isfile(timescale_path):
    df_timescale = pd.read_csv(timescale_path, low_memory=False)
    df_timescale = df_timescale.rename(columns={'area': 'ecephys_structure_acronym'})
else:
    df_timescale = pd.DataFrame()

# Concatenate, allowing missing columns for some rows
df = pd.concat([df_change, df_timescale], ignore_index=True, sort=False)

# Drop rows without area information
if 'ecephys_structure_acronym' in df.columns:
    df = df[~df.ecephys_structure_acronym.isna()]

# Map fine VIS* labels to coarse labels
fine_to_coarse = {'VISp':'V1','VISl':'LM','VISrl':'RL','VISal':'AL','VISpm':'PM','VISam':'AM'}
df['area_coarse'] = df['ecephys_structure_acronym'].apply(lambda a: fine_to_coarse.get(a,a))

# Diagnostics: coarse coverage
metrics_present = ['time_to_first_spike_fl','mod_idx_dg','timescale_ac']
print('Metric coverage by area (coarse):')
for m in metrics_present:
    if m in df.columns:
        counts = df.groupby('area_coarse')[m].apply(lambda x: int(np.sum(np.isfinite(x))))
        ordered = [f"{a}:{counts[a]}" for a in ['LGd','V1','LM','RL','LP','AL','PM','AM'] if a in counts]
        print(f'  {m}: ' + ', '.join(ordered))
    else:
        print(f'  {m}: (column missing)')

# %%

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.figure(14781, figsize=(20, 12))
plt.clf()

areas = ('LGd','V1','LM','RL','LP','AL','PM','AM')

color_palette = None  # unused with simplified palette

hierarchy_score = {'LGd' : -0.515,
                   'V1' : -0.357,
                   'LM' : -0.093,
                   'RL' : -0.059,
                   'LP' : 0.105,
                   'AL' : 0.152,
                   'PM' : 0.327,
                   'AM' : 0.441}

HS = [hierarchy_score[a] for a in areas]

num_units = 0
mouse_count = np.zeros((4,))

def get_bootstrap_95ci(M, measure_of_central_tendency, N=500):
    if len(M) < 5:
        return np.nan
    n = max(1, int(len(M)/2))
    est = np.zeros((N,))
    for i in range(N):
        boot = M[np.random.permutation(len(M))[:n]]
        est[i] = measure_of_central_tendency(boot)
    return np.percentile(est,97.5) - np.nanmean(est)

def convert_to_ms(value_in_s):
    return value_in_s*1000

def take_log(original_value):
    return np.log10(original_value)

def do_not_change(original_value):
    return original_value

measure_of_central_tendency = np.nanmean

np.random.seed(10)

num_per_area = np.zeros((8,))
num_with_rfs = np.zeros((8,))
num_after_filter = np.zeros((8,))
num_after_fl = np.zeros((8,))
num_after_ac = np.zeros((8,))
mice_per_area = np.zeros((8,))

if True:
    # Reduced metric set – only those available in precomputed CSVs
    metrics = ['time_to_first_spike_fl', 'mod_idx_dg', 'timescale_ac']
    labels = ['Time to first spike (ms)', 'Modulation index', 'Response decay timescale (ms)']
    bins = [np.linspace(15,120,30), np.linspace(-1.0,1.0,50), np.linspace(0,150,50)]
    function_to_apply = [convert_to_ms, do_not_change, do_not_change]
    y_vals = [60, 0.5, 35]

else:
    metrics = [ 'firing_rate']
    labels = ['$log_{10}$ Firing rate']
    bins = [np.linspace(-1,2)]
    function_to_apply = [take_log]
    y_vals = [0.75]

unit_count = np.zeros((len(metrics),))
centers = np.zeros((8,len(metrics)))
errorbars = np.zeros((8,len(metrics)))

max_values = np.zeros((len(metrics),))

all_values = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

#all_df = []

for area_idx, area in enumerate(areas):

    base_selection = (df.area_coarse == area)
    num_per_area[area_idx] = np.sum(base_selection)
    num_with_rfs[area_idx] = np.sum(base_selection)
    num_after_filter[area_idx] = np.sum(base_selection)
    mice_per_area[area_idx] = np.sum(base_selection)

    for metric_idx, metric in enumerate(metrics):
        selection_metric = base_selection.copy()

        if metric_idx == 0:
            selection_metric &= (df.time_to_first_spike_fl < 0.1)
            num_after_fl[area_idx] = np.sum(selection_metric)
        elif metric_idx == 2:  # timescale_ac filters
            selection_metric &= (df[metric] < 300)
            selection_metric &= (df[metric] > 1)
            selection_metric &= (df.spike_count_ac > 50)
            selection_metric &= (df.err_ac < 20)
            num_after_ac[area_idx] = np.sum(selection_metric)

        if metric not in df.columns:
            M = np.array([])
        else:
            M = function_to_apply[metric_idx](df[selection_metric][metric].values)

        all_values[metric_idx][area] = M
        if len(M) == 0 or np.all(np.isnan(M)):
            centers[area_idx, metric_idx] = np.nan
            errorbars[area_idx, metric_idx] = np.nan
            continue
        h, b = np.histogram(M, bins=bins[metric_idx], density=True)
        h_filt = gaussian_filter1d(h,1.5)
        unit_count[metric_idx] += len(M)
        max_values[metric_idx] = np.max([np.max(h_filt), max_values[metric_idx]])
        plt.subplot(len(metrics), 4, metric_idx*4+1)
        plt.plot(b[:-1], h_filt, color=get_color_palette(areas[area_idx]), lw=2, alpha=0.85)
        if area_idx == len(areas)-1:
            plt.xlabel(labels[metric_idx])
        plt.subplot(len(metrics), 4, metric_idx*4+2)
        plt.plot(b[:-1], np.cumsum(h_filt), color=get_color_palette(areas[area_idx]), lw=2, alpha=0.85)
        if area_idx == len(areas)-1:
            plt.xlabel(labels[metric_idx])
        centers[area_idx, metric_idx] = measure_of_central_tendency(M)
        errorbars[area_idx,  metric_idx] = get_bootstrap_95ci(M, measure_of_central_tendency)
        
print('TOTAL: ' + str(np.sum(num_per_area)))
    
from scipy.stats import linregress, pearsonr, spearmanr


x = HS
    
for i in range(len(metrics)):
    
    plt.subplot(len(metrics),4,i*4+3)
    y = centers[:,i]
    finite_mask = np.isfinite(y)
    
    for k in range(8):
        plt.plot(x[k], centers[k,i], 'o', ms=6, color=get_color_palette(areas[k]))
        plt.errorbar(x[k], centers[k,i], yerr=errorbars[k,i], fmt='none', ecolor=get_color_palette(areas[k]), alpha=0.8, lw=1)

    if np.sum(finite_mask) < 3:
        continue
    slope,intercept,r,p,std = linregress(np.array(x)[finite_mask], y[finite_mask])
    x2 = np.linspace(-0.75,0.5,10)
    
    plt.plot(x2, x2*slope+intercept, '--', color='k', alpha=0.6, lw=1.5)

    r_s,p_s = spearmanr(np.array(x)[finite_mask], y[finite_mask])
    r_p,p_p = pearsonr(np.array(x)[finite_mask], y[finite_mask])
    text =  '$r_P$ = ' + str(np.around(r_p,2)) + '; $P_P$ = ' + str(np.around(p_p,6)) + '\n' + \
             '$r_S$ = ' + str(np.around(r_s,2)) + '; $P_S$ = ' + str(np.around(p_s,6))
    
    plt.text(-0.30,y_vals[i],text,fontsize=8)
    plt.ylabel(labels[i])
        
for i in range(len(metrics)):
    for j in range(2):
        plt.subplot(len(metrics),4,i*4+1+j)
        ax = plt.gca()
        #plt.gca().get_yaxis().set_visible(False)
        [ax.spines[loc].set_visible(False) for loc in ['top', 'right']]        
    
    plt.subplot(len(metrics),4,i*4+3)
    ax = plt.gca()
    [ax.spines[loc].set_visible(False) for loc in ['right', 'top']]   
    plt.xlim([-0.85,0.65])
        

from scipy.stats import ks_2samp, ranksums
from statsmodels.stats.multitest import multipletests

alpha = 0.05

common_names = ['LGN', 'V1', 'LM', 'RL', 'LP', 'AL', 'PM', 'AM']

for metric_idx, metric in enumerate(metrics):
    comparison_matrix = np.zeros((len(areas),len(areas)))
    
    for area_idx1, area1 in enumerate(areas):
        
        for area_idx2, area2 in enumerate(areas):
            
            if area_idx2 > area_idx1:
                
                v1 = all_values[metric_idx][area1]
                v2 = all_values[metric_idx][area2]
                v1 = v1[np.isfinite(v1)]
                v2 = v2[np.isfinite(v2)]
                if len(v1) < 5 or len(v2) < 5:
                    continue
                from scipy.stats import ranksums as rs_test
                _, p = rs_test(v1, v2)
                comparison_matrix[area_idx1, area_idx2] = p if p > 0 else 1e-10
       
    p_values = comparison_matrix.flatten()
    ok_inds = np.where(p_values > 0)[0]
    inds = np.where(comparison_matrix > 0)
    indx = inds[0]
    indy = inds[1]
    
    reject, p_values_corrected, alphaSidak, alphacBonf = multipletests(p_values[ok_inds], alpha=alpha, method='fdr_bh')
            
    p_values_corrected2 = np.zeros((len(p_values),))
    p_values_corrected2[ok_inds] = p_values_corrected
    comparison_corrected = np.reshape(p_values_corrected2, comparison_matrix.shape)
    
    sig_thresh = np.log10(alpha)
    plot_range = 8
    
    plt.subplot(len(metrics),4,metric_idx*4+4)
    safe_matrix = comparison_corrected.copy()
    safe_matrix[safe_matrix <= 0] = 1e-10
    plt.imshow(np.log10(safe_matrix), cmap='bone', vmin=-5, vmax=np.log10(0.05))
    
    plt.colorbar(fraction=0.026, pad=0.04)
    
    plt.xticks(ticks=np.arange(len(common_names)), labels=common_names)
    plt.yticks(ticks=np.arange(len(common_names)), labels=common_names)
    plt.ylim([-0.5,len(common_names)-0.5])
    plt.xlim([-0.5,len(common_names)-0.5])    
    
plt.tight_layout()

# Apply uniform y-limits for histogram & cumulative subplots per metric
cumulative_ylims = [0.3, 14, 0.35]  # time_to_first_spike, mod_idx, timescale
for metric_idx in range(len(metrics)):
    # Collect max density across all areas for metric
    # max_values were tracked already; use small padding
    ymax = max_values[metric_idx] * 1.05 if max_values[metric_idx] > 0 else None
    if ymax:
        ax_hist = plt.subplot(len(metrics),4,metric_idx*4+1)
        ax_hist.set_ylim(0, ymax)
        # cumulative custom limits per metric
        ax_cum = plt.subplot(len(metrics),4,metric_idx*4+2)
        ax_cum.set_ylim(0, cumulative_ylims[metric_idx])

plt.savefig('Figure3_restored_style.png', dpi=600, bbox_inches='tight')
print('Saved figure to Figure3_restored_style.png')