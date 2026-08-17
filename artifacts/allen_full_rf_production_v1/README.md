# Allen full-population RF production v1

This directory is prepared for the 58-session Allen Visual Coding Neuropixels
RF production run. The dry run resolves 36,430 V1/HVA units and 20,879
published-like fitting units from the downloaded NWB inventory.

The workflow is resumable and runs in two stages:

1. Two concurrent NWB extraction workers write compact Gabor trial and spike
   count caches. Eye tracking is not loaded and all 3,645 Gabor presentations
   are retained.
2. Six concurrent fitting workers fit axis-aligned and freely rotated point and
   analytic circular-aperture models for every selected unit. Each session
   checkpoints every ten units.

Each child process is restricted to one numerical thread. NWB extraction and
fitting have independent completion validators. Shared aggregate tables and
the production summary figure are written only after every session fit passes.
No LFP/raw data are accessed and the runner never deletes existing files.

## Launch

```bash
env MPLCONFIGDIR=/tmp/allen_rf_production_mpl \
    XDG_CACHE_HOME=/tmp/allen_rf_production_xdg \
    /home/huklaban5/anaconda3/envs/allensdk/bin/python \
    scripts/run_allen_full_rf_production.py --mode all
```

## Status

```bash
env MPLCONFIGDIR=/tmp/allen_rf_production_mpl \
    XDG_CACHE_HOME=/tmp/allen_rf_production_xdg \
    /home/huklaban5/anaconda3/envs/allensdk/bin/python \
    scripts/run_allen_full_rf_production.py --mode status
```

Rerunning `--mode all` skips valid completed sessions, resumes partial fitting
tables, and retries only missing or invalid stages.
