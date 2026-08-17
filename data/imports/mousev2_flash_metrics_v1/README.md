# MouseV2 full-field flash metrics v1

Per-unit pooled, bright, and dark TTFS and response-decay timescales
computed from the eight versioned NWBs.

- TTFS is the median first occupied 1-ms bin in the released 30–200 ms window.
- Timescale uses 10-ms AllenSDK bin centers in the released 40–290 ms selection
  (25 bins centered at 45–285 ms), then the released bounded exponential fit.
- Bright is contrast +1 (white); dark is contrast −1 (black).
- Latencies are raw relative to NWB interval `start_time`; no cross-dataset
  mean matching or display-latency correction is applied.
- NWB starts exactly match the processed stimulus timestamp series, but no
  photodiode trace or physical light-onset provenance is encoded.

See `timing_audit.csv`, `site_metric_summary.csv`, and `import_manifest.json`.
