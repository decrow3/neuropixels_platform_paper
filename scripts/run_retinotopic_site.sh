#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_retinotopic_site.sh \
#     --nwb "/path/to/sub-XXX/...ecephys.nwb" \
#     --out_dir "./data/siteX_processed" \
#     --site_name "V1_siteX" \
#     --id_offset 1000000
#
# Notes:
# - Ensure conda env is active (see README).
# - id_offset must be unique per site to avoid unit ID collisions.

# Default values (can be overridden by args)
NWB_PATH=""
OUT_DIR=""
SITE_NAME=""
ID_OFFSET=""
PYTHON_BIN="python"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nwb)
      NWB_PATH="$2"; shift 2;;
    --out_dir)
      OUT_DIR="$2"; shift 2;;
    --site_name)
      SITE_NAME="$2"; shift 2;;
    --id_offset)
      ID_OFFSET="$2"; shift 2;;
    --python)
      PYTHON_BIN="$2"; shift 2;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

if [[ -z "$NWB_PATH" || -z "$OUT_DIR" || -z "$SITE_NAME" || -z "$ID_OFFSET" ]]; then
  echo "Missing required args. See usage with --help." >&2
  exit 1
fi

# Create output directory if missing
mkdir -p "$OUT_DIR"

# Run generator
set -x
"$PYTHON_BIN" generate_retinotopic_csvs.py \
  --nwb "$NWB_PATH" \
  --out_dir "$OUT_DIR" \
  --site_name "$SITE_NAME" \
  --id_offset "$ID_OFFSET"
set +x

echo "Done. Outputs written to: $OUT_DIR"
