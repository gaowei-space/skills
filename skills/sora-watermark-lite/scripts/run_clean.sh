#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage (single): $0 <input_video> <output_video> [extra args]"
  echo "Usage (batch):  $0 --input-dir <dir> --output-dir <dir> [extra args]"
  exit 1
fi

if [[ "$1" == --* ]]; then
  PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n sorawm_m2 \
    python "$(dirname "$0")/clean_lite.py" "$@"
  exit 0
fi

if [ "$#" -lt 2 ]; then
  echo "Single mode requires: <input_video> <output_video>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n sorawm_m2 \
  python "$(dirname "$0")/clean_lite.py" -i "$INPUT" -o "$OUTPUT" "$@"
