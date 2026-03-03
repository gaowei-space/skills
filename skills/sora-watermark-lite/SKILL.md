---
name: sora-watermark-lite
description: LAMA-based Sora watermark remover for single videos or full directories. Use this whenever the user asks for better visual quality watermark removal, batch-process many MP4 files, or wants results closer to SoraWatermarkCleaner quality.
---

# sora-watermark-lite

Use the bundled script for LAMA-based watermark removal.

## Install prerequisites

1. Ensure `ffmpeg` is available in PATH.
2. Ensure SoraWatermarkCleaner repository exists locally and expose it via `SORAWM_PATH`:

```bash
export SORAWM_PATH=/path/to/SoraWatermarkCleaner
```

3. Install this script's helper deps:

```bash
pip install -r scripts/requirements.txt
```

## Single video

```bash
python scripts/clean_lite.py \
  -i "/absolute/path/input.MP4" \
  -o "/absolute/path/output_cleaned.MP4"
```

## Batch directory

```bash
python scripts/clean_lite.py \
  --input-dir "/absolute/path/input_dir" \
  --output-dir "/absolute/path/output_dir" \
  --pattern "*.MP4"
```

## Wrapper mode

If the user already has conda environment `sorawm_m2`, use:

```bash
bash scripts/run_clean.sh --input-dir "/absolute/path/input_dir" --output-dir "/absolute/path/output_dir"
```

`run_clean.sh` will auto-set `SORAWM_PATH` on this machine if it finds the default local clone path.

Or for single file:

```bash
bash scripts/run_clean.sh "/absolute/path/input.MP4" "/absolute/path/output_cleaned.MP4"
```

## Useful options

- `--detect-batch-size 4`
- `--report /path/to/report.csv`

## Output expectations

The script prints:

- input/output path
- processed frame counts
- detected frame counts
- elapsed seconds

Batch mode also prints total batch elapsed seconds and writes a CSV report.
