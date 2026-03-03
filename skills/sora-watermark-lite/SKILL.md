---
name: sora-watermark-lite
description: Lightweight local Sora watermark remover for single videos or full directories. Use this whenever the user asks to remove Sora watermarks, batch-process many MP4 files, compare processing time, or wants an easier alternative to the full SoraWatermarkCleaner stack.
---

# sora-watermark-lite

Use the bundled script for fast local watermark removal with minimal dependencies.

## Install prerequisites

1. Ensure `ffmpeg` is available in PATH.
2. Install Python deps:

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

Or for single file:

```bash
bash scripts/run_clean.sh "/absolute/path/input.MP4" "/absolute/path/output_cleaned.MP4"
```

## Useful options

- `--device auto|cpu|mps`
- `--conf 0.25`
- `--expand 0.2`
- `--max-gap 8`
- `--method telea|ns`
- `--radius 3`
- `--report /path/to/report.csv`

## Output expectations

The script prints:

- input/output path
- processed frame counts
- detected frame counts
- selected device
- elapsed seconds

Batch mode also prints total batch elapsed seconds and writes a CSV report.
