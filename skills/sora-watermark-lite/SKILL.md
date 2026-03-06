---
name: sora-watermark-lite
description: Self-contained LAMA-based Sora watermark remover for single videos or full directories. Use this whenever the user asks for higher-quality local watermark removal without depending on the full SoraWatermarkCleaner repository.
---

# sora-watermark-lite

Use the bundled script for self-contained LAMA-based watermark removal.

## Recommended flow

Prefer the wrapper script when the machine already has a prepared conda environment. In that case, do not reinstall Python dependencies.

Prerequisites:

1. Ensure `ffmpeg` is available in PATH.
2. Ensure `conda` is available in PATH.
3. Ensure the runtime environment already contains the packages from `scripts/requirements.txt`.

The wrapper defaults to conda env `sorawm_m2`.

If the environment uses a different name, set `SORA_WM_CONDA_ENV` first:

```bash
export SORA_WM_CONDA_ENV=my_env_name
```

Batch mode:

```bash
bash scripts/run_clean.sh --input-dir "/absolute/path/input_dir" --output-dir "/absolute/path/output_dir"
```

Single file:

```bash
bash scripts/run_clean.sh "/absolute/path/input.MP4" "/absolute/path/output_cleaned.MP4"
```

## Fallback: direct Python mode

Use this only when there is no ready conda environment.

1. Ensure `ffmpeg` is available in PATH.
2. Install Python deps into the current interpreter:

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

## Notes

- Prefer wrapper mode before suggesting `pip install`.
- Only suggest reinstalling dependencies if wrapper mode fails because the conda environment is missing or incomplete.
- On Apple Silicon, wrapper mode already sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Useful options

- `--device auto|cpu|mps`
- `--conf 0.25`
- `--expand 0.0`
- `--max-gap 8`
- `--detect-batch-size 4`
- `--report /path/to/report.csv`

## Output expectations

The script prints:

- input/output path
- processed frame counts
- detected frame counts
- selected device
- elapsed seconds

Batch mode also prints total batch elapsed seconds and writes a CSV report.
