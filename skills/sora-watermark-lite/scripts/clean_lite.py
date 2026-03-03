#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Callable

import ffmpeg
import numpy as np


def bootstrap_sorawm_path() -> None:
    env_path = os.environ.get("SORAWM_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())
    candidates.extend(
        [
            Path.cwd() / "SoraWatermarkCleaner",
            Path.home() / "SoraWatermarkCleaner",
            Path("/Users/techman/Library/Mobile Documents/com~apple~CloudDocs/Pics/SoraWatermarkCleaner"),
        ]
    )
    for candidate in candidates:
        if (candidate / "sorawm").exists():
            sys.path.insert(0, str(candidate))
            return


bootstrap_sorawm_path()

try:
    from sorawm.cleaner.lama_cleaner import LamaCleaner
    from sorawm.utils.imputation_utils import (
        find_2d_data_bkps,
        find_idxs_interval,
        get_interval_average_bbox,
    )
    from sorawm.utils.video_utils import VideoLoader
    from sorawm.watermark_detector import SoraWaterMarkDetector
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing SoraWatermarkCleaner runtime. Provide repository path via SORAWM_PATH, for example:\n"
        "  export SORAWM_PATH=/path/to/SoraWatermarkCleaner\n"
        "  conda activate sorawm_m2\n"
        "  python scripts/clean_lite.py -i input.MP4 -o output.MP4\n"
        f"Import error: {exc}"
    )


def has_encoder(name: str) -> bool:
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return name in res.stdout


def pick_encode_options() -> dict:
    if has_encoder("libx264"):
        return {"pix_fmt": "yuv420p", "vcodec": "libx264", "preset": "slow", "crf": "18"}
    return {"pix_fmt": "yuv420p", "vcodec": "mpeg4", "qscale": 2}


def merge_audio(src: Path, no_audio_video: Path, out: Path) -> None:
    video_stream = ffmpeg.input(str(no_audio_video))
    audio_stream = ffmpeg.input(str(src)).audio
    (
        ffmpeg.output(video_stream, audio_stream, str(out), vcodec="copy", acodec="aac")
        .overwrite_output()
        .run(quiet=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAMA-based Sora watermark cleaner")
    parser.add_argument("-i", "--input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pattern", default="*.MP4")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--detect-batch-size", type=int, default=4)
    args = parser.parse_args()

    single_mode = args.input is not None or args.output is not None
    batch_mode = args.input_dir is not None or args.output_dir is not None

    if single_mode and batch_mode:
        parser.error("Use single mode (-i/-o) OR batch mode (--input-dir/--output-dir), not both")
    if single_mode:
        if args.input is None or args.output is None:
            parser.error("Single mode requires both -i/--input and -o/--output")
    elif batch_mode:
        if args.input_dir is None or args.output_dir is None:
            parser.error("Batch mode requires both --input-dir and --output-dir")
    else:
        parser.error("Please provide single mode (-i/-o) or batch mode (--input-dir/--output-dir)")
    return args


def fill_missed_bboxes(frame_bboxes: dict[int, dict], total_frames: int) -> None:
    detect_missed: list[int] = []
    bbox_centers: list[tuple[int, int] | None] = []
    bboxes: list[tuple[int, int, int, int] | None] = []

    for idx in range(total_frames):
        bbox = frame_bboxes[idx]["bbox"]
        if bbox is None:
            detect_missed.append(idx)
            bbox_centers.append(None)
            bboxes.append(None)
        else:
            x1, y1, x2, y2 = bbox
            bbox_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            bboxes.append((x1, y1, x2, y2))

    if not detect_missed:
        return

    try:
        bkps = find_2d_data_bkps(bbox_centers)
        bkps_full = [0] + bkps + [total_frames]
        interval_bboxes = get_interval_average_bbox(bboxes, bkps_full)
        missed_intervals = find_idxs_interval(detect_missed, bkps_full)
    except Exception:
        interval_bboxes = []
        missed_intervals = [-1] * len(detect_missed)

    for missed_idx, interval_idx in zip(detect_missed, missed_intervals):
        if interval_idx >= 0 and interval_idx < len(interval_bboxes) and interval_bboxes[interval_idx] is not None:
            frame_bboxes[missed_idx]["bbox"] = interval_bboxes[interval_idx]
            continue
        before = max(missed_idx - 1, 0)
        after = min(missed_idx + 1, total_frames - 1)
        before_box = frame_bboxes[before]["bbox"]
        after_box = frame_bboxes[after]["bbox"]
        if before_box is not None:
            frame_bboxes[missed_idx]["bbox"] = before_box
        elif after_box is not None:
            frame_bboxes[missed_idx]["bbox"] = after_box


def process_video(input_video: Path, output_video: Path, detect_batch_size: int) -> dict:
    start = time.perf_counter()

    input_loader = VideoLoader(input_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    width = input_loader.width
    height = input_loader.height
    fps = input_loader.fps
    total_frames = input_loader.total_frames

    detector = SoraWaterMarkDetector()
    cleaner = LamaCleaner()

    frame_bboxes: dict[int, dict] = {}
    batch_frames = []
    batch_indices = []
    for idx, frame in enumerate(input_loader):
        batch_frames.append(frame)
        batch_indices.append(idx)
        if len(batch_frames) >= detect_batch_size:
            batch_results = detector.detect_batch(batch_frames, batch_size=detect_batch_size)
            for batch_idx, result in zip(batch_indices, batch_results):
                frame_bboxes[batch_idx] = {"bbox": result["bbox"] if result["detected"] else None}
            batch_frames.clear()
            batch_indices.clear()

    if batch_frames:
        batch_results = detector.detect_batch(batch_frames, batch_size=detect_batch_size)
        for batch_idx, result in zip(batch_indices, batch_results):
            frame_bboxes[batch_idx] = {"bbox": result["bbox"] if result["detected"] else None}

    fill_missed_bboxes(frame_bboxes, total_frames)

    temp_output = output_video.parent / f"temp_{output_video.name}"
    encode_options = pick_encode_options()
    process_out = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{width}x{height}", r=fps)
        .output(str(temp_output), **encode_options)
        .overwrite_output()
        .global_args("-loglevel", "error")
        .run_async(pipe_stdin=True)
    )

    input_loader_2 = VideoLoader(input_video)
    detected_frames = 0
    for idx, frame in enumerate(input_loader_2):
        bbox = frame_bboxes[idx]["bbox"]
        if bbox is not None:
            detected_frames += 1
            x1, y1, x2, y2 = bbox
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            cleaned_frame = cleaner.clean(frame, mask)
        else:
            cleaned_frame = frame
        process_out.stdin.write(cleaned_frame.tobytes())

    process_out.stdin.close()
    process_out.wait()
    merge_audio(input_video, temp_output, output_video)
    temp_output.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    return {
        "input": str(input_video),
        "output": str(output_video),
        "frames": total_frames,
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "elapsed": elapsed,
    }


def write_batch_report(report_path: Path, rows: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input",
                "output",
                "frames",
                "total_frames",
                "detected_frames",
                "elapsed",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["elapsed"] = f"{out['elapsed']:.3f}"
            writer.writerow(out)


def run_batch(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    detect_batch_size: int,
    report_path: Path | None,
    progress: Callable[[str], None],
) -> int:
    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"No files matched: {input_dir}/{pattern}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_start = time.perf_counter()
    rows: list[dict] = []
    for src in files:
        if not src.is_file():
            continue
        dst = output_dir / f"{src.stem}_cleaned{src.suffix}"
        progress(f"[Batch] Processing: {src.name}")
        result = process_video(src, dst, detect_batch_size)
        rows.append(result)
        progress(f"[Batch] Done: {src.name} ({result['elapsed']:.3f}s)")

    total_elapsed = time.perf_counter() - batch_start
    report = report_path if report_path is not None else output_dir / "batch_report.csv"
    write_batch_report(report, rows)
    print(f"BatchCount: {len(rows)}")
    print(f"BatchReport: {report}")
    print(f"BatchElapsedSeconds: {total_elapsed:.3f}")
    return 0


def main() -> int:
    args = parse_args()
    if args.input is not None:
        in_video = args.input.expanduser().resolve()
        out_video = args.output.expanduser().resolve()
        if not in_video.exists():
            print(f"Input not found: {in_video}", file=sys.stderr)
            return 1
        result = process_video(in_video, out_video, args.detect_batch_size)
        print(f"Input: {result['input']}")
        print(f"Output: {result['output']}")
        print(f"Frames: {result['frames']}/{result['total_frames']}")
        print(f"DetectedFrames: {result['detected_frames']}")
        print(f"ElapsedSeconds: {result['elapsed']:.3f}")
        return 0

    return run_batch(
        args.input_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.pattern,
        args.detect_batch_size,
        args.report.expanduser().resolve() if args.report is not None else None,
        print,
    )


if __name__ == "__main__":
    raise SystemExit(main())
