#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

MODEL_URL = "https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/V0.0.1/best.pt"


def ensure_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    with requests.get(MODEL_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with model_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return model_path


def detect_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    try:
        import torch  # noqa: PLC0415

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
    except Exception:
        pass
    return "cpu"


def expand_bbox(
    bbox: Tuple[int, int, int, int], width: int, height: int, expand: float
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * expand)
    dy = int(bh * expand)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    )


def merge_audio(src: Path, no_audio_video: Path, out: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(no_audio_video),
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight Sora watermark cleaner")
    parser.add_argument("-i", "--input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pattern", default="*.MP4")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--expand", type=float, default=0.2)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--method", choices=["telea", "ns"], default="telea")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / ".cache" / "sora-watermark-lite" / "best.pt",
    )
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


def process_video(
    model: YOLO,
    in_video: Path,
    out_video: Path,
    *,
    device: str,
    conf: float,
    expand: float,
    max_gap: int,
    radius: int,
    inpaint_flag: int,
) -> dict:
    start = time.perf_counter()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {in_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp = out_video.parent / f"tmp_{out_video.name}"
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_fn is None:
        raise RuntimeError("OpenCV build does not provide VideoWriter_fourcc")

    writer = cv2.VideoWriter(
        str(tmp),
        fourcc_fn(*"mp4v"),
        fps,
        (width, height),
    )

    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_seen_idx = -10**9
    detected_frames = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, verbose=False, device=device, conf=conf)[0]
        chosen: Optional[Tuple[int, int, int, int]] = None
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
            best = int(np.argmax(confs))
            x1, y1, x2, y2 = boxes[best].tolist()
            chosen = expand_bbox((x1, y1, x2, y2), width, height, expand)
            last_bbox = chosen
            last_seen_idx = frame_idx
            detected_frames += 1
        elif last_bbox is not None and (frame_idx - last_seen_idx) <= max_gap:
            chosen = last_bbox

        if chosen is not None:
            x1, y1, x2, y2 = chosen
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            frame = cv2.inpaint(frame, mask, radius, inpaint_flag)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    try:
        merge_audio(in_video, tmp, out_video)
    finally:
        if tmp.exists():
            tmp.unlink()

    elapsed = time.perf_counter() - start
    return {
        "input": str(in_video),
        "output": str(out_video),
        "frames": frame_idx,
        "total_frames": total,
        "detected_frames": detected_frames,
        "device": device,
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
                "device",
                "elapsed",
            ],
        )
        writer.writeheader()
        for row in rows:
            row2 = dict(row)
            row2["elapsed"] = f"{row2['elapsed']:.3f}"
            writer.writerow(row2)


def main() -> int:
    args = parse_args()
    model_path = ensure_model(args.model_path.expanduser().resolve())
    device = detect_device(args.device)
    inpaint_flag = cv2.INPAINT_TELEA if args.method == "telea" else cv2.INPAINT_NS
    model = YOLO(str(model_path))

    if args.input is not None:
        in_video = args.input.expanduser().resolve()
        out_video = args.output.expanduser().resolve()

        if not in_video.exists():
            print(f"Input not found: {in_video}", file=sys.stderr)
            return 1

        result = process_video(
            model,
            in_video,
            out_video,
            device=device,
            conf=args.conf,
            expand=args.expand,
            max_gap=args.max_gap,
            radius=args.radius,
            inpaint_flag=inpaint_flag,
        )
        print(f"Input: {result['input']}")
        print(f"Output: {result['output']}")
        print(f"Frames: {result['frames']}/{result['total_frames']}")
        print(f"DetectedFrames: {result['detected_frames']}")
        print(f"Device: {result['device']}")
        print(f"ElapsedSeconds: {result['elapsed']:.3f}")
        return 0

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matched: {input_dir}/{args.pattern}", file=sys.stderr)
        return 1

    batch_start = time.perf_counter()
    rows: list[dict] = []
    for src in files:
        if not src.is_file():
            continue
        dst = output_dir / f"{src.stem}_cleaned{src.suffix}"
        print(f"[Batch] Processing: {src.name}")
        result = process_video(
            model,
            src,
            dst,
            device=device,
            conf=args.conf,
            expand=args.expand,
            max_gap=args.max_gap,
            radius=args.radius,
            inpaint_flag=inpaint_flag,
        )
        rows.append(result)
        print(f"[Batch] Done: {src.name} ({result['elapsed']:.3f}s)")

    total_elapsed = time.perf_counter() - batch_start
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output_dir / "batch_report.csv"
    )
    write_batch_report(report_path, rows)
    print(f"BatchCount: {len(rows)}")
    print(f"BatchReport: {report_path}")
    print(f"BatchElapsedSeconds: {total_elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
