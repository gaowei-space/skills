#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import ffmpeg
import numpy as np
import requests
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLO

MODEL_URL = "https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/V0.0.1/best.pt"


def ensure_detector_model(model_path: Path) -> Path:
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


def lama_device_from_name(device_name: str) -> torch.device:
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained LAMA Sora watermark cleaner")
    parser.add_argument("-i", "--input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pattern", default="*.MP4")
    parser.add_argument("--report", type=Path)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--expand", type=float, default=0.0)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--detect-batch-size", type=int, default=4)
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


def has_encoder(name: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return name in result.stdout


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


def expand_bbox(bbox: tuple[int, int, int, int], width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    if ratio <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * ratio)
    dy = int(bh * ratio)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    )


def detect_bboxes(
    model: YOLO,
    frames: list[np.ndarray],
    device: str,
    conf: float,
    width: int,
    height: int,
    expand: float,
) -> list[Optional[tuple[int, int, int, int]]]:
    results = model(frames, verbose=False, device=device, conf=conf)
    out: list[Optional[tuple[int, int, int, int]]] = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            out.append(None)
            continue
        confs = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        best = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[best].tolist()
        out.append(expand_bbox((x1, y1, x2, y2), width, height, expand))
    return out


def fill_missed_bboxes(bboxes: list[Optional[tuple[int, int, int, int]]], max_gap: int) -> list[Optional[tuple[int, int, int, int]]]:
    out = list(bboxes)
    last_box: Optional[tuple[int, int, int, int]] = None
    last_seen = -10**9
    for idx, box in enumerate(out):
        if box is not None:
            last_box = box
            last_seen = idx
            continue
        if last_box is not None and (idx - last_seen) <= max_gap:
            out[idx] = last_box
    return out


def lama_clean_frame(
    lama: SimpleLama,
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    crop_padding: int = 48,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    cx1 = max(0, x1 - crop_padding)
    cy1 = max(0, y1 - crop_padding)
    cx2 = min(w, x2 + crop_padding)
    cy2 = min(h, y2 + crop_padding)

    crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
    crop_h, crop_w = crop_bgr.shape[:2]

    local_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    lx1 = x1 - cx1
    ly1 = y1 - cy1
    lx2 = x2 - cx1
    ly2 = y2 - cy1
    local_mask[ly1:ly2, lx1:lx2] = 255

    crop_rgb = crop_bgr[:, :, ::-1]
    out_crop_img = lama(Image.fromarray(crop_rgb), Image.fromarray(local_mask))
    out_crop_rgb = np.array(out_crop_img)
    out_crop_bgr = out_crop_rgb[:, :, ::-1]
    if out_crop_bgr.shape[0] != crop_h or out_crop_bgr.shape[1] != crop_w:
        import cv2  # noqa: PLC0415

        out_crop_bgr = cv2.resize(out_crop_bgr, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    out = frame_bgr.copy()
    out[cy1:cy2, cx1:cx2] = out_crop_bgr
    return out


def process_video(
    model: YOLO,
    lama: SimpleLama,
    in_video: Path,
    out_video: Path,
    device: str,
    conf: float,
    expand: float,
    max_gap: int,
    detect_batch_size: int,
) -> dict:
    import cv2  # noqa: PLC0415

    start = time.perf_counter()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {in_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()

    bboxes: list[Optional[tuple[int, int, int, int]]] = [None] * len(all_frames)
    for start_idx in range(0, len(all_frames), detect_batch_size):
        chunk = all_frames[start_idx : start_idx + detect_batch_size]
        detected = detect_bboxes(model, chunk, device, conf, width, height, expand)
        bboxes[start_idx : start_idx + len(detected)] = detected

    bboxes = fill_missed_bboxes(bboxes, max_gap=max_gap)

    temp_output = out_video.parent / f"temp_{out_video.name}"
    process_out = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{width}x{height}", r=fps)
        .output(str(temp_output), **pick_encode_options())
        .overwrite_output()
        .global_args("-loglevel", "error")
        .run_async(pipe_stdin=True)
    )

    detected_frames = 0
    for idx, (frame, bbox) in enumerate(zip(all_frames, bboxes), start=1):
        if bbox is not None:
            detected_frames += 1
            cleaned = lama_clean_frame(lama, frame, bbox)
        else:
            cleaned = frame
        process_out.stdin.write(cleaned.tobytes())
        if idx % 10 == 0 or idx == len(all_frames):
            print(f"[Clean] {idx}/{len(all_frames)} frames", flush=True)

    process_out.stdin.close()
    process_out.wait()

    merge_audio(in_video, temp_output, out_video)
    temp_output.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    return {
        "input": str(in_video),
        "output": str(out_video),
        "frames": len(all_frames),
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "device": device,
        "elapsed": elapsed,
    }


def write_batch_report(report_path: Path, rows: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["input", "output", "frames", "total_frames", "detected_frames", "device", "elapsed"],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["elapsed"] = f"{out['elapsed']:.3f}"
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    model_path = ensure_detector_model(args.model_path.expanduser().resolve())
    device = detect_device(args.device)

    model = YOLO(str(model_path))
    lama_device = lama_device_from_name(device)
    lama = SimpleLama(device=lama_device)

    if args.input is not None:
        in_video = args.input.expanduser().resolve()
        out_video = args.output.expanduser().resolve()
        if not in_video.exists():
            print(f"Input not found: {in_video}", file=sys.stderr)
            return 1
        result = process_video(
            model,
            lama,
            in_video,
            out_video,
            device,
            args.conf,
            args.expand,
            args.max_gap,
            args.detect_batch_size,
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
    files = sorted([p for p in input_dir.glob(args.pattern) if p.is_file()])
    if not files:
        print(f"No files matched: {input_dir}/{args.pattern}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_start = time.perf_counter()
    rows: list[dict] = []
    for src in files:
        dst = output_dir / f"{src.stem}_cleaned{src.suffix}"
        print(f"[Batch] Processing: {src.name}")
        row = process_video(
            model,
            lama,
            src,
            dst,
            device,
            args.conf,
            args.expand,
            args.max_gap,
            args.detect_batch_size,
        )
        rows.append(row)
        print(f"[Batch] Done: {src.name} ({row['elapsed']:.3f}s)")

    report = args.report.expanduser().resolve() if args.report else output_dir / "batch_report.csv"
    write_batch_report(report, rows)
    print(f"BatchCount: {len(rows)}")
    print(f"BatchReport: {report}")
    print(f"BatchElapsedSeconds: {time.perf_counter() - batch_start:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
