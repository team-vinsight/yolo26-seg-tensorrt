from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO segmentation models in PT, ONNX, and TensorRT engine formats.",
    )
    parser.add_argument("--pt", default="models/yolo26n-seg.pt", help="Path to .pt model")
    parser.add_argument("--onnx", default="models/yolo26n-seg.onnx", help="Path to .onnx model")
    parser.add_argument("--engine", default="models/yolo26n-seg.engine", help="Path to .engine model")
    parser.add_argument("--video", default="media/car.mp4", help="Video used to extract one benchmark frame")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=50, help="Timed iterations")
    return parser.parse_args()


def first_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    return frame


def benchmark_model(model_path: Path, frame, imgsz: int, warmup: int, runs: int) -> dict:
    model = YOLO(str(model_path))

    for _ in range(warmup):
        model.predict(frame, imgsz=imgsz, verbose=False)

    latencies_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(frame, imgsz=imgsz, verbose=False)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    p90_ms = statistics.quantiles(latencies_ms, n=10)[8] if runs >= 10 else max(latencies_ms)
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "model": model_path.name,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "throughput": throughput,
    }


def main() -> None:
    args = parse_args()
    root = REPO_ROOT

    model_paths = [
        root / args.pt,
        root / args.onnx,
        root / args.engine,
    ]
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")

    video_path = root / args.video
    frame = first_frame(video_path)

    print(f"Benchmark input frame: {frame.shape[1]}x{frame.shape[0]} from {video_path.name}")
    print(f"Warmup: {args.warmup} | Runs: {args.runs} | imgsz: {args.imgsz}")
    print()
    print(
        f"{'Model':<20} {'Mean(ms)':>10} {'Median(ms)':>11} {'P90(ms)':>9} {'Min(ms)':>9} {'Max(ms)':>9} {'QPS':>9}"
    )
    print("-" * 82)

    for path in model_paths:
        try:
            stats = benchmark_model(path, frame, args.imgsz, args.warmup, args.runs)
            print(
                f"{stats['model']:<20} {stats['mean_ms']:>10.2f} {stats['median_ms']:>11.2f}"
                f" {stats['p90_ms']:>9.2f} {stats['min_ms']:>9.2f} {stats['max_ms']:>9.2f} {stats['throughput']:>9.2f}"
            )
        except Exception as exc:
            print(f"{path.name:<20} ERROR: {exc}")


if __name__ == "__main__":
    main()