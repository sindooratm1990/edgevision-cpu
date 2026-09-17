import argparse
import csv
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
from ultralytics import YOLO


def percentile(values: list[float], value: int) -> float:
    return float(np.percentile(values, value)) if values else 0.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO CPU inference on a fixed video."
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Path to the YOLO model",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to the input video",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help="Inference image size, such as 320 or 640",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detection confidence threshold",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of initial frames excluded from metrics",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of measured frames",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    source_path = Path(args.source)

    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process = psutil.Process(os.getpid())
    logical_cpu_count = psutil.cpu_count(logical=True) or 1

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print("Execution device: CPU")
    print(f"Input video: {source_path}")
    print(f"Requested image size: {args.imgsz}")
    print(f"Warm-up frames: {args.warmup}")
    print(f"Measured frames: {args.max_frames}")

    result_stream = iter(
        model.predict(
            source=str(source_path),
            device="cpu",
            imgsz=args.imgsz,
            conf=args.conf,
            stream=True,
            verbose=False,
            save=False,
	    rect=False,
        )
    )

    # Warm-up allows model initialization and internal caches to stabilize.
    print("\nWarming up...")

    for _ in range(args.warmup):
        try:
            next(result_stream)
        except StopIteration:
            raise RuntimeError("Video ended during warm-up")

    pipeline_times_ms = []
    preprocess_times_ms = []
    inference_times_ms = []
    postprocess_times_ms = []
    rss_values_mb = []
    detection_counts = []
    class_counts = Counter()

    process_cpu_start = process.cpu_times()
    benchmark_start = time.perf_counter()

    print("Benchmarking...")

    for frame_number in range(args.max_frames):
        frame_start = time.perf_counter()

        try:
            result = next(result_stream)
        except StopIteration:
            break

        frame_end = time.perf_counter()

        pipeline_ms = (frame_end - frame_start) * 1000.0
        pipeline_times_ms.append(pipeline_ms)

        preprocess_times_ms.append(float(result.speed["preprocess"]))
        inference_times_ms.append(float(result.speed["inference"]))
        postprocess_times_ms.append(float(result.speed["postprocess"]))

        rss_mb = process.memory_info().rss / (1024 * 1024)
        rss_values_mb.append(rss_mb)

        number_of_detections = len(result.boxes)
        detection_counts.append(number_of_detections)

        if result.boxes is not None:
            for class_id in result.boxes.cls.tolist():
                class_name = result.names[int(class_id)]
                class_counts[class_name] += 1

    benchmark_end = time.perf_counter()
    process_cpu_end = process.cpu_times()

    if not pipeline_times_ms:
        raise RuntimeError("No frames were measured")

    wall_time_seconds = benchmark_end - benchmark_start

    cpu_time_seconds = (
        process_cpu_end.user
        + process_cpu_end.system
        - process_cpu_start.user
        - process_cpu_start.system
    )

    # One fully occupied core equals 1.0 core-equivalent.
    cpu_core_equivalents = cpu_time_seconds / wall_time_seconds

    # Percentage of the Mac's total logical CPU capacity used by this process.
    total_cpu_capacity_percent = (
        cpu_core_equivalents / logical_cpu_count
    ) * 100.0

    measured_frames = len(pipeline_times_ms)
    end_to_end_fps = measured_frames / wall_time_seconds

    metrics = {
        "model": args.model,
        "source": str(source_path),
        "device": "cpu",
        "requested_imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "warmup_frames": args.warmup,
        "measured_frames": measured_frames,
        "wall_time_seconds": round(wall_time_seconds, 4),
        "end_to_end_fps": round(end_to_end_fps, 2),
        "pipeline_latency_ms": {
            "mean": round(statistics.mean(pipeline_times_ms), 3),
            "median": round(statistics.median(pipeline_times_ms), 3),
            "p95": round(percentile(pipeline_times_ms, 95), 3),
            "p99": round(percentile(pipeline_times_ms, 99), 3),
            "minimum": round(min(pipeline_times_ms), 3),
            "maximum": round(max(pipeline_times_ms), 3),
        },
        "preprocess_latency_ms": {
            "mean": round(statistics.mean(preprocess_times_ms), 3),
            "p95": round(percentile(preprocess_times_ms, 95), 3),
        },
        "inference_latency_ms": {
            "mean": round(statistics.mean(inference_times_ms), 3),
            "p95": round(percentile(inference_times_ms, 95), 3),
        },
        "postprocess_latency_ms": {
            "mean": round(statistics.mean(postprocess_times_ms), 3),
            "p95": round(percentile(postprocess_times_ms, 95), 3),
        },
        "memory_mb": {
            "mean_rss": round(statistics.mean(rss_values_mb), 2),
            "peak_rss": round(max(rss_values_mb), 2),
        },
        "cpu": {
            "logical_cpu_count": logical_cpu_count,
            "core_equivalents": round(cpu_core_equivalents, 2),
            "total_capacity_percent": round(total_cpu_capacity_percent, 2),
        },
        "detections": {
            "mean_per_frame": round(statistics.mean(detection_counts), 2),
            "total": sum(detection_counts),
            "classes": dict(class_counts.most_common()),
        },
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)

    frame_csv_path = output_path.with_suffix(".frames.csv")

    with frame_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "frame",
                "pipeline_ms",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "rss_mb",
                "detections",
            ]
        )

        for index in range(measured_frames):
            writer.writerow(
                [
                    index + 1,
                    round(pipeline_times_ms[index], 3),
                    round(preprocess_times_ms[index], 3),
                    round(inference_times_ms[index], 3),
                    round(postprocess_times_ms[index], 3),
                    round(rss_values_mb[index], 2),
                    detection_counts[index],
                ]
            )

    print("\nBenchmark complete")
    print(json.dumps(metrics, indent=2))
    print(f"\nSummary: {output_path}")
    print(f"Per-frame data: {frame_csv_path}")


if __name__ == "__main__":
    main()
