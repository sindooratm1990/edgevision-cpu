import argparse
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from coco_names import COCO_NAMES
from onnx_detector import ONNXDetector, preprocess


def percentile(values, percentage):
    if not values:
        return 0.0

    return float(np.percentile(values, percentage))


def latency_summary(values):
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "minimum": round(min(values), 3),
        "maximum": round(max(values), 3),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a complete CPU-only ONNX YOLO pipeline, "
            "including decoding and NMS."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the ONNX model",
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
        help="Model input size, such as 320 or 640",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold used by NMS",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Number of ONNX Runtime inference threads. "
            "Use 0 for the runtime default."
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up frames",
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
        help="Output JSON path",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_path = Path(args.model)
    source_path = Path(args.source)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Video not found: {source_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    process = psutil.Process(os.getpid())
    logical_cpu_count = psutil.cpu_count(logical=True) or 1

    # A value of zero means ONNX Runtime chooses its default.
    configured_threads = (
        None if args.threads == 0 else args.threads
    )

    print(f"Loading model: {model_path}")

    model_load_start = time.perf_counter()

    detector = ONNXDetector(
        model_path=str(model_path),
        image_size=args.imgsz,
        class_names=COCO_NAMES,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        threads=configured_threads,
    )

    model_load_ms = (
        time.perf_counter() - model_load_start
    ) * 1000.0

    input_metadata = detector.session.get_inputs()[0]
    output_metadata = detector.session.get_outputs()[0]

    print(f"Providers: {detector.session.get_providers()}")
    print(f"Input shape: {input_metadata.shape}")
    print(f"Output shape: {output_metadata.shape}")
    print(f"Threads: {args.threads or 'runtime default'}")
    print(f"Confidence threshold: {args.conf}")
    print(f"NMS IoU threshold: {args.iou}")
    print(f"Model load time: {model_load_ms:.2f} ms")

    capture = cv2.VideoCapture(str(source_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {source_path}"
        )

    print(f"Warming up with {args.warmup} frames...")

    for _ in range(args.warmup):
        success, frame = capture.read()

        if not success:
            capture.release()
            raise RuntimeError(
                "Video ended during warm-up"
            )

        tensor, letterbox_info = preprocess(
            frame,
            args.imgsz,
        )

        outputs = detector.session.run(
            None,
            {detector.input_name: tensor},
        )

        # Warm up decoding and NMS as well as model inference.
        detector.decode(
            outputs[0],
            frame.shape,
            letterbox_info,
        )

    decode_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    pipeline_times = []
    rss_values = []
    detection_counts = []

    class_counts = Counter()

    process_cpu_start = process.cpu_times()
    benchmark_start = time.perf_counter()

    print(
        f"Benchmarking up to "
        f"{args.max_frames} frames..."
    )

    output_shape = None

    for _ in range(args.max_frames):
        pipeline_start = time.perf_counter()

        # Stage 1: Read and decode one video frame.
        decode_start = time.perf_counter()

        success, frame = capture.read()

        decode_end = time.perf_counter()

        if not success:
            break

        # Stage 2: Letterbox, convert and normalize.
        preprocess_start = time.perf_counter()

        tensor, letterbox_info = preprocess(
            frame,
            args.imgsz,
        )

        preprocess_end = time.perf_counter()

        # Stage 3: Execute the ONNX neural network.
        inference_start = time.perf_counter()

        outputs = detector.session.run(
            None,
            {detector.input_name: tensor},
        )

        inference_end = time.perf_counter()

        # Stage 4: Decode raw YOLO output and apply NMS.
        postprocess_start = time.perf_counter()

        detections = detector.decode(
            outputs[0],
            frame.shape,
            letterbox_info,
        )

        postprocess_end = time.perf_counter()
        pipeline_end = postprocess_end

        decode_times.append(
            (decode_end - decode_start) * 1000.0
        )

        preprocess_times.append(
            (preprocess_end - preprocess_start)
            * 1000.0
        )

        inference_times.append(
            (inference_end - inference_start)
            * 1000.0
        )

        postprocess_times.append(
            (postprocess_end - postprocess_start)
            * 1000.0
        )

        pipeline_times.append(
            (pipeline_end - pipeline_start)
            * 1000.0
        )

        rss_values.append(
            process.memory_info().rss / (1024 * 1024)
        )

        detection_counts.append(len(detections))

        for detection in detections:
            class_counts[detection.class_name] += 1

        output_shape = list(outputs[0].shape)

    benchmark_end = time.perf_counter()
    process_cpu_end = process.cpu_times()

    capture.release()

    if not pipeline_times:
        raise RuntimeError(
            "No frames were benchmarked"
        )

    wall_time_seconds = (
        benchmark_end - benchmark_start
    )

    cpu_time_seconds = (
        process_cpu_end.user
        + process_cpu_end.system
        - process_cpu_start.user
        - process_cpu_start.system
    )

    cpu_core_equivalents = (
        cpu_time_seconds / wall_time_seconds
    )

    total_cpu_capacity_percent = (
        cpu_core_equivalents
        / logical_cpu_count
        * 100.0
    )

    measured_frames = len(pipeline_times)

    metrics = {
        "runtime": "onnxruntime",
        "provider": detector.session.get_providers(),
        "model": str(model_path),
        "model_size_mb": round(
            model_path.stat().st_size
            / (1024 * 1024),
            2,
        ),
        "source": str(source_path),
        "requested_imgsz": args.imgsz,
        "input_shape": input_metadata.shape,
        "output_shape": output_shape,
        "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "configured_threads": (
            args.threads
            if args.threads > 0
            else "runtime_default"
        ),
        "model_load_ms": round(model_load_ms, 3),
        "warmup_frames": args.warmup,
        "measured_frames": measured_frames,
        "wall_time_seconds": round(
            wall_time_seconds,
            4,
        ),
        "end_to_end_fps": round(
            measured_frames / wall_time_seconds,
            2,
        ),
        "decode_latency_ms": latency_summary(
            decode_times
        ),
        "preprocess_latency_ms": latency_summary(
            preprocess_times
        ),
        "inference_latency_ms": latency_summary(
            inference_times
        ),
        "postprocess_latency_ms": latency_summary(
            postprocess_times
        ),
        "pipeline_latency_ms": latency_summary(
            pipeline_times
        ),
        "memory_mb": {
            "mean_rss": round(
                statistics.mean(rss_values),
                2,
            ),
            "peak_rss": round(
                max(rss_values),
                2,
            ),
        },
        "cpu": {
            "logical_cpu_count": logical_cpu_count,
            "core_equivalents": round(
                cpu_core_equivalents,
                2,
            ),
            "total_capacity_percent": round(
                total_cpu_capacity_percent,
                2,
            ),
        },
        "detections": {
            "mean_per_frame": round(
                statistics.mean(detection_counts),
                2,
            ),
            "minimum_per_frame": min(
                detection_counts
            ),
            "maximum_per_frame": max(
                detection_counts
            ),
            "total": sum(detection_counts),
            "classes": dict(
                class_counts.most_common()
            ),
        },
        "pipeline_scope": [
            "video_decode",
            "preprocess",
            "onnx_inference",
            "confidence_filtering",
            "output_decoding",
            "class_aware_nms",
            "coordinate_mapping",
        ],
        "excluded_from_pipeline": [
            "drawing_boxes",
            "display",
            "video_encoding",
            "mqtt_publishing",
            "sqlite_persistence",
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metrics,
            output_file,
            indent=2,
        )

    print("\nBenchmark complete")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
