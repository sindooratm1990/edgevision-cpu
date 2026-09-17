import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil


def percentile(values, percentage):
    return float(np.percentile(values, percentage)) if values else 0.0


def letterbox(image, size):
    original_height, original_width = image.shape[:2]

    scale = min(
        size / original_width,
        size / original_height,
    )

    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal_padding = size - resized_width
    vertical_padding = size - resized_height

    left = horizontal_padding // 2
    right = horizontal_padding - left
    top = vertical_padding // 2
    bottom = vertical_padding - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded


def preprocess(frame, size):
    image = letterbox(frame, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)

    return image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pure ONNX Runtime CPU benchmark"
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_path = Path(args.model)
    source_path = Path(args.source)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    process = psutil.Process(os.getpid())
    logical_cpu_count = psutil.cpu_count(logical=True) or 1

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    print(f"Loading model: {model_path}")

    model_load_start = time.perf_counter()

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0

    input_metadata = session.get_inputs()[0]
    output_metadata = session.get_outputs()[0]
    input_name = input_metadata.name

    print(f"Provider: {session.get_providers()}")
    print(f"Input shape: {input_metadata.shape}")
    print(f"Output shape: {output_metadata.shape}")
    print(f"Model load time: {model_load_ms:.2f} ms")

    capture = cv2.VideoCapture(str(source_path))

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    print(f"Warming up with {args.warmup} frames...")

    for _ in range(args.warmup):
        success, frame = capture.read()

        if not success:
            raise RuntimeError("Video ended during warm-up")

        tensor = preprocess(frame, args.imgsz)
        session.run(None, {input_name: tensor})

    decode_times = []
    preprocess_times = []
    inference_times = []
    pipeline_times = []
    rss_values = []

    process_cpu_start = process.cpu_times()
    benchmark_start = time.perf_counter()

    print(f"Benchmarking up to {args.max_frames} frames...")

    output_shape = None

    for _ in range(args.max_frames):
        pipeline_start = time.perf_counter()

        decode_start = time.perf_counter()
        success, frame = capture.read()
        decode_end = time.perf_counter()

        if not success:
            break

        preprocess_start = time.perf_counter()
        tensor = preprocess(frame, args.imgsz)
        preprocess_end = time.perf_counter()

        inference_start = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        inference_end = time.perf_counter()

        pipeline_end = time.perf_counter()

        decode_times.append((decode_end - decode_start) * 1000.0)
        preprocess_times.append(
            (preprocess_end - preprocess_start) * 1000.0
        )
        inference_times.append(
            (inference_end - inference_start) * 1000.0
        )
        pipeline_times.append(
            (pipeline_end - pipeline_start) * 1000.0
        )

        rss_values.append(
            process.memory_info().rss / (1024 * 1024)
        )

        output_shape = list(outputs[0].shape)

    benchmark_end = time.perf_counter()
    process_cpu_end = process.cpu_times()
    capture.release()

    if not pipeline_times:
        raise RuntimeError("No frames were benchmarked")

    wall_time_seconds = benchmark_end - benchmark_start

    cpu_seconds = (
        process_cpu_end.user
        + process_cpu_end.system
        - process_cpu_start.user
        - process_cpu_start.system
    )

    core_equivalents = cpu_seconds / wall_time_seconds
    total_cpu_capacity_percent = (
        core_equivalents / logical_cpu_count
    ) * 100.0

    measured_frames = len(pipeline_times)

    metrics = {
        "runtime": "onnxruntime",
        "provider": session.get_providers(),
        "model": str(model_path),
        "model_size_mb": round(
            model_path.stat().st_size / (1024 * 1024),
            2,
        ),
        "source": str(source_path),
        "requested_imgsz": args.imgsz,
        "input_shape": input_metadata.shape,
        "output_shape": output_shape,
        "model_load_ms": round(model_load_ms, 3),
        "warmup_frames": args.warmup,
        "measured_frames": measured_frames,
        "wall_time_seconds": round(wall_time_seconds, 4),
        "end_to_end_fps": round(
            measured_frames / wall_time_seconds,
            2,
        ),
        "decode_latency_ms": {
            "mean": round(statistics.mean(decode_times), 3),
            "p95": round(percentile(decode_times, 95), 3),
        },
        "preprocess_latency_ms": {
            "mean": round(statistics.mean(preprocess_times), 3),
            "p95": round(percentile(preprocess_times, 95), 3),
        },
        "inference_latency_ms": {
            "mean": round(statistics.mean(inference_times), 3),
            "median": round(statistics.median(inference_times), 3),
            "p95": round(percentile(inference_times, 95), 3),
            "p99": round(percentile(inference_times, 99), 3),
            "minimum": round(min(inference_times), 3),
            "maximum": round(max(inference_times), 3),
        },
        "pipeline_latency_ms": {
            "mean": round(statistics.mean(pipeline_times), 3),
            "median": round(statistics.median(pipeline_times), 3),
            "p95": round(percentile(pipeline_times, 95), 3),
            "p99": round(percentile(pipeline_times, 99), 3),
            "minimum": round(min(pipeline_times), 3),
            "maximum": round(max(pipeline_times), 3),
        },
        "memory_mb": {
            "mean_rss": round(statistics.mean(rss_values), 2),
            "peak_rss": round(max(rss_values), 2),
        },
        "cpu": {
            "logical_cpu_count": logical_cpu_count,
            "core_equivalents": round(core_equivalents, 2),
            "total_capacity_percent": round(
                total_cpu_capacity_percent,
                2,
            ),
        },
        "note": (
            "Pipeline includes video decoding, preprocessing and ONNX "
            "inference. Detection decoding and NMS are not yet included."
        ),
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)

    print("\nBenchmark complete")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
