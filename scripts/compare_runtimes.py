import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def nested_value(data, *keys):
    value = data

    for key in keys:
        value = value[key]

    return value


def percentage_change(candidate, baseline):
    if baseline == 0:
        return 0.0

    return ((candidate - baseline) / baseline) * 100.0


def format_change(candidate, baseline):
    change = percentage_change(candidate, baseline)

    if change > 0:
        return f"+{change:.1f}%"

    return f"{change:.1f}%"


def verify_equivalent(pytorch_data, onnx_data):
    checks = [
        ("source",),
        ("requested_imgsz",),
        ("confidence_threshold",),
        ("nms_iou_threshold",),
        ("warmup_frames",),
        ("measured_frames",),
    ]

    mismatches = []

    for keys in checks:
        pytorch_value = nested_value(
            pytorch_data,
            *keys,
        )

        onnx_value = nested_value(
            onnx_data,
            *keys,
        )

        if pytorch_value != onnx_value:
            mismatches.append(
                (
                    ".".join(keys),
                    pytorch_value,
                    onnx_value,
                )
            )

    if mismatches:
        print("Warning: benchmark settings differ:")

        for name, pytorch_value, onnx_value in mismatches:
            print(
                f"  {name}: "
                f"PyTorch={pytorch_value}, "
                f"ONNX={onnx_value}"
            )

        raise SystemExit(
            "Comparison stopped because the runs "
            "are not equivalent."
        )


def print_row(
    name,
    pytorch_value,
    onnx_value,
    unit="",
):
    change = format_change(
        onnx_value,
        pytorch_value,
    )

    print(
        f"| {name} "
        f"| {pytorch_value:.2f}{unit} "
        f"| {onnx_value:.2f}{unit} "
        f"| {change} |"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare equivalent PyTorch and "
            "ONNX Runtime benchmarks"
        )
    )

    parser.add_argument(
        "--pytorch",
        required=True,
    )

    parser.add_argument(
        "--onnx",
        required=True,
    )

    args = parser.parse_args()

    pytorch_data = load_json(args.pytorch)
    onnx_data = load_json(args.onnx)

    verify_equivalent(
        pytorch_data,
        onnx_data,
    )

    image_size = pytorch_data["requested_imgsz"]

    print(
        f"\n## Baseline vs Candidate — "
        f"{image_size} × {image_size}\n"
    )

    print(
        "| Metric | Baseline | Candidate | Candidate change |"
    )
    print(
        "|---|---:|---:|---:|"
    )

    print_row(
        "Mean inference latency",
        pytorch_data[
            "inference_latency_ms"
        ]["mean"],
        onnx_data[
            "inference_latency_ms"
        ]["mean"],
        " ms",
    )

    print_row(
        "p95 inference latency",
        pytorch_data[
            "inference_latency_ms"
        ]["p95"],
        onnx_data[
            "inference_latency_ms"
        ]["p95"],
        " ms",
    )

    print_row(
        "Mean postprocess latency",
        pytorch_data[
            "postprocess_latency_ms"
        ]["mean"],
        onnx_data[
            "postprocess_latency_ms"
        ]["mean"],
        " ms",
    )

    print_row(
        "Mean pipeline latency",
        pytorch_data[
            "pipeline_latency_ms"
        ]["mean"],
        onnx_data[
            "pipeline_latency_ms"
        ]["mean"],
        " ms",
    )

    print_row(
        "p95 pipeline latency",
        pytorch_data[
            "pipeline_latency_ms"
        ]["p95"],
        onnx_data[
            "pipeline_latency_ms"
        ]["p95"],
        " ms",
    )

    print_row(
        "Throughput",
        pytorch_data["end_to_end_fps"],
        onnx_data["end_to_end_fps"],
        " FPS",
    )

    print_row(
        "Peak RSS",
        pytorch_data[
            "memory_mb"
        ]["peak_rss"],
        onnx_data[
            "memory_mb"
        ]["peak_rss"],
        " MB",
    )

    print_row(
        "CPU core-equivalents",
        pytorch_data[
            "cpu"
        ]["core_equivalents"],
        onnx_data[
            "cpu"
        ]["core_equivalents"],
    )

    print_row(
        "Mean detections/frame",
        pytorch_data[
            "detections"
        ]["mean_per_frame"],
        onnx_data[
            "detections"
        ]["mean_per_frame"],
    )

    print_row(
        "Total detections",
        float(
            pytorch_data[
                "detections"
            ]["total"]
        ),
        float(
            onnx_data[
                "detections"
            ]["total"]
        ),
    )

    print(
        "\nNegative latency or memory change is an "
        "improvement. Positive FPS change is an "
        "improvement."
    )
    baseline_classes = (
        pytorch_data
        .get("detections", {})
        .get("classes", {})
    )

    candidate_classes = (
        onnx_data
        .get("detections", {})
        .get("classes", {})
    )

    all_classes = sorted(
        set(baseline_classes)
        | set(candidate_classes)
    )

    print("\n### Per-class detection counts\n")
    print(
        "| Class | Baseline | Candidate "
        "| Difference | Retention |"
    )
    print("|---|---:|---:|---:|---:|")

    for class_name in all_classes:
        baseline_count = baseline_classes.get(
            class_name,
            0,
        )

        candidate_count = candidate_classes.get(
            class_name,
            0,
        )

        difference = (
            candidate_count - baseline_count
        )

        if baseline_count > 0:
            retention = (
                candidate_count
                / baseline_count
                * 100.0
            )

            retention_text = (
                f"{retention:.1f}%"
            )
        else:
            retention_text = "N/A"

        print(
            f"| {class_name} "
            f"| {baseline_count} "
            f"| {candidate_count} "
            f"| {difference:+d} "
            f"| {retention_text} |"
        )

if __name__ == "__main__":
    main()
