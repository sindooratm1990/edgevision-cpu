import argparse

import cv2
import numpy as np
import onnxruntime as ort

from onnx_detector import preprocess


def summarize(name, output, threshold):
    predictions = output[0].transpose(1, 0)

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    maximum_class_scores = np.max(
        class_scores,
        axis=1,
    )

    print(f"\n{name}")
    print("-" * len(name))

    print("Output shape:", output.shape)
    print("Output dtype:", output.dtype)

    print(
        "All finite:",
        bool(np.all(np.isfinite(output))),
    )

    print(
        "Output minimum:",
        float(np.min(output)),
    )

    print(
        "Output maximum:",
        float(np.max(output)),
    )

    print(
        "Box minimum:",
        float(np.min(boxes)),
    )

    print(
        "Box maximum:",
        float(np.max(boxes)),
    )

    print(
        "Maximum class score:",
        float(np.max(maximum_class_scores)),
    )

    print(
        "Mean maximum class score:",
        float(np.mean(maximum_class_scores)),
    )

    print(
        "p95 maximum class score:",
        float(
            np.percentile(
                maximum_class_scores,
                95,
            )
        ),
    )

    print(
        "p99 maximum class score:",
        float(
            np.percentile(
                maximum_class_scores,
                99,
            )
        ),
    )

    print(
        f"Candidates >= {threshold}:",
        int(
            np.sum(
                maximum_class_scores >= threshold
            )
        ),
    )

    for test_threshold in [
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
    ]:
        count = int(
            np.sum(
                maximum_class_scores
                >= test_threshold
            )
        )

        print(
            f"Candidates >= "
            f"{test_threshold:.2f}: {count}"
        )


def run_model(model_path, tensor):
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    return session.run(
        None,
        {input_name: tensor},
    )[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp32",
        required=True,
    )

    parser.add_argument(
        "--int8",
        required=True,
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
    )

    args = parser.parse_args()

    capture = cv2.VideoCapture(args.source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open: {args.source}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        args.frame,
    )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            f"Unable to read frame {args.frame}"
        )

    tensor, _ = preprocess(
        frame,
        args.imgsz,
    )

    fp32_output = run_model(
        args.fp32,
        tensor,
    )

    int8_output = run_model(
        args.int8,
        tensor,
    )

    summarize(
        "FP32 output",
        fp32_output,
        args.conf,
    )

    summarize(
        "INT8 output",
        int8_output,
        args.conf,
    )

    difference = np.abs(
        fp32_output - int8_output
    )

    print("\nFP32 versus INT8")
    print("-----------------")
    print(
        "Mean absolute difference:",
        float(np.mean(difference)),
    )
    print(
        "Maximum absolute difference:",
        float(np.max(difference)),
    )


if __name__ == "__main__":
    main()
