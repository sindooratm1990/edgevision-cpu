import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import onnx

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import (
    quant_pre_process,
)

from onnx_detector import preprocess


class VideoCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        model_path,
        video_path,
        image_size,
        number_of_samples,
    ):
        self.model_path = str(model_path)
        self.video_path = str(video_path)
        self.image_size = image_size
        self.number_of_samples = number_of_samples

        session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = session.get_inputs()[0].name

        self.capture = None
        self.frame_indices = []
        self.position = 0

        self._initialize_video()

    def _initialize_video(self):
        if self.capture is not None:
            self.capture.release()

        self.capture = cv2.VideoCapture(
            self.video_path
        )

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Unable to open calibration video: "
                f"{self.video_path}"
            )

        frame_count = int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if frame_count <= 0:
            raise RuntimeError(
                "Unable to determine calibration "
                "video frame count"
            )

        sample_count = min(
            self.number_of_samples,
            frame_count,
        )

        # Select frames across the entire video instead
        # of taking only consecutive frames at the start.
        self.frame_indices = np.linspace(
            0,
            frame_count - 1,
            sample_count,
            dtype=int,
        ).tolist()

        self.position = 0

    def get_next(self):
        if self.position >= len(self.frame_indices):
            self.capture.release()
            return None

        frame_index = self.frame_indices[
            self.position
        ]

        self.position += 1

        self.capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = self.capture.read()

        if not success:
            raise RuntimeError(
                f"Unable to read calibration frame "
                f"{frame_index}"
            )

        tensor, _ = preprocess(
            frame,
            self.image_size,
        )

        return {
            self.input_name: tensor
        }

    def rewind(self):
        self._initialize_video()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Apply static INT8 quantization "
            "to a YOLO ONNX model"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input FP32 ONNX model",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output INT8 ONNX model",
    )

    parser.add_argument(
        "--calibration-video",
        required=True,
        help="Representative calibration video",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help="Static model input size",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of calibration frames",
    )
    parser.add_argument(
        "--quant-types",
        choices=["s8s8", "u8u8"],
        default="s8s8",
        help="Activation and weight quantization types",
    )
    parser.add_argument(
	"--exclude-detect-head",
	action="store_true",
	help=(
	    "Keep the YOLO model.23 detection "
	    "head in floating point"
        ),
    )

    return parser.parse_args()

def find_detection_head_nodes(model_path):
    model = onnx.load(str(model_path))

    node_names = [
        node.name
        for node in model.graph.node
        if node.name
        and "model.23" in node.name
    ]

    return node_names
def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)
    calibration_path = Path(
        args.calibration_video
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Model not found: {input_path}"
        )

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration video not found: "
            f"{calibration_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessed_path = output_path.with_name(
        f"{output_path.stem}_preprocessed.onnx"
    )

    print("Preprocessing ONNX graph...")
    print(f"Input: {input_path}")
    print(f"Preprocessed: {preprocessed_path}")

    quant_pre_process(
        input_model_path=str(input_path),
        output_model_path=str(
            preprocessed_path
        ),
    )

    print(
        f"Preparing {args.samples} "
        f"calibration frames..."
    )

    calibration_reader = VideoCalibrationReader(
        model_path=preprocessed_path,
        video_path=calibration_path,
        image_size=args.imgsz,
        number_of_samples=args.samples,
    )

    print("Running static INT8 quantization...")

    if args.quant_types == "s8s8":
        activation_type = QuantType.QInt8
        weight_type = QuantType.QInt8
    else:
        activation_type = QuantType.QUInt8
        weight_type = QuantType.QUInt8
    nodes_to_exclude = []

    if args.exclude_detect_head:
        nodes_to_exclude = (
            find_detection_head_nodes(
                preprocessed_path
            ) 
        )

        if not nodes_to_exclude:
            raise RuntimeError(
                "No model.23 detection-head nodes "
                "were found. Inspect the ONNX node "
                "names before continuing."
            ) 

        print(
            f"Keeping {len(nodes_to_exclude)} "
            f"detection-head nodes in FP32"
        )

        for node_name in nodes_to_exclude:
            print(f"  Excluding: {node_name}")
    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=(
            calibration_reader
        ),
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
	weight_type=weight_type,
        per_channel=True,
        calibrate_method=(
            CalibrationMethod.MinMax
        ),
        nodes_to_exclude=nodes_to_exclude,
    )

    print("\nQuantization complete")
    print(f"FP32 model: {input_path}")
    print(f"INT8 model: {output_path}")

    fp32_size = (
        input_path.stat().st_size
        / (1024 * 1024)
    )

    int8_size = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    reduction = (
        (fp32_size - int8_size)
        / fp32_size
        * 100.0
    )

    print(f"FP32 size: {fp32_size:.2f} MB")
    print(f"INT8 size: {int8_size:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
