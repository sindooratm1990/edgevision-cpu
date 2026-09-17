from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class LetterboxInfo:
    scale: float
    pad_left: int
    pad_top: int


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def letterbox(image: np.ndarray, size: int):
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

    pad_left = horizontal_padding // 2
    pad_right = horizontal_padding - pad_left
    pad_top = vertical_padding // 2
    pad_bottom = vertical_padding - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    info = LetterboxInfo(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )

    return padded, info


def preprocess(image: np.ndarray, size: int):
    padded, letterbox_info = letterbox(image, size)

    tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = tensor.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    tensor /= 255.0
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, letterbox_info


class ONNXDetector:
    def __init__(
        self,
        model_path: str,
        image_size: int,
        class_names: dict[int, str],
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        threads: int | None = None,
    ):
        self.image_size = image_size
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if threads is not None:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame: np.ndarray):
        tensor, letterbox_info = preprocess(
            frame,
            self.image_size,
        )

        outputs = self.session.run(
            None,
            {self.input_name: tensor},
        )

        detections = self.decode(
            outputs[0],
            frame.shape,
            letterbox_info,
        )

        return detections

    def decode(
        self,
        output: np.ndarray,
        original_shape,
        letterbox_info: LetterboxInfo,
    ):
        # [1, 84, candidates] -> [candidates, 84]
        predictions = output[0].transpose(1, 0)

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        keep = confidences >= self.confidence_threshold

        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        if len(boxes_xywh) == 0:
            return []

        # YOLO returns center x, center y, width and height.
        center_x = boxes_xywh[:, 0]
        center_y = boxes_xywh[:, 1]
        widths = boxes_xywh[:, 2]
        heights = boxes_xywh[:, 3]

        x = center_x - widths / 2
        y = center_y - heights / 2

        nms_boxes = np.column_stack(
            [x, y, widths, heights]
        ).tolist()

        selected_indices = []

        # Perform NMS separately for each class.
        for class_id in np.unique(class_ids):
            class_positions = np.where(class_ids == class_id)[0]

            class_boxes = [
                nms_boxes[position]
                for position in class_positions
            ]

            class_confidences = [
                float(confidences[position])
                for position in class_positions
            ]

            retained = cv2.dnn.NMSBoxes(
                class_boxes,
                class_confidences,
                self.confidence_threshold,
                self.iou_threshold,
            )

            if len(retained) == 0:
                continue

            retained = np.array(retained).reshape(-1)

            for retained_index in retained:
                selected_indices.append(
                    class_positions[int(retained_index)]
                )

        original_height, original_width = original_shape[:2]
        detections = []

        for index in selected_indices:
            box_x, box_y, box_width, box_height = nms_boxes[index]

            # Remove padding and map to original frame.
            x1 = (
                box_x - letterbox_info.pad_left
            ) / letterbox_info.scale

            y1 = (
                box_y - letterbox_info.pad_top
            ) / letterbox_info.scale

            x2 = (
                box_x
                + box_width
                - letterbox_info.pad_left
            ) / letterbox_info.scale

            y2 = (
                box_y
                + box_height
                - letterbox_info.pad_top
            ) / letterbox_info.scale

            x1 = int(np.clip(x1, 0, original_width - 1))
            y1 = int(np.clip(y1, 0, original_height - 1))
            x2 = int(np.clip(x2, 0, original_width - 1))
            y2 = int(np.clip(y2, 0, original_height - 1))

            class_id = int(class_ids[index])

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(
                        class_id,
                        str(class_id),
                    ),
                    confidence=float(confidences[index]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections
