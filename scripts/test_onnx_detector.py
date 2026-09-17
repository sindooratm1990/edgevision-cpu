import cv2

from coco_names import COCO_NAMES
from onnx_detector import ONNXDetector


MODEL_PATH = "models/yolo11n_320.onnx"
VIDEO_PATH = "data/test_video.mp4"
OUTPUT_PATH = "results/onnx_320_annotated.mp4"


detector = ONNXDetector(
    model_path=MODEL_PATH,
    image_size=320,
    class_names=COCO_NAMES,
    confidence_threshold=0.4,
    iou_threshold=0.45,
)

capture = cv2.VideoCapture(VIDEO_PATH)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

while True:
    success, frame = capture.read()

    if not success:
        break

    detections = detector.infer(frame)

    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (detection.x1, max(20, detection.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    writer.write(frame)

capture.release()
writer.release()

print(f"Saved: {OUTPUT_PATH}")
