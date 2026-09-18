# EdgeVision-CPU

CPU-only object detection and performance optimization for resource-constrained edge devices.

## Overview

EdgeVision-CPU explores how a video analytics pipeline can be deployed without a GPU. The project compares input resolution, inference runtime, latency, memory consumption, CPU utilization and detection behavior using YOLO11n.

Development and benchmarking were performed on an Apple M4 system using CPU-only execution. The longer-term goal is to develop a containerized video analytics service suitable for ARM-based edge gateways and router-class devices.

## Objectives

* Run object detection without GPU acceleration
* Compare 320 and 640 input resolutions
* Measure mean, median, p95 and p99 latency
* Measure throughput, CPU utilization and peak RSS memory
* Compare PyTorch/Ultralytics with pure ONNX Runtime
* Study the trade-off between performance and detection detail
* Progress toward a deployment footprint suitable for constrained devices

## Hardware and Environment

* Processor: Apple M4
* Architecture: ARM64
* Memory: 24 GB
* Python: 3.11.16
* Ultralytics: 8.4.154
* PyTorch: 2.14.0
* Execution mode: CPU only
* ONNX provider: `CPUExecutionProvider`

GPU, Apple MPS and CoreML acceleration were not used for the reported benchmarks.

## Pipeline

```text
Video
  -> Frame decoding
  -> Letterbox resize
  -> BGR-to-RGB conversion
  -> Normalization
  -> CPU model inference
  -> Detection postprocessing
  -> Annotated output or benchmark metrics
```

## Phase 1: PyTorch Baseline

The pretrained YOLO11n PyTorch model was run against the same fixed video at two requested input sizes.

Annotated videos were generated under:

```text
results/baseline_320/
results/baseline_640/
```

The saved outputs were used to visually inspect missed objects, false detections and bounding-box stability.

### PyTorch benchmark results

| Metric                    | 320 request | 640 request |
| ------------------------- | ----------: | ----------: |
| Actual inference tensor   |   192 × 320 |   384 × 640 |
| Mean pipeline latency     |     8.00 ms |    18.33 ms |
| p95 pipeline latency      |     8.44 ms |    19.93 ms |
| Mean inference latency    |     6.86 ms |    16.84 ms |
| Throughput                |  124.70 FPS |   54.48 FPS |
| Peak RSS                  |   476.09 MB |   518.00 MB |
| Mean detections per frame |        3.61 |        8.18 |

Reducing the requested input size from 640 to 320 decreased pipeline latency by approximately 56% and increased throughput by approximately 2.29 times. However, the smaller input produced substantially fewer detections.

Detection count is not equivalent to accuracy. Labeled ground-truth data is required to determine whether additional detections are correct or false positives.

## Phase 2: ONNX Export

The PyTorch model was exported into two fixed-shape ONNX models:

```text
yolo11n_320.onnx -> input [1, 3, 320, 320]
yolo11n_640.onnx -> input [1, 3, 640, 640]
```

The output tensors were:

```text
320 model -> [1, 84, 2100]
640 model -> [1, 84, 8400]
```

The 84 output channels contain four bounding-box values and 80 COCO class scores. The 640 model produces four times as many candidate locations as the 320 model.

Both ONNX files are approximately 10 MB because input resolution changes the inference workload rather than the learned parameter count.

## Phase 3: Pure ONNX Runtime

A standalone benchmark was implemented using OpenCV, NumPy and ONNX Runtime. PyTorch and Ultralytics were not loaded into the inference process.

The runtime was explicitly restricted to:

```python
providers=["CPUExecutionProvider"]
```

### ONNX Runtime results

| Metric                 |   ONNX 320 |  ONNX 640 |
| ---------------------- | ---------: | --------: |
| Input tensor           |  320 × 320 | 640 × 640 |
| Mean pipeline latency  |    7.78 ms |  18.51 ms |
| p95 pipeline latency   |    8.33 ms |  20.17 ms |
| Mean inference latency |    5.84 ms |  16.03 ms |
| Throughput             | 127.94 FPS | 53.95 FPS |
| Peak RSS               |  247.98 MB | 338.58 MB |
| CPU core-equivalents   |       5.93 |      5.37 |

The standalone ONNX pipeline reduced peak process memory from approximately 476–518 MB to 248–339 MB.

The ONNX benchmark currently includes:

* Video decoding
* Letterbox resizing
* Color conversion
* Normalization
* ONNX inference

It does not yet include:

* Confidence filtering
* Bounding-box conversion
* Non-maximum suppression
* Tracking
* Drawing or video encoding

## Important Comparison Limitation

The original PyTorch baseline used rectangular inference tensors, while the exported ONNX models use fixed square tensors. Additionally, PyTorch included detection postprocessing while the current ONNX benchmark stops at raw model output.

The existing measurements demonstrate runtime behavior and memory improvement, but should not be presented as a perfectly controlled framework-to-framework speed comparison.

A future benchmark will run both runtimes with identical square inputs and equivalent postprocessing.

## Findings

1. Both tested resolutions satisfied a theoretical 30 FPS processing deadline on the development machine.
2. A 320 input provided significantly lower latency and memory consumption.
3. A 640 input retained more detection candidates and appeared to detect more small objects.
4. Removing PyTorch and Ultralytics from the runtime substantially reduced memory.
5. ONNX Runtime used more than five CPU core-equivalents, which may be unsuitable for a router sharing CPU resources with networking services.
6. Resolution reduction alone is insufficient to reach a 100 MB process-memory target.

## Next Steps

* Implement YOLO output decoding and non-maximum suppression
* Compare equivalent square-input PyTorch and ONNX pipelines
* Benchmark ONNX Runtime with 1, 2 and 4 threads
* Quantize the ONNX model to INT8
* Add RTSP input and reconnection handling
* Introduce frame dropping and bounded queues
* Publish detection events using MQTT
* Persist unsent events in SQLite
* Containerize the application for ARM64
* Build a lightweight C++ inference version
* Test on Raspberry Pi or router-class ARM hardware

## Disclaimer

These results were measured on an Apple M4 development system and do not represent Raspberry Pi, Cradlepoint or production router performance. Target-device benchmarking is required before making deployment claims.

## INT8 Quantization Experiment

Static post-training quantization was evaluated using ONNX Runtime on an
Apple M4 CPU. The YOLO detection head was retained in FP32 because fully
quantizing the model caused all class-confidence outputs to become zero.

Both mixed-precision models were calibrated using 6,000 sequential frames.
FP32, U8U8 and S8S8 models were benchmarked over the same 3,000-frame video
segment at 640 × 640 with two ONNX Runtime CPU threads.

| Metric | FP32 | U8U8 | S8S8 |
|---|---:|---:|---:|
| Mean inference latency | 16.69 ms | 18.57 ms | 17.46 ms |
| Mean pipeline latency | 18.66 ms | 20.56 ms | 19.41 ms |
| Throughput | 53.53 FPS | 48.58 FPS | 51.46 FPS |
| Peak RSS | 349.70 MB | 283.53 MB | 276.38 MB |
| Total detections | 27,972 | 27,911 | 26,532 |
| Person detections | 45 | 34 | 25 |
| Truck detections | 2,333 | 1,516 | 1,590 |
| Motorcycle detections | 15 | 7 | 3 |

U8U8 reduced peak memory by 18.9%, while S8S8 reduced it by 21.0%.
Neither INT8 variant improved inference speed on Apple M4. Both models also
introduced class-level detection drift, particularly for people, trucks and
motorcycles.

The aggregate detection count was insufficient for evaluating model
equivalence because increases in car and bus detections concealed losses in
other classes. These counts measure agreement with FP32 predictions rather
than true precision or recall because labeled ground-truth annotations were
not available.

FP32 ONNX Runtime with two CPU threads was therefore retained as the selected
configuration.
