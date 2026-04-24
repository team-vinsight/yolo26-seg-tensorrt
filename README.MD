# Video Segmentation Workspace

This workspace contains a complete pipeline to:

1. Run YOLO segmentation on video.
2. Black out segmented dynamic objects (cars, people, buses, trucks, etc.).
3. Display processed output in a Tkinter window.
4. Convert models between PT, ONNX, and TensorRT engine formats.
5. Benchmark model formats.

## Workspace Contents

- `car.mp4`: Sample video input.
- `webcam_viewer.py`: Basic OpenCV webcam viewer.
- `car_segmentation_viewer.py`: Main TensorRT-backed Tkinter segmentation viewer.
- `benchmark_models.py`: Benchmark script for PT, ONNX, and TensorRT paths.
- `yolo26n-seg.pt`: Source PyTorch model.
- `yolo26n-seg.onnx`: Exported ONNX model.
- `yolo26n-seg.engine`: Exported TensorRT engine.
- Additional model variants (`yolo26s/m/l/x-seg.pt`, `yolov8s-seg.pt`).

## Environment Setup

Create or activate your virtual environment, then install required packages.

```bash
cd /home/intellisense05/akinduid/mi

# Example venv activation
source .venv/bin/activate

# Core dependencies
pip install opencv-python pillow ultralytics onnx onnxruntime

# TensorRT Python runtime (required for engine execution from Python)
pip install tensorrt cuda-python
```

## Run the Main Program (TensorRT Engine)

Default run:

```bash
/home/intellisense05/akinduid/mi/.venv/bin/python /home/intellisense05/akinduid/mi/car_segmentation_viewer.py
```

Custom run with explicit arguments:

```bash
/home/intellisense05/akinduid/mi/.venv/bin/python /home/intellisense05/akinduid/mi/car_segmentation_viewer.py \
	--model /home/intellisense05/akinduid/mi/yolo26n-seg.engine \
	--video /home/intellisense05/akinduid/mi/car.mp4 \
	--scale 0.5
```

Show CLI help:

```bash
/home/intellisense05/akinduid/mi/.venv/bin/python /home/intellisense05/akinduid/mi/car_segmentation_viewer.py --help
```

## Convert PT -> ONNX

```bash
cd /home/intellisense05/akinduid/mi
/home/intellisense05/akinduid/mi/.venv/bin/python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolo26n-seg.pt')
out = model.export(format='onnx')
print(out)
PY
```

## Convert ONNX -> TensorRT Engine

```bash
cd /home/intellisense05/akinduid/mi
trtexec --onnx=yolo26n-seg.onnx --saveEngine=yolo26n-seg.engine --fp16
```

## Benchmark Models

Run Python benchmark script:

```bash
cd /home/intellisense05/akinduid/mi
/home/intellisense05/akinduid/mi/.venv/bin/python benchmark_models.py --warmup 10 --runs 40 --imgsz 640
```

Run TensorRT engine-only benchmark:

```bash
cd /home/intellisense05/akinduid/mi
trtexec --loadEngine=yolo26n-seg.engine --duration=3 --warmUp=200
```

## Copy-and-Run Quick Start

Use this block exactly:

```bash
cd /home/intellisense05/akinduid/mi
source .venv/bin/activate

# Run viewer
python car_segmentation_viewer.py --model yolo26n-seg.engine --video car.mp4 --scale 0.5

# Optional benchmark
python benchmark_models.py --warmup 10 --runs 40 --imgsz 640
```

## Notes

- Press `q` in the viewer window (or close the window) to exit.
- If TensorRT Python binding is missing, install with `pip install tensorrt cuda-python`.
- If ONNX GPU provider is unstable on your machine, ONNX CPU benchmarking may still work reliably.
