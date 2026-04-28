# Video Segmentation Workspace

This workspace contains a complete pipeline to:

1. Run YOLO segmentation on video.
2. Black out segmented dynamic objects (cars, people, buses, trucks, etc.).
3. Display processed output in a Tkinter window.
4. Convert models between PT, ONNX, and TensorRT engine formats.
5. Benchmark model formats.

## Workspace Contents

- `scripts/`: Python entry points for viewers and benchmarks.
- `models/`: PyTorch, ONNX, and TensorRT model files.
- `media/`: Sample video input.
- `reports/`: Generated benchmark outputs and plots.
- `datasets/`: Cityscapes-style image and label data.

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
/home/intellisense05/akinduid/mi/.venv/bin/python /home/intellisense05/akinduid/mi/scripts/car_segmentation_viewer.py
```

Custom run with explicit arguments:

```bash
/home/intellisense05/akinduid/mi/.venv/bin/python /home/intellisense05/akinduid/mi/scripts/car_segmentation_viewer.py \
	--model /home/intellisense05/akinduid/mi/models/yolo26n-seg.engine \
	--video /home/intellisense05/akinduid/mi/media/car.mp4 \
	--scale 0.5
```

Show CLI help:

```bash
/home/intellisense05/akinduid/mi/.venv/bin/python /home/intellisense05/akinduid/mi/scripts/car_segmentation_viewer.py --help
```

## Convert PT -> ONNX

```bash
cd /home/intellisense05/akinduid/mi
/home/intellisense05/akinduid/mi/.venv/bin/python - <<'PY'
from ultralytics import YOLO
model = YOLO('models/yolo26n-seg.pt')
out = model.export(format='onnx')
print(out)
PY
```

## Convert ONNX -> TensorRT Engine

```bash
cd /home/intellisense05/akinduid/mi
trtexec --onnx=models/yolo26n-seg.onnx --saveEngine=models/yolo26n-seg.engine --fp16
```

## Benchmark Models

Run Python benchmark script:

```bash
cd /home/intellisense05/akinduid/mi
/home/intellisense05/akinduid/mi/.venv/bin/python scripts/benchmark_models.py --warmup 10 --runs 40 --imgsz 640
```

Run TensorRT engine-only benchmark:

```bash
cd /home/intellisense05/akinduid/mi
trtexec --loadEngine=models/yolo26n-seg.engine --duration=3 --warmUp=200
```

## Benchmark on Dataset (PT/ONNX/Engine)

Use `benchmark_cityscapes.py` when you want to compare mixed model formats on Cityscapes-style data under `datasets/`.

Example run:

```bash
cd /home/intellisense05/akinduid/mi
/home/intellisense05/akinduid/mi/.venv/bin/python scripts/benchmark_cityscapes.py \
	--dataset-root datasets \
	--split val \
	--max-images 50 \
	--output-dir reports/benchmark_report_cityscapes_50
```

Default model list in this script includes:

- `models/yolo26n-seg.pt`
- `models/yolo26n-seg.onnx`
- `models/yolo26n-seg-fp32.engine`
- `models/yolo26n-seg-fp16.engine`

## Benchmark on Dataset (Engine-Only, Native TensorRT Decode)

Use `benchmark_engines.py` for reliable engine benchmarking and metrics. This script decodes TensorRT outputs natively (not via Ultralytics engine parsing) and prints per-image progress so you can verify the run is active.

Example run:

```bash
cd /home/intellisense05/akinduid/mi
/home/intellisense05/akinduid/mi/.venv/bin/python scripts/benchmark_engines.py \
	--dataset-root datasets \
	--split val \
	--max-images 50 \
	--output-dir reports/benchmark_report_engines_50
```

You will see logs like:

- `[model] image X/N: <file_name>` during inference collection.
- `evaluating metrics for class=<name> ...` during AP/IoU computation.

## Benchmark Output Files

Both dataset benchmark scripts write the same output structure:

- `benchmark_report.md`: Main markdown report with summary table and plot embeds.
- `benchmark_results.csv`: Flat metrics table for spreadsheet analysis.
- `benchmark_results.json`: Full structured metrics payload.
- `plots/`: Generated figures.

Important plots include:

- `plots/inference_time.png`
- `plots/map_scores.png`
- `plots/iou_scores.png`
- `plots/class_ap50.png`
- `plots/confusion_matrix.png` (engine benchmark)

## How to Read the Results

- `Prec@0.5` / `Rec@0.5` in the summary are end-of-curve values after ranking by confidence.
- `mAP@0.5` and `mAP@0.5:0.95` summarize the full precision-recall behavior and are usually more stable than a single operating point.
- `Best F1@0.5` and `Best conf@0.5` show the confidence operating point with strongest F1 and are useful when precision/recall look imbalanced.
- Confusion matrix heatmap is row-normalized (`True` class on Y, `Predicted` on X), so brighter off-diagonal cells indicate class confusions.

## Copy-and-Run Quick Start

Use this block exactly:

```bash
cd /home/intellisense05/akinduid/mi
source .venv/bin/activate

# Run viewer
python scripts/car_segmentation_viewer.py --model models/yolo26n-seg.engine --video media/car.mp4 --scale 0.5

# Optional benchmark
python scripts/benchmark_models.py --warmup 10 --runs 40 --imgsz 640

# Dataset mixed-format benchmark (PT/ONNX)
python scripts/benchmark_cityscapes.py --max-images 50 --output-dir reports/benchmark_report_cityscapes_50

# Dataset engine benchmark (native TensorRT decode)
python scripts/benchmark_engines.py --max-images 50 --output-dir reports/benchmark_report_engines_50
```

## Notes

- Press `q` in the viewer window (or close the window) to exit.
- If TensorRT Python binding is missing, install with `pip install tensorrt cuda-python`.
- If ONNX GPU provider is unstable on your machine, ONNX CPU benchmarking may still work reliably.
