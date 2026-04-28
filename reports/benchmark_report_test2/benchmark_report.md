# Cityscapes Segmentation Benchmark

- Dataset root: `/home/intellisense05/akinduid/mi/datasets`
- Split: `val`
- Image pairs evaluated: `1`
- Max images: `1`

## Summary

| Model                   | Mean ms | Median ms | P95 ms | FPS   | Mean IoU | Prec@0.5 | Rec@0.5 | F1@0.5 | mAP@0.5 | mAP@0.5:0.95 | Eval mode      |
| ----------------------- | ------- | --------- | ------ | ----- | -------- | -------- | ------- | ------ | ------- | ------------ | -------------- |
| yolo26n-seg.pt          | 787.14  | 787.14    | 787.14 | 1.27  | 0.6829   | 0.0254   | 0.7143  | 0.0490 | 0.6508  | 0.2792       | class-agnostic |
| yolo26n-seg.onnx        | 317.77  | 317.77    | 317.77 | 3.15  | 0.6832   | 0.0256   | 0.7143  | 0.0495 | 0.6429  | 0.2784       | class-agnostic |
| yolo26n-seg-fp32.engine | 176.78  | 176.78    | 176.78 | 5.66  | 0.0000   | 0.0000   | 0.0000  | 0.0000 | 0.0000  | 0.0000       | class-agnostic |
| yolo26n-seg-fp16.engine | 54.94   | 54.94     | 54.94  | 18.20 | 0.0000   | 0.0000   | 0.0000  | 0.0000 | 0.0000  | 0.0000       | class-agnostic |

Engine models may use class-agnostic fallback when class/conf fields are incompatible.

## Plots

![Inference Time](plots/inference_time.png)

![mAP Scores](plots/map_scores.png)

![Mean IoU](plots/iou_scores.png)

![Per-Class AP50](plots/class_ap50.png)

## Per-Class AP50

| Model                   | all-objects |
| ----------------------- | ----------- |
| yolo26n-seg.pt          | 0.6508      |
| yolo26n-seg.onnx        | 0.6429      |
| yolo26n-seg-fp32.engine | 0.0000      |
| yolo26n-seg-fp16.engine | 0.0000      |

## Outputs

- JSON: [`benchmark_results.json`](benchmark_results.json)
- CSV: [`benchmark_results.csv`](benchmark_results.csv)
- Plots directory: [`plots/`](plots)