from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_DEFAULTS = [
    "yolo26n-seg.pt",
    "yolo26n-seg.onnx",
    "yolo26s-seg.pt",
    "yolo26s-seg.onnx",
    "yolo26m-seg.pt",
    "yolo26m-seg.onnx"
    
]

COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
}

CITYSCAPES_INSTANCE_TO_COCO = {
    24: 0,  # person
    26: 2,  # car
    27: 7,  # truck
    28: 5,  # bus
    31: 6,  # train
    32: 3,  # motorcycle
    33: 1,  # bicycle
}

DEFAULT_TARGET_CLASS_IDS = list(COCO_CLASS_NAMES.keys())
CLASS_AGNOSTIC_ID = -1


@dataclass(frozen=True)
class InstanceRecord:
    image_index: int
    class_id: int
    mask: np.ndarray


@dataclass(frozen=True)
class PredictionRecord:
    image_index: int
    class_id: int
    score: float
    mask: np.ndarray


@dataclass
class ModelSummary:
    model_name: str
    mean_inference_ms: float
    median_inference_ms: float
    p95_inference_ms: float
    throughput_fps: float
    mean_iou: float
    precision_50: float
    recall_50: float
    f1_50: float
    map_50: float
    map_5095: float
    per_class_ap50: dict[int, float]
    per_class_iou: dict[int, float]
    evaluation_mode: str
    best_f1_50: float = 0.0
    best_conf_50: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO segmentation models on a Cityscapes-style dataset.")
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Root folder containing images/leftImg8bit and gt/gtFine",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_DEFAULTS,
        help="Model paths to benchmark",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold used during prediction")
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="NMS IoU threshold during prediction")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images to evaluate")
    parser.add_argument("--onnx-device", default="cpu", choices=["cpu", "gpu"], help="Device to use for ONNX models")
    parser.add_argument(
        "--output-dir",
        default="benchmark_report",
        help="Directory where the markdown report, CSV, JSON, and plots are written",
    )
    return parser.parse_args()


def discover_pairs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_root = dataset_root / "images" / "leftImg8bit" / split
    label_root = dataset_root / "gt" / "gtFine" / split

    if not image_root.exists():
        raise FileNotFoundError(f"Image split not found: {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Label split not found: {label_root}")

    image_paths = sorted(image_root.glob("**/*_leftImg8bit.png"))
    pairs: list[tuple[Path, Path]] = []

    for image_path in image_paths:
        relative = image_path.relative_to(image_root)
        label_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
        label_path = label_root / relative.parent / label_name
        if label_path.exists():
            pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError(f"No matching image/label pairs found under {image_root}")

    return pairs


def build_mask_from_polygons(polygons: Iterable[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        if polygon is None:
            continue
        points = np.asarray(polygon, dtype=np.int32)
        if points.size == 0:
            continue
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def extract_ground_truth_instances(instance_ids: np.ndarray, image_index: int) -> list[InstanceRecord]:
    records: list[InstanceRecord] = []
    for instance_id in np.unique(instance_ids):
        if instance_id < 1000:
            continue
        class_id = int(instance_id // 1000)
        coco_class_id = CITYSCAPES_INSTANCE_TO_COCO.get(class_id)
        if coco_class_id is None:
            continue
        mask = instance_ids == instance_id
        if mask.any():
            records.append(InstanceRecord(image_index=image_index, class_id=coco_class_id, mask=mask))
    return records


def extract_predictions(result, image_index: int, target_class_ids: set[int]) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    if result.boxes is None or result.masks is None:
        return records

    boxes = result.boxes
    masks_xy = result.masks.xy

    for det_index, polygon_list in enumerate(masks_xy):
        class_id = int(boxes.cls[det_index])
        if class_id not in target_class_ids:
            continue

        score = float(boxes.conf[det_index])
        mask = build_mask_from_polygons([polygon_list], result.orig_shape[:2])
        records.append(PredictionRecord(image_index=image_index, class_id=class_id, score=score, mask=mask))

    return records


def has_valid_class_conf(result) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return True

    sample_count = min(len(result.boxes), 10)
    classes = result.boxes.cls[:sample_count].tolist()
    confidences = result.boxes.conf[:sample_count].tolist()
    for class_id, confidence in zip(classes, confidences):
        if class_id < 0 or class_id > 1000:
            return False
        if confidence < 0.0 or confidence > 1.0:
            return False
    return True


def extract_predictions_class_agnostic(result, image_index: int) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    if result.masks is None:
        return records

    for det_index, polygon_list in enumerate(result.masks.xy):
        score = 1.0 - (det_index * 1e-6)
        mask = build_mask_from_polygons([polygon_list], result.orig_shape[:2])
        records.append(PredictionRecord(image_index=image_index, class_id=CLASS_AGNOSTIC_ID, score=score, mask=mask))

    return records


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    area = np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1])
    return float(area)


def evaluate_class(
    predictions: list[PredictionRecord],
    ground_truths: list[InstanceRecord],
    iou_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray, float, float, float, float]:
    total_gt = len(ground_truths)
    if total_gt == 0:
        return 0.0, np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

    gt_by_image: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for gt_index, gt in enumerate(ground_truths):
        gt_by_image[gt.image_index].append((gt_index, gt.mask))

    matched_gts: set[int] = set()
    sorted_predictions = sorted(predictions, key=lambda item: item.score, reverse=True)
    true_positives: list[int] = []
    false_positives: list[int] = []
    matched_ious: list[float] = []

    for prediction in sorted_predictions:
        candidates = gt_by_image.get(prediction.image_index, [])
        best_gt_index = None
        best_iou = 0.0

        for gt_index, gt_mask in candidates:
            if gt_index in matched_gts:
                continue
            iou = mask_iou(prediction.mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gts.add(best_gt_index)
            true_positives.append(1)
            false_positives.append(0)
            matched_ious.append(best_iou)
        else:
            true_positives.append(0)
            false_positives.append(1)

    tp = np.asarray(true_positives, dtype=np.float64)
    fp = np.asarray(false_positives, dtype=np.float64)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / max(total_gt, np.finfo(np.float64).eps)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    ap = compute_ap(recalls, precisions)

    precision = float(precisions[-1]) if precisions.size else 0.0
    recall = float(recalls[-1]) if recalls.size else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return ap, recalls, precisions, precision, recall, f1, mean_iou


def evaluate_model(
    model_path: Path,
    pairs: list[tuple[Path, Path]],
    imgsz: int,
    conf: float,
    iou_threshold: float,
    onnx_device: str,
    target_class_ids: set[int],
) -> tuple[ModelSummary, dict[int, float]]:
    model = YOLO(str(model_path))

    if model_path.suffix.lower() == ".onnx":
        device = "cpu" if onnx_device == "cpu" else 0
    else:
        device = 0

    inference_times: list[float] = []
    gt_records: list[InstanceRecord] = []
    pred_records: list[PredictionRecord] = []
    class_agnostic_mode = False

    for image_index, (image_path, label_path) in enumerate(pairs):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        instance_ids = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if instance_ids is None:
            raise RuntimeError(f"Could not read label: {label_path}")

        gt_records.extend(extract_ground_truth_instances(instance_ids, image_index))

        start = time.perf_counter()
        result = model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            iou=iou_threshold,
            verbose=False,
            device=device,
        )[0]
        inference_times.append((time.perf_counter() - start) * 1000.0)

        if not class_agnostic_mode and not has_valid_class_conf(result):
            class_agnostic_mode = True

        if class_agnostic_mode:
            pred_records.extend(extract_predictions_class_agnostic(result, image_index))
        else:
            pred_records.extend(extract_predictions(result, image_index, target_class_ids))

    per_class_ap50: dict[int, float] = {}
    per_class_iou: dict[int, float] = {}
    per_class_ap_curve: dict[int, list[float]] = {}

    iou_thresholds = [round(value, 2) for value in np.arange(0.5, 1.0, 0.05)]
    ap_per_threshold: list[float] = []

    precision_50 = recall_50 = f1_50 = mean_iou_50 = 0.0
    precision_values_50: list[float] = []
    recall_values_50: list[float] = []
    f1_values_50: list[float] = []
    iou_values_50: list[float] = []

    if class_agnostic_mode:
        class_gts = [InstanceRecord(image_index=gt.image_index, class_id=CLASS_AGNOSTIC_ID, mask=gt.mask) for gt in gt_records]
        class_preds = pred_records
        class_ap_values = []
        for threshold in iou_thresholds:
            ap, _, _, precision, recall, f1, mean_iou = evaluate_class(class_preds, class_gts, threshold)
            class_ap_values.append(ap)
            if math.isclose(threshold, 0.5, abs_tol=1e-9):
                precision_50 = precision
                recall_50 = recall
                f1_50 = f1
                mean_iou_50 = mean_iou
                precision_values_50.append(precision)
                recall_values_50.append(recall)
                f1_values_50.append(f1)
                iou_values_50.append(mean_iou)

        per_class_ap_curve[CLASS_AGNOSTIC_ID] = class_ap_values
        per_class_ap50[CLASS_AGNOSTIC_ID] = class_ap_values[0] if class_ap_values else 0.0
        per_class_iou[CLASS_AGNOSTIC_ID] = mean_iou_50
        ap_per_threshold.append(float(np.mean(class_ap_values)))
    else:
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)
        for gt in gt_records:
            gt_by_class[gt.class_id].append(gt)
        for pred in pred_records:
            pred_by_class[pred.class_id].append(pred)

        for class_id in sorted(target_class_ids):
            class_gts = gt_by_class.get(class_id, [])
            class_preds = pred_by_class.get(class_id, [])
            if not class_gts:
                continue

            class_ap_values = []
            for threshold in iou_thresholds:
                ap, _, _, precision, recall, f1, mean_iou = evaluate_class(class_preds, class_gts, threshold)
                class_ap_values.append(ap)
                if math.isclose(threshold, 0.5, abs_tol=1e-9):
                    precision_50 = precision
                    recall_50 = recall
                    f1_50 = f1
                    mean_iou_50 = mean_iou
                    precision_values_50.append(precision)
                    recall_values_50.append(recall)
                    f1_values_50.append(f1)
                    iou_values_50.append(mean_iou)

            per_class_ap_curve[class_id] = class_ap_values
            per_class_ap50[class_id] = class_ap_values[0] if class_ap_values else 0.0
            per_class_iou[class_id] = mean_iou_50 if class_gts else 0.0
            ap_per_threshold.append(float(np.mean(class_ap_values)))

    map_50 = float(np.mean([values[0] for values in per_class_ap_curve.values()])) if per_class_ap_curve else 0.0
    map_5095 = float(np.mean(ap_per_threshold)) if ap_per_threshold else 0.0
    precision_50 = float(np.mean(precision_values_50)) if precision_values_50 else 0.0
    recall_50 = float(np.mean(recall_values_50)) if recall_values_50 else 0.0
    f1_50 = float(np.mean(f1_values_50)) if f1_values_50 else 0.0
    mean_iou_50 = float(np.mean(iou_values_50)) if iou_values_50 else 0.0

    mean_inference_ms = statistics.mean(inference_times) if inference_times else 0.0
    median_inference_ms = statistics.median(inference_times) if inference_times else 0.0
    p95_inference_ms = statistics.quantiles(inference_times, n=20)[18] if len(inference_times) >= 20 else max(inference_times, default=0.0)
    throughput_fps = 1000.0 / mean_inference_ms if mean_inference_ms > 0 else 0.0

    summary = ModelSummary(
        model_name=model_path.name,
        mean_inference_ms=mean_inference_ms,
        median_inference_ms=median_inference_ms,
        p95_inference_ms=p95_inference_ms,
        throughput_fps=throughput_fps,
        mean_iou=mean_iou_50,
        precision_50=precision_50,
        recall_50=recall_50,
        f1_50=f1_50,
        map_50=map_50,
        map_5095=map_5095,
        per_class_ap50=per_class_ap50,
        per_class_iou=per_class_iou,
        evaluation_mode="class-agnostic" if class_agnostic_mode else "class-aware",
    )
    return summary, per_class_ap50


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def save_plots(output_dir: Path, summaries: list[ModelSummary], class_tables: dict[str, dict[int, float]]) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_names = [summary.model_name for summary in summaries]
    indices = np.arange(len(model_names))
    width = 0.25

    inference_plot = plots_dir / "inference_time.png"
    map_plot = plots_dir / "map_scores.png"
    iou_plot = plots_dir / "iou_scores.png"
    class_ap_plot = plots_dir / "class_ap50.png"

    plt.figure(figsize=(10, 6))
    means = [summary.mean_inference_ms for summary in summaries]
    medians = [summary.median_inference_ms for summary in summaries]
    p95s = [summary.p95_inference_ms for summary in summaries]
    plt.bar(indices - width, means, width=width, label="Mean")
    plt.bar(indices, medians, width=width, label="Median")
    plt.bar(indices + width, p95s, width=width, label="P95")
    plt.xticks(indices, model_names, rotation=20, ha="right")
    plt.ylabel("Inference time (ms)")
    plt.title("Inference Time by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(inference_plot, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    map50 = [summary.map_50 for summary in summaries]
    map5095 = [summary.map_5095 for summary in summaries]
    plt.bar(indices - width / 2, map50, width=width, label="mAP@0.5")
    plt.bar(indices + width / 2, map5095, width=width, label="mAP@0.5:0.95")
    plt.xticks(indices, model_names, rotation=20, ha="right")
    plt.ylabel("AP")
    plt.title("Segmentation AP by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(map_plot, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    mean_ious = [summary.mean_iou for summary in summaries]
    plt.bar(indices, mean_ious, width=0.6)
    plt.xticks(indices, model_names, rotation=20, ha="right")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU by Model")
    plt.tight_layout()
    plt.savefig(iou_plot, dpi=200)
    plt.close()

    class_ids = sorted({class_id for table in class_tables.values() for class_id in table})
    if class_ids:
        heatmap = np.zeros((len(summaries), len(class_ids)), dtype=np.float32)
        for row_index, summary in enumerate(summaries):
            for col_index, class_id in enumerate(class_ids):
                heatmap[row_index, col_index] = class_tables[summary.model_name].get(class_id, 0.0)

        plt.figure(figsize=(12, 6))
        im = plt.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(im, label="AP50")
        plt.yticks(np.arange(len(summaries)), model_names)
        labels = ["all-objects" if class_id == CLASS_AGNOSTIC_ID else COCO_CLASS_NAMES[class_id] for class_id in class_ids]
        plt.xticks(np.arange(len(class_ids)), labels, rotation=25, ha="right")
        plt.title("Per-Class AP50")
        plt.tight_layout()
        plt.savefig(class_ap_plot, dpi=200)
        plt.close()

    return {
        "inference": str(inference_plot.relative_to(output_dir)),
        "map": str(map_plot.relative_to(output_dir)),
        "iou": str(iou_plot.relative_to(output_dir)),
        "class_ap": str(class_ap_plot.relative_to(output_dir)),
    }


def write_outputs(
    output_dir: Path,
    dataset_root: Path,
    split: str,
    summaries: list[ModelSummary],
    class_tables: dict[str, dict[int, float]],
    plot_paths: dict[str, str],
    pairs_count: int,
    max_images: int,
    extra_report_lines: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = output_dir / "benchmark_results.json"
    results_csv = output_dir / "benchmark_results.csv"
    report_md = output_dir / "benchmark_report.md"

    json_payload = {
        "dataset_root": str(dataset_root),
        "split": split,
        "pairs_count": pairs_count,
        "max_images": max_images,
        "models": [
            {
                "model_name": summary.model_name,
                "mean_inference_ms": summary.mean_inference_ms,
                "median_inference_ms": summary.median_inference_ms,
                "p95_inference_ms": summary.p95_inference_ms,
                "throughput_fps": summary.throughput_fps,
                "mean_iou": summary.mean_iou,
                "precision_50": summary.precision_50,
                "recall_50": summary.recall_50,
                "f1_50": summary.f1_50,
                "map_50": summary.map_50,
                "map_5095": summary.map_5095,
                "evaluation_mode": summary.evaluation_mode,
                "best_f1_50": summary.best_f1_50,
                "best_conf_50": summary.best_conf_50,
                "per_class_ap50": {
                    ("all-objects" if class_id == CLASS_AGNOSTIC_ID else COCO_CLASS_NAMES[class_id]): value
                    for class_id, value in summary.per_class_ap50.items()
                },
            }
            for summary in summaries
        ],
    }
    results_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "model",
            "mean_inference_ms",
            "median_inference_ms",
            "p95_inference_ms",
            "throughput_fps",
            "mean_iou",
            "precision_50",
            "recall_50",
            "f1_50",
            "map_50",
            "map_5095",
            "evaluation_mode",
            "best_f1_50",
            "best_conf_50",
        ])
        for summary in summaries:
            writer.writerow([
                summary.model_name,
                f"{summary.mean_inference_ms:.4f}",
                f"{summary.median_inference_ms:.4f}",
                f"{summary.p95_inference_ms:.4f}",
                f"{summary.throughput_fps:.4f}",
                f"{summary.mean_iou:.4f}",
                f"{summary.precision_50:.4f}",
                f"{summary.recall_50:.4f}",
                f"{summary.f1_50:.4f}",
                f"{summary.map_50:.4f}",
                f"{summary.map_5095:.4f}",
                summary.evaluation_mode,
                f"{summary.best_f1_50:.4f}",
                f"{summary.best_conf_50:.4f}",
            ])

    summary_headers = [
        "Model",
        "Mean ms",
        "Median ms",
        "P95 ms",
        "FPS",
        "Mean IoU",
        "Prec@0.5",
        "Rec@0.5",
        "F1@0.5",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "Eval mode",
        "Best F1@0.5",
        "Best conf@0.5",
    ]
    summary_rows = [
        [
            summary.model_name,
            f"{summary.mean_inference_ms:.2f}",
            f"{summary.median_inference_ms:.2f}",
            f"{summary.p95_inference_ms:.2f}",
            f"{summary.throughput_fps:.2f}",
            f"{summary.mean_iou:.4f}",
            f"{summary.precision_50:.4f}",
            f"{summary.recall_50:.4f}",
            f"{summary.f1_50:.4f}",
            f"{summary.map_50:.4f}",
            f"{summary.map_5095:.4f}",
            summary.evaluation_mode,
            f"{summary.best_f1_50:.4f}",
            f"{summary.best_conf_50:.4f}",
        ]
        for summary in summaries
    ]

    class_ids = sorted({class_id for table in class_tables.values() for class_id in table})
    class_headers = [
        "Model",
        *[("all-objects" if class_id == CLASS_AGNOSTIC_ID else COCO_CLASS_NAMES[class_id]) for class_id in class_ids],
    ]
    class_rows = []
    for summary in summaries:
        row = [summary.model_name]
        for class_id in class_ids:
            row.append(f"{class_tables[summary.model_name].get(class_id, 0.0):.4f}")
        class_rows.append(row)

    report_lines = [
        "# Cityscapes Segmentation Benchmark",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Split: `{split}`",
        f"- Image pairs evaluated: `{pairs_count}`",
        f"- Max images: `{max_images if max_images > 0 else 'all'}`",
        "",
        "## Summary",
        "",
        markdown_table(summary_headers, summary_rows),
        "",
        "Engine models may use class-agnostic fallback when class/conf fields are incompatible.",
        "",
        "## Plots",
        "",
        f"![Inference Time]({plot_paths['inference']})",
        "",
        f"![mAP Scores]({plot_paths['map']})",
        "",
        f"![Mean IoU]({plot_paths['iou']})",
        "",
        f"![Per-Class AP50]({plot_paths['class_ap']})",
    ]

    if plot_paths.get("confusion"):
        report_lines.extend(["", f"![Confusion Matrix]({plot_paths['confusion']})"])

    report_lines.extend([
        "",
        "## Per-Class AP50",
        "",
        markdown_table(class_headers, class_rows) if class_rows else "No classes were evaluated.",
        "",
        "## Threshold View",
        "",
        markdown_table(
            ["Model", "Best F1@0.5", "Best conf@0.5"],
            [[summary.model_name, f"{summary.best_f1_50:.4f}", f"{summary.best_conf_50:.4f}"] for summary in summaries],
        ),
        "",
        "## Outputs",
        "",
        f"- JSON: [`{results_json.name}`]({results_json.name})",
        f"- CSV: [`{results_csv.name}`]({results_csv.name})",
        f"- Plots directory: [`plots/`](plots)",
    ])

    if extra_report_lines:
        report_lines.extend(["", *extra_report_lines])

    report_md.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    dataset_root = (root / args.dataset_root).resolve()
    output_dir = (root / args.output_dir).resolve()

    pairs = discover_pairs(dataset_root, args.split)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    print(f"Evaluating {len(pairs)} image pairs from {dataset_root} split={args.split}")

    summaries: list[ModelSummary] = []
    class_tables: dict[str, dict[int, float]] = {}

    for model_index, model_name in enumerate(args.models, start=1):
        model_path = (root / model_name).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        print(f"[{model_index}/{len(args.models)}] Evaluating {model_path.name}")
        summary, class_ap50 = evaluate_model(
            model_path=model_path,
            pairs=pairs,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou_threshold,
            onnx_device=args.onnx_device,
            target_class_ids=set(DEFAULT_TARGET_CLASS_IDS),
        )
        summaries.append(summary)
        class_tables[summary.model_name] = class_ap50

    plot_paths = save_plots(output_dir, summaries, class_tables)
    write_outputs(
        output_dir=output_dir,
        dataset_root=dataset_root,
        split=args.split,
        summaries=summaries,
        class_tables=class_tables,
        plot_paths=plot_paths,
        pairs_count=len(pairs),
        max_images=args.max_images,
    )

    print(f"Report written to: {output_dir / 'benchmark_report.md'}")
    print(f"CSV written to: {output_dir / 'benchmark_results.csv'}")
    print(f"JSON written to: {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()