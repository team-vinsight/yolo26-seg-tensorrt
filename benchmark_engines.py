from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from benchmark_cityscapes import (
    COCO_CLASS_NAMES,
    InstanceRecord,
    ModelSummary,
    PredictionRecord,
    DEFAULT_TARGET_CLASS_IDS,
    discover_pairs,
    extract_ground_truth_instances,
    save_plots,
    write_outputs,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENGINE_MODELS = [
    "yolo26n-seg-fp32.engine",
    "yolo26n-seg-fp16.engine",
    "yolo26s-seg-fp32.engine",
    "yolo26s-seg-fp16.engine",
    "yolo26m-seg-fp32.engine",
    "yolo26m-seg-fp16.engine",
]

IOU_THRESHOLDS = [round(value, 2) for value in np.arange(0.5, 1.0, 0.05)]
CONFUSION_THRESHOLD = 0.5


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _letterbox(frame: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, float, float, float, int, int]:
    original_height, original_width = frame.shape[:2]
    target_height, target_width = new_shape

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    pad_width = target_width - resized_width
    pad_height = target_height - resized_height
    pad_left = pad_width / 2
    pad_top = pad_height / 2

    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        int(round(pad_top - 0.1)),
        int(round(pad_height - pad_top - 0.1)),
        int(round(pad_left - 0.1)),
        int(round(pad_width - pad_left - 0.1)),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, pad_left, pad_top, resized_width, resized_height


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    area = np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1])
    return float(area)


@dataclass(frozen=True)
class DetailedClassMetrics:
    ap: float
    precision: float
    recall: float
    f1: float
    mean_iou: float
    best_f1: float
    best_conf: float
    precisions: np.ndarray
    recalls: np.ndarray
    scores: np.ndarray


def evaluate_class_detailed(
    predictions: list[PredictionRecord],
    ground_truths: list[InstanceRecord],
    iou_threshold: float,
) -> DetailedClassMetrics:
    total_gt = len(ground_truths)
    if total_gt == 0:
        empty = np.array([], dtype=np.float64)
        return DetailedClassMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, empty, empty, empty)

    gt_by_image: dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int]]]] = defaultdict(list)
    for gt_index, gt in enumerate(ground_truths):
        bbox = _mask_bbox(gt.mask)
        if bbox is None:
            continue
        gt_by_image[gt.image_index].append((gt_index, gt.mask, bbox))

    predictions_with_bbox: list[tuple[PredictionRecord, tuple[int, int, int, int]]] = []
    for prediction in predictions:
        bbox = _mask_bbox(prediction.mask)
        if bbox is None:
            continue
        predictions_with_bbox.append((prediction, bbox))

    predictions_with_bbox.sort(key=lambda item: item[0].score, reverse=True)

    matched_gts: set[int] = set()
    true_positives: list[int] = []
    false_positives: list[int] = []
    matched_ious: list[float] = []
    scores: list[float] = []

    for prediction, pred_bbox in predictions_with_bbox:
        candidates = gt_by_image.get(prediction.image_index, [])
        best_gt_index = None
        best_iou = 0.0

        for gt_index, gt_mask, gt_bbox in candidates:
            if gt_index in matched_gts:
                continue
            if not _bbox_intersects(pred_bbox, gt_bbox):
                continue

            iou = _mask_iou(prediction.mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        scores.append(prediction.score)
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
    ap = _compute_ap(recalls, precisions)

    precision = float(precisions[-1]) if precisions.size else 0.0
    recall = float(recalls[-1]) if recalls.size else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    if precisions.size:
        f1_curve = np.divide(
            2.0 * precisions * recalls,
            precisions + recalls,
            out=np.zeros_like(precisions),
            where=(precisions + recalls) > 0,
        )
        best_index = int(np.argmax(f1_curve))
        best_f1 = float(f1_curve[best_index])
        best_conf = float(scores[best_index])
    else:
        best_f1 = 0.0
        best_conf = 0.0

    return DetailedClassMetrics(
        ap=ap,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        best_f1=best_f1,
        best_conf=best_conf,
        precisions=precisions,
        recalls=recalls,
        scores=np.asarray(scores, dtype=np.float64),
    )


def evaluate_class_fast(
    predictions: list[PredictionRecord],
    ground_truths: list[InstanceRecord],
    iou_threshold: float,
) -> tuple[float, float, float, float, float]:
    total_gt = len(ground_truths)
    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    gt_by_image: dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int]]]] = {}
    for gt_index, gt in enumerate(ground_truths):
        bbox = _mask_bbox(gt.mask)
        if bbox is None:
            continue
        gt_by_image.setdefault(gt.image_index, []).append((gt_index, gt.mask, bbox))

    predictions_with_bbox: list[tuple[PredictionRecord, tuple[int, int, int, int]]] = []
    for prediction in predictions:
        bbox = _mask_bbox(prediction.mask)
        if bbox is None:
            continue
        predictions_with_bbox.append((prediction, bbox))

    predictions_with_bbox.sort(key=lambda item: item[0].score, reverse=True)

    matched_gts: set[int] = set()
    true_positives: list[int] = []
    false_positives: list[int] = []
    matched_ious: list[float] = []

    for prediction, pred_bbox in predictions_with_bbox:
        candidates = gt_by_image.get(prediction.image_index, [])
        best_gt_index = None
        best_iou = 0.0

        for gt_index, gt_mask, gt_bbox in candidates:
            if gt_index in matched_gts:
                continue
            if not _bbox_intersects(pred_bbox, gt_bbox):
                continue

            iou = _mask_iou(prediction.mask, gt_mask)
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
    ap = _compute_ap(recalls, precisions)

    precision = float(precisions[-1]) if precisions.size else 0.0
    recall = float(recalls[-1]) if recalls.size else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return ap, precision, recall, f1, mean_iou


def build_confusion_matrix(
    predictions: list[PredictionRecord],
    ground_truths: list[InstanceRecord],
    class_ids: list[int],
    iou_threshold: float = CONFUSION_THRESHOLD,
) -> tuple[np.ndarray, list[str]]:
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    background_index = len(class_ids)
    matrix = np.zeros((len(class_ids) + 1, len(class_ids) + 1), dtype=np.int64)

    gt_by_image: dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int], int]]] = defaultdict(list)
    for gt_index, gt in enumerate(ground_truths):
        bbox = _mask_bbox(gt.mask)
        if bbox is None:
            continue
        gt_by_image[gt.image_index].append((gt_index, gt.mask, bbox, gt.class_id))

    predictions_with_bbox: list[tuple[PredictionRecord, tuple[int, int, int, int]]] = []
    for prediction in predictions:
        bbox = _mask_bbox(prediction.mask)
        if bbox is None:
            continue
        predictions_with_bbox.append((prediction, bbox))

    predictions_with_bbox.sort(key=lambda item: item[0].score, reverse=True)
    matched_gts: set[int] = set()

    for prediction, pred_bbox in predictions_with_bbox:
        candidates = gt_by_image.get(prediction.image_index, [])
        best_gt_index = None
        best_iou = 0.0
        best_true_class = None

        for gt_index, gt_mask, gt_bbox, true_class_id in candidates:
            if gt_index in matched_gts:
                continue
            if not _bbox_intersects(pred_bbox, gt_bbox):
                continue

            iou = _mask_iou(prediction.mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
                best_true_class = true_class_id

        pred_index = class_to_index.get(prediction.class_id)
        if pred_index is None:
            continue

        if best_gt_index is not None and best_iou >= iou_threshold and best_true_class in class_to_index:
            matched_gts.add(best_gt_index)
            matrix[class_to_index[best_true_class], pred_index] += 1
        else:
            matrix[background_index, pred_index] += 1

    for gt_index, gt in enumerate(ground_truths):
        if gt_index not in matched_gts and gt.class_id in class_to_index:
            matrix[class_to_index[gt.class_id], background_index] += 1

    labels = [COCO_CLASS_NAMES[class_id] for class_id in class_ids] + ["background"]
    return matrix, labels


def save_confusion_matrix_plot(
    output_dir: Path,
    model_matrices: dict[str, tuple[np.ndarray, list[str]]],
) -> str:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(model_matrices.keys())
    count = len(model_names)
    cols = 2 if count > 1 else 1
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    image = None
    for axis, model_name in zip(axes.ravel(), model_names):
        matrix, labels = model_matrices[model_name]
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums > 0)
        image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(model_name)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_yticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_yticklabels(labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")

    for axis in axes.ravel()[count:]:
        axis.axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Row-normalized fraction")
    fig.suptitle(f"Confusion Matrix @ IoU {CONFUSION_THRESHOLD:.2f}")
    fig.tight_layout()

    output_path = plots_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path.relative_to(output_dir))


class NativeTensorRTEngine:
    def __init__(self, engine_path: Path) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with engine_path.open("rb") as engine_file:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")

        stream_status, stream = cudart.cudaStreamCreate()
        if stream_status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Could not create CUDA stream: {stream_status}")
        self.stream = stream

        self.input_name: str | None = None
        self.output_names: list[str] = []
        self.host_buffers: dict[str, np.ndarray] = {}
        self.device_buffers: dict[str, int] = {}
        self._initialize_tensors()

    def _initialize_tensors(self) -> None:
        for tensor_index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(tensor_index)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_name = tensor_name
                self.context.set_input_shape(tensor_name, (1, 3, 640, 640))
            else:
                self.output_names.append(tensor_name)

        if self.input_name is None:
            raise RuntimeError("TensorRT engine does not expose an input tensor")

        for tensor_name in [self.input_name, *self.output_names]:
            tensor_shape = tuple(self.context.get_tensor_shape(tensor_name))
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            host_buffer = np.empty(tensor_shape, dtype=tensor_dtype)
            self.host_buffers[tensor_name] = host_buffer

            status, device_ptr = cudart.cudaMalloc(host_buffer.nbytes)
            if status != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for {tensor_name}: {status}")
            self.device_buffers[tensor_name] = int(device_ptr)
            self.context.set_tensor_address(tensor_name, int(device_ptr))

    def close(self) -> None:
        for device_ptr in self.device_buffers.values():
            cudart.cudaFree(device_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)

    def infer_predictions(
        self,
        frame: np.ndarray,
        target_class_ids: set[int],
        conf_threshold: float,
        image_index: int,
    ) -> tuple[list[PredictionRecord], float]:
        inference_start = time.perf_counter()

        letterboxed_frame, scale, pad_left, pad_top, resized_width, resized_height = _letterbox(frame)
        rgb_frame = cv2.cvtColor(letterboxed_frame, cv2.COLOR_BGR2RGB)
        input_tensor = rgb_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        np.copyto(self.host_buffers[self.input_name], input_tensor)
        cudart.cudaMemcpy(
            self.device_buffers[self.input_name],
            self.host_buffers[self.input_name].ctypes.data,
            self.host_buffers[self.input_name].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT engine execution failed")

        cudart.cudaStreamSynchronize(self.stream)
        for tensor_name in self.output_names:
            cudart.cudaMemcpy(
                self.host_buffers[tensor_name].ctypes.data,
                self.device_buffers[tensor_name],
                self.host_buffers[tensor_name].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )

        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        first_output = self.host_buffers[self.output_names[0]][0]
        second_output = self.host_buffers[self.output_names[1]][0]
        if first_output.ndim == 2 and second_output.ndim == 3:
            detections, prototypes = first_output, second_output
        elif second_output.ndim == 2 and first_output.ndim == 3:
            detections, prototypes = second_output, first_output
        else:
            raise RuntimeError(f"Unexpected output tensor shapes: {first_output.shape}, {second_output.shape}")

        original_height, original_width = frame.shape[:2]
        proto_height, proto_width = prototypes.shape[1], prototypes.shape[2]
        proto_flat = prototypes.reshape(prototypes.shape[0], -1)

        predictions: list[PredictionRecord] = []
        for detection in detections:
            confidence = float(detection[4])
            class_id = int(round(float(detection[5])))
            if confidence < conf_threshold or class_id not in target_class_ids:
                continue

            x1_model, y1_model, x2_model, y2_model = map(float, detection[:4])
            coeffs = detection[6:]

            mask = _sigmoid(coeffs @ proto_flat).reshape(proto_height, proto_width)
            mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_LINEAR)

            left = int(round(pad_left))
            top = int(round(pad_top))
            mask = mask[top : top + resized_height, left : left + resized_width]
            if mask.size == 0:
                continue
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

            x1 = int(round((x1_model - pad_left) / scale))
            y1 = int(round((y1_model - pad_top) / scale))
            x2 = int(round((x2_model - pad_left) / scale))
            y2 = int(round((y2_model - pad_top) / scale))
            x1 = max(0, min(original_width, x1))
            y1 = max(0, min(original_height, y1))
            x2 = max(0, min(original_width, x2))
            y2 = max(0, min(original_height, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            box_mask = np.zeros((original_height, original_width), dtype=bool)
            box_mask[y1:y2, x1:x2] = True
            final_mask = (mask > 0.5) & box_mask
            if not final_mask.any():
                continue

            predictions.append(
                PredictionRecord(
                    image_index=image_index,
                    class_id=class_id,
                    score=confidence,
                    mask=final_mask,
                )
            )

        return predictions, inference_ms


def evaluate_engine_model(
    model_path: Path,
    pairs: list[tuple[Path, Path]],
    conf: float,
    target_class_ids: set[int],
) -> tuple[ModelSummary, dict[int, float], np.ndarray, list[str]]:
    engine = NativeTensorRTEngine(model_path)

    inference_times: list[float] = []
    gt_records = []
    pred_records = []
    total_pairs = len(pairs)
    try:
        for image_index, (image_path, label_path) in enumerate(pairs):
            print(
                f"  [{model_path.name}] image {image_index + 1}/{total_pairs}: {image_path.name}",
                flush=True,
            )
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")

            instance_ids = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if instance_ids is None:
                raise RuntimeError(f"Could not read label: {label_path}")

            gt_records.extend(extract_ground_truth_instances(instance_ids, image_index))
            predictions, inference_ms = engine.infer_predictions(
                frame=image,
                target_class_ids=target_class_ids,
                conf_threshold=conf,
                image_index=image_index,
            )
            pred_records.extend(predictions)
            inference_times.append(inference_ms)
    finally:
        engine.close()

    per_class_ap50: dict[int, float] = {}
    per_class_iou: dict[int, float] = {}
    per_class_ap_curve: dict[int, list[float]] = {}
    per_class_best_f1: list[float] = []
    per_class_best_conf: list[float] = []

    ap_per_threshold: list[float] = []
    precision_values_50: list[float] = []
    recall_values_50: list[float] = []
    f1_values_50: list[float] = []
    iou_values_50: list[float] = []

    gt_by_class = {class_id: [] for class_id in target_class_ids}
    pred_by_class = {class_id: [] for class_id in target_class_ids}
    for gt in gt_records:
        if gt.class_id in gt_by_class:
            gt_by_class[gt.class_id].append(gt)
    for pred in pred_records:
        if pred.class_id in pred_by_class:
            pred_by_class[pred.class_id].append(pred)

    for class_id in sorted(target_class_ids):
        class_gts = gt_by_class.get(class_id, [])
        class_preds = pred_by_class.get(class_id, [])
        if not class_gts:
            continue

        class_name = COCO_CLASS_NAMES.get(class_id, str(class_id))
        print(
            f"  [{model_path.name}] evaluating metrics for class={class_name} with {len(class_preds)} predictions and {len(class_gts)} gt",
            flush=True,
        )

        detailed = evaluate_class_detailed(class_preds, class_gts, CONFUSION_THRESHOLD)
        class_ap_values = [detailed.ap]
        class_iou_50 = 0.0
        for threshold in IOU_THRESHOLDS[1:]:
            ap, precision, recall, f1, mean_iou = evaluate_class_fast(class_preds, class_gts, threshold)
            class_ap_values.append(ap)
        precision_values_50.append(detailed.precision)
        recall_values_50.append(detailed.recall)
        f1_values_50.append(detailed.f1)
        iou_values_50.append(detailed.mean_iou)
        per_class_best_f1.append(detailed.best_f1)
        per_class_best_conf.append(detailed.best_conf)
        class_iou_50 = detailed.mean_iou

        per_class_ap_curve[class_id] = class_ap_values
        per_class_ap50[class_id] = class_ap_values[0] if class_ap_values else 0.0
        per_class_iou[class_id] = class_iou_50
        ap_per_threshold.append(float(np.mean(class_ap_values)))

    map_50 = float(np.mean([values[0] for values in per_class_ap_curve.values()])) if per_class_ap_curve else 0.0
    map_5095 = float(np.mean(ap_per_threshold)) if ap_per_threshold else 0.0
    precision_50 = float(np.mean(precision_values_50)) if precision_values_50 else 0.0
    recall_50 = float(np.mean(recall_values_50)) if recall_values_50 else 0.0
    f1_50 = float(np.mean(f1_values_50)) if f1_values_50 else 0.0
    mean_iou_50 = float(np.mean(iou_values_50)) if iou_values_50 else 0.0
    best_f1_50 = float(np.mean(per_class_best_f1)) if per_class_best_f1 else 0.0
    best_conf_50 = float(np.mean(per_class_best_conf)) if per_class_best_conf else 0.0

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
        evaluation_mode="native-trt-class-aware",
        best_f1_50=best_f1_50,
        best_conf_50=best_conf_50,
    )
    confusion_matrix, confusion_labels = build_confusion_matrix(
        predictions=pred_records,
        ground_truths=gt_records,
        class_ids=sorted(target_class_ids),
        iou_threshold=CONFUSION_THRESHOLD,
    )
    return summary, per_class_ap50, confusion_matrix, confusion_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT segmentation engines on a Cityscapes-style dataset.")
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
        default=ENGINE_MODELS,
        help="TensorRT engine model paths to benchmark",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold used during prediction")
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="NMS IoU threshold during prediction")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images to evaluate")
    parser.add_argument(
        "--output-dir",
        default="benchmark_report_engines",
        help="Directory where markdown report, CSV, JSON, and plots are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    dataset_root = (root / args.dataset_root).resolve()
    output_dir = (root / args.output_dir).resolve()

    pairs = discover_pairs(dataset_root, args.split)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    print(f"Evaluating {len(pairs)} image pairs from {dataset_root} split={args.split}")

    summaries = []
    class_tables = {}
    confusion_matrices: dict[str, tuple[np.ndarray, list[str]]] = {}

    for index, model_name in enumerate(args.models, start=1):
        model_path = (root / model_name).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if model_path.suffix.lower() != ".engine":
            raise ValueError(f"Only .engine models are supported by this script: {model_path.name}")

        print(f"[{index}/{len(args.models)}] Evaluating {model_path.name}")
        summary, class_ap50, confusion_matrix, confusion_labels = evaluate_engine_model(
            model_path=model_path,
            pairs=pairs,
            conf=args.conf,
            target_class_ids=set(DEFAULT_TARGET_CLASS_IDS),
        )
        summaries.append(summary)
        class_tables[summary.model_name] = class_ap50
        confusion_matrices[summary.model_name] = (confusion_matrix, confusion_labels)

    plot_paths = save_plots(output_dir, summaries, class_tables)
    plot_paths["confusion"] = save_confusion_matrix_plot(output_dir, confusion_matrices)
    write_outputs(
        output_dir=output_dir,
        dataset_root=dataset_root,
        split=args.split,
        summaries=summaries,
        class_tables=class_tables,
        plot_paths=plot_paths,
        pairs_count=len(pairs),
        max_images=args.max_images,
        extra_report_lines=[
            "## Notes",
            "",
            "- `Prec@0.5` and `Rec@0.5` are the final-point values on the ranked prediction curve, so low precision with high recall can happen when many low-confidence false positives are retained.",
            "- `Best F1@0.5` shows the strongest confidence operating point for each model and is usually a better sanity check for imbalance than the final-point precision alone.",
        ],
    )

    print(f"Report written to: {output_dir / 'benchmark_report.md'}")
    print(f"CSV written to: {output_dir / 'benchmark_results.csv'}")
    print(f"JSON written to: {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
