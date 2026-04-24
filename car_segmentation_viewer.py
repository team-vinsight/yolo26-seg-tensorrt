import os
import time
import warnings
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
warnings.filterwarnings("ignore", message="CUDA initialization.*", category=UserWarning)

from ultralytics import YOLO


import time
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorrt as trt
from cuda.bindings import runtime as cudart


VIDEO_PATH = Path(__file__).with_name("car.mp4")
MODEL_PATH = Path(__file__).with_name("yolo26n-seg.engine")
DISPLAY_SCALE = 0.5
DYNAMIC_CLASS_IDS = {0, 1, 2, 3, 5, 6, 7, 8}


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


class TensorRTSegmentationModel:
    def __init__(self, engine_path: Path) -> None:
        self.engine_path = engine_path

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with engine_path.open("rb") as engine_file:
            engine_bytes = engine_file.read()

        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
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

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, int, float, float]:
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

        detections = self.host_buffers[self.output_names[0]][0]
        prototypes = self.host_buffers[self.output_names[1]][0]
        masked_frame, dynamic_count, masked_ratio = self._postprocess(
            frame,
            detections,
            prototypes,
            scale,
            pad_left,
            pad_top,
            resized_width,
            resized_height,
        )
        return masked_frame, dynamic_count, masked_ratio, inference_ms

    def _postprocess(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
        prototypes: np.ndarray,
        scale: float,
        pad_left: float,
        pad_top: float,
        resized_width: int,
        resized_height: int,
    ) -> tuple[np.ndarray, int, float]:
        original_height, original_width = frame.shape[:2]
        proto_height, proto_width = prototypes.shape[1], prototypes.shape[2]
        proto_flat = prototypes.reshape(prototypes.shape[0], -1)

        selected_mask = np.zeros((original_height, original_width), dtype=bool)
        dynamic_count = 0

        for detection in detections:
            confidence = float(detection[4])
            class_id = int(round(float(detection[5])))
            if confidence < 0.25 or class_id not in DYNAMIC_CLASS_IDS:
                continue

            x_center, y_center, box_width, box_height = map(float, detection[:4])
            coeffs = detection[6:]

            mask = _sigmoid(coeffs @ proto_flat).reshape(proto_height, proto_width)
            mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_LINEAR)

            left = int(round(pad_left))
            top = int(round(pad_top))
            mask = mask[top : top + resized_height, left : left + resized_width]
            if mask.size == 0:
                continue

            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

            x1 = int(round((x_center - box_width / 2 - pad_left) / scale))
            y1 = int(round((y_center - box_height / 2 - pad_top) / scale))
            x2 = int(round((x_center + box_width / 2 - pad_left) / scale))
            y2 = int(round((y_center + box_height / 2 - pad_top) / scale))
            x1 = max(0, min(original_width, x1))
            y1 = max(0, min(original_height, y1))
            x2 = max(0, min(original_width, x2))
            y2 = max(0, min(original_height, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            dynamic_count += 1
            box_mask = np.zeros((original_height, original_width), dtype=bool)
            box_mask[y1:y2, x1:x2] = True
            selected_mask |= (mask > 0.5) & box_mask

        masked_frame = frame.copy()
        masked_frame[selected_mask] = 0
        masked_ratio = float(selected_mask.mean() * 100.0)
        return masked_frame, dynamic_count, masked_ratio


class SegmentationViewer:
    def __init__(self, video_path: Path, model_path: Path) -> None:
        self.video_path = video_path
        self.model = TensorRTSegmentationModel(model_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.root = tk.Tk()
        self.root.title("TensorRT Dynamic Object Masking")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.Label(
            self.root,
            text="Press q or close the window to exit",
            justify="left",
            anchor="w",
            font=("TkDefaultFont", 10),
        )
        self.status.pack(pady=6)
        self.photo = None
        self.delay_ms = self._frame_delay_ms()
        self.prev_frame_time: float | None = None
        self.display_fps = 0.0
        self.frame_index = 0

        self.root.bind("q", lambda _event: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _frame_delay_ms(self) -> int:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return max(1, int(1000 / fps))
        return 33

    def _update_status(self, frame: np.ndarray, dynamic_count: int, masked_ratio: float, inference_ms: float) -> None:
        now = time.perf_counter()
        if self.prev_frame_time is None:
            instantaneous_fps = 0.0
        else:
            dt = now - self.prev_frame_time
            instantaneous_fps = (1.0 / dt) if dt > 0 else 0.0
        self.prev_frame_time = now

        if self.display_fps == 0.0:
            self.display_fps = instantaneous_fps
        else:
            self.display_fps = (0.9 * self.display_fps) + (0.1 * instantaneous_fps)

        source_size = f"{frame.shape[1]}x{frame.shape[0]}"
        display_size = f"{int(frame.shape[1] * DISPLAY_SCALE)}x{int(frame.shape[0] * DISPLAY_SCALE)}"
        progress = f"{self.frame_index}/{self.total_frames}" if self.total_frames > 0 else str(self.frame_index)
        video_fps_text = f"{self.video_fps:.2f}" if self.video_fps > 0 else "unknown"

        self.status.configure(
            text=(
                f"Playback FPS: {self.display_fps:.2f} | Source FPS: {video_fps_text} | Inference: {inference_ms:.1f} ms\n"
                f"Dynamic objects: {dynamic_count} | Masked area: {masked_ratio:.2f}%\n"
                f"Frame: {progress} | Source: {source_size} | Display: {display_size} (scale={DISPLAY_SCALE:.2f})"
            )
        )

    def _update_frame(self) -> None:
        if not self.cap.isOpened():
            self.close()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.close()
            return

        self.frame_index += 1
        masked_frame, dynamic_count, masked_ratio, inference_ms = self.model.infer(frame)
        display_frame = cv2.resize(
            masked_frame,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        self.photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=self.photo)
        self._update_status(frame, dynamic_count, masked_ratio, inference_ms)
        self.root.after(self.delay_ms, self._update_frame)

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "model") and self.model is not None:
            self.model.close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.model.engine_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model.engine_path}")

        self._update_frame()
        self.root.mainloop()


def main() -> None:
    viewer = SegmentationViewer(VIDEO_PATH, MODEL_PATH)
    viewer.run()


if __name__ == "__main__":
    main()