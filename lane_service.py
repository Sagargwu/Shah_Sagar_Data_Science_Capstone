from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "full_CNN_model.h5"

_model = None


class LaneSmoother:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.recent_fit = []

    def smooth(self, prediction: np.ndarray) -> np.ndarray:
        self.recent_fit.append(prediction)
        if len(self.recent_fit) > self.window_size:
            self.recent_fit = self.recent_fit[-self.window_size:]
        return np.mean(np.array(self.recent_fit), axis=0)


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. Keep full_CNN_model.h5 in the project root."
            )
        _model = load_model(str(MODEL_PATH))
    return _model


def resize_image(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    image = Image.fromarray(arr)
    height, width = int(size_hw[0]), int(size_hw[1])
    image = image.resize((width, height), resample=Image.BILINEAR)
    return np.array(image)


def is_night_frame(frame_bgr: np.ndarray, threshold: float = 85.0) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < threshold


def enhance_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def prepare_model_input(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small_img = resize_image(rgb, (80, 160))

    if small_img.ndim == 2:
        small_img = np.stack([small_img] * 3, axis=-1)

    if small_img.shape[-1] == 4:
        small_img = small_img[:, :, :3]

    return np.expand_dims(small_img, axis=0)


def predict_lane_mask(frame_bgr: np.ndarray, smoother: LaneSmoother) -> tuple[np.ndarray, float]:
    model = get_model()
    input_batch = prepare_model_input(frame_bgr)

    prediction = model.predict(input_batch, verbose=0)[0]

    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[:, :, 0]

    prediction = np.clip(prediction, 0.0, 1.0)
    smoothed = smoother.smooth(prediction)

    mean_confidence = float(np.mean(smoothed))
    return smoothed, mean_confidence


def build_visual_overlay(
    frame_bgr: np.ndarray,
    lane_mask_small: np.ndarray,
    lane_alpha: float = 0.85,
    heatmap_alpha: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame_bgr.shape[:2]

    lane_mask_uint8 = (lane_mask_small * 255.0).astype(np.uint8)
    lane_mask_full = resize_image(lane_mask_uint8, (height, width))

    green_overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
    green_overlay[:, :, 1] = lane_mask_full

    heatmap_color = cv2.applyColorMap(lane_mask_full, cv2.COLORMAP_TURBO)

    blended = cv2.addWeighted(frame_bgr, 1.0, heatmap_color, heatmap_alpha, 0)
    blended = cv2.addWeighted(blended, 1.0, green_overlay, lane_alpha, 0)

    return blended, lane_mask_full


def save_summary_heatmap(accumulator: np.ndarray, output_path: str) -> None:
    norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    color_map = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(output_path, color_map)


def save_frame_plot(values, out_path, title, ylabel):
    plt.figure(figsize=(10, 4))
    plt.plot(values, linewidth=2)
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_hist_plot(values, out_path, title, xlabel):
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_day_night_bar(day_count, night_count, out_path):
    labels = ["Day Frames", "Night Frames"]
    values = [day_count, night_count]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Day vs Night Frame Distribution")
    plt.ylabel("Frame Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_best_worst_frame(best_frame, worst_frame, best_path, worst_path):
    if best_frame is not None:
        cv2.imwrite(best_path, best_frame)
    if worst_frame is not None:
        cv2.imwrite(worst_path, worst_frame)


def process_video(input_path: str, output_path: str, asset_prefix: str) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    heatmap_path = f"{asset_prefix}_heatmap.png"
    conf_plot_path = f"{asset_prefix}_confidence.png"
    latency_plot_path = f"{asset_prefix}_latency.png"
    coverage_hist_path = f"{asset_prefix}_coverage.png"
    day_night_path = f"{asset_prefix}_daynight.png"
    best_frame_path = f"{asset_prefix}_bestframe.png"
    worst_frame_path = f"{asset_prefix}_worstframe.png"
    json_path = f"{asset_prefix}_metrics.json"

    video = cv2.VideoCapture(str(input_file))
    if not video.isOpened():
        raise RuntimeError("Could not open the input video.")

    input_fps = float(video.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if frame_width <= 0 or frame_height <= 0:
        video.release()
        raise RuntimeError("Invalid input video resolution.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, input_fps, (frame_width, frame_height))

    if not writer.isOpened():
        video.release()
        raise RuntimeError("Could not create output video writer.")

    smoother = LaneSmoother(window_size=5)

    started_at = time.time()
    processed_frames = 0
    inference_times_ms = []
    confidence_values = []
    lane_coverage_values = []
    brightness_values = []
    night_frames_detected = 0
    day_frames_detected = 0
    heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    best_frame = None
    worst_frame = None
    best_conf = -1.0
    worst_conf = 999.0

    try:
        while True:
            ret, frame_bgr = video.read()
            if not ret:
                break

            original_frame = frame_bgr.copy()
            frame_for_model = frame_bgr.copy()

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            brightness_values.append(mean_brightness)

            if is_night_frame(frame_bgr):
                frame_for_model = enhance_low_light(frame_bgr)
                night_frames_detected += 1
            else:
                day_frames_detected += 1

            infer_start = time.time()
            lane_mask_small, mean_conf = predict_lane_mask(frame_for_model, smoother)
            infer_ms = (time.time() - infer_start) * 1000.0

            output_frame, lane_mask_full = build_visual_overlay(original_frame, lane_mask_small)
            writer.write(output_frame)

            lane_coverage = float(np.mean(lane_mask_full > 120))
            lane_coverage_values.append(lane_coverage)

            if mean_conf > best_conf:
                best_conf = mean_conf
                best_frame = output_frame.copy()

            if mean_conf < worst_conf:
                worst_conf = mean_conf
                worst_frame = output_frame.copy()

            heatmap_accumulator += lane_mask_full.astype(np.float32)
            inference_times_ms.append(infer_ms)
            confidence_values.append(mean_conf)
            processed_frames += 1

    finally:
        video.release()
        writer.release()

    total_processing_seconds = time.time() - started_at
    duration_seconds = processed_frames / input_fps if input_fps > 0 else 0.0

    avg_latency_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    mean_lane_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
    mean_lane_coverage = float(np.mean(lane_coverage_values)) if lane_coverage_values else 0.0
    mean_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    processing_fps = processed_frames / total_processing_seconds if total_processing_seconds > 0 else 0.0

    save_summary_heatmap(heatmap_accumulator, heatmap_path)
    save_frame_plot(confidence_values, conf_plot_path, "Frame-wise Lane Confidence", "Confidence")
    save_frame_plot(inference_times_ms, latency_plot_path, "Frame-wise Inference Latency", "Latency (ms)")
    save_hist_plot(lane_coverage_values, coverage_hist_path, "Lane Coverage Distribution", "Lane Coverage Ratio")
    save_day_night_bar(day_frames_detected, night_frames_detected, day_night_path)
    save_best_worst_frame(best_frame, worst_frame, best_frame_path, worst_frame_path)

    metrics = {
        "input_video": input_file.name,
        "output_video": output_file.name,
        "duration_seconds": round(duration_seconds, 2),
        "input_fps": round(input_fps, 2),
        "processed_frames": processed_frames,
        "avg_inference_latency_ms": round(avg_latency_ms, 2),
        "processing_seconds": round(total_processing_seconds, 2),
        "processing_fps": round(processing_fps, 2),
        "mean_lane_confidence": round(mean_lane_confidence, 4),
        "mean_lane_coverage": round(mean_lane_coverage, 4),
        "mean_brightness": round(mean_brightness, 2),
        "night_frames_detected": night_frames_detected,
        "day_frames_detected": day_frames_detected,
        "assets": {
            "heatmap": Path(heatmap_path).name,
            "confidence_plot": Path(conf_plot_path).name,
            "latency_plot": Path(latency_plot_path).name,
            "coverage_histogram": Path(coverage_hist_path).name,
            "day_night_chart": Path(day_night_path).name,
            "best_frame": Path(best_frame_path).name,
            "worst_frame": Path(worst_frame_path).name,
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
