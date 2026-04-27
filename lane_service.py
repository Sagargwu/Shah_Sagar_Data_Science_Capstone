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


# Fast LUTs for faster night enhancement than per-frame heavy CLAHE
_GAMMA_LUT_16 = np.array([((i / 255.0) ** (1.0 / 1.6)) * 255 for i in range(256)]).astype("uint8")
_GAMMA_LUT_20 = np.array([((i / 255.0) ** (1.0 / 2.0)) * 255 for i in range(256)]).astype("uint8")


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


def get_frame_stats(frame_bgr: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    return mean_brightness, std_brightness


def detect_condition(frame_bgr: np.ndarray) -> str:
    mean_brightness, std_brightness = get_frame_stats(frame_bgr)

    if mean_brightness < 85:
        return "night"

    if mean_brightness > 145 and std_brightness < 60:
        return "snow"

    return "normal"


def fast_enhance_night(frame_bgr: np.ndarray, very_dark: bool = False) -> np.ndarray:
    lut = _GAMMA_LUT_20 if very_dark else _GAMMA_LUT_16
    gamma_corrected = cv2.LUT(frame_bgr, lut)
    enhanced = cv2.convertScaleAbs(gamma_corrected, alpha=1.10, beta=6)
    return enhanced


def fast_enhance_snow(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)
    sharpened_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.addWeighted(frame_bgr, 0.55, sharpened_bgr, 0.45, 0)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.08, beta=-4)
    return enhanced


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


def save_line_plot(values, out_path, title, ylabel):
    plt.figure(figsize=(10, 4))
    plt.plot(values, linewidth=2)
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_bar_chart(labels, values, out_path, title, ylabel):
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_pie_chart(labels, values, out_path, title):
    plt.figure(figsize=(6, 6))
    total = sum(values)
    if total <= 0:
        values = [1 for _ in values]
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_summary_heatmap(accumulator: np.ndarray, output_path: str) -> None:
    norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    color_map = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(output_path, color_map)


def classify_risk_zone(risk_score: float) -> str:
    if risk_score < 35:
        return "SAFE"
    if risk_score < 70:
        return "CAUTION"
    return "DANGER"


def get_zone_color(zone_name: str) -> tuple[int, int, int]:
    # BGR
    if zone_name == "SAFE":
        return (40, 200, 80)
    if zone_name == "CAUTION":
        return (0, 215, 255)
    return (50, 60, 255)


def refine_lane_mask(lane_mask_full: np.ndarray) -> np.ndarray:
    mask = lane_mask_full.astype(np.uint8)

    blurred = cv2.GaussianBlur(mask, (9, 9), 0)

    roi = blurred[int(mask.shape[0] * 0.55):, :]
    thresh_val = int(max(65, min(150, np.percentile(roi, 72)))) if roi.size > 0 else 90

    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return binary


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) == 0:
        return values
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(values)]


def estimate_lane_polygon(binary_mask: np.ndarray) -> dict:
    height, width = binary_mask.shape[:2]
    y_top = int(height * 0.56)
    y_bottom = int(height * 0.98)

    ys = []
    lefts = []
    rights = []

    for y in range(y_bottom, y_top, -6):
        row = binary_mask[y]
        xs = np.where(row > 0)[0]

        if len(xs) < 12:
            continue

        left_x = int(xs[0])
        right_x = int(xs[-1])

        if right_x - left_x < max(40, width * 0.08):
            continue

        ys.append(y)
        lefts.append(left_x)
        rights.append(right_x)

    if len(ys) < 8:
        return {"detected": False}

    ys = np.array(ys[::-1], dtype=np.int32)
    lefts = np.array(lefts[::-1], dtype=np.float32)
    rights = np.array(rights[::-1], dtype=np.float32)

    lefts = moving_average(lefts, window=7)
    rights = moving_average(rights, window=7)

    corridor_expand = max(6, int(width * 0.008))
    lefts = np.clip(lefts - corridor_expand, 0, width - 1)
    rights = np.clip(rights + corridor_expand, 0, width - 1)

    left_points = np.array([[int(x), int(y)] for x, y in zip(lefts, ys)], dtype=np.int32)
    right_points = np.array([[int(x), int(y)] for x, y in zip(rights, ys)], dtype=np.int32)

    polygon = np.vstack([left_points, right_points[::-1]])

    bottom_count = min(4, len(lefts))
    left_bottom = float(np.mean(lefts[-bottom_count:]))
    right_bottom = float(np.mean(rights[-bottom_count:]))
    lane_center_bottom = (left_bottom + right_bottom) / 2.0
    lane_width_bottom = max(1.0, right_bottom - left_bottom)

    return {
        "detected": True,
        "polygon": polygon,
        "left_points": left_points,
        "right_points": right_points,
        "left_bottom": left_bottom,
        "right_bottom": right_bottom,
        "lane_center_bottom": lane_center_bottom,
        "lane_width_bottom": lane_width_bottom,
    }


def build_visual_overlay(
    frame_bgr: np.ndarray,
    lane_mask_small: np.ndarray,
    lane_confidence: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = frame_bgr.shape[:2]

    lane_mask_uint8 = (lane_mask_small * 255.0).astype(np.uint8)
    lane_mask_full = resize_image(lane_mask_uint8, (height, width))
    refined_binary = refine_lane_mask(lane_mask_full)
    polygon_info = estimate_lane_polygon(refined_binary)

    vehicle_center_x = width // 2

    if polygon_info["detected"]:
        lane_center_x = float(polygon_info["lane_center_bottom"])
        lane_width = float(polygon_info["lane_width_bottom"])
        offset_pixels = float(vehicle_center_x - lane_center_x)
        departure_ratio = min(1.6, abs(offset_pixels) / max(1.0, lane_width / 2.0))
    else:
        lane_center_x = float(vehicle_center_x)
        lane_width = float(width * 0.22)
        offset_pixels = float(width * 0.30)
        departure_ratio = 1.3

    confidence_factor = max(0.0, 1.0 - min(1.0, lane_confidence / 0.22))
    risk_score = min(100.0, departure_ratio * 72.0 + confidence_factor * 28.0)

    if not polygon_info["detected"]:
        risk_score = max(risk_score, 88.0)

    zone_name = classify_risk_zone(risk_score)
    zone_color = get_zone_color(zone_name)

    output = frame_bgr.copy()

    if polygon_info["detected"]:
        fill_overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
        cv2.fillPoly(fill_overlay, [polygon_info["polygon"]], zone_color)
        output = cv2.addWeighted(output, 1.0, fill_overlay, 0.42, 0)

        cv2.polylines(output, [polygon_info["left_points"]], False, (255, 255, 255), 2)
        cv2.polylines(output, [polygon_info["right_points"]], False, (255, 255, 255), 2)

        center_line_top = (int(lane_center_x), int(height * 0.58))
        center_line_bottom = (int(lane_center_x), int(height * 0.98))
        cv2.line(output, center_line_top, center_line_bottom, (255, 240, 0), 2)

    heatmap_color = cv2.applyColorMap(lane_mask_full, cv2.COLORMAP_TURBO)
    output = cv2.addWeighted(output, 0.94, heatmap_color, 0.12, 0)

    cv2.line(output, (vehicle_center_x, int(height * 0.58)), (vehicle_center_x, int(height * 0.98)), (0, 255, 255), 2)

    panel_x1, panel_y1 = 20, 20
    panel_x2, panel_y2 = 370, 145
    cv2.rectangle(output, (panel_x1, panel_y1), (panel_x2, panel_y2), (10, 18, 32), -1)
    cv2.rectangle(output, (panel_x1, panel_y1), (panel_x2, panel_y2), zone_color, 2)

    cv2.putText(output, f"Risk Zone: {zone_name}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, zone_color, 2)
    cv2.putText(output, f"Risk Score: {risk_score:.1f}%", (35, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (230, 240, 255), 2)
    cv2.putText(output, f"Lane Departure: {departure_ratio * 100:.1f}%", (35, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 240, 255), 2)

    risk_meta = {
        "risk_score": float(risk_score),
        "risk_zone": zone_name,
        "departure_ratio": float(departure_ratio),
        "offset_pixels": float(offset_pixels),
        "lane_detected": bool(polygon_info["detected"]),
        "lane_width": float(lane_width),
    }

    return output, lane_mask_full, risk_meta


def process_video(input_path: str, output_path: str, asset_prefix: str) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    heatmap_path = f"{asset_prefix}_heatmap.png"
    risk_plot_path = f"{asset_prefix}_risktrend.png"
    offset_plot_path = f"{asset_prefix}_offsettrend.png"
    zone_chart_path = f"{asset_prefix}_zonebar.png"
    day_night_path = f"{asset_prefix}_daynight.png"
    risk_pie_path = f"{asset_prefix}_riskpie.png"
    best_frame_path = f"{asset_prefix}_bestframe.png"
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
    brightness_values = []
    risk_scores = []
    offset_values = []
    departure_values = []

    night_frames_detected = 0
    day_frames_detected = 0
    snow_frames_detected = 0
    heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    safe_count = 0
    caution_count = 0
    danger_count = 0
    lane_departure_events = 0

    best_frame = None
    best_score = 999.0

    try:
        while True:
            ret, frame_bgr = video.read()
            if not ret:
                break

            original_frame = frame_bgr.copy()
            frame_for_model = frame_bgr.copy()

            mean_brightness, _ = get_frame_stats(frame_bgr)
            brightness_values.append(mean_brightness)

            condition = detect_condition(frame_bgr)
            if condition == "night":
                very_dark = mean_brightness < 55
                frame_for_model = fast_enhance_night(frame_bgr, very_dark=very_dark)
                night_frames_detected += 1
            elif condition == "snow":
                frame_for_model = fast_enhance_snow(frame_bgr)
                snow_frames_detected += 1
                day_frames_detected += 1
            else:
                day_frames_detected += 1

            infer_start = time.time()
            lane_mask_small, mean_conf = predict_lane_mask(frame_for_model, smoother)
            infer_ms = (time.time() - infer_start) * 1000.0

            output_frame, lane_mask_full, risk_meta = build_visual_overlay(original_frame, lane_mask_small, mean_conf)
            writer.write(output_frame)

            heatmap_accumulator += lane_mask_full.astype(np.float32)

            risk_score = float(risk_meta["risk_score"])
            risk_zone = risk_meta["risk_zone"]
            departure_ratio = float(risk_meta["departure_ratio"])
            offset_pixels = float(risk_meta["offset_pixels"])

            if risk_zone == "SAFE":
                safe_count += 1
            elif risk_zone == "CAUTION":
                caution_count += 1
            else:
                danger_count += 1

            if departure_ratio >= 1.0:
                lane_departure_events += 1

            if risk_score < best_score:
                best_score = risk_score
                best_frame = output_frame.copy()

            inference_times_ms.append(infer_ms)
            confidence_values.append(mean_conf)
            risk_scores.append(risk_score)
            offset_values.append(offset_pixels)
            departure_values.append(departure_ratio * 100.0)
            processed_frames += 1

    finally:
        video.release()
        writer.release()

    total_processing_seconds = time.time() - started_at
    duration_seconds = processed_frames / input_fps if input_fps > 0 else 0.0

    avg_latency_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    mean_lane_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
    mean_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    avg_risk_score = float(np.mean(risk_scores)) if risk_scores else 0.0
    max_risk_score = float(np.max(risk_scores)) if risk_scores else 0.0
    max_departure_percent = float(np.max(departure_values)) if departure_values else 0.0
    processing_fps = processed_frames / total_processing_seconds if total_processing_seconds > 0 else 0.0
    road_position_bias = float(np.mean(offset_values)) if offset_values else 0.0

    if avg_risk_score < 35:
        overall_risk = "LOW"
    elif avg_risk_score < 70:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "HIGH"

    save_summary_heatmap(heatmap_accumulator, heatmap_path)
    save_line_plot(risk_scores, risk_plot_path, "Frame-wise Driving Risk Score", "Risk Score (%)")
    save_line_plot(offset_values, offset_plot_path, "Vehicle Offset from Lane Center", "Offset (pixels)")
    save_bar_chart(
        ["Safe", "Caution", "Danger"],
        [safe_count, caution_count, danger_count],
        zone_chart_path,
        "Risk Zone Distribution",
        "Frame Count"
    )
    save_bar_chart(
        ["Day", "Night", "Snow"],
        [day_frames_detected, night_frames_detected, snow_frames_detected],
        day_night_path,
        "Driving Condition Distribution",
        "Frame Count"
    )
    save_pie_chart(
        ["Safe", "Caution", "Danger"],
        [safe_count, caution_count, danger_count],
        risk_pie_path,
        "Driving Risk Proportion"
    )

    if best_frame is not None:
        cv2.imwrite(best_frame_path, best_frame)

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
        "mean_brightness": round(mean_brightness, 2),
        "night_frames_detected": night_frames_detected,
        "day_frames_detected": day_frames_detected,
        "snow_frames_detected": snow_frames_detected,
        "avg_risk_score": round(avg_risk_score, 2),
        "max_risk_score": round(max_risk_score, 2),
        "max_departure_percent": round(max_departure_percent, 2),
        "lane_departure_events": int(lane_departure_events),
        "safe_frames": int(safe_count),
        "caution_frames": int(caution_count),
        "danger_frames": int(danger_count),
        "overall_risk": overall_risk,
        "road_position_bias_pixels": round(road_position_bias, 2),
        "assets": {
            "heatmap": Path(heatmap_path).name,
            "risk_plot": Path(risk_plot_path).name,
            "offset_plot": Path(offset_plot_path).name,
            "zone_chart": Path(zone_chart_path).name,
            "day_night_chart": Path(day_night_path).name,
            "risk_pie_chart": Path(risk_pie_path).name,
            "best_frame": Path(best_frame_path).name,
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
