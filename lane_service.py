from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
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
                f"Model file not found: {MODEL_PATH}. Keep full_CNN_model.h5 in MLND-Capstone folder."
            )
        _model = load_model(str(MODEL_PATH))
    return _model


def resize_image(arr, size_hw):
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


def build_lane_overlay(frame: np.ndarray, smoother: LaneSmoother, alpha: float = 0.8) -> np.ndarray:
    model = get_model()

    small_img = resize_image(frame, (80, 160))

    if small_img.ndim == 2:
        small_img = np.stack([small_img] * 3, axis=-1)

    if small_img.shape[-1] == 4:
        small_img = small_img[:, :, :3]

    input_batch = np.expand_dims(small_img, axis=0)

    prediction = model.predict(input_batch, verbose=0)[0]

    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[:, :, 0]

    prediction = prediction * 255.0
    avg_fit = smoother.smooth(prediction)

    blanks = np.zeros_like(avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, avg_fit.astype(np.uint8), blanks))

    height, width = frame.shape[:2]
    lane_image = resize_image(lane_drawn, (height, width))

    if lane_image.ndim == 2:
        lane_image = np.stack([lane_image] * 3, axis=-1)

    output = cv2.addWeighted(frame, 1.0, lane_image, alpha, 0)
    return output


def process_video(input_path: str, output_path: str) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    smoother = LaneSmoother(window_size=5)
    started_at = time.time()

    clip = VideoFileClip(str(input_file))
    processed_clip = clip.fl_image(lambda frame: build_lane_overlay(frame, smoother))
    processed_clip.write_videofile(str(output_file), audio=False, logger=None)

    try:
        duration = float(clip.duration or 0)
        fps = float(clip.fps or 0)
        frame_count = int(duration * fps) if duration and fps else 0
    finally:
        processed_clip.close()
        clip.close()

    return {
        "input_video": input_file.name,
        "output_video": output_file.name,
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "approx_frames": frame_count,
        "processing_seconds": round(time.time() - started_at, 2),
    }