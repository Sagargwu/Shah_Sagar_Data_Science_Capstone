import numpy as np
import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model


def imresize(arr, size):
    """
    Safe imresize using PIL.
    - Accepts uint8 or float arrays
    - If float, assumes either [0,1] or [0,255] and converts to uint8
    """
    arr = np.asarray(arr)

    # Convert float arrays to uint8 for PIL
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # If grayscale with shape (H,W,1), squeeze it
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    img = Image.fromarray(arr)

    # If size is tuple/list -> treat as (H,W,channels) or (H,W)
    if isinstance(size, (tuple, list)):
        h, w = int(size[0]), int(size[1])
        img = img.resize((w, h), resample=Image.BILINEAR)
        return np.array(img)

    # scalar scaling (percent or ratio)
    scale = size / 100.0 if size > 1 else size
    w0, h0 = img.size
    img = img.resize((int(w0 * scale), int(h0 * scale)), resample=Image.BILINEAR)
    return np.array(img)


class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    # Resize frame for model input (80x160)
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)[None, :, :, :]

    # Predict lane mask (model outputs 0..1). Scale to 0..255 for visualization.
    prediction = model.predict(small_img, verbose=0)[0] * 255.0

    # Temporal smoothing over last 5 frames
    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]
    lanes.avg_fit = np.mean(np.array(lanes.recent_fit), axis=0)

    # Build green overlay mask (RGB)
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit.astype(np.uint8), blanks))

    # Resize overlay to match the current frame size
    h, w = image.shape[:2]
    lane_image = imresize(lane_drawn, (h, w, 3))

    # Blend overlay with original frame
    result = cv2.addWeighted(image, 1.0, lane_image, 1.0, 0)

    return result


if __name__ == "__main__":
    # Load model
    model = load_model("full_CNN_model.h5")
    lanes = Lanes()

    # Input/Output videos (must exist in the SAME folder as this script)
    input_video = "challenge.mp4"
    output_video = "challenge_lanes.mp4"

    clip1 = VideoFileClip(input_video)
    vid_clip = clip1.fl_image(road_lines)
    vid_clip.write_videofile(output_video, audio=False)