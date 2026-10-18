"""Image preprocessing — mirrors `gradute_preoject.ipynb` exactly.

Notebook reference:
- Training images: cv2.imread (BGR) -> cv2.resize(128, 128) -> BGR2GRAY (CNN) or BGR2RGB (EfficientNet).
- Pixels stay in 0-255 as float32; every model has its own Rescaling layer inside.
- Ages are normalized by MAX_AGE = 90, so age = model output * 90.
- Uploaded photos: YuNet face detection (score_threshold=0.7), crop with FACE_MARGINS, then as above.

FACE_MARGINS reproduce UTKFace's framing. Measured by running YuNet on 1,000 UTKFace images: the
detected box nearly fills the image, with median margins of 9% (left), 0% (top), 8% (right) and
2% (bottom) of the box size. A looser crop shows the model a smaller face than it was trained on.

The only addition is that detection may run on a downscaled copy of very large photos;
the crop itself is always taken from the full-resolution image.
"""
import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 128
MAX_AGE = 90
# (left, top, right, bottom) as a fraction of the detected box width/height
FACE_MARGINS = (0.09, 0.0, 0.08, 0.02)
FACE_SCORE_THRESHOLD = 0.7
MAX_DETECTION_SIDE = 1280

# YuNet logs a warning on every detector creation with OpenCV 5
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int


def decode_image(data: bytes) -> np.ndarray | None:
    """Bytes -> BGR image (same channel order as cv2.imread), or None if undecodable."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def detect_faces(img: np.ndarray, detector_path: Path) -> list[np.ndarray]:
    """YuNet face boxes [x, y, w, h] in original-image pixels, largest first."""
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DETECTION_SIDE / max(h, w))
    small = cv2.resize(img, (round(w * scale), round(h * scale))) if scale < 1.0 else img

    sh, sw = small.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (sw, sh), score_threshold=FACE_SCORE_THRESHOLD)
    _, faces = detector.detect(small)
    if faces is None:
        return []
    boxes = [f[:4] / scale for f in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_with_margin(
    img: np.ndarray, box: np.ndarray, margins: tuple[float, float, float, float] = FACE_MARGINS
) -> tuple[np.ndarray, BBox]:
    """Same crop as the notebook's crop_face(): expand the box by `margins`, clip to the image."""
    h, w = img.shape[:2]
    x, y, fw, fh = box
    left, top, right, bottom = margins
    x0, y0 = int(max(0, x - left * fw)), int(max(0, y - top * fh))
    x1, y1 = int(min(w, x + fw + right * fw)), int(min(h, y + fh + bottom * fh))
    return img[y0:y1, x0:x1], BBox(x0, y0, x1 - x0, y1 - y0)


def to_model_input(face_bgr: np.ndarray, channels: int) -> np.ndarray:
    """BGR crop -> (128, 128, channels) uint8 array, exactly as in training."""
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    if channels == 1:
        return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)[..., None]
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def to_batch(model_input: np.ndarray) -> np.ndarray:
    """(128, 128, C) uint8 -> (1, 128, 128, C) float32 in 0-255 (models rescale internally)."""
    return model_input[None].astype(np.float32)


def denormalize_age(normalized_age):
    """Works on a float or a numpy array."""
    return normalized_age * MAX_AGE


def normalize_age(age: float) -> float:
    return age / MAX_AGE


def generator_output_to_uint8(img: np.ndarray) -> np.ndarray:
    """Generator tanh output in [-1, 1] -> uint8 0-255 (notebook: (fake + 1) * 127.5)."""
    return np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)


def encode_png_base64(img: np.ndarray) -> str:
    # cv2 expects BGR for 3-channel images
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3:
        img = img[..., 0]
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
