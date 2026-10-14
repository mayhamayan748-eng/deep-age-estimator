"""Loads every model once at startup and runs inference."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services import preprocessing as pp

logger = logging.getLogger(__name__)

# name -> (file, label). Order is the fallback preference when no metrics.json is present.
AGE_MODELS: dict[str, tuple[str, str]] = {
    "efficientnet": ("age_efficientnet.keras", "EfficientNetB0 · RGB · transfer learning"),
    "cnn_gan": ("age_cnn_gan.keras", "CNN + GAN augmentation · grayscale"),
    "cnn": ("age_cnn.keras", "CNN · grayscale"),
}
GENERATOR_FILE = "gan_generator.keras"
FACE_DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
BUNDLED_FACE_DETECTOR = Path(__file__).resolve().parent.parent / "assets" / FACE_DETECTOR_FILE
METRICS_FILE = "metrics.json"
NOISE_DIM = 100


class InvalidImageError(Exception):
    pass


class NoFaceError(Exception):
    pass


class ModelUnavailableError(Exception):
    pass


@dataclass
class FaceResult:
    bbox: pp.BBox
    age: float
    crop_png: str


@dataclass
class ModelRegistry:
    models_dir: Path
    age_models: dict = field(default_factory=dict)
    generator: object | None = None
    face_detector: Path | None = None
    metrics: dict = field(default_factory=dict)
    default_model: str | None = None

    def load(self, preferred_default: str = "") -> None:
        import keras  # imported lazily: TensorFlow startup is slow

        for name, (filename, _) in AGE_MODELS.items():
            path = self.models_dir / filename
            if not path.exists():
                logger.warning("Age model %s not found at %s", name, path)
                continue
            try:
                model = keras.saving.load_model(path, compile=False)
                channels = model.input_shape[-1]
                model(np.zeros((1, pp.IMG_SIZE, pp.IMG_SIZE, channels), np.float32), training=False)  # warm-up
                self.age_models[name] = model
                logger.info("Loaded age model %s (%d channel input)", name, channels)
            except Exception:
                logger.exception("Failed to load age model %s", name)

        gen_path = self.models_dir / GENERATOR_FILE
        if gen_path.exists():
            try:
                self.generator = keras.saving.load_model(gen_path, compile=False)
                logger.info("Loaded GAN generator")
            except Exception:
                logger.exception("Failed to load GAN generator")
        else:
            logger.warning("GAN generator not found at %s — /api/generate disabled", gen_path)

        # A copy in models/ wins; otherwise use the one bundled with the backend
        for det_path in (self.models_dir / FACE_DETECTOR_FILE, BUNDLED_FACE_DETECTOR):
            if det_path.exists():
                self.face_detector = det_path
                break
        else:
            logger.warning("Face detector %s not found — /api/predict disabled", FACE_DETECTOR_FILE)

        metrics_path = self.models_dir / METRICS_FILE
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.default_model = self._pick_default(preferred_default)

    def _pick_default(self, preferred: str) -> str | None:
        if preferred in self.age_models:
            return preferred
        val_mae = {
            name: m["val_mae"]
            for name, m in self.metrics.get("models", {}).items()
            if name in self.age_models and "val_mae" in m
        }
        if val_mae:
            return min(val_mae, key=val_mae.get)
        return next(iter(self.age_models), None)

    @property
    def ready(self) -> bool:
        return bool(self.age_models) and self.face_detector is not None

    def predict(self, image_bytes: bytes, model_name: str | None = None) -> tuple[str, tuple[int, int], list[FaceResult]]:
        name = model_name or self.default_model
        if name is None:
            raise ModelUnavailableError("No age model is loaded")
        if name not in self.age_models:
            raise ModelUnavailableError(f"Model '{name}' is not loaded")
        if self.face_detector is None:
            raise ModelUnavailableError("Face detector is not available")

        img = pp.decode_image(image_bytes)
        if img is None:
            raise InvalidImageError("Could not decode the image")

        boxes = pp.detect_faces(img, self.face_detector)
        if not boxes:
            raise NoFaceError("No face detected in the image")

        model = self.age_models[name]
        channels = model.input_shape[-1]
        crops, bboxes = [], []
        for box in boxes:
            face, bbox = pp.crop_with_margin(img, box)
            crops.append(pp.to_model_input(face, channels))
            bboxes.append(bbox)

        batch = np.stack(crops).astype(np.float32)
        preds = np.asarray(model(batch, training=False)).ravel()
        faces = [
            FaceResult(bbox=b, age=round(pp.denormalize_age(p), 1), crop_png=pp.encode_png_base64(c))
            for b, p, c in zip(bboxes, preds, crops)
        ]
        h, w = img.shape[:2]
        return name, (w, h), faces

    def generate(self, age: int, count: int, seed: int | None) -> tuple[int, list[str]]:
        if self.generator is None:
            raise ModelUnavailableError("GAN generator is not available")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((count, NOISE_DIM)).astype(np.float32)
        ages = np.full((count, 1), pp.normalize_age(age), dtype=np.float32)
        fake = np.asarray(self.generator([noise, ages], training=False))
        return seed, [pp.encode_png_base64(pp.generator_output_to_uint8(img)) for img in fake]
