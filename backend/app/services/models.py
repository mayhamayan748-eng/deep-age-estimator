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
    "efficientnet_diverse": ("age_efficientnet_diverse.keras", "EfficientNetB0 + FairFace · RGB · diverse training data"),
    "efficientnet": ("age_efficientnet.keras", "EfficientNetB0 · RGB · transfer learning"),
    "cnn_gan": ("age_cnn_gan.keras", "CNN + GAN augmentation · grayscale"),
    "cnn": ("age_cnn.keras", "CNN · grayscale"),
}
# Weighted average of two complementary models (weights chosen on the UTKFace validation set)
ENSEMBLE_NAME = "ensemble"
ENSEMBLE_WEIGHTS = {"efficientnet_diverse": 0.8, "cnn_gan": 0.2}
ENSEMBLE_LABEL = "Ensemble · 80% EfficientNet + FairFace, 20% CNN + GAN"
MODEL_LABELS = {ENSEMBLE_NAME: ENSEMBLE_LABEL, **{name: label for name, (_, label) in AGE_MODELS.items()}}
# Optional monotonic correction of the ensemble's output, fitted on UTKFace validation (x = predicted, y = corrected)
CALIBRATION_FILE = "calibration.json"
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
    calibration: dict | None = None
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
        calibration_path = self.models_dir / CALIBRATION_FILE
        if calibration_path.exists():
            self.calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            logger.info("Loaded ensemble calibration (%d points)", len(self.calibration["x"]))

        self.default_model = self._pick_default(preferred_default)

    def _pick_default(self, preferred: str) -> str | None:
        if preferred in self.model_names:
            return preferred
        val_mae = {
            name: m["val_mae"]
            for name, m in self.metrics.get("models", {}).items()
            if name in self.model_names and "val_mae" in m
        }
        if val_mae:
            return min(val_mae, key=val_mae.get)
        return next(iter(self.model_names), None)

    @property
    def model_names(self) -> list[str]:
        """Loaded models, plus the ensemble when all its members are loaded."""
        names = [n for n in AGE_MODELS if n in self.age_models]
        if all(m in self.age_models for m in ENSEMBLE_WEIGHTS):
            names.insert(0, ENSEMBLE_NAME)
        return names

    @property
    def ready(self) -> bool:
        return bool(self.age_models) and self.face_detector is not None

    def _years(self, name: str, faces_bgr: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Ages for each face from one model, averaged with its horizontally flipped copy (test-time augmentation)."""
        model = self.age_models[name]
        inputs = [pp.to_model_input(face, model.input_shape[-1]) for face in faces_bgr]
        batch = np.stack(inputs).astype(np.float32)
        preds = np.asarray(model(np.concatenate([batch, batch[:, :, ::-1]]), training=False)).ravel()
        n = len(inputs)
        return pp.denormalize_age((preds[:n] + preds[n:]) / 2), inputs

    def predict(self, image_bytes: bytes, model_name: str | None = None) -> tuple[str, tuple[int, int], list[FaceResult]]:
        name = model_name or self.default_model
        if name is None:
            raise ModelUnavailableError("No age model is loaded")
        if name not in self.model_names:
            raise ModelUnavailableError(f"Model '{name}' is not loaded")
        if self.face_detector is None:
            raise ModelUnavailableError("Face detector is not available")

        img = pp.decode_image(image_bytes)
        if img is None:
            raise InvalidImageError("Could not decode the image")

        boxes = pp.detect_faces(img, self.face_detector)
        if not boxes:
            raise NoFaceError("No face detected in the image")

        faces_bgr, bboxes = [], []
        for box in boxes:
            face, bbox = pp.crop_with_margin(img, box)
            faces_bgr.append(face)
            bboxes.append(bbox)

        if name == ENSEMBLE_NAME:
            ages, crops = 0.0, None
            for member, weight in ENSEMBLE_WEIGHTS.items():
                years, inputs = self._years(member, faces_bgr)
                ages = ages + weight * years
                crops = crops or inputs  # show the first member's input (RGB)
            if self.calibration:
                ages = np.interp(ages, self.calibration["x"], self.calibration["y"])
        else:
            ages, crops = self._years(name, faces_bgr)

        faces = [
            FaceResult(bbox=b, age=round(float(a), 1), crop_png=pp.encode_png_base64(c))
            for b, a, c in zip(bboxes, ages, crops)
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
