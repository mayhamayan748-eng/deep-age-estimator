"""Flip test-time augmentation, the ensemble and its calibration."""
import json
import shutil

import keras
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.models import ModelRegistry
from tests.conftest import ENSEMBLE_AGE


def test_flip_tta_averages_original_and_mirrored(tmp_path):
    # Stand-in model that only looks at the left half: mean brightness of the left half, in years
    inp = keras.Input((128, 128, 1))
    x = keras.layers.Cropping2D(((0, 0), (0, 64)))(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(1, kernel_initializer=keras.initializers.Constant(1 / 255), bias_initializer="zeros")(x)
    keras.Model(inp, out).save(tmp_path / "age_cnn.keras")
    reg = ModelRegistry(tmp_path)
    reg.load()

    face = np.zeros((200, 200, 3), np.uint8)
    face[:, 100:] = 255  # left half black, right half white
    years, _ = reg._years("cnn", [face])
    # Original: left half black -> 0 years; mirrored: left half white -> 90 years; averaged -> 45
    assert abs(years[0] - 45) < 1


def test_calibration_is_applied_to_the_ensemble(models_dir, tmp_path, face_bytes):
    d = tmp_path / "models"
    shutil.copytree(models_dir, d)
    (d / "calibration.json").write_text(json.dumps({"x": [0, 90], "y": [10, 100]}))  # +10 years everywhere
    with TestClient(create_app(Settings(models_dir=d))) as c:
        files = {"file": ("face.jpg", face_bytes, "image/jpeg")}
        assert c.post("/api/predict", files=files).json()["faces"][0]["age"] == ENSEMBLE_AGE + 10
        # single models are not calibrated
        assert c.post("/api/predict?model=cnn", files=files).json()["faces"][0]["age"] == 30.0
