"""Tests run against tiny stand-in models with known outputs, so they need no trained weights.

`test_real_models.py` additionally checks the real models when they are present in ../models.
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

DATA_DIR = Path(__file__).parent / "data"
FACE_IMAGE = DATA_DIR / "face.jpg"  # NASA portrait of Neil Armstrong (public domain)

CNN_AGE = 30.0           # stand-in grayscale models always predict 30 years
EFFICIENTNET_AGE = 45.0  # stand-in RGB model always predicts 45 years
DIVERSE_AGE = 40.0       # stand-in diverse RGB model always predicts 40 years
ENSEMBLE_AGE = 0.8 * DIVERSE_AGE + 0.2 * CNN_AGE  # efficientnet_diverse 80% + cnn_gan 20% = 38


def _constant_age_model(channels: int, age: float) -> keras.Model:
    inp = keras.Input((128, 128, channels))
    x = keras.layers.GlobalAveragePooling2D()(inp)
    out = keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer=keras.initializers.Constant(age / 90))(x)
    return keras.Model(inp, out)


def _tiny_generator() -> keras.Model:
    noise = keras.Input((100,))
    age = keras.Input((1,))
    x = keras.layers.Concatenate()([noise, age])
    x = keras.layers.Dense(128 * 128, activation="tanh")(x)
    return keras.Model([noise, age], keras.layers.Reshape((128, 128, 1))(x))


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("models")
    _constant_age_model(1, CNN_AGE).save(d / "age_cnn.keras")
    _constant_age_model(1, CNN_AGE).save(d / "age_cnn_gan.keras")
    _constant_age_model(3, EFFICIENTNET_AGE).save(d / "age_efficientnet.keras")
    _constant_age_model(3, DIVERSE_AGE).save(d / "age_efficientnet_diverse.keras")
    _tiny_generator().save(d / "gan_generator.keras")
    return d


@pytest.fixture(scope="session")
def client(models_dir):
    app = create_app(Settings(models_dir=models_dir, max_upload_mb=1))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def client_without_generator(tmp_path_factory):
    d = tmp_path_factory.mktemp("models_no_gan")
    _constant_age_model(1, CNN_AGE).save(d / "age_cnn.keras")
    app = create_app(Settings(models_dir=d))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def face_bytes() -> bytes:
    return FACE_IMAGE.read_bytes()


@pytest.fixture
def blank_png() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.full((300, 300, 3), 127, np.uint8))
    return buf.tobytes()
