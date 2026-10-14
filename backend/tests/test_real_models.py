"""Sanity checks with the real trained models (skipped when ../models has none).

Set REAL_MODELS_DIR to test models stored elsewhere.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

REAL_MODELS = Path(os.environ.get("REAL_MODELS_DIR", Path(__file__).resolve().parents[2] / "models"))
ARMSTRONG_AGE = 38  # age in the 1969 NASA portrait used as test image

pytestmark = pytest.mark.skipif(not any(REAL_MODELS.glob("age_*.keras")), reason="trained models not present")


@pytest.fixture(scope="module")
def real_client():
    with TestClient(create_app(Settings(models_dir=REAL_MODELS))) as c:
        yield c


def test_real_models_predict_plausible_age(real_client, face_bytes):
    loaded = [m for m, ok in real_client.get("/api/health").json()["models"].items() if ok]
    for name in loaded:
        r = real_client.post(f"/api/predict?model={name}", files={"file": ("face.jpg", face_bytes, "image/jpeg")})
        assert r.status_code == 200, r.text
        age = r.json()["faces"][0]["age"]
        assert abs(age - ARMSTRONG_AGE) < 20, f"{name} predicted {age}"
