import base64

import cv2
import numpy as np

from tests.conftest import CNN_AGE, EFFICIENTNET_AGE


def _decode_png(b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)


def test_health_reports_loaded_models(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["models"] == {"efficientnet_diverse": False, "efficientnet": True, "cnn_gan": True, "cnn": True}
    assert body["generator"] is True and body["face_detector"] is True
    # No metrics.json -> first model in preference order
    assert body["default_model"] == "efficientnet"


def test_health_without_generator(client_without_generator):
    body = client_without_generator.get("/api/health").json()
    assert body["generator"] is False
    assert body["models"]["cnn"] is True and body["models"]["efficientnet"] is False
    assert body["default_model"] == "cnn"


def test_predict_default_model(client, face_bytes):
    r = client.post("/api/predict", files={"file": ("face.jpg", face_bytes, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "efficientnet"
    assert (body["image_width"], body["image_height"]) == (960, 1200)
    assert len(body["faces"]) == 1
    face = body["faces"][0]
    assert face["age"] == EFFICIENTNET_AGE
    assert set(face["bbox"]) == {"x", "y", "w", "h"}
    assert _decode_png(face["crop"]).shape == (128, 128, 3)


def test_predict_grayscale_model_returns_grayscale_crop(client, face_bytes):
    r = client.post("/api/predict?model=cnn", files={"file": ("face.jpg", face_bytes, "image/jpeg")})
    assert r.status_code == 200, r.text
    face = r.json()["faces"][0]
    assert face["age"] == CNN_AGE
    assert _decode_png(face["crop"]).shape == (128, 128)


def test_predict_no_face_returns_422(client, blank_png):
    r = client.post("/api/predict", files={"file": ("blank.png", blank_png, "image/png")})
    assert r.status_code == 422
    assert "No face" in r.json()["detail"]


def test_predict_rejects_non_image(client):
    r = client.post("/api/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_rejects_undecodable_image(client):
    r = client.post("/api/predict", files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400


def test_predict_rejects_large_file(client):
    big = b"\xff" * (1024 * 1024 + 10)  # test client is configured with a 1 MB limit
    r = client.post("/api/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_predict_unknown_model(client, face_bytes):
    r = client.post("/api/predict?model=resnet", files={"file": ("face.jpg", face_bytes, "image/jpeg")})
    assert r.status_code == 400


def test_generate_returns_requested_count(client):
    r = client.post("/api/generate", json={"age": 30, "count": 3, "seed": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed"] == 7 and len(body["images"]) == 3
    assert _decode_png(body["images"][0]).shape == (128, 128)


def test_generate_is_reproducible_with_seed(client):
    a = client.post("/api/generate", json={"age": 50, "count": 2, "seed": 123}).json()
    b = client.post("/api/generate", json={"age": 50, "count": 2, "seed": 123}).json()
    assert a["images"] == b["images"]


def test_generate_validates_input(client):
    assert client.post("/api/generate", json={"age": 0}).status_code == 422
    assert client.post("/api/generate", json={"age": 30, "count": 9}).status_code == 422


def test_generate_without_generator_returns_503(client_without_generator):
    r = client_without_generator.post("/api/generate", json={"age": 30})
    assert r.status_code == 503


def test_model_info(client):
    body = client.get("/api/model-info").json()
    assert [m["name"] for m in body["models"]] == ["efficientnet_diverse", "efficientnet", "cnn_gan", "cnn"]
    assert body["gan"]["loaded"] is True
    assert body["preprocessing"]
