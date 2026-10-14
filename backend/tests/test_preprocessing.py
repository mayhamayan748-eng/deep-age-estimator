import cv2
import numpy as np

from app.services import preprocessing as pp
from app.services.models import BUNDLED_FACE_DETECTOR
from tests.conftest import FACE_IMAGE


def test_decode_matches_cv2_imread():
    decoded = pp.decode_image(FACE_IMAGE.read_bytes())
    np.testing.assert_array_equal(decoded, cv2.imread(str(FACE_IMAGE)))


def test_decode_invalid_bytes_returns_none():
    assert pp.decode_image(b"not an image") is None
    assert pp.decode_image(b"") is None


def test_grayscale_input_matches_notebook_pipeline():
    img = cv2.imread(str(FACE_IMAGE))
    expected = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2GRAY)[..., None]
    out = pp.to_model_input(img, channels=1)
    assert out.shape == (128, 128, 1) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)


def test_rgb_input_matches_notebook_pipeline():
    img = cv2.imread(str(FACE_IMAGE))
    expected = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(pp.to_model_input(img, channels=3), expected)


def test_batch_is_float32_in_0_255():
    batch = pp.to_batch(np.full((128, 128, 1), 255, np.uint8))
    assert batch.shape == (1, 128, 128, 1) and batch.dtype == np.float32
    assert batch.max() == 255.0


def test_crop_adds_margins_and_clips_to_image():
    img = np.zeros((200, 300, 3), np.uint8)
    _, bbox = pp.crop_with_margin(img, np.array([100, 50, 80, 100], np.float32), (0.25, 0.25, 0.25, 0.25))
    assert (bbox.x, bbox.y, bbox.w, bbox.h) == (80, 25, 120, 150)
    crop, bbox = pp.crop_with_margin(img, np.array([0, 0, 100, 100], np.float32), (0.25, 0.25, 0.25, 0.25))
    assert (bbox.x, bbox.y) == (0, 0) and crop.shape[:2] == (125, 125)


def test_default_margins_match_utkface_framing():
    img = np.zeros((1000, 1000, 3), np.uint8)
    _, bbox = pp.crop_with_margin(img, np.array([400, 300, 100, 120], np.float32))
    # 9% left, 0% top, 8% right, 2% bottom of the box size
    assert (bbox.x, bbox.y, bbox.w, bbox.h) == (391, 300, 117, 122)


def test_detects_one_face_in_portrait():
    boxes = pp.detect_faces(cv2.imread(str(FACE_IMAGE)), BUNDLED_FACE_DETECTOR)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    # The face is in the upper-right part of the 960x1200 photo
    assert 450 < x + w / 2 < 750 and 200 < y + h / 2 < 500


def test_detection_on_large_image_returns_original_coordinates():
    img = cv2.imread(str(FACE_IMAGE))
    big = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))
    small_box = pp.detect_faces(img, BUNDLED_FACE_DETECTOR)[0]
    big_box = pp.detect_faces(big, BUNDLED_FACE_DETECTOR)[0]
    np.testing.assert_allclose(big_box, small_box * 3, rtol=0.1)


def test_no_face_in_blank_image():
    assert pp.detect_faces(np.full((300, 300, 3), 127, np.uint8), BUNDLED_FACE_DETECTOR) == []


def test_age_normalization_round_trip():
    assert pp.denormalize_age(0.5) == 45.0
    assert pp.normalize_age(90) == 1.0


def test_generator_output_scaling():
    out = pp.generator_output_to_uint8(np.array([-1.0, 0.0, 1.0], np.float32))
    np.testing.assert_array_equal(out, [0, 127, 255])
