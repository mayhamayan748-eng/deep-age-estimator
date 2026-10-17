"""Export the trained Keras models to ONNX for in-browser inference, and check the outputs match.

    python deploy/browser/export_onnx.py models/ models/onnx/ [UTKFace/]

Augmentation layers only act during training, so they are dropped from the exported graphs.
"""
import sys
from pathlib import Path

import keras
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

AGE_MODELS = ["age_cnn.keras", "age_cnn_gan.keras", "age_efficientnet.keras", "age_efficientnet_diverse.keras"]
OPSET = 17


def freeze_normalization(layer: keras.layers.Layer) -> None:
    """EfficientNet's Normalization keeps mean/variance as eager tensors, which tf2onnx would turn into
    graph inputs. As numpy arrays they are embedded as constants instead."""
    for sub in getattr(layer, "layers", []):
        freeze_normalization(sub)
    if isinstance(layer, keras.layers.Normalization):
        layer.mean = np.asarray(layer.mean)
        layer.variance = np.asarray(layer.variance)


def inference_only(model: keras.Model) -> keras.Model:
    """Same layers and weights, without the 'augmentation' block (age models are linear chains)."""
    freeze_normalization(model)
    inp = keras.Input(model.input_shape[1:])
    x = inp
    for layer in model.layers[1:]:
        if layer.name != "augmentation":
            x = layer(x, training=False)
    return keras.Model(inp, x)


def to_onnx(model: keras.Model, signature: list[tf.TensorSpec], out: Path) -> None:
    @tf.function(input_signature=signature)
    def serve(*inputs):
        return model(list(inputs) if len(inputs) > 1 else inputs[0], training=False)

    tf2onnx.convert.from_function(serve, input_signature=signature, opset=OPSET, output_path=str(out))


def max_diff(keras_model, onnx_path: Path, inputs: list[np.ndarray]) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {i.name: x for i, x in zip(sess.get_inputs(), inputs)}
    onnx_out = sess.run(None, feeds)[0]
    keras_out = np.asarray(keras_model(inputs if len(inputs) > 1 else inputs[0], training=False))
    return float(np.abs(onnx_out - keras_out).max())


def sample_images(utk_dir: Path | None, channels: int, n: int = 32) -> np.ndarray:
    import cv2

    if utk_dir is None:
        return np.random.default_rng(0).uniform(0, 255, (n, 128, 128, channels)).astype(np.float32)
    files = sorted(utk_dir.glob("*.jpg"))[:: max(1, len(list(utk_dir.glob("*.jpg"))) // n)][:n]
    imgs = []
    for f in files:
        img = cv2.resize(cv2.imread(str(f)), (128, 128))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None] if channels == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(imgs, dtype=np.float32)


def dynamic_yunet(src: Path, out: Path) -> None:
    """YuNet ships with a fixed 640x640 input. OpenCV re-infers shapes at runtime; onnxruntime needs the
    spatial dims (and the per-stride output lengths) declared symbolic to accept any padded size."""
    import onnx

    m = onnx.load(str(src))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param, dims[3].dim_param = "height", "width"
    for o in m.graph.output:
        o.type.tensor_type.shape.dim[1].dim_param = f"{o.name}_cells"
    del m.graph.value_info[:]
    onnx.save(m, str(out))


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    utk = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    dst.mkdir(parents=True, exist_ok=True)

    yunet = Path(__file__).resolve().parents[2] / "backend" / "app" / "assets" / "face_detection_yunet_2023mar.onnx"
    dynamic_yunet(yunet, dst / "face_detection_yunet.onnx")
    print("face_detection_yunet.onnx  (dynamic input size)")

    for name in AGE_MODELS:
        model = inference_only(keras.saving.load_model(src / name, compile=False))
        channels = model.input_shape[-1]
        out = dst / name.replace(".keras", ".onnx")
        to_onnx(model, [tf.TensorSpec([None, 128, 128, channels], tf.float32, name="image")], out)
        diff = max_diff(model, out, [sample_images(utk, channels)])
        print(f"{out.name:<26}{out.stat().st_size / 1e6:6.1f} MB   max |onnx - keras| = {diff * 90:.5f} years")

    gen = keras.saving.load_model(src / "gan_generator.keras", compile=False)
    out = dst / "gan_generator.onnx"
    to_onnx(gen, [tf.TensorSpec([None, 100], tf.float32, name="noise"), tf.TensorSpec([None, 1], tf.float32, name="age")], out)
    rng = np.random.default_rng(0)
    diff = max_diff(gen, out, [rng.standard_normal((8, 100)).astype(np.float32), rng.uniform(0, 1, (8, 1)).astype(np.float32)])
    print(f"{out.name:<26}{out.stat().st_size / 1e6:6.1f} MB   max |onnx - keras| = {diff * 127.5:.4f} gray levels")


if __name__ == "__main__":
    main()
