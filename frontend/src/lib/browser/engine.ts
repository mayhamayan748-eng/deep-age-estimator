/**
 * In-browser implementation of the API (same responses as the FastAPI backend), so the site can be
 * hosted as static files. Mirrors backend/app/services/models.py and preprocessing.py.
 */
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import type { FacePrediction, GenerateResponse, Health, ModelInfo, ModelName, PredictResponse } from "../api";
import { bgrToGray, bgrToRgb, crop, decodeImage, encodePngBase64, resizeLinear, roundHalfEven, type Image } from "./imageops";
import { detectFaces, type Box } from "./yunet";

ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
// Static hosts don't send the COOP/COEP headers that multi-threaded WASM needs
ort.env.wasm.numThreads = 1;

const MODEL_BASE = (import.meta.env.VITE_MODEL_BASE ?? `${import.meta.env.BASE_URL}models/`).replace(/\/?$/, "/");

// Same constants as backend/app/services/preprocessing.py
const IMG_SIZE = 128;
const MAX_AGE = 90;
const FACE_MARGINS = [0.09, 0.0, 0.08, 0.02] as const;
const FACE_SCORE_THRESHOLD = 0.7;
const MAX_DETECTION_SIDE = 1280;
const NOISE_DIM = 100;

const MODEL_FILES: Record<ModelName, { file: string; label: string; channels: 1 | 3 }> = {
  efficientnet_diverse: { file: "age_efficientnet_diverse.onnx", label: "EfficientNetB0 + FairFace · RGB · diverse training data", channels: 3 },
  efficientnet: { file: "age_efficientnet.onnx", label: "EfficientNetB0 · RGB · transfer learning", channels: 3 },
  cnn_gan: { file: "age_cnn_gan.onnx", label: "CNN + GAN augmentation · grayscale", channels: 1 },
  cnn: { file: "age_cnn.onnx", label: "CNN · grayscale", channels: 1 },
};

class BrowserError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

// Each model is downloaded once, on first use
const sessions = new Map<string, Promise<ort.InferenceSession>>();
function session(file: string): Promise<ort.InferenceSession> {
  if (!sessions.has(file)) {
    const s = ort.InferenceSession.create(`${MODEL_BASE}${file}`, { executionProviders: ["wasm"] });
    s.catch(() => sessions.delete(file));
    sessions.set(file, s);
  }
  return sessions.get(file)!;
}

interface Metrics {
  dataset: Record<string, unknown>;
  best_model: ModelName;
  gan: Record<string, unknown>;
  fairface_test?: ModelInfo["fairface_test"];
  models: Record<ModelName, { val_mae: number; test_mae: number; test_rmse: number; mae_by_age_group: Record<string, number> }>;
}
let metricsPromise: Promise<Metrics> | null = null;
const metrics = (): Promise<Metrics> =>
  (metricsPromise ??= fetch(`${MODEL_BASE}metrics.json`).then((r) => r.json() as Promise<Metrics>));

async function health(): Promise<Health> {
  const m = await metrics();
  // Start downloading the detector and the default model while the visitor reads the page
  void session("face_detection_yunet.onnx").catch(() => {});
  void session(MODEL_FILES[m.best_model].file).catch(() => {});
  return {
    status: "ok",
    models: { efficientnet_diverse: true, efficientnet: true, cnn_gan: true, cnn: true },
    generator: true,
    face_detector: true,
    default_model: m.best_model,
  };
}

async function modelInfo(): Promise<ModelInfo> {
  const m = await metrics();
  return {
    dataset: m.dataset,
    preprocessing: [
      "Face detection with OpenCV YuNet (score ≥ 0.7)",
      "Crop tightly around the detected face, matching how UTKFace images are framed",
      "Resize to 128×128 (grayscale for the CNNs, RGB for EfficientNet)",
      "Pixels kept in 0-255; each model rescales internally",
      "Predicted value × 90 = age in years",
      "Runs entirely in your browser (ONNX Runtime Web); the photo never leaves your device",
    ],
    models: (Object.keys(MODEL_FILES) as ModelName[]).map((name) => ({
      name,
      label: MODEL_FILES[name].label,
      loaded: true,
      input: MODEL_FILES[name].channels === 3 ? "128×128 RGB, 0-255" : "128×128 grayscale, 0-255",
      val_mae: m.models[name]?.val_mae ?? null,
      test_mae: m.models[name]?.test_mae ?? null,
      test_rmse: m.models[name]?.test_rmse ?? null,
      mae_by_age_group: m.models[name]?.mae_by_age_group ?? {},
    })),
    default_model: m.best_model,
    gan: { ...m.gan, loaded: true },
    fairface_test: m.fairface_test,
  };
}

async function findFaces(img: Image): Promise<Box[]> {
  const { width: w, height: h } = img;
  const scale = Math.min(1, MAX_DETECTION_SIDE / Math.max(h, w));
  const small = scale < 1 ? resizeLinear(img, roundHalfEven(w * scale), roundHalfEven(h * scale)) : img;
  const boxes = await detectFaces(await session("face_detection_yunet.onnx"), ort.Tensor, small, FACE_SCORE_THRESHOLD);
  return boxes
    .map((b) => b.map((v) => v / scale) as Box)
    .sort((a, b) => b[2] * b[3] - a[2] * a[3]);
}

async function predict(blob: Blob, model?: ModelName): Promise<PredictResponse> {
  const name = model ?? (await metrics()).best_model;
  const spec = MODEL_FILES[name];
  if (!spec) throw new BrowserError(400, `Unknown or unavailable model '${model}'`);

  const img = await decodeImage(blob);
  if (!img) throw new BrowserError(400, "Could not decode the image");

  const boxes = await findFaces(img);
  if (boxes.length === 0) throw new BrowserError(422, "No face detected in the image");

  const [left, top, right, bottom] = FACE_MARGINS;
  const inputs: Image[] = [];
  const bboxes = boxes.map(([x, y, fw, fh]) => {
    const x0 = Math.trunc(Math.max(0, x - left * fw)), y0 = Math.trunc(Math.max(0, y - top * fh));
    const x1 = Math.trunc(Math.min(img.width, x + fw + right * fw)), y1 = Math.trunc(Math.min(img.height, y + fh + bottom * fh));
    const face = resizeLinear(crop(img, x0, y0, x1, y1), IMG_SIZE, IMG_SIZE);
    inputs.push(spec.channels === 1 ? bgrToGray(face) : bgrToRgb(face));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  });

  const batch = new Float32Array(inputs.length * IMG_SIZE * IMG_SIZE * spec.channels);
  inputs.forEach((m, i) => batch.set(m.data, i * m.data.length));
  const sess = await session(spec.file);
  const out = await sess.run({
    [sess.inputNames[0]]: new ort.Tensor("float32", batch, [inputs.length, IMG_SIZE, IMG_SIZE, spec.channels]),
  });
  const preds = out[sess.outputNames[0]].data as Float32Array;

  const faces: FacePrediction[] = await Promise.all(
    bboxes.map(async (bbox, i) => ({
      bbox,
      age: Math.round(preds[i] * MAX_AGE * 10) / 10,
      crop: await encodePngBase64(inputs[i]),
    })),
  );
  return { model: name, image_width: img.width, image_height: img.height, faces };
}

/** Seeded standard normal noise (mulberry32 + Box-Muller). */
function normalNoise(n: number, seed: number): Float32Array {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - uniform()));
    const theta = 2 * Math.PI * uniform();
    out[i] = r * Math.cos(theta);
    if (i + 1 < n) out[i + 1] = r * Math.sin(theta);
  }
  return out;
}

async function generate(age: number, count: number, seed?: number): Promise<GenerateResponse> {
  const usedSeed = seed ?? Math.floor(Math.random() * 2 ** 31);
  const sess = await session("gan_generator.onnx");
  const out = await sess.run({
    noise: new ort.Tensor("float32", normalNoise(count * NOISE_DIM, usedSeed), [count, NOISE_DIM]),
    age: new ort.Tensor("float32", new Float32Array(count).fill(age / MAX_AGE), [count, 1]),
  });
  const pixels = out[sess.outputNames[0]].data as Float32Array;
  const size = IMG_SIZE * IMG_SIZE;
  const images = await Promise.all(
    Array.from({ length: count }, (_, i) => {
      const data = new Uint8Array(size);
      // generator tanh output in [-1, 1] -> uint8, as np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)
      for (let p = 0; p < size; p++) data[p] = Math.min(255, Math.max(0, (pixels[i * size + p] + 1) * 127.5));
      return encodePngBase64({ data, width: IMG_SIZE, height: IMG_SIZE, channels: 1 });
    }),
  );
  return { age, seed: usedSeed, images };
}

export const browserApi = { health, modelInfo, predict, generate, BrowserError };
