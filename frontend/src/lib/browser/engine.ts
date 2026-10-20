/**
 * In-browser implementation of the API (same responses as the FastAPI backend), so the site can be
 * hosted as static files. Mirrors backend/app/services/models.py and preprocessing.py.
 */
// onnxruntime-web 1.18 "wasm-core": the last release with single-threaded WebAssembly builds. Newer releases
// only ship multi-threaded builds, which declare shared WebAssembly memory; iOS Safari fails to allocate it
// ("RangeError: Out of memory"), so the models could not load on iPhones.
import * as ort from "onnxruntime-web/wasm-core";
// The package's exports map hides dist/, so the files are referenced by path; Vite emits them as assets
import simdWasmUrl from "../../../node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm?url";
import plainWasmUrl from "../../../node_modules/onnxruntime-web/dist/ort-wasm.wasm?url";
import type { FacePrediction, GenerateResponse, Health, ModelInfo, ModelName, PredictResponse } from "../api";
import { bgrToGray, bgrToRgb, crop, decodeImage, encodePngBase64, resizeLinear, roundHalfEven, type Image } from "./imageops";
import { detectFaces, type Box } from "./yunet";

// SIMD build where supported (iOS 16.4+), plain build otherwise
ort.env.wasm.wasmPaths = { "ort-wasm-simd.wasm": simdWasmUrl, "ort-wasm.wasm": plainWasmUrl };
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const MODEL_BASE = (import.meta.env.VITE_MODEL_BASE ?? `${import.meta.env.BASE_URL}models/`).replace(/\/?$/, "/");
// Static hosts send no cache headers. Data files get a per-build URL; model files get a URL tied to their
// content hash (metrics.json "file_versions", written by deploy_static.py), so a redeploy doesn't make
// phones download ~40 MB of unchanged models again.
let fileVersions: Record<string, string> = {};
const versioned = (file: string) => `${MODEL_BASE}${file}?v=${fileVersions[file] ?? __BUILD_ID__}`;

// Same constants as backend/app/services/preprocessing.py
const IMG_SIZE = 128;
const MAX_AGE = 90;
const FACE_MARGINS = [0.09, 0.0, 0.08, 0.02] as const;
const FACE_SCORE_THRESHOLD = 0.7;
const MAX_DETECTION_SIDE = 1280;
const NOISE_DIM = 100;

type SingleModel = Exclude<ModelName, "ensemble">;

const MODEL_FILES: Record<SingleModel, { file: string; label: string; channels: 1 | 3 }> = {
  efficientnet_diverse: { file: "age_efficientnet_diverse.onnx", label: "EfficientNetB0 + FairFace · RGB · diverse training data", channels: 3 },
  efficientnet: { file: "age_efficientnet.onnx", label: "EfficientNetB0 · RGB · transfer learning", channels: 3 },
  cnn_gan: { file: "age_cnn_gan.onnx", label: "CNN + GAN augmentation · grayscale", channels: 1 },
  cnn: { file: "age_cnn.onnx", label: "CNN · grayscale", channels: 1 },
};

// Same as ENSEMBLE_WEIGHTS in backend/app/services/models.py
const ENSEMBLE_WEIGHTS: [SingleModel, number][] = [["efficientnet_diverse", 0.8], ["cnn_gan", 0.2]];
const LABELS: Record<ModelName, string> = {
  ensemble: "Ensemble · 80% EfficientNet + FairFace, 20% CNN + GAN",
  efficientnet_diverse: MODEL_FILES.efficientnet_diverse.label,
  efficientnet: MODEL_FILES.efficientnet.label,
  cnn_gan: MODEL_FILES.cnn_gan.label,
  cnn: MODEL_FILES.cnn.label,
};

class BrowserError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

// Download progress across all model files, so the page can show it on slow (mobile) connections
export type DownloadProgress = { loaded: number; total: number; active: number };
const progress: DownloadProgress = { loaded: 0, total: 0, active: 0 };
const listeners = new Set<(p: DownloadProgress) => void>();
const notify = () => listeners.forEach((cb) => cb({ ...progress }));

function subscribeDownload(cb: (p: DownloadProgress) => void): () => void {
  listeners.add(cb);
  cb({ ...progress });
  return () => listeners.delete(cb);
}

async function download(file: string): Promise<Uint8Array> {
  const res = await fetch(versioned(file));
  if (!res.ok || !res.body) throw new Error(`Failed to download ${file} (HTTP ${res.status})`);
  const size = Number(res.headers.get("content-length")) || 0;
  progress.total += size;
  progress.active++;
  notify();
  const chunks: Uint8Array[] = [];
  let received = 0;
  try {
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      progress.loaded += value.length;
      notify();
    }
  } finally {
    progress.active--;
    if (size) progress.loaded += size - received; // keep totals consistent if the stream ended early
    notify();
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

// Each model is downloaded once, on first use
const sessions = new Map<string, Promise<ort.InferenceSession>>();
function session(file: string): Promise<ort.InferenceSession> {
  if (!sessions.has(file)) {
    const s = download(file).then((bytes) => ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] }));
    s.catch(() => sessions.delete(file));
    sessions.set(file, s);
  }
  return sessions.get(file)!;
}

interface Metrics {
  file_versions?: Record<string, string>;
  dataset: Record<string, unknown>;
  best_model: ModelName;
  gan: Record<string, unknown>;
  fairface_test?: ModelInfo["fairface_test"];
  models: Record<ModelName, { val_mae: number; test_mae: number; test_rmse: number; mae_by_age_group: Record<string, number> }>;
}
let metricsPromise: Promise<Metrics> | null = null;
const metrics = (): Promise<Metrics> =>
  (metricsPromise ??= fetch(versioned("metrics.json"))
    .then((r) => r.json() as Promise<Metrics>)
    .then((m) => {
      fileVersions = m.file_versions ?? {};
      return m;
    }));

// Optional monotonic correction of the ensemble output (calibration.json, same file the backend uses)
type Calibration = { x: number[]; y: number[] } | null;
let calibrationPromise: Promise<Calibration> | null = null;
const calibration = (): Promise<Calibration> =>
  (calibrationPromise ??= fetch(versioned("calibration.json"))
    .then((r) => (r.ok ? (r.json() as Promise<Calibration>) : null))
    .catch(() => null));

/** np.interp: piecewise-linear, clamped at both ends. */
function interp(v: number, xs: number[], ys: number[]): number {
  if (v <= xs[0]) return ys[0];
  if (v >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (xs[i] < v) i++;
  const t = (v - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

const modelFiles = (name: ModelName): string[] =>
  name === "ensemble" ? ENSEMBLE_WEIGHTS.map(([m]) => MODEL_FILES[m].file) : [MODEL_FILES[name].file];

async function health(): Promise<Health> {
  const m = await metrics();
  // Start downloading the detector and the default model while the visitor reads the page
  void session("face_detection_yunet.onnx").catch(() => {});
  for (const file of modelFiles(m.best_model)) void session(file).catch(() => {});
  return {
    status: "ok",
    models: { ensemble: true, efficientnet_diverse: true, efficientnet: true, cnn_gan: true, cnn: true },
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
      "Each model also sees the mirrored face; the two predictions are averaged",
      "Predicted value × 90 = age in years",
      "Runs entirely in your browser (ONNX Runtime Web); the photo never leaves your device",
    ],
    models: (Object.keys(LABELS) as ModelName[]).map((name) => ({
      name,
      label: LABELS[name],
      loaded: true,
      input: name === "ensemble"
        ? "128×128 RGB + grayscale, flip-averaged"
        : MODEL_FILES[name].channels === 3 ? "128×128 RGB, 0-255" : "128×128 grayscale, 0-255",
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

/** Mirror an image left-right (img[:, ::-1]). */
function flipHorizontal(img: Image): Image {
  const { width: w, height: h, channels: cn } = img;
  const out = new Uint8Array(img.data.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < cn; c++) out[(y * w + x) * cn + c] = img.data[(y * w + (w - 1 - x)) * cn + c];
  return { ...img, data: out };
}

/** Ages from one model, averaged with its prediction on the mirrored faces (test-time augmentation). */
async function modelYears(name: SingleModel, faces: Image[]): Promise<{ years: number[]; inputs: Image[] }> {
  const spec = MODEL_FILES[name];
  const inputs = faces.map((f) => (spec.channels === 1 ? bgrToGray(f) : bgrToRgb(f)));
  const all = [...inputs, ...inputs.map(flipHorizontal)];
  const batch = new Float32Array(all.length * IMG_SIZE * IMG_SIZE * spec.channels);
  all.forEach((m, i) => batch.set(m.data, i * m.data.length));
  const sess = await session(spec.file);
  const out = await sess.run({
    [sess.inputNames[0]]: new ort.Tensor("float32", batch, [all.length, IMG_SIZE, IMG_SIZE, spec.channels]),
  });
  const preds = out[sess.outputNames[0]].data as Float32Array;
  const n = inputs.length;
  return { years: inputs.map((_, i) => ((preds[i] + preds[n + i]) / 2) * MAX_AGE), inputs };
}

async function predict(blob: Blob, model?: ModelName): Promise<PredictResponse> {
  const name = model ?? (await metrics()).best_model;
  if (!(name in LABELS)) throw new BrowserError(400, `Unknown or unavailable model '${model}'`);

  const img = await decodeImage(blob);
  if (!img) throw new BrowserError(400, "Could not decode the image");

  const boxes = await findFaces(img);
  if (boxes.length === 0) throw new BrowserError(422, "No face detected in the image");

  const [left, top, right, bottom] = FACE_MARGINS;
  const facesBgr: Image[] = [];
  const bboxes = boxes.map(([x, y, fw, fh]) => {
    const x0 = Math.trunc(Math.max(0, x - left * fw)), y0 = Math.trunc(Math.max(0, y - top * fh));
    const x1 = Math.trunc(Math.min(img.width, x + fw + right * fw)), y1 = Math.trunc(Math.min(img.height, y + fh + bottom * fh));
    facesBgr.push(resizeLinear(crop(img, x0, y0, x1, y1), IMG_SIZE, IMG_SIZE));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  });

  let ages: number[];
  let shown: Image[];
  if (name === "ensemble") {
    ages = facesBgr.map(() => 0);
    shown = [];
    for (const [member, weight] of ENSEMBLE_WEIGHTS) {
      const { years, inputs } = await modelYears(member, facesBgr);
      years.forEach((y, i) => (ages[i] += weight * y));
      if (shown.length === 0) shown = inputs; // first member's input (RGB)
    }
    const cal = await calibration();
    if (cal) ages = ages.map((a) => interp(a, cal.x, cal.y));
  } else {
    ({ years: ages, inputs: shown } = await modelYears(name, facesBgr));
  }

  const faces: FacePrediction[] = await Promise.all(
    bboxes.map(async (bbox, i) => ({
      bbox,
      age: Math.round(ages[i] * 10) / 10,
      crop: await encodePngBase64(shown[i]),
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

export const browserApi = { health, modelInfo, predict, generate, subscribeDownload, BrowserError };
