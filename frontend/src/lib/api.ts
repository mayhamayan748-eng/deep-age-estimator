// Relative by default so the nginx /api proxy works; override with VITE_API_BASE for a separate API host
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export const MAX_UPLOAD_MB = 5;

export type ModelName = "ensemble" | "efficientnet_diverse" | "efficientnet" | "cnn_gan" | "cnn";

export interface Health {
  status: "ok" | "degraded";
  models: Record<ModelName, boolean>;
  generator: boolean;
  face_detector: boolean;
  default_model: ModelName | null;
}

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FacePrediction {
  bbox: BBox;
  age: number;
  crop: string;
}

export interface PredictResponse {
  model: ModelName;
  image_width: number;
  image_height: number;
  faces: FacePrediction[];
}

export interface GenerateResponse {
  age: number;
  seed: number;
  images: string[];
}

export interface ModelSummary {
  name: ModelName;
  label: string;
  loaded: boolean;
  input: string;
  val_mae: number | null;
  test_mae: number | null;
  test_rmse: number | null;
  mae_by_age_group: Record<string, number>;
}

export interface ModelInfo {
  dataset: Record<string, unknown>;
  preprocessing: string[];
  models: ModelSummary[];
  default_model: ModelName | null;
  gan: Record<string, unknown>;
  /** Per-group results on FairFace, before and after training on diverse data */
  fairface_test?: Record<string, FairFaceGroup>;
}

export interface FairFaceGroup {
  faces: number;
  outside_before: number;
  outside_after: number;
  in_range_before: number;
  in_range_after: number;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api${path}`, init);
  } catch {
    throw new ApiError(0, "Can't reach the server. Is the backend running?");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const serverApi = {
  health: () => request<Health>("/health"),
  modelInfo: () => request<ModelInfo>("/model-info"),
  predict: (file: Blob, model?: ModelName) => {
    const form = new FormData();
    form.append("file", file, file instanceof File ? file.name : "capture.jpg");
    return request<PredictResponse>(`/predict${model ? `?model=${model}` : ""}`, { method: "POST", body: form });
  },
  generate: (age: number, count: number, seed?: number) =>
    request<GenerateResponse>("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ age, count, seed }),
    }),
};

/** VITE_INFERENCE=browser: run the models in the visitor's browser (static hosting, no backend). */
export const IN_BROWSER = import.meta.env.VITE_INFERENCE === "browser";

// Loaded only in browser mode, so the server build doesn't ship onnxruntime
const engine = () => import("./browser/engine").then((m) => m.browserApi);

async function inBrowser<T>(call: (e: Awaited<ReturnType<typeof engine>>) => Promise<T>): Promise<T> {
  const e = await engine();
  try {
    return await call(e);
  } catch (err) {
    if (err instanceof e.BrowserError) throw new ApiError(err.status, err.message);
    throw new ApiError(0, "Couldn't load or run the model in this browser. Check your connection and try again.");
  }
}

export const api: typeof serverApi = IN_BROWSER
  ? {
      health: () => inBrowser((e) => e.health()),
      modelInfo: () => inBrowser((e) => e.modelInfo()),
      predict: (file, model) => inBrowser((e) => e.predict(file, model)),
      generate: (age, count, seed) => inBrowser((e) => e.generate(age, count, seed)),
    }
  : serverApi;

export const pngSrc = (b64: string) => `data:image/png;base64,${b64}`;

export const MODEL_LABELS: Record<ModelName, string> = {
  ensemble: "Ensemble",
  efficientnet_diverse: "EfficientNet + FairFace",
  efficientnet: "EfficientNetB0",
  cnn_gan: "CNN + GAN",
  cnn: "CNN",
};
