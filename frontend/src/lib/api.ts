// Relative by default so the nginx /api proxy works; override with VITE_API_BASE for a separate API host
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export const MAX_UPLOAD_MB = 5;

export type ModelName = "efficientnet" | "cnn_gan" | "cnn";

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

export const api = {
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

export const pngSrc = (b64: string) => `data:image/png;base64,${b64}`;

export const MODEL_LABELS: Record<ModelName, string> = {
  efficientnet: "EfficientNetB0",
  cnn_gan: "CNN + GAN",
  cnn: "CNN",
};
