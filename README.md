# Deep Age Estimator

Estimate a person's age from a face photo, and generate faces of any age with a conditional GAN.
Graduation project by **Ayham Ayan**: a training notebook plus a full web app (FastAPI + React, packaged with Docker).

**Live demo: https://ayhamayan-deep-age-estimator.static.hf.space**. The models run in your browser, so photos
never leave your device.

![Predict section](docs/screenshots/predict.png)

## Results

Three models were trained on [UTKFace](https://susanqq.github.io/UTKFace/) (ages 0–90, 23,621 images: 17,006 train,
1,890 validation, 4,725 test). All numbers are on the same held-out test set.

| Model | Input | Test MAE | Test RMSE |
|---|---|---|---|
| CNN | 128×128 grayscale | 6.16 years | 8.12 |
| CNN + GAN augmentation | 128×128 grayscale | 5.83 years | 8.05 |
| EfficientNetB0 (transfer learning) | 128×128 RGB | 4.97 years | 7.07 |
| EfficientNetB0 + FairFace (see below) | 128×128 RGB | 4.83 years | 6.80 |
| **Ensemble** + flip + calibration (see below) | both | **4.65 years** | **6.60** |

The first version of the notebook reached 6.3 years. Faces added by the conditional GAN improve the same CNN by 0.33 years.
The web app uses the ensemble by default.

MAE by age group (years):

| Model | 0–12 | 13–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–90 |
|---|---|---|---|---|---|---|---|
| CNN | 8.63 | 5.81 | 3.91 | 5.40 | 6.85 | 8.24 | 8.68 |
| CNN + GAN | 4.28 | 4.71 | 3.74 | 5.79 | 7.55 | 8.98 | 10.14 |
| EfficientNetB0 | 2.22 | 4.92 | 3.69 | 5.59 | 6.76 | 7.44 | 7.37 |
| EfficientNetB0 + FairFace | 2.14 | 4.25 | 3.65 | 5.13 | 6.33 | 7.18 | 8.11 |
| Ensemble + flip + calibration | 1.69 | 4.93 | 3.67 | 5.00 | 5.93 | 7.10 | 7.37 |

### Reducing the error without retraining

Three inference-time techniques, applied by both the backend and the in-browser engine. Each choice was fitted on
the UTKFace validation set; the numbers below are on the test sets.

| | UTKFace test MAE | FairFace years outside range |
|---|---|---|
| EfficientNetB0 + FairFace | 4.83 | 2.32 |
| + **flip TTA**: average the prediction on the face and its mirror image | 4.81 | 2.28 |
| + **ensemble**: 80% EfficientNetB0 + FairFace, 20% CNN + GAN | 4.72 | 2.36 |
| + **isotonic calibration** of the ensemble output (`models/calibration.json`) | **4.65** | **2.35** |

- CNN + GAN is weaker alone (5.83) but adds information the EfficientNet lacks; adding the UTKFace-only EfficientNet
  to the ensemble instead improved UTKFace (4.62) but hurt diverse faces (FairFace 2.82), so it was left out.
- All models regress towards the mean: they overestimate children (+2 years) and underestimate people over 60
  (−6.7 years). Calibration halves the error for children (2.43 → 1.69) and reduces it for 60+ (8.13 → 7.37), at the
  cost of ages 13–29 (13–19: 4.13 → 4.93).
- Calibration improved the validation MAE more (4.55 → 4.30) than the test MAE (4.72 → 4.65): with 178 breakpoints
  fitted on 1,890 faces it partly overfits, so the test number is the one to trust.
- Single models in the web app also use flip TTA, so they are slightly better than the notebook numbers above.

### More diverse training data

UTKFace is 42% White and has no Middle Eastern category. [`train_diverse.ipynb`](train_diverse.ipynb) fine-tunes the
EfficientNetB0 on UTKFace plus [FairFace](https://github.com/joojs/fairface) (CC BY 4.0, 97k faces balanced across seven
groups). FairFace labels age as ranges (20–29, 30–39, …), so training uses an interval loss: a prediction inside the
range costs nothing, outside it the distance to the range is penalised. FairFace faces are cropped by
[`tools/prepare_fairface.py`](tools/prepare_fairface.py) with the web app's own preprocessing.

FairFace test set (official validation split, 10,925 faces never used for training). Years outside the labelled
range, and share of predictions inside it:

| Group | UTKFace only | + FairFace | In range |
|---|---|---|---|
| Black | 5.20 | **2.68** | 35% → 44% |
| East Asian | 3.94 | **2.03** | 41% → 53% |
| Indian | 4.33 | **2.20** | 36% → 49% |
| Latino / Hispanic | 4.49 | **2.40** | 39% → 46% |
| Middle Eastern | 4.26 | **2.21** | 39% → 48% |
| Southeast Asian | 4.43 | **2.26** | 40% → 50% |
| White | 4.98 | **2.40** | 36% → 47% |
| **All** | 4.55 | **2.32** | 38% → 48% |

On the UTKFace test set every ethnic group improved as well (overall 4.97 → 4.83). The trade-off: ages 60–90 got
worse (7.37 → 8.11), because FairFace's oldest label is the wide range "70+".

> **Photos from outside the datasets are harder.** On four official portraits with known ages (studio lighting, unlike
> the mostly casual dataset photos) the default model is off by about 10 years on average. Four photos are too few to
> rank models; the test sets above are the reliable comparison.

## Architecture

```mermaid
flowchart LR
    U[Browser] -->|:3000| N[nginx<br/>React build]
    N -->|/api/*| B[FastAPI<br/>TensorFlow CPU]
    B --> D[YuNet face detector]
    B --> M[(./models<br/>read-only volume)]
    M --> E[age_efficientnet.keras]
    M --> C[age_cnn*.keras]
    M --> G[gan_generator.keras]
```

```
├── gradute_preoject.ipynb   # training: data, 3 models, GAN, evaluation, export
├── backend/                 # FastAPI app, tests, Dockerfile
│   └── app/
│       ├── routers/         # health, predict, generate, model-info
│       ├── services/        # preprocessing (mirrors the notebook) and model registry
│       └── assets/          # YuNet face detector (ONNX, 230 KB)
├── frontend/                # React + Vite + TypeScript + Tailwind + Motion, nginx config
├── models/                  # trained weights (not in git, see models/README.md)
└── docker-compose.yml
```

**Prediction pipeline** (identical in the notebook and the API):
1. Detect faces with OpenCV's YuNet (score ≥ 0.7).
2. Crop each face with margins of 9% left, 0% top, 8% right and 2% bottom of the detected box. These were measured by
   running YuNet on 1,000 UTKFace images, so uploaded photos are framed like the training images.
3. Resize to 128×128 (grayscale for the CNNs, RGB for EfficientNet). Pixels stay in 0–255; each model rescales internally.
4. Age = model output × 90.

## Run it with Docker

1. Put the trained models in `models/` (see [models/README.md](models/README.md)). Running the last cells of the
   notebook copies them there along with `metrics.json`.
2. Start the app:
   ```bash
   docker compose up --build
   ```
3. Open http://localhost:3000.

The backend loads all models once at startup (about 20 s) and the frontend waits for it to be healthy.
Only port 3000 is published; the API is reached through nginx at `/api`. Missing models are reported by
`/api/health` and the matching feature is disabled in the UI instead of failing.

## Local development

```bash
# Backend (Python 3.11+)
cd backend
pip install -r requirements-dev.txt
cp .env.example .env            # MODELS_DIR=../models by default
uvicorn app.main:app --reload   # http://localhost:8000/api/docs
pytest                          # 27 tests; the real-model test runs when ./models has weights

# Frontend (Node 20+)
cd frontend
npm install
npm run dev                     # http://localhost:5173, /api is proxied to :8000
```

| Variable | Default | Meaning |
|---|---|---|
| `MODELS_DIR` | `../models` (`/models` in Docker) | Folder with the `.keras` files and `metrics.json` |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated origins, only needed when the browser calls the API directly |
| `MAX_UPLOAD_MB` | `5` | Upload size limit |
| `DEFAULT_MODEL` | best by validation MAE | `cnn`, `cnn_gan`, `efficientnet` or `efficientnet_diverse` |
| `VITE_API_BASE` (frontend, build time) | empty | API origin; empty means same origin |

## Free hosting: models in the browser

The live demo has no server. The models are exported to ONNX and run in the visitor's browser with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), and the site is a free Hugging Face Static Space.
`frontend/src/lib/browser/` re-implements the backend pipeline in TypeScript (YuNet decoding and NMS,
OpenCV's fixed-point `INTER_LINEAR` resize and grayscale conversion). On the test photos it returns the same
face boxes and ages as the FastAPI backend (largest difference 0.00 years).

```bash
# 1. Export the models (Linux/WSL, same environment as training); checks ONNX vs Keras outputs
python deploy/browser/export_onnx.py models models/onnx UTKFace
cp models/metrics.json models/onnx/

# 2. Build with VITE_INFERENCE=browser and upload to a Static Space
hf auth login
python deploy/browser/deploy_static.py <username>/deep-age-estimator
```

The same frontend talks to the FastAPI backend when built without `VITE_INFERENCE=browser` (Docker setup above).
`deploy/huggingface/` holds a single-container Docker Space as an alternative (needs Hugging Face PRO).

## API

Interactive docs: `/api/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Status and which models are loaded |
| `POST` | `/api/predict?model=` | Multipart `file` (image, ≤ 5 MB). Returns every detected face |
| `POST` | `/api/generate` | `{"age": 1–90, "count": 1–8, "seed"?: int}` → GAN faces |
| `GET` | `/api/model-info` | Dataset, preprocessing steps, metrics per model |

Errors: `400` undecodable image or unknown model, `413` file too large, `415` not an image, `422` no face detected,
`503` model not loaded.

```bash
curl -F "file=@backend/tests/data/face.jpg;type=image/jpeg" http://localhost:3000/api/predict
```
```json
{
  "model": "efficientnet",
  "image_width": 960,
  "image_height": 1200,
  "faces": [
    {
      "bbox": { "x": 498, "y": 231, "w": 179, "h": 203 },
      "age": 49.6,
      "crop": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPac..."
    }
  ]
}
```
`bbox` is in original image pixels; `crop` is a base64 PNG of the exact 128×128 image the model received.

## Screenshots

| | |
|---|---|
| ![Hero](docs/screenshots/hero.png) | ![Generate](docs/screenshots/generate.png) |
| ![How it works](docs/screenshots/how-it-works.png) | ![Light theme](docs/screenshots/predict-light.png) |

<img src="docs/screenshots/mobile-predict.png" alt="Mobile" width="300">

## Training

Open `gradute_preoject.ipynb`, set `UTKFACE_DIR` (or put the `UTKFace` folder next to the notebook) and run all cells.
On an RTX 3050 Laptop GPU a full run takes about 2.5 hours (CNN ~25 min, GAN ~25 min, CNN + GAN ~25 min,
EfficientNetB0 ~70 min). The project report is
[التنبؤ بعمر االنسان باستخدام التعلم العميق.pdf](التنبؤ%20بعمر%20االنسان%20باستخدام%20التعلم%20العميق.pdf).

The test photo `backend/tests/data/face.jpg` is NASA's 1969 portrait of Neil Armstrong (public domain).
