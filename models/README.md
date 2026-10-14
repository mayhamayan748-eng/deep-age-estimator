# models/

Trained weights go here. They are not committed to git; this folder is mounted read-only into the backend container.

| File | Produced by the notebook | Needed for |
|---|---|---|
| `age_efficientnet.keras` | section 5 | `/api/predict?model=efficientnet` |
| `age_cnn_gan.keras` | section 4 | `/api/predict?model=cnn_gan` |
| `age_cnn.keras` | section 2 | `/api/predict?model=cnn` |
| `gan_generator.keras` | section 3 | `/api/generate` |
| `metrics.json` | section 6 | metrics in `/api/model-info`, choice of the default model |

Any subset works: missing models are reported by `/api/health` and the matching feature is disabled.
The YuNet face detector is bundled with the backend (`backend/app/assets/`).
