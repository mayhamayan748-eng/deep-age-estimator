from fastapi import APIRouter, Depends

from app.core.deps import get_registry
from app.schemas import ModelInfoResponse, ModelSummary
from app.services.models import MODEL_LABELS, ModelRegistry

router = APIRouter(tags=["info"])

INPUTS = {
    "ensemble": "128×128 RGB + grayscale, flip-averaged",
    "efficientnet_diverse": "128×128 RGB, 0-255",
    "efficientnet": "128×128 RGB, 0-255",
    "cnn_gan": "128×128 grayscale, 0-255",
    "cnn": "128×128 grayscale, 0-255",
}


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(registry: ModelRegistry = Depends(get_registry)) -> ModelInfoResponse:
    metrics = registry.metrics.get("models", {})
    models = [
        ModelSummary(
            name=name,
            label=label,
            loaded=name in registry.model_names,
            input=INPUTS[name],
            val_mae=metrics.get(name, {}).get("val_mae"),
            test_mae=metrics.get(name, {}).get("test_mae"),
            test_rmse=metrics.get(name, {}).get("test_rmse"),
            mae_by_age_group=metrics.get(name, {}).get("mae_by_age_group", {}),
        )
        for name, label in MODEL_LABELS.items()
    ]
    return ModelInfoResponse(
        dataset=registry.metrics.get("dataset", {
            "name": "UTKFace (aligned & cropped)",
            "url": "https://susanqq.github.io/UTKFace/",
            "age_range": [0, 90],
        }),
        preprocessing=[
            "Face detection with OpenCV YuNet (score ≥ 0.7)",
            "Crop tightly around the detected face, matching how UTKFace images are framed",
            "Resize to 128×128 (grayscale for the CNNs, RGB for EfficientNet)",
            "Pixels kept in 0-255; each model rescales internally",
            "Each model also sees the mirrored face; the two predictions are averaged",
            "Predicted value × 90 = age in years",
        ],
        models=models,
        default_model=registry.default_model,
        gan=registry.metrics.get("gan", {
            "type": "Conditional DCGAN with projection discriminator",
            "noise_dim": 100,
            "output": "128×128 grayscale",
        }) | {"loaded": registry.generator is not None},
        fairface_test=registry.metrics.get("fairface_test", {}),
    )
