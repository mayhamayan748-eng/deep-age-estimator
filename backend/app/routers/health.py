from fastapi import APIRouter, Depends

from app.core.deps import get_registry
from app.schemas import HealthResponse
from app.services.models import MODEL_LABELS, ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    return HealthResponse(
        status="ok" if registry.ready else "degraded",
        models={name: name in registry.model_names for name in MODEL_LABELS},
        generator=registry.generator is not None,
        face_detector=registry.face_detector is not None,
        default_model=registry.default_model,
    )
