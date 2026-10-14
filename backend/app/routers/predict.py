from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.deps import get_registry
from app.schemas import BBoxSchema, ErrorResponse, FacePrediction, PredictResponse
from app.services.models import InvalidImageError, ModelRegistry, ModelUnavailableError, NoFaceError

router = APIRouter(tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Unknown model or undecodable image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Not an image"},
        422: {"model": ErrorResponse, "description": "No face detected"},
        503: {"model": ErrorResponse, "description": "Model or face detector not loaded"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Photo containing one or more faces"),
    model: str | None = Query(default=None, description="cnn | cnn_gan | efficientnet (default: best model)"),
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only image files are accepted")

    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Image must be at most {settings.max_upload_mb} MB")

    if model is not None and model not in registry.age_models:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown or unavailable model '{model}'")

    try:
        name, (width, height), faces = await run_in_threadpool(registry.predict, data, model)
    except InvalidImageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except NoFaceError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    except ModelUnavailableError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))

    return PredictResponse(
        model=name,
        image_width=width,
        image_height=height,
        faces=[FacePrediction(bbox=BBoxSchema(**vars(f.bbox)), age=f.age, crop=f.crop_png) for f in faces],
    )
