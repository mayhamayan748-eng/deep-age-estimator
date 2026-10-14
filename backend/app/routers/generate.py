from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.core.deps import get_registry
from app.schemas import ErrorResponse, GenerateRequest, GenerateResponse
from app.services.models import ModelRegistry, ModelUnavailableError

router = APIRouter(tags=["generate"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={503: {"model": ErrorResponse, "description": "GAN generator not loaded"}},
)
async def generate(body: GenerateRequest, registry: ModelRegistry = Depends(get_registry)) -> GenerateResponse:
    try:
        seed, images = await run_in_threadpool(registry.generate, body.age, body.count, body.seed)
    except ModelUnavailableError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))
    return GenerateResponse(age=body.age, seed=seed, images=images)
