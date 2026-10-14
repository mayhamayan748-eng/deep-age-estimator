from fastapi import Request

from app.services.models import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry
