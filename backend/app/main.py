import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import Settings, get_settings
from app.routers import generate, health, info, predict
from app.services.models import ModelRegistry

logger = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Models are loaded once per process, not per request
        registry = ModelRegistry(settings.models_dir)
        registry.load(settings.default_model)
        app.state.registry = registry
        logger.info("Ready: models=%s default=%s generator=%s",
                    list(registry.age_models), registry.default_model, registry.generator is not None)
        yield

    app = FastAPI(
        title="Deep Age Estimator API",
        description="Age estimation from face photos (CNN / CNN+GAN / EfficientNetB0) and conditional GAN face generation.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code,
                    (time.perf_counter() - start) * 1000)
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    for r in (health.router, predict.router, generate.router, info.router):
        app.include_router(r, prefix="/api")

    if settings.static_dir:
        mount_frontend(app, settings.static_dir)
    return app


def mount_frontend(app: FastAPI, static_dir: Path) -> None:
    """Serve the built React app, with index.html as the fallback for client-side routes."""
    root = static_dir.resolve()
    index = root / "index.html"
    app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        file = (root / path).resolve()
        if path and file.is_file() and root in file.parents:
            return FileResponse(file)
        return FileResponse(index)


app = create_app()
