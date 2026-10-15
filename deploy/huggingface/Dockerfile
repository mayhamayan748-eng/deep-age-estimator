# Single-container image for Hugging Face Spaces: FastAPI serves both the API and the built React app.
# Build context is the folder assembled by deploy.py (backend/, frontend/, models/).

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODELS_DIR=/models \
    STATIC_DIR=/app/static

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Spaces run containers as uid 1000
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app backend/app ./app
COPY --chown=app:app models /models
COPY --from=web --chown=app:app /web/dist ./static
USER app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
