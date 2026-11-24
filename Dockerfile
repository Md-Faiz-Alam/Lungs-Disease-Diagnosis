# CPU inference image for the LungXray API (FastAPI + PyTorch model).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/production/model.pt \
    PREDICTION_LOG=/app/logs/predictions.jsonl \
    LOCAL_STORAGE_DIR=/app/var/storage

WORKDIR /app

# CPU-only torch keeps the image ~4x smaller than the default CUDA wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,azure]"

COPY models ./models

# The model is not committed to git: CI downloads it from a GitHub Release and verifies its SHA-256
ARG MODEL_URL=""
ARG MODEL_SHA256=""
RUN if [ -n "$MODEL_URL" ]; then python -m lungxray.fetch_model; fi

RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/logs /app/var/storage && chown -R appuser /app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# One process per container; Azure Container Apps scales out with replicas.
CMD ["uvicorn", "lungxray.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
