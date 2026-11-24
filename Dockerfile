# CPU inference image for the FastAPI service (and, with a different command, the Streamlit UI).
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/production/model.pt \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

# CPU-only torch keeps the image ~4x smaller than the default CUDA wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,ui]"

COPY app ./app
COPY samples ./samples
COPY models ./models

# Optional: fetch the model at build time instead of copying it (e.g. from a GitHub release)
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then python -c "import urllib.request,os; os.makedirs('models/production',exist_ok=True); urllib.request.urlretrieve('$MODEL_URL','models/production/model.pt')"; fi

RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "lungxray.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
