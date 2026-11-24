"""FastAPI service: clinical studies API (/api/*) for the web app plus the original inference API.

Run:  uvicorn lungxray.api:app --host 0.0.0.0 --port 8000
Configuration is read from environment variables, see lungxray/service/settings.py.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from lungxray import __version__
from lungxray.inference.predictor import InvalidImageError, Predictor
from lungxray.service.auth import get_current_user
from lungxray.service.ratelimit import SlidingWindowLimiter, client_key, parse_rate
from lungxray.service.routes import router as clinical_router
from lungxray.service.settings import Settings, get_settings
from lungxray.service.storage import build_repository
from lungxray.utils import get_logger

log = get_logger("lungxray.api")

if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):  # traces, requests, exceptions -> App Insights
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(logger_name="lungxray")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "application/octet-stream"}
MAX_BATCH = 16
INFERENCE_PATHS = {"/api/studies", "/predict", "/predict/batch"}


class Prediction(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool
    recommendation: str
    warnings: list[str]
    model: dict
    latency_ms: float
    gradcam_png_base64: str | None = None


class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = self.errors = self.rejected = 0
        self.latency_sum = 0.0
        self.by_class: dict[str, int] = {}

    def record(self, label: str | None, seconds: float, status: str) -> None:
        with self.lock:
            self.requests += 1
            self.latency_sum += seconds
            if status == "error":
                self.errors += 1
            elif status == "rejected":
                self.rejected += 1
            if label:
                self.by_class[label] = self.by_class.get(label, 0) + 1


state: dict = {"predictor": None, "stats": _Stats()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings.cache_clear()
    settings = get_settings()
    state["predictor"] = None
    try:
        state["predictor"] = Predictor(settings.model_path, prediction_log=settings.prediction_log)
        log.info("Loaded model %s from %s", state["predictor"].info["run_id"], settings.model_path)
    except FileNotFoundError:
        log.error("Model file not found at %s - prediction endpoints will return 503", settings.model_path)
    app.state.predictor = state["predictor"]
    app.state.repo = build_repository(settings)
    app.state.limiter = SlidingWindowLimiter(*parse_rate(settings.rate_limit_inference))
    log.info("auth=%s storage=%s", settings.auth_mode, settings.storage_mode)
    yield


app = FastAPI(
    title="Lung X-ray Classifier API",
    version=__version__,
    description="COVID-19 / Normal / Pneumonia classification of chest radiographs with calibrated "
    "confidence, radiologist-referral flag and Grad-CAM explanations. Research use only - not a "
    "medical device.",
    lifespan=lifespan,
)


@app.middleware("http")
async def guard_inference(request: Request, call_next):
    """Reject oversized uploads before reading the body, and rate-limit model inference per client."""
    if request.method == "POST" and request.url.path in INFERENCE_PATHS:
        settings = get_settings()
        cap_mb = settings.max_upload_mb * (MAX_BATCH if request.url.path.endswith("/batch") else 1) + 1
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > cap_mb * 1024 * 1024:
            return JSONResponse({"detail": f"Request exceeds {cap_mb:.0f} MB"}, status_code=413)
        limiter = getattr(request.app.state, "limiter", None)
        retry = limiter.check(client_key(request)) if limiter else 0.0
        if retry:
            return JSONResponse(
                {"detail": "Too many analyses from this client; please wait and try again."},
                status_code=429,
                headers={"Retry-After": str(int(retry) + 1)},
            )
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Correlation ID: echoed back and attached to error logs so a user report can be traced.
    rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/") and "Cache-Control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"  # never cache clinical data in shared caches
    return response


app.include_router(clinical_router)
# Added last = outermost, so CORS headers are present even on 413/429 responses from the guards above.
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.from_env().cors_origins,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Demo-Session", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Retry-After"],
    max_age=3600,
)


def require_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Guard for the original inference endpoints: API key if configured, Entra token in entra mode."""
    settings = get_settings()
    expected = settings.legacy_api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    if not expected and settings.auth_mode == "entra":
        get_current_user(request, settings)


def get_predictor() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["predictor"]


def _read(file: UploadFile) -> bytes:
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type {file.content_type}")
    limit = int(get_settings().max_upload_mb * 1024 * 1024)
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="File too large")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return data


def _predict_one(predictor: Predictor, data: bytes, explain: bool) -> dict:
    t0 = time.perf_counter()
    try:
        result = predictor.predict(Predictor.open_image(data), explain=explain)
    except InvalidImageError as e:
        state["stats"].record(None, time.perf_counter() - t0, "rejected")
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - defensive
        state["stats"].record(None, time.perf_counter() - t0, "error")
        log.exception("Prediction failed")  # request id is in the X-Request-ID response header
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    state["stats"].record(result["prediction"], time.perf_counter() - t0, "ok")
    return result


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state["predictor"] is not None, "version": __version__}


@app.get("/model", dependencies=[Depends(require_key)])
def model_info(predictor: Predictor = Depends(get_predictor)) -> dict:
    return predictor.info


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_key)])
def predict(
    file: UploadFile = File(..., description="Frontal chest X-ray (JPEG/PNG)"),
    explain: bool = Query(False, description="include a Grad-CAM heatmap (base64 PNG)"),
    predictor: Predictor = Depends(get_predictor),
) -> dict:
    return _predict_one(predictor, _read(file), explain)


@app.post("/predict/batch", dependencies=[Depends(require_key)])
def predict_batch(
    files: list[UploadFile] = File(...), predictor: Predictor = Depends(get_predictor)
) -> list[dict]:
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"At most {MAX_BATCH} files per batch")
    out = []
    for f in files:
        try:
            out.append({"filename": f.filename, **_predict_one(predictor, _read(f), False)})
        except HTTPException as e:
            out.append({"filename": f.filename, "error": e.detail, "status_code": e.status_code})
    return out


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text format."""
    s = state["stats"]
    with s.lock:
        lines = [
            "# TYPE lungxray_requests_total counter",
            f"lungxray_requests_total {s.requests}",
            f"lungxray_errors_total {s.errors}",
            f"lungxray_rejected_inputs_total {s.rejected}",
            f"lungxray_latency_seconds_sum {s.latency_sum:.4f}",
        ]
        lines += [f'lungxray_predictions_total{{label="{k}"}} {v}' for k, v in s.by_class.items()]
    return "\n".join(lines) + "\n"
