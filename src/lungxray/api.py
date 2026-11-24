"""FastAPI inference service.

Run:  uvicorn lungxray.api:app --host 0.0.0.0 --port 8000
Env:  MODEL_PATH (default models/production/model.pt), API_KEY (optional),
      PREDICTION_LOG (default logs/predictions.jsonl), MAX_UPLOAD_MB (default 10)
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from lungxray import __version__
from lungxray.inference.predictor import InvalidImageError, Predictor
from lungxray.utils import get_logger

log = get_logger("lungxray.api")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "application/octet-stream"}
MAX_BATCH = 16


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
async def lifespan(_app: FastAPI):
    path = os.environ.get("MODEL_PATH", "models/production/model.pt")
    try:
        state["predictor"] = Predictor(
            path, prediction_log=os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
        )
        log.info("Loaded model %s from %s", state["predictor"].info["run_id"], path)
    except FileNotFoundError:
        log.error("Model file not found at %s - /predict will return 503", path)
    yield


app = FastAPI(
    title="Lung X-ray Classifier API",
    version=__version__,
    description="COVID-19 / Normal / Pneumonia classification of chest radiographs with calibrated "
    "confidence, radiologist-referral flag and Grad-CAM explanations. Research use only - not a "
    "medical device.",
    lifespan=lifespan,
)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def get_predictor() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["predictor"]


def _read(file: UploadFile) -> bytes:
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type {file.content_type}")
    limit = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
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
        log.exception("Prediction failed")
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
