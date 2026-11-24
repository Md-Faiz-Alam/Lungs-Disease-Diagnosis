"""Production predictor: input validation, calibrated probabilities, referral decision,
optional Grad-CAM, and structured prediction logging for drift monitoring."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, UnidentifiedImageError

from lungxray.data.dataset import Preprocessing, eval_transform
from lungxray.evaluation.calibration import softmax
from lungxray.explain.gradcam import GradCAM, overlay
from lungxray.models.bundle import load_bundle
from lungxray.models.factory import gradcam_target_layer

MIN_SIDE = 128
MAX_COLORFULNESS = 0.06  # radiographs are grayscale; colour photos score far higher


class InvalidImageError(ValueError):
    """The input cannot (or should not) be analysed as a chest radiograph."""


def colorfulness(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    return float(np.abs(a - a.mean(axis=2, keepdims=True)).mean())


class Predictor:
    def __init__(
        self, bundle_path: str | Path, device: str | None = None, prediction_log: str | Path | None = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.meta = load_bundle(bundle_path, self.device)
        self.classes: list[str] = self.meta["classes"]
        self.temperature: float = self.meta.get("temperature", 1.0)
        self.threshold: float = self.meta.get("referral_threshold", 0.7)
        self.transform = eval_transform(Preprocessing.from_dict(self.meta["preprocessing"]))
        self.prediction_log = Path(prediction_log) if prediction_log else None
        self._cam = GradCAM(self.model, gradcam_target_layer(self.model, self.meta["arch"]))
        self._lock = threading.Lock()  # Grad-CAM hooks keep per-call state

    @property
    def info(self) -> dict:
        return {
            "arch": self.meta["arch"],
            "run_id": self.meta.get("run_id"),
            "created_at": self.meta.get("created_at"),
            "classes": self.classes,
            "temperature": self.temperature,
            "referral_threshold": self.threshold,
            "metrics": self.meta.get("metrics", {}),
        }

    @staticmethod
    def open_image(data: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImageError("File is not a readable image") from e
        return img

    def validate(self, img: Image.Image) -> list[str]:
        if min(img.size) < MIN_SIDE:
            raise InvalidImageError(f"Image too small ({img.width}x{img.height}); need >= {MIN_SIDE}px")
        if colorfulness(img) > MAX_COLORFULNESS:
            raise InvalidImageError("Image looks like a colour photograph, not a radiograph")
        warnings = []
        ratio = img.width / img.height
        if not 0.6 <= ratio <= 1.7:
            warnings.append(f"Unusual aspect ratio {ratio:.2f}; is this a frontal chest X-ray?")
        return warnings

    def _run(self, img: Image.Image, need_heatmap: bool) -> tuple[dict, Image.Image, np.ndarray | None]:
        t0 = time.perf_counter()
        warnings = self.validate(img)
        gray = img.convert("L").convert("RGB")
        x = self.transform(gray).unsqueeze(0).to(self.device)
        heat = None
        with self._lock:
            if need_heatmap:
                heat, logits = self._cam(x)
            else:
                with torch.no_grad():
                    logits = self.model(x)
        probs = softmax(logits.cpu().numpy(), self.temperature)[0]
        k = int(probs.argmax())
        confidence = float(probs[k])
        needs_review = confidence < self.threshold
        result = {
            "prediction": self.classes[k],
            "confidence": round(confidence, 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs, strict=True)},
            "needs_review": needs_review,
            "recommendation": "Low confidence - refer to a radiologist"
            if needs_review
            else "Model is confident - still requires clinician confirmation",
            "warnings": warnings,
            "model": {"arch": self.meta["arch"], "run_id": self.meta.get("run_id")},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        self._log(gray, result)
        return result, gray, heat

    def predict(self, img: Image.Image, explain: bool = False) -> dict:
        result, gray, heat = self._run(img, need_heatmap=explain)
        if heat is not None:
            buf = io.BytesIO()
            overlay(gray, heat).save(buf, format="PNG")
            result["gradcam_png_base64"] = base64.b64encode(buf.getvalue()).decode()
        return result

    def analyze(self, img: Image.Image) -> tuple[dict, np.ndarray]:
        """Prediction plus the raw Grad-CAM heatmap (model-input coordinates, values in [0, 1])."""
        result, _gray, heat = self._run(img, need_heatmap=True)
        return result, heat

    def heatmap_layer(self, heat: np.ndarray, size: tuple[int, int]) -> Image.Image:
        """RGBA heatmap aligned to an image of ``size`` (w, h); transparent where the model saw nothing.

        With center-crop preprocessing the model only sees the central square, so the heatmap is
        placed there instead of being stretched over the whole radiograph.
        """
        w, h = size
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        if self.meta["preprocessing"].get("resize") == "center_crop":
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            box_w, box_h = side, side
        else:
            left, top, box_w, box_h = 0, 0, w, h
        hm = (
            np.asarray(
                Image.fromarray((heat * 255).astype(np.uint8)).resize((box_w, box_h), Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
        rgb = (colormaps["jet"](hm)[..., :3] * 255).astype(np.uint8)
        alpha = (np.clip((hm - 0.15) / 0.85, 0, 1) ** 0.8 * 255).astype(np.uint8)
        layer[top : top + box_h, left : left + box_w, :3] = rgb
        layer[top : top + box_h, left : left + box_w, 3] = alpha
        return Image.fromarray(layer, mode="RGBA")

    def _log(self, gray: Image.Image, result: dict) -> None:
        if not self.prediction_log:
            return
        a = np.asarray(gray.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "needs_review": result["needs_review"],
            "intensity_mean": float(a.mean()),
            "intensity_std": float(a.std()),
            "latency_ms": result["latency_ms"],
            "run_id": result["model"]["run_id"],
        }
        self.prediction_log.parent.mkdir(parents=True, exist_ok=True)
        with self.prediction_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
