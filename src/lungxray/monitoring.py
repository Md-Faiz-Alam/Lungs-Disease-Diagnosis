"""Production drift monitoring from the API's prediction log.

Compares live input statistics and confidence against the validation-set reference
stored in the model bundle, using the Population Stability Index (PSI).
Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def psi(reference_counts, live_counts, eps: float = 1e-4) -> float:
    ref = np.asarray(reference_counts, dtype=float)
    live = np.asarray(live_counts, dtype=float)
    ref = ref / max(ref.sum(), 1) + eps
    live = live / max(live.sum(), 1) + eps
    return float(np.sum((live - ref) * np.log(live / ref)))


def drift_report(bundle_path: str | Path, log_path: str | Path, min_samples: int = 100) -> dict:
    ref = torch.load(bundle_path, map_location="cpu", weights_only=True).get("reference_stats", {})
    rows = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line]
    if not rows or not ref:
        return {"status": "insufficient data", "n": len(rows)}
    out: dict = {"n": len(rows), "features": {}}
    for feat, key in (
        ("intensity_mean", "intensity_mean"),
        ("intensity_std", "intensity_std"),
        ("confidence", "confidence"),
    ):
        bins = ref[feat]["bins"]
        live = np.histogram([r[key] for r in rows], bins=bins)[0]
        value = psi(ref[feat]["values"], live)
        out["features"][feat] = {
            "psi": round(value, 4),
            "status": "stable" if value < 0.1 else "moderate shift" if value < 0.25 else "SIGNIFICANT shift",
        }
    out["referral_rate"] = round(float(np.mean([r["needs_review"] for r in rows])), 4)
    out["prediction_distribution"] = dict(Counter(r["prediction"] for r in rows))
    out["p95_latency_ms"] = float(np.percentile([r["latency_ms"] for r in rows], 95))
    enough = len(rows) >= min_samples
    out["alert"] = enough and any(f["psi"] >= 0.25 for f in out["features"].values())
    if not enough:
        out["note"] = (
            f"only {len(rows)} predictions logged; PSI is unreliable below {min_samples}, no alert raised"
        )
    return out
