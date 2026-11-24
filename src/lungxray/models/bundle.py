"""Self-describing model bundle.

One file carries weights + architecture + class names + preprocessing + calibration
temperature + evaluation metrics, so training and serving can never disagree.
Loaded with ``weights_only=True`` (no arbitrary pickle execution).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from lungxray.data.dataset import Preprocessing
from lungxray.models.factory import build_model

FORMAT_VERSION = 1


def save_bundle(
    path: str | Path,
    model: nn.Module,
    arch: str,
    classes: list[str],
    preprocessing: Preprocessing,
    temperature: float = 1.0,
    referral_threshold: float = 0.7,
    metrics: dict | None = None,
    reference_stats: dict | None = None,
    run_id: str = "",
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "arch": arch,
            "classes": list(classes),
            "preprocessing": preprocessing.to_dict(),
            "temperature": float(temperature),
            "referral_threshold": float(referral_threshold),
            "metrics": metrics or {},
            "reference_stats": reference_stats or {},
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extra": extra or {},
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        path,
    )


def load_bundle(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported bundle format in {path}")
    model = build_model(bundle["arch"], len(bundle["classes"]), pretrained=False)
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in bundle.items() if k != "state_dict"}
    return model, meta


def update_bundle(path: str | Path, **fields) -> None:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    bundle.update(fields)
    torch.save(bundle, path)


def import_legacy_state_dict(state_dict_path: str | Path, out_path: str | Path, classes: list[str]) -> None:
    """Wrap the original notebook's weights (``models/legacy/custom_cnn_v0.pth``, a LegacyNet
    state_dict) into a bundle with its original preprocessing (resize short side + center crop),
    so it can be evaluated fairly."""
    model = build_model("custom_cnn", len(classes))
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=True))
    save_bundle(
        out_path, model, "custom_cnn", classes, Preprocessing(resize="center_crop"), run_id="legacy-v0"
    )
