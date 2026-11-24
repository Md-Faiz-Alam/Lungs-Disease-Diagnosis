"""Quality-gated model promotion (a minimal model registry).

A run is copied to models/production only if it clears absolute quality bars AND beats
the model currently in production on the same clean test set.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from lungxray.config import Config
from lungxray.utils import get_logger, load_json, save_json

log = get_logger(__name__)
PRODUCTION = Path("models/production/model.pt")
REGISTRY = Path("models/registry.json")


def _summary(bundle_path: Path) -> dict:
    return torch.load(bundle_path, map_location="cpu", weights_only=True).get("metrics", {})


def check_gate(metrics: dict, cfg: Config) -> list[str]:
    failures = []
    gate = cfg.promotion
    if metrics.get("balanced_accuracy", 0) < gate.min_balanced_accuracy:
        failures.append(
            f"balanced_accuracy {metrics.get('balanced_accuracy', 0):.4f} < {gate.min_balanced_accuracy}"
        )
    for c, r in metrics.get("per_class_recall", {}).items():
        if r < gate.min_class_recall:
            failures.append(f"recall[{c}] {r:.4f} < {gate.min_class_recall}")
    if gate.require_better_than_production and PRODUCTION.exists():
        current = _summary(PRODUCTION).get("balanced_accuracy", 0)
        if metrics.get("balanced_accuracy", 0) <= current:
            failures.append(f"not better than production ({current:.4f})")
    return failures


def promote(run_dir: str | Path, cfg: Config, force: bool = False) -> bool:
    run_dir = Path(run_dir)
    bundle = run_dir / "model.pt"
    metrics = _summary(bundle)
    failures = check_gate(metrics, cfg)
    if failures and not force:
        log.warning("Promotion REJECTED for %s:\n  - %s", run_dir.name, "\n  - ".join(failures))
        return False
    PRODUCTION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle, PRODUCTION)
    registry = load_json(REGISTRY) if REGISTRY.exists() else []
    registry.append(
        {
            "run": run_dir.name,
            "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "forced": bool(failures),
            "metrics": metrics,
        }
    )
    save_json(registry, REGISTRY)
    log.info("Promoted %s -> %s", run_dir.name, PRODUCTION)
    return True
