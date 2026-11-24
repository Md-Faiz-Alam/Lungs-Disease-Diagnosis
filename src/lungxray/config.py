"""Typed, validated configuration loaded from YAML with dotted CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train_dir: str = "data/train"
    test_dir: str = "data/test"
    classes: list[str] = ["COVID19", "NORMAL", "PNEUMONIA"]
    val_fraction: float = Field(0.15, gt=0, lt=0.5)
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    balance: Literal["none", "weighted_loss", "weighted_sampler"] = "weighted_loss"
    near_duplicate_threshold: int = 4
    drop_test_leaks: bool = True
    drop_conflicting_labels: bool = True
    resize: Literal["square", "center_crop"] = "center_crop"


class ModelConfig(BaseModel):
    name: Literal["custom_cnn", "resnet18", "densenet121", "efficientnet_b0", "convnext_tiny"] = (
        "efficientnet_b0"
    )
    pretrained: bool = True
    dropout: float = 0.3


class TrainConfig(BaseModel):
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    freeze_backbone_epochs: int = 1
    early_stopping_patience: int = 4
    monitor: str = "val_balanced_accuracy"
    amp: bool = True
    max_train_batches: int | None = None


class InferenceConfig(BaseModel):
    referral_threshold: float = 0.70


class PromotionConfig(BaseModel):
    min_balanced_accuracy: float = 0.90
    min_class_recall: float = 0.85
    require_better_than_production: bool = True


class TrackingConfig(BaseModel):
    mlflow: bool = False


class Config(BaseModel):
    project: str = "lung-xray"
    seed: int = 42
    output_dir: str = "artifacts"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    inference: InferenceConfig = InferenceConfig()
    promotion: PromotionConfig = PromotionConfig()
    tracking: TrackingConfig = TrackingConfig()


def _apply_override(raw: dict, dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"Override must look like key.sub=value, got {dotted!r}")
    key, value = dotted.split("=", 1)
    node = raw
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(value)  # "3" -> 3, "true" -> True, "null" -> None


def load_config(path: str | Path = "configs/config.yaml", overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    for item in overrides or []:
        _apply_override(raw, item)
    return Config.model_validate(raw)
