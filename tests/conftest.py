from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lungxray.data.dataset import Preprocessing
from lungxray.models.bundle import save_bundle
from lungxray.models.factory import build_model

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA"]


def fake_xray(seed: int, label_idx: int, size: int = 160) -> Image.Image:
    """Grayscale image with a class-dependent blob, so a tiny model can learn something."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.35, 0.08, (size, size))
    yy, xx = np.mgrid[:size, :size]
    cx = size * (0.3 + 0.2 * label_idx)
    img += 0.4 * np.exp(-((xx - cx) ** 2 + (yy - size / 2) ** 2) / (2 * (size / 8) ** 2))
    return Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8), mode="L")


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    seed = 0
    for split, n in (("train", 14), ("test", 5)):
        for k, c in enumerate(CLASSES):
            folder = root / split / c
            folder.mkdir(parents=True)
            for i in range(n):
                seed += 1
                fake_xray(seed, k).save(folder / f"{c}_{i}.png")
    # plant a train->test leak: identical image in both splits
    shutil.copy(root / "train" / "NORMAL" / "NORMAL_0.png", root / "test" / "NORMAL" / "leak.png")
    return root


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "model.pt"
    model = build_model("resnet18", len(CLASSES), pretrained=False)
    save_bundle(
        path, model, "resnet18", CLASSES, Preprocessing(image_size=64), temperature=1.5, run_id="test-run"
    )
    return path
