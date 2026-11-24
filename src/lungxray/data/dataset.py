"""Preprocessing, datasets, leakage-safe splitting and data loaders."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from lungxray.data.audit import Record

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Preprocessing:
    """Everything inference needs to reproduce training-time preprocessing.

    Stored inside every model bundle so the API can never drift from training.
    """

    image_size: int = 224
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD
    resize: str = "square"  # "square" (whole image) | "center_crop" (legacy baseline)

    def to_dict(self) -> dict:
        return {
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "resize": self.resize,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Preprocessing:
        return cls(d["image_size"], tuple(d["mean"]), tuple(d["std"]), d.get("resize", "square"))


def load_xray(path_or_file, max_side: int | None = None) -> Image.Image:
    """Open an X-ray as 3-channel grayscale (pretrained CNNs expect 3 channels)."""
    im = Image.open(path_or_file)
    if max_side:
        im.draft("L", (max_side, max_side))  # cheap JPEG downscale while decoding
    return im.convert("L").convert("RGB")


def _geometry(pp: Preprocessing) -> list:
    s = pp.image_size
    return [T.Resize((s, s))] if pp.resize == "square" else [T.Resize(s), T.CenterCrop(s)]


def eval_transform(pp: Preprocessing) -> T.Compose:
    return T.Compose([*_geometry(pp), T.ToTensor(), T.Normalize(pp.mean, pp.std)])


def train_transform(pp: Preprocessing) -> T.Compose:
    # No horizontal flip: it moves the heart to the wrong side (anatomically implausible).
    # Hue/saturation jitter is meaningless on grayscale radiographs -> intensity jitter only.
    return T.Compose(
        [
            *_geometry(pp),
            T.RandomAffine(degrees=7, translate=(0.06, 0.06), scale=(0.92, 1.08)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(pp.mean, pp.std),
        ]
    )


class XrayDataset(Dataset):
    def __init__(self, records: list[Record], classes: list[str], transform, max_side: int = 512):
        self.records = records
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        return self.transform(load_xray(r.path, self.max_side)), self.class_to_idx[r.label]

    @property
    def targets(self) -> list[int]:
        return [self.class_to_idx[r.label] for r in self.records]


def split_train_val(records: list[Record], val_fraction: float, seed: int) -> None:
    """Mark a stratified subset of train as 'val', keeping duplicate groups together so
    near-identical images never straddle train and validation."""
    train = [r for r in records if r.split == "train"]
    n_splits = max(2, round(1 / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(train, [r.label for r in train], [r.group for r in train]))
    for i in val_idx:
        train[i].split = "val"


def class_weights(records: list[Record], classes: list[str]) -> torch.Tensor:
    counts = Counter(r.label for r in records)
    w = [len(records) / (len(classes) * max(counts[c], 1)) for c in classes]
    return torch.tensor(np.array(w, dtype=np.float32))


def make_loaders(
    records: list[Record],
    classes: list[str],
    pp: Preprocessing,
    batch_size: int,
    num_workers: int,
    balance: str,
    exclude_test_paths: set[str] | None = None,
) -> dict[str, DataLoader]:
    exclude = exclude_test_paths or set()
    by_split = {
        "train": [r for r in records if r.split == "train"],
        "val": [r for r in records if r.split == "val"],
        "test": [r for r in records if r.split == "test" and r.path not in exclude],
    }
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_ds = XrayDataset(by_split["train"], classes, train_transform(pp))
    sampler = None
    if balance == "weighted_sampler":
        w = class_weights(by_split["train"], classes).numpy()
        sampler = WeightedRandomSampler(
            [float(w[t]) for t in train_ds.targets], num_samples=len(train_ds), replacement=True
        )
    loaders = {}
    if len(train_ds):
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=len(train_ds) > batch_size,
            **common,
        )
    for name in ("val", "test"):
        ds = XrayDataset(by_split[name], classes, eval_transform(pp))
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, **common)
    return loaders
