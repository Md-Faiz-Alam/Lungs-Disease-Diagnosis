"""Dataset auditing: manifest building, near-duplicate / leakage detection, shortcut probing.

Why this matters: public chest X-ray datasets are stitched together from several
sources. The same image often appears in both train and test (inflating accuracy),
and image *metadata* (resolution, file format) can differ by class, which lets a CNN
"cheat" by recognising the source hospital rather than the disease.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from lungxray.utils import get_logger, load_json, save_json

log = get_logger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Record:
    path: str
    split: str  # train | val | test
    label: str
    width: int
    height: int
    mode: str
    ext: str
    bytes: int
    phash: str
    group: int = -1


def _describe(path: Path, split: str, label: str) -> Record:
    with Image.open(path) as im:
        width, height, mode = im.width, im.height, im.mode
        im.draft("L", (256, 256))  # fast JPEG decode at reduced size; pHash is size invariant
        ph = str(imagehash.phash(im.convert("L")))
    return Record(
        path=str(path).replace("\\", "/"),
        split=split,
        label=label,
        width=width,
        height=height,
        mode=mode,
        ext=path.suffix.lower(),
        bytes=path.stat().st_size,
        phash=ph,
    )


def build_manifest(
    split_dirs: dict[str, str], classes: list[str], cache_path: str | Path | None = None
) -> list[Record]:
    """Scan class folders and describe every image. Cached by (path, size, mtime)."""
    cache: dict[str, dict] = {}
    if cache_path and Path(cache_path).exists():
        cache = {r["key"]: r for r in load_json(cache_path)}

    todo = []
    for split, root in split_dirs.items():
        for label in classes:
            folder = Path(root) / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            for name in sorted(os.listdir(folder)):
                p = folder / name
                if p.suffix.lower() in IMAGE_EXTS:
                    todo.append((p, split, label))

    records, cache_out = [], []
    for p, split, label in tqdm(todo, desc="Scanning images", leave=False):
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key in cache:
            rec = Record(**{k: v for k, v in cache[key].items() if k != "key"})
        else:
            rec = _describe(p, split, label)
        rec.split, rec.label, rec.group = split, label, -1
        records.append(rec)
        cache_out.append({"key": key, **asdict(rec)})

    if cache_path:
        save_json(cache_out, cache_path)
    log.info("Manifest: %d images", len(records))
    return records


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = x.copy()  # numpy < 2.0 fallback
    count = np.zeros(x.shape, dtype=np.uint8)
    while np.any(x):
        count += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return count


def assign_duplicate_groups(records: list[Record], threshold: int) -> int:
    """Union near-duplicates (pHash Hamming distance <= threshold) into groups.

    Returns how many images belong to a group of size > 1.
    """
    n = len(records)
    hashes = np.array([int(r.phash, 16) for r in records], dtype=np.uint64)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    chunk = 512
    for start in range(0, n, chunk):
        dist = _popcount(hashes[start : start + chunk, None] ^ hashes[None, :])
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows + start, cols, strict=True):
            if r < c:
                ra, rb = find(int(r)), find(int(c))
                if ra != rb:
                    parent[rb] = ra

    roots = [find(i) for i in range(n)]
    remap = {root: gid for gid, root in enumerate(dict.fromkeys(roots))}
    sizes: dict[int, int] = {}
    for rec, root in zip(records, roots, strict=True):
        rec.group = remap[root]
        sizes[rec.group] = sizes.get(rec.group, 0) + 1
    return sum(1 for r in records if sizes[r.group] > 1)


def leakage_report(records: list[Record]) -> dict:
    """Test images that have a near-duplicate in train/val, plus label conflicts in groups."""
    by_group: dict[int, list[Record]] = {}
    for r in records:
        by_group.setdefault(r.group, []).append(r)

    leaked_test: list[str] = []
    conflicting_paths: list[str] = []
    conflicting_groups = dup_images = 0
    for members in by_group.values():
        if len(members) < 2:
            continue
        dup_images += len(members)
        splits = {m.split for m in members}
        if "test" in splits and splits - {"test"}:
            leaked_test += [m.path for m in members if m.split == "test"]
        if len({m.label for m in members}) > 1:
            conflicting_groups += 1
            conflicting_paths += [m.path for m in members]

    n_test = sum(r.split == "test" for r in records)
    return {
        "images_in_duplicate_groups": dup_images,
        "test_images_with_train_duplicate": len(leaked_test),
        "test_leak_fraction": round(len(leaked_test) / max(n_test, 1), 4),
        "groups_with_conflicting_labels": conflicting_groups,
        "leaked_test_paths": sorted(leaked_test),
        "conflicting_label_paths": sorted(conflicting_paths),
    }


def metadata_shortcut_probe(records: list[Record], classes: list[str], seed: int = 42) -> dict:
    """Train a classifier on file metadata ONLY (no pixels).

    If resolution / file format alone predicts the diagnosis far above chance, the
    dataset contains source shortcuts that a CNN can exploit.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    def feats(r: Record) -> list[float]:
        return [
            r.width,
            r.height,
            r.width / r.height,
            r.bytes / (r.width * r.height),
            float(r.mode == "L"),
            float(r.mode == "RGB"),
            float(r.ext in {".jpg", ".jpeg"}),
            float(r.ext == ".png"),
        ]

    train = [r for r in records if r.split != "test"]
    test = [r for r in records if r.split == "test"]
    idx = {c: i for i, c in enumerate(classes)}
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit([feats(r) for r in train], [idx[r.label] for r in train])
    bal = balanced_accuracy_score([idx[r.label] for r in test], clf.predict([feats(r) for r in test]))
    if bal > 0.75:
        verdict = "STRONG shortcut risk: file metadata alone separates the classes"
    elif bal > 0.5:
        verdict = "moderate shortcut risk"
    else:
        verdict = "low shortcut risk"
    return {
        "metadata_only_balanced_accuracy": round(float(bal), 4),
        "chance_level": round(1 / len(classes), 4),
        "interpretation": verdict,
    }


def class_metadata_summary(records: list[Record], classes: list[str]) -> dict:
    out = {}
    for c in classes:
        rs = [r for r in records if r.label == c]
        if rs:
            out[c] = {
                "count": len(rs),
                "median_width": int(np.median([r.width for r in rs])),
                "median_height": int(np.median([r.height for r in rs])),
                "fraction_png": round(float(np.mean([r.ext == ".png" for r in rs])), 3),
                "fraction_rgb_mode": round(float(np.mean([r.mode == "RGB" for r in rs])), 3),
            }
    return out
