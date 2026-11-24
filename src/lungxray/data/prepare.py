"""Deterministic data preparation shared by audit, training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lungxray.config import Config
from lungxray.data.audit import Record, assign_duplicate_groups, build_manifest, leakage_report
from lungxray.data.dataset import split_train_val
from lungxray.utils import get_logger

log = get_logger(__name__)


@dataclass
class PreparedData:
    records: list[Record]
    leakage: dict
    excluded_test: set[str]

    def counts(self) -> dict:
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            out.setdefault(r.split, {}).setdefault(r.label, 0)
            out[r.split][r.label] += 1
        out["test_evaluated"] = {
            "total": sum(1 for r in self.records if r.split == "test" and r.path not in self.excluded_test)
        }
        return out


def prepare_data(cfg: Config) -> PreparedData:
    d = cfg.data
    records = build_manifest(
        {"train": d.train_dir, "test": d.test_dir},
        d.classes,
        cache_path=Path(cfg.output_dir) / "cache" / "manifest.json",
    )
    n_dup = assign_duplicate_groups(records, d.near_duplicate_threshold)
    leakage = leakage_report(records)  # computed before val split: train vs test leakage
    if d.drop_conflicting_labels and leakage["conflicting_label_paths"]:
        bad = set(leakage["conflicting_label_paths"])
        records = [r for r in records if r.path not in bad]
        leakage["dropped_conflicting_label_images"] = len(bad)
        log.info("Dropped %d images whose duplicates carry a different label", len(bad))
    split_train_val(records, d.val_fraction, cfg.seed)
    kept = {r.path for r in records}
    excluded = {p for p in leakage["leaked_test_paths"] if p in kept} if d.drop_test_leaks else set()
    log.info(
        "Near-duplicate images: %d | test images duplicated in train: %d (%.1f%%)%s",
        n_dup,
        leakage["test_images_with_train_duplicate"],
        100 * leakage["test_leak_fraction"],
        " -> excluded from clean test set" if excluded else "",
    )
    return PreparedData(records, leakage, excluded)
