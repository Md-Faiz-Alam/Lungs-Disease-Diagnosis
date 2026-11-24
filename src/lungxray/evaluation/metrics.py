"""Clinically meaningful metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from lungxray.evaluation.calibration import expected_calibration_error


def _safe_auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> tuple[float, list[float]]:
    per_class = []
    for k in range(n_classes):
        yk = (y_true == k).astype(int)
        per_class.append(float(roc_auc_score(yk, probs[:, k])) if 0 < yk.sum() < len(yk) else float("nan"))
    return float(np.nanmean(per_class)), per_class


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    n = len(classes)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    macro_auc, auc_per_class = _safe_auc(y_true, probs, n)

    per_class = {}
    for k, c in enumerate(classes):
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        per_class[c] = {
            "precision": float(prec[k]),
            "recall_sensitivity": float(rec[k]),
            "specificity": float(tn / max(tn + fp, 1)),
            "f1": float(f1[k]),
            "roc_auc": auc_per_class[k],
            "support": int(support[k]),
        }

    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_roc_auc": macro_auc,
        "log_loss": float(log_loss(y_true, probs / probs.sum(1, keepdims=True), labels=list(range(n)))),
        "ece": expected_calibration_error(probs, y_true),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    # Screening view: "any disease" vs NORMAL. A missed disease is the costly error.
    if "NORMAL" in classes:
        k = classes.index("NORMAL")
        abnormal_true = y_true != k
        abnormal_pred = y_pred != k
        tp = int((abnormal_true & abnormal_pred).sum())
        fn = int((abnormal_true & ~abnormal_pred).sum())
        tn = int((~abnormal_true & ~abnormal_pred).sum())
        fp = int((~abnormal_true & abnormal_pred).sum())
        out["screening"] = {
            "abnormal_sensitivity": tp / max(tp + fn, 1),
            "abnormal_specificity": tn / max(tn + fp, 1),
            "missed_disease_cases": fn,
            "abnormal_roc_auc": float(roc_auc_score(abnormal_true, 1 - probs[:, k]))
            if 0 < abnormal_true.sum() < len(abnormal_true)
            else float("nan"),
        }
    return out


def bootstrap_ci(
    y_true: np.ndarray, probs: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Percentile bootstrap 95% CIs: a single test-set number hides how uncertain it is."""
    rng = np.random.default_rng(seed)
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    n_classes = probs.shape[1]
    stats: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "macro_roc_auc": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, pr = y_true[idx], probs[idx]
        yp = pr.argmax(1)
        stats["accuracy"].append(accuracy_score(yt, yp))
        stats["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        stats["macro_f1"].append(f1_score(yt, yp, average="macro", zero_division=0))
        stats["macro_roc_auc"].append(_safe_auc(yt, pr, n_classes)[0])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))] for k, v in stats.items()}
