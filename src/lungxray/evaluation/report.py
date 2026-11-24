"""Evaluation figures: confusion matrix, ROC curves, reliability diagram, training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from lungxray.evaluation.calibration import reliability_bins  # noqa: E402


def plot_confusion(cm: list[list[int]], classes: list[str], path: Path, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{norm[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.6 else "black",
                fontsize=9,
            )
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, probs: np.ndarray, classes: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for k, c in enumerate(classes):
        yk = (y_true == k).astype(int)
        if 0 < yk.sum() < len(yk):
            fpr, tpr, _ = roc_curve(yk, probs[:, k])
            ax.plot(fpr, tpr, label=c)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(y_true, probs_raw, probs_cal, path: Path, ece_raw: float, ece_cal: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    for probs, name, ece in ((probs_raw, "before", ece_raw), (probs_cal, "after temperature", ece_cal)):
        rows = reliability_bins(probs, y_true)
        ax.plot([r[1] for r in rows], [r[2] for r in rows], "o-", label=f"{name} (ECE {ece:.3f})")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [h["val_balanced_accuracy"] for h in history], label="val balanced acc")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="val macro F1")
    axes[1].set_title("Validation metrics")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
