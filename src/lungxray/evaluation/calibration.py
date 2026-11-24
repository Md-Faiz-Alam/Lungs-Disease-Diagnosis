"""Temperature scaling (Guo et al., 2017) and calibration error.

A model that says "97% confident" should be right ~97% of the time. Modern CNNs are
usually over-confident; one scalar temperature fitted on the validation set fixes most
of it without changing the predicted class.
"""

from __future__ import annotations

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((float((lo + hi) / 2), float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows  # (bin_center, mean_confidence, accuracy, count)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    rows = reliability_bins(np.asarray(probs), np.asarray(labels), n_bins)
    n = sum(r[3] for r in rows)
    return float(sum(r[3] / n * abs(r[1] - r[2]) for r in rows)) if n else 0.0
