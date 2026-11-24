"""Temperature scaling (Guo et al., 2017) and calibration error.

A model that says "97% confident" should be right ~97% of the time. Modern CNNs are
usually over-confident; one scalar temperature fitted on the validation set fixes most
of it without changing the predicted class.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising validation NLL.

    NLL is unimodal in log(T), so a bounded 1-D search over log T in [log 0.05, log 20]
    is deterministic and always converges (unlike a single LBFGS step, which can stop early).
    """
    labels = np.asarray(labels, dtype=int)
    res = minimize_scalar(
        lambda log_t: _nll(logits, labels, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(np.exp(res.x))


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
