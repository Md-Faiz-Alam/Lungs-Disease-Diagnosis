import numpy as np

from lungxray.evaluation.calibration import expected_calibration_error, fit_temperature, softmax
from lungxray.evaluation.metrics import bootstrap_ci, compute_metrics
from lungxray.monitoring import psi

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA"]


def test_perfect_predictions():
    y = np.array([0, 1, 2, 2, 1, 0])
    probs = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = compute_metrics(y, probs, CLASSES)
    assert m["accuracy"] == 1.0 and m["balanced_accuracy"] == 1.0 and m["macro_roc_auc"] == 1.0
    assert m["screening"]["missed_disease_cases"] == 0


def test_missed_disease_is_counted():
    y = np.array([2, 2, 1])
    probs = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])  # pneumonia -> normal
    m = compute_metrics(y, probs, CLASSES)
    assert m["screening"]["missed_disease_cases"] == 1
    assert m["per_class"]["PNEUMONIA"]["recall_sensitivity"] == 0.5


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(0, 1, (2000, 3))
    logits[np.arange(2000), y] += 1.5
    logits *= 4  # make it badly over-confident
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert expected_calibration_error(softmax(logits, t), y) < expected_calibration_error(softmax(logits), y)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    probs = softmax(np.eye(3)[y] * 2 + rng.normal(0, 1, (300, 3)))
    point = compute_metrics(y, probs, CLASSES)["balanced_accuracy"]
    lo, hi = bootstrap_ci(y, probs, n_boot=200)["balanced_accuracy"]
    assert lo <= point <= hi


def test_psi():
    assert psi([10, 10, 10], [10, 10, 10]) < 1e-6
    assert psi([30, 0, 0], [0, 0, 30]) > 0.25


def test_drift_needs_minimum_sample(tiny_bundle, tmp_path):
    import json

    import torch

    from lungxray.monitoring import drift_report

    bundle = torch.load(tiny_bundle, weights_only=True)
    bins = [i / 10 for i in range(11)]
    bundle["reference_stats"] = {
        f: {"values": [10] * 10, "bins": bins} for f in ("intensity_mean", "intensity_std", "confidence")
    }
    path = tmp_path / "m.pt"
    torch.save(bundle, path)
    log = tmp_path / "log.jsonl"
    row = {
        "prediction": "NORMAL",
        "confidence": 0.99,
        "needs_review": False,
        "intensity_mean": 0.95,
        "intensity_std": 0.95,
        "latency_ms": 5,
    }
    log.write_text("\n".join(json.dumps(row) for _ in range(5)))
    assert drift_report(path, log)["alert"] is False
    log.write_text("\n".join(json.dumps(row) for _ in range(150)))
    assert drift_report(path, log)["alert"] is True
