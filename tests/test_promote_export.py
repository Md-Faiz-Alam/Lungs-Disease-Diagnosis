"""Quality-gated promotion and ONNX export."""

import json

import torch

from lungxray.config import load_config
from lungxray.export import export_onnx
from lungxray.training.promote import check_gate, promote


def _run(tmp_path, name, bundle, bal_acc, recalls):
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    b = torch.load(bundle, weights_only=True)
    b["metrics"] = {"balanced_accuracy": bal_acc, "per_class_recall": recalls}
    torch.save(b, run / "model.pt")
    return run


def test_quality_gate_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(tmp_path / "none.yaml")
    assert check_gate({"balanced_accuracy": 0.95, "per_class_recall": {"A": 0.9}}, cfg) == []
    failures = check_gate({"balanced_accuracy": 0.80, "per_class_recall": {"A": 0.5}}, cfg)
    assert any("balanced_accuracy" in f for f in failures) and any("recall[A]" in f for f in failures)


def test_promote_only_better_models(tiny_bundle, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(tmp_path / "none.yaml")
    good = {"A": 0.95, "B": 0.96}
    assert promote(_run(tmp_path, "v1", tiny_bundle, 0.95, good), cfg) is True
    assert promote(_run(tmp_path, "v2-worse", tiny_bundle, 0.93, good), cfg) is False  # not better
    assert promote(_run(tmp_path, "v3-weak-class", tiny_bundle, 0.99, {"A": 0.99, "B": 0.5}), cfg) is False
    assert promote(_run(tmp_path, "v4", tiny_bundle, 0.97, good), cfg) is True

    prod = torch.load(tmp_path / "models/production/model.pt", weights_only=True)
    assert prod["metrics"]["balanced_accuracy"] == 0.97
    registry = json.loads((tmp_path / "models/registry.json").read_text())
    assert [r["run"] for r in registry] == ["v1", "v4"]

    assert promote(_run(tmp_path, "v5-forced", tiny_bundle, 0.5, good), cfg, force=True) is True
    assert json.loads((tmp_path / "models/registry.json").read_text())[-1]["forced"] is True


def test_onnx_export_matches_pytorch(tiny_bundle, tmp_path):
    report = export_onnx(tiny_bundle, tmp_path / "model.onnx", runs=2)
    assert report["parity_ok"] and report["max_abs_logit_diff"] < 1e-3
    assert (tmp_path / "model.json").exists()
