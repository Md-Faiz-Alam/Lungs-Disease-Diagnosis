"""End-to-end: audit -> split -> train -> calibrate -> evaluate -> bundle -> predict."""

import json

from lungxray.config import load_config
from lungxray.inference.predictor import Predictor
from lungxray.training.trainer import train
from tests.conftest import fake_xray


def test_train_evaluate_predict_end_to_end(synthetic_data, tmp_path):
    cfg = load_config(
        tmp_path / "none.yaml",
        [
            f"data.train_dir={synthetic_data / 'train'}",
            f"data.test_dir={synthetic_data / 'test'}",
            "data.image_size=64",
            "data.batch_size=8",
            "data.num_workers=0",
            "data.val_fraction=0.25",
            "model.name=resnet18",
            "model.pretrained=false",
            "train.epochs=2",
            f"output_dir={tmp_path / 'artifacts'}",
        ],
    )
    run_dir = train(cfg, run_name="smoke")
    for f in (
        "model.pt",
        "metrics.json",
        "history.json",
        "confusion_matrix.png",
        "reliability.png",
        "roc_curves.png",
        "training_curves.png",
    ):
        assert (run_dir / f).exists(), f
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["leakage"]["test_images_with_train_duplicate"] == 1
    assert metrics["n"] == 15  # the leaked test image was excluded
    assert metrics["full_test_with_leaks"]["n"] == 16

    res = Predictor(run_dir / "model.pt").predict(fake_xray(99, 0))
    assert res["prediction"] in cfg.data.classes
