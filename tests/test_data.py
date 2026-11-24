from lungxray.config import load_config
from lungxray.data.audit import assign_duplicate_groups, build_manifest, leakage_report
from lungxray.data.dataset import Preprocessing, eval_transform, split_train_val, train_transform
from tests.conftest import CLASSES, fake_xray


def _records(root):
    return build_manifest({"train": str(root / "train"), "test": str(root / "test")}, CLASSES)


def test_config_overrides(tmp_path):
    cfg = load_config(
        tmp_path / "missing.yaml", ["model.name=resnet18", "train.epochs=2", "data.balance=none"]
    )
    assert cfg.model.name == "resnet18" and cfg.train.epochs == 2 and cfg.data.balance == "none"


def test_leak_is_detected(synthetic_data):
    records = _records(synthetic_data)
    assert assign_duplicate_groups(records, threshold=0) >= 2
    report = leakage_report(records)
    assert report["test_images_with_train_duplicate"] == 1
    assert report["leaked_test_paths"][0].endswith("leak.png")


def test_val_split_keeps_duplicate_groups_together(synthetic_data):
    records = _records(synthetic_data)
    assign_duplicate_groups(records, threshold=4)
    split_train_val(records, 0.2, seed=0)
    splits_per_group = {}
    for r in records:
        if r.split != "test":
            splits_per_group.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in splits_per_group.values())
    assert any(r.split == "val" for r in records)


def test_transforms_shape_and_determinism():
    pp = Preprocessing(image_size=96)
    img = fake_xray(1, 0, size=200).convert("RGB")
    a, b = eval_transform(pp)(img), eval_transform(pp)(img)
    assert a.shape == (3, 96, 96) and (a == b).all()
    assert train_transform(pp)(img).shape == (3, 96, 96)


def test_cli_accepts_overrides_after_subcommand(tmp_path, monkeypatch, capsys):
    from lungxray import cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_audit", lambda args, cfg: seen.setdefault("cfg", cfg))
    cli.main(["audit", "--config", str(tmp_path / "none.yaml"), "--set", "data.batch_size=7"])
    assert seen["cfg"].data.batch_size == 7
