"""Train on a Kaggle GPU notebook.

In a Kaggle notebook (Settings -> Accelerator: GPU T4 x2 or P100, Internet: on), add the
dataset "prashant268/chest-xray-covid19-pneumonia" as input, then run:

    !git clone https://github.com/Md-Faiz-Alam/Lungs-Disease-Diagnosis.git
    %cd Lungs-Disease-Diagnosis
    !pip install -q -e .
    !python kaggle/run_kaggle.py --models efficientnet_b0 resnet18 densenet121 custom_cnn

Outputs go to /kaggle/working/artifacts (download from the notebook's Output tab):
  runs/<run>/model.pt + metrics/plots for every model, and comparison.md with a results table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lungxray.config import load_config
from lungxray.training.trainer import train


def find_data_root(base: Path = Path("/kaggle/input")) -> Path:
    """Locate the folder that contains train/<class> and test/<class>."""
    for train_dir in base.rglob("train"):
        if (train_dir / "COVID19").is_dir() and (train_dir.parent / "test").is_dir():
            return train_dir.parent
    sys.exit(f"Could not find train/ and test/ class folders under {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", default="/kaggle/working/artifacts")
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides")
    args = ap.parse_args()

    root = args.data_root or find_data_root()
    print(f"Data root: {root}")
    rows = []
    for name in args.models:
        cfg = load_config(
            "configs/config.yaml",
            [
                f"data.train_dir={root / 'train'}",
                f"data.test_dir={root / 'test'}",
                f"output_dir={args.out}",
                f"model.name={name}",
                f"train.epochs={args.epochs}",
                "data.batch_size=64",
                "data.num_workers=4",
                *args.set,
            ],
        )
        if name == "custom_cnn":  # trained from scratch: needs a higher LR, no warm-up freeze
            cfg.train.lr, cfg.model.pretrained = 3e-3, False
        run_dir = train(cfg, run_name=f"kaggle-{name}")
        m = json.loads((run_dir / "metrics.json").read_text())
        rows.append((name, m))

    lines = [
        "| Model | Accuracy | Balanced acc. (95% CI) | Macro F1 | Macro AUC | Abnormal sensitivity | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lo, hi = m["ci95"]["balanced_accuracy"]
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} ({lo:.3f}-{hi:.3f}) | "
            f"{m['macro_f1']:.3f} | {m['macro_roc_auc']:.3f} | "
            f"{m['screening']['abnormal_sensitivity']:.3f} | {m['ece']:.3f} |"
        )
    table = "\n".join(lines)
    Path(args.out, "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
