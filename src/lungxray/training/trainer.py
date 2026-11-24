"""Training loop: class-weighted loss, backbone warm-up, warmup+cosine LR, AMP on GPU,
early stopping on validation balanced accuracy, best-checkpoint bundle."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lungxray.config import Config
from lungxray.data.dataset import Preprocessing, class_weights, make_loaders
from lungxray.data.prepare import prepare_data
from lungxray.evaluation.calibration import softmax
from lungxray.evaluation.metrics import compute_metrics
from lungxray.evaluation.report import plot_history
from lungxray.models.bundle import save_bundle
from lungxray.models.factory import build_model, count_parameters, set_backbone_trainable
from lungxray.utils import get_device, get_logger, save_json, seed_everything

log = get_logger(__name__)


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    if not logits:
        return np.zeros((0, 0)), np.zeros(0, dtype=int)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


class _Tracker:
    """Optional MLflow logging that degrades to a no-op."""

    def __init__(self, enabled: bool, cfg: Config, run_name: str):
        self.mlflow = None
        if enabled:
            try:
                import mlflow

                mlflow.set_experiment(cfg.project)
                mlflow.start_run(run_name=run_name)
                mlflow.log_params(
                    {
                        "model": cfg.model.name,
                        "lr": cfg.train.lr,
                        "epochs": cfg.train.epochs,
                        "batch_size": cfg.data.batch_size,
                        "balance": cfg.data.balance,
                        "seed": cfg.seed,
                    }
                )
                self.mlflow = mlflow
            except ImportError:
                log.warning("tracking.mlflow=true but mlflow is not installed; skipping")

    def metrics(self, d: dict, step: int) -> None:
        if self.mlflow:
            self.mlflow.log_metrics({k: v for k, v in d.items() if isinstance(v, (int, float))}, step=step)

    def artifacts(self, path: Path) -> None:
        if self.mlflow:
            self.mlflow.log_artifacts(str(path))
            self.mlflow.end_run()


def train(cfg: Config, run_name: str | None = None) -> Path:
    seed_everything(cfg.seed)
    device = get_device()
    run_name = run_name or f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.model.name}"
    run_dir = Path(cfg.output_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg.model_dump(), run_dir / "config.json")

    data = prepare_data(cfg)
    save_json(
        {
            "counts": data.counts(),
            "leakage": {k: v for k, v in data.leakage.items() if not k.endswith("_paths")},
        },
        run_dir / "data_summary.json",
    )
    pp = Preprocessing(image_size=cfg.data.image_size, resize=cfg.data.resize)
    loaders = make_loaders(
        data.records,
        cfg.data.classes,
        pp,
        cfg.data.batch_size,
        cfg.data.num_workers,
        cfg.data.balance,
        data.excluded_test,
    )

    model = build_model(cfg.model.name, len(cfg.data.classes), cfg.model.pretrained, cfg.model.dropout)
    model.to(device)
    log.info(
        "Model %s | %s params | device %s | run %s",
        cfg.model.name,
        f"{count_parameters(model):,}",
        device,
        run_dir,
    )

    weight = None
    if cfg.data.balance == "weighted_loss":
        weight = class_weights([r for r in data.records if r.split == "train"], cfg.data.classes)
        log.info("Class weights: %s", dict(zip(cfg.data.classes, weight.tolist(), strict=True)))
    criterion = nn.CrossEntropyLoss(
        weight=weight.to(device) if weight is not None else None,
        label_smoothing=cfg.train.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = len(loaders["train"])
    if cfg.train.max_train_batches:
        steps_per_epoch = min(steps_per_epoch, cfg.train.max_train_batches)
    total, warmup = cfg.train.epochs * steps_per_epoch, steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else nullcontext

    tracker = _Tracker(cfg.tracking.mlflow, cfg, run_name)
    history: list[dict] = []
    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    best_path = run_dir / "model.pt"

    for epoch in range(1, cfg.train.epochs + 1):
        warm = epoch <= cfg.train.freeze_backbone_epochs and cfg.model.name != "custom_cnn"
        set_backbone_trainable(model, cfg.model.name, trainable=not warm)
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        pbar = tqdm(loaders["train"], total=steps_per_epoch, desc=f"epoch {epoch}", leave=False)
        for step, (x, y) in enumerate(pbar):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)
            pbar.set_postfix(loss=f"{running / seen:.4f}")

        val_logits, val_y = predict_logits(model, loaders["val"], device)
        val_m = compute_metrics(val_y, softmax(val_logits), cfg.data.classes)
        val_loss = float(nn.functional.cross_entropy(torch.tensor(val_logits), torch.tensor(val_y)))
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_macro_roc_auc": val_m["macro_roc_auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "backbone_frozen": warm,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        tracker.metrics(row, epoch)
        log.info(
            "epoch %d | train_loss %.4f | val_loss %.4f | val_bal_acc %.4f | val_f1 %.4f | %.0fs",
            epoch,
            row["train_loss"],
            val_loss,
            row["val_balanced_accuracy"],
            row["val_macro_f1"],
            row["seconds"],
        )

        score = row[cfg.train.monitor]
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_bundle(
                best_path,
                model,
                cfg.model.name,
                cfg.data.classes,
                pp,
                referral_threshold=cfg.inference.referral_threshold,
                run_id=run_name,
                extra={"best_epoch": epoch, "val_score": score},
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stopping_patience:
                log.info("Early stopping: no improvement for %d epochs", bad_epochs)
                break

    save_json(history, run_dir / "history.json")
    plot_history(history, run_dir / "training_curves.png")
    log.info("Best %s = %.4f at epoch %d", cfg.train.monitor, best_score, best_epoch)

    # Calibrate + full evaluation report on the best checkpoint
    from lungxray.evaluation.evaluate import evaluate_bundle

    evaluate_bundle(best_path, cfg, run_dir, data=data)
    tracker.artifacts(run_dir)
    return run_dir
