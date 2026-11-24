"""Evaluate any model bundle: calibrate on validation, report on the leakage-free test set."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lungxray.config import Config  # noqa: E402
from lungxray.data.dataset import Preprocessing, eval_transform, load_xray, make_loaders  # noqa: E402
from lungxray.data.prepare import PreparedData, prepare_data  # noqa: E402
from lungxray.evaluation.calibration import expected_calibration_error, fit_temperature, softmax  # noqa: E402
from lungxray.evaluation.metrics import bootstrap_ci, compute_metrics  # noqa: E402
from lungxray.evaluation.report import plot_confusion, plot_reliability, plot_roc  # noqa: E402
from lungxray.explain.gradcam import GradCAM, overlay  # noqa: E402
from lungxray.models.bundle import load_bundle, update_bundle  # noqa: E402
from lungxray.models.factory import gradcam_target_layer  # noqa: E402
from lungxray.utils import get_device, get_logger, save_json  # noqa: E402

log = get_logger(__name__)
CONF_BINS = np.linspace(0, 1, 11)


def image_stats(img) -> tuple[float, float]:
    """Mean intensity and contrast of a radiograph (inputs to drift monitoring)."""
    a = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    return float(a.mean()), float(a.std())


def reference_stats(records, conf: np.ndarray, limit: int = 600) -> dict:
    stats = np.array([image_stats(load_xray(r.path, 256)) for r in records[:limit]])
    return {
        "intensity_mean": {
            "values": np.histogram(stats[:, 0], bins=CONF_BINS)[0].tolist(),
            "bins": CONF_BINS.tolist(),
        },
        "intensity_std": {
            "values": np.histogram(stats[:, 1], bins=np.linspace(0, 0.5, 11))[0].tolist(),
            "bins": np.linspace(0, 0.5, 11).tolist(),
        },
        "confidence": {"values": np.histogram(conf, bins=CONF_BINS)[0].tolist(), "bins": CONF_BINS.tolist()},
    }


def gradcam_gallery(model, arch, records, y_true, probs, classes, pp, device, path: Path, n=8) -> None:
    pred = probs.argmax(1)
    wrong = [i for i in np.argsort(-probs.max(1)) if pred[i] != y_true[i]][: n // 2]
    right = []
    for k in range(len(classes)):
        right += [i for i in np.where((pred == k) & (y_true == k))[0][:2]]
    picks = (wrong + right)[:n]
    if not picks:
        return
    cam = GradCAM(model, gradcam_target_layer(model, arch))
    tf = eval_transform(pp)
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), picks, strict=False):
        img = load_xray(records[i].path, 512)
        heat, _ = cam(tf(img).unsqueeze(0).to(device), int(pred[i]))
        ax.imshow(overlay(img, heat))
        ok = "OK" if pred[i] == y_true[i] else "WRONG"
        ax.set_title(
            f"{ok} true={classes[y_true[i]]}\npred={classes[pred[i]]} ({probs[i].max():.0%})",
            fontsize=8,
            color="green" if ok == "OK" else "red",
        )
    cam.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_bundle(
    bundle_path: str | Path, cfg: Config, out_dir: str | Path, data: PreparedData | None = None
) -> dict:
    from lungxray.training.trainer import predict_logits

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, meta = load_bundle(bundle_path, device)
    classes, pp = meta["classes"], Preprocessing.from_dict(meta["preprocessing"])
    data = data or prepare_data(cfg)
    loaders = make_loaders(
        data.records, classes, pp, cfg.data.batch_size, cfg.data.num_workers, "none", data.excluded_test
    )

    val_logits, val_y = predict_logits(model, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_y)
    test_logits, test_y = predict_logits(model, loaders["test"], device)
    raw, cal = softmax(test_logits), softmax(test_logits, temperature)

    metrics = compute_metrics(test_y, cal, classes)
    metrics["temperature"] = temperature
    metrics["ece_before_calibration"] = expected_calibration_error(raw, test_y)
    metrics["ci95"] = bootstrap_ci(test_y, cal)
    metrics["test_set"] = (
        "clean (near-duplicates of training images removed)" if data.excluded_test else "full"
    )
    metrics["leakage"] = {k: v for k, v in data.leakage.items() if not k.endswith("_paths")}
    thr = meta.get("referral_threshold", cfg.inference.referral_threshold)
    referred = cal.max(1) < thr
    metrics["referral"] = {
        "threshold": thr,
        "fraction_referred": float(referred.mean()),
        "accuracy_on_non_referred": float((cal.argmax(1) == test_y)[~referred].mean())
        if (~referred).any()
        else None,
    }

    if data.excluded_test:  # also report the inflated number, for transparency
        full_recs = [r for r in data.records if r.split == "test"]
        full = make_loaders(full_recs, classes, pp, cfg.data.batch_size, cfg.data.num_workers, "none")
        fl, fy = predict_logits(model, full["test"], device)
        fm = compute_metrics(fy, softmax(fl, temperature), classes)
        metrics["full_test_with_leaks"] = {
            k: fm[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1")
        }

    plot_confusion(
        metrics["confusion_matrix"], classes, out_dir / "confusion_matrix.png", f"{meta['arch']} - test"
    )
    plot_roc(test_y, cal, classes, out_dir / "roc_curves.png")
    plot_reliability(
        test_y, raw, cal, out_dir / "reliability.png", metrics["ece_before_calibration"], metrics["ece"]
    )
    test_records = loaders["test"].dataset.records
    gradcam_gallery(
        model, meta["arch"], test_records, test_y, cal, classes, pp, device, out_dir / "gradcam_gallery.png"
    )

    val_records = loaders["val"].dataset.records
    ref = reference_stats(val_records, softmax(val_logits, temperature).max(1))
    summary = {k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc", "ece")}
    summary["per_class_recall"] = {c: v["recall_sensitivity"] for c, v in metrics["per_class"].items()}
    if "screening" in metrics:
        summary["abnormal_sensitivity"] = metrics["screening"]["abnormal_sensitivity"]
    update_bundle(bundle_path, temperature=temperature, metrics=summary, reference_stats=ref)
    save_json(metrics, out_dir / "metrics.json")
    log.info(
        "TEST (%s, n=%d): acc %.4f | bal_acc %.4f [%.4f-%.4f] | macro_f1 %.4f | AUC %.4f | "
        "ECE %.3f -> %.3f (T=%.2f)",
        metrics["test_set"].split()[0],
        metrics["n"],
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        *metrics["ci95"]["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["macro_roc_auc"],
        metrics["ece_before_calibration"],
        metrics["ece"],
        temperature,
    )
    return metrics
