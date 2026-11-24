"""Export a bundle to ONNX, verify numerical parity, and benchmark CPU latency."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from lungxray.models.bundle import load_bundle
from lungxray.utils import get_logger, save_json

log = get_logger(__name__)


def export_onnx(bundle_path: str | Path, out_path: str | Path, runs: int = 30) -> dict:
    model, meta = load_bundle(bundle_path, "cpu")
    size = meta["preprocessing"]["image_size"]
    dummy = torch.randn(1, 3, size, size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import onnxscript  # noqa: F401  (required by the torch.export-based exporter)

        # Modern exporter (default since PyTorch 2.9): traces with torch.export, dynamic batch size.
        program = torch.onnx.export(
            model,
            (dummy,),
            dynamo=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},
            verbose=False,
        )
        program.save(str(out_path))
        exporter = "torch.export"
    except ImportError:  # onnxscript not installed: fall back to the legacy TorchScript exporter
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        exporter = "torchscript (legacy)"

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())

    one = x[:1]
    with torch.no_grad():
        for _ in range(3):
            model(one)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(one)
        torch_ms = (time.perf_counter() - t0) / runs * 1000
    one_np = one.numpy()
    for _ in range(3):
        sess.run(None, {"image": one_np})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"image": one_np})
    onnx_ms = (time.perf_counter() - t0) / runs * 1000

    report = {
        "onnx_path": out_path.as_posix(),
        "exporter": exporter,
        "size_mb": round(out_path.stat().st_size / 1e6, 2),
        "max_abs_logit_diff": max_diff,
        "parity_ok": max_diff < 1e-3,
        "cpu_latency_ms": {"pytorch": round(torch_ms, 2), "onnxruntime": round(onnx_ms, 2)},
    }
    save_json(report, out_path.with_suffix(".json"))
    log.info("ONNX export: %s", report)
    return report
