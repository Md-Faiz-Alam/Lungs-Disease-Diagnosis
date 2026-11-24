"""Command line interface:  lungxray <command> --help"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lungxray.config import load_config
from lungxray.utils import get_logger, save_json

log = get_logger("lungxray")


def cmd_audit(args, cfg) -> None:
    from lungxray.data.audit import class_metadata_summary, metadata_shortcut_probe
    from lungxray.data.prepare import prepare_data

    data = prepare_data(cfg)
    report = {
        "counts": data.counts(),
        "class_metadata": class_metadata_summary(data.records, cfg.data.classes),
        "leakage": data.leakage,
        "shortcut_probe": metadata_shortcut_probe(data.records, cfg.data.classes, cfg.seed),
    }
    out = Path(cfg.output_dir) / "audit" / "audit.json"
    save_json(report, out)
    printable = {
        **report,
        "leakage": {k: v for k, v in report["leakage"].items() if not k.endswith("_paths")},
    }
    print(json.dumps(printable, indent=2))
    log.info("Full audit written to %s", out)


def cmd_train(args, cfg) -> None:
    from lungxray.training.promote import promote
    from lungxray.training.trainer import train

    run_dir = train(cfg, args.run_name)
    if args.promote:
        promote(run_dir, cfg)


def cmd_evaluate(args, cfg) -> None:
    from lungxray.evaluation.evaluate import evaluate_bundle

    out = args.out or str(Path(args.model).parent / "eval")
    evaluate_bundle(args.model, cfg, out)


def cmd_import_legacy(args, cfg) -> None:
    from lungxray.models.bundle import import_legacy_state_dict

    import_legacy_state_dict(args.weights, args.out, cfg.data.classes)
    log.info("Wrapped legacy weights into %s", args.out)


def cmd_promote(args, cfg) -> None:
    from lungxray.training.promote import promote

    promote(args.run, cfg, force=args.force)


def cmd_export(args, cfg) -> None:
    from lungxray.export import export_onnx

    export_onnx(args.model, args.out)


def cmd_predict(args, cfg) -> None:
    from lungxray.inference.predictor import Predictor

    p = Predictor(args.model)
    for path in args.images:
        res = p.predict(Predictor.open_image(Path(path).read_bytes()))
        print(path, json.dumps({k: v for k, v in res.items() if k != "gradcam_png_base64"}))


def cmd_drift(args, cfg) -> None:
    from lungxray.monitoring import drift_report

    print(json.dumps(drift_report(args.model, args.log), indent=2))


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/config.yaml")
    common.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config, e.g. --set model.name=resnet18 train.epochs=3",
    )
    parser = argparse.ArgumentParser(prog="lungxray", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_, parents=[common])

    add("audit", "leakage, duplicates and shortcut audit of the dataset")
    p = add("train", "train, calibrate and evaluate a model")
    p.add_argument("--run-name")
    p.add_argument("--promote", action="store_true", help="promote if the quality gate passes")
    p = add("evaluate", "evaluate a model bundle on the clean test set")
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    p = add("import-legacy", "wrap the original v0 weights into a bundle")
    p.add_argument("--weights", default="models/legacy/custom_cnn_v0.pth")
    p.add_argument("--out", default="models/legacy/custom_cnn_v0.pt")
    p = add("promote", "quality-gated promotion of a run to production")
    p.add_argument("--run", required=True)
    p.add_argument("--force", action="store_true")
    p = add("export-onnx", "export to ONNX with parity check and benchmark")
    p.add_argument("--model", default="models/production/model.pt")
    p.add_argument("--out", default="models/production/model.onnx")
    p = add("predict", "predict on image files")
    p.add_argument("--model", default="models/production/model.pt")
    p.add_argument("images", nargs="+")
    p = add("drift", "drift report from the API prediction log")
    p.add_argument("--model", default="models/production/model.pt")
    p.add_argument("--log", default="logs/predictions.jsonl")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    {
        "audit": cmd_audit,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "import-legacy": cmd_import_legacy,
        "promote": cmd_promote,
        "export-onnx": cmd_export,
        "predict": cmd_predict,
        "drift": cmd_drift,
    }[args.command](args, cfg)


if __name__ == "__main__":
    main()
