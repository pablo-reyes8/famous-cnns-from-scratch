"""Command-line interface for training and inference across all architectures."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .data import IMAGE_SUFFIXES, build_loaders, image_transform
from .factory import list_models, model_info
from .orchestrator import CNNOrchestrator


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"JSON inválido: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("El valor debe ser un objeto JSON.")
    return parsed


def _add_model_arguments(parser: argparse.ArgumentParser, *, model_required: bool = True) -> None:
    parser.add_argument("--model", choices=list_models(), required=model_required)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="cpu, cuda, cuda:0; por defecto se detecta automáticamente"
    )
    parser.add_argument("--model-kwargs", type=_json_object, default={}, metavar="JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="famous-cnns", description="Unified CLI for famous CNNs")
    parser.add_argument("--version", action="version", version="famous-cnns 0.3.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List registered architectures")
    info_parser = subparsers.add_parser("info", help="Show architecture metadata")
    info_parser.add_argument("model", choices=list_models())

    train_parser = subparsers.add_parser(
        "train", help="Train a classification or segmentation model"
    )
    _add_model_arguments(train_parser)
    train_parser.add_argument("--data-dir", type=Path)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument(
        "--optimizer", choices=("adam", "adamw", "sgd", "rmsprop"), default="adamw"
    )
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--optimizer-kwargs", type=_json_object, default={}, metavar="JSON")
    train_parser.add_argument("--output", type=Path, default=None)
    train_parser.add_argument(
        "--smoke-test", action="store_true", help="Use a tiny synthetic dataset"
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference from a saved checkpoint")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--input", type=Path)
    infer_parser.add_argument("--device", default=None)
    infer_parser.add_argument("--top-k", type=int, default=5)
    infer_parser.add_argument("--class-names", type=Path, help="JSON list with class names")
    infer_parser.add_argument("--output", type=Path)
    infer_parser.add_argument("--smoke-test", action="store_true")
    return parser


def _resolved_image_size(model: str, requested: int | None) -> int:
    return requested or model_info(model).recommended_input_size or 128


def train_command(args: argparse.Namespace) -> int:
    info = model_info(args.model)
    in_channels = args.in_channels or info.default_input_channels
    image_size = _resolved_image_size(args.model, args.image_size)
    train_loader, val_loader, class_names = build_loaders(
        model_name=args.model,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        in_channels=in_channels,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
    )
    cnn = CNNOrchestrator(
        args.model,
        num_classes=args.num_classes,
        in_channels=in_channels,
        optimizer=args.optimizer,
        lr=args.lr,
        optimizer_kwargs=args.optimizer_kwargs,
        device=args.device,
        **args.model_kwargs,
    )
    cnn.fit(train_loader, epochs=args.epochs, val_loader=val_loader)
    output = args.output or Path("outputs") / f"{args.model}.pt"
    cnn.configuration.update(image_size=image_size, class_names=class_names)
    cnn.save(output)
    print(json.dumps({"checkpoint": str(output), **cnn.summary()}, indent=2))
    return 0


def _input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(path)


def infer_command(args: argparse.Namespace) -> int:
    cnn = CNNOrchestrator.from_checkpoint(args.checkpoint, device=args.device)
    image_size = int(cnn.configuration.get("image_size") or cnn.info.recommended_input_size or 128)
    in_channels = int(cnn.model.input_channels)
    if args.smoke_test:
        images = torch.randn(1, in_channels, image_size, image_size)
        paths = [Path("synthetic")]
    else:
        if args.input is None:
            raise ValueError("--input es obligatorio salvo cuando se usa --smoke-test.")
        paths = _input_paths(args.input)
        transform = image_transform(image_size, in_channels)
        images = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
    probabilities = cnn.predict(images).cpu()
    output_path = args.output
    if cnn.info.task == "segmentation":
        masks = (
            (probabilities >= 0.5).to(torch.uint8)
            if cnn.num_classes == 1
            else probabilities.argmax(dim=1)
        )
        output_path = output_path or Path("outputs") / "segmentation_masks.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"inputs": [str(path) for path in paths], "masks": masks}, output_path)
        result: Any = {"output": str(output_path), "shape": list(masks.shape)}
    else:
        class_names = cnn.configuration.get("class_names")
        if args.class_names:
            class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
        k = min(args.top_k, probabilities.shape[1])
        values, indices = probabilities.topk(k, dim=1)
        result = []
        for path, sample_values, sample_indices in zip(paths, values, indices, strict=False):
            predictions = [
                {
                    "class_id": int(index),
                    "class_name": class_names[int(index)] if class_names else None,
                    "probability": float(value),
                }
                for value, index in zip(sample_values, sample_indices, strict=False)
            ]
            result.append({"input": str(path), "predictions": predictions})
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for name in list_models():
            info = model_info(name)
            print(f"{name:16} {info.task:14} {info.description}")
        return 0
    if args.command == "info":
        print(json.dumps(model_info(args.model).__dict__, indent=2))
        return 0
    if args.command == "train":
        return train_command(args)
    return infer_command(args)


def architecture_main(model: str, command: str, argv: Sequence[str] | None = None) -> int:
    """Entry point used by the thin architecture-specific scripts."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    return main(
        [command, "--model", model, *arguments] if command == "train" else [command, *arguments]
    )
