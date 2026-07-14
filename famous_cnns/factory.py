"""Unified model registry and model-construction helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelInfo:
    """Public metadata for an architecture in the registry."""

    name: str
    task: str
    default_input_channels: int
    recommended_input_size: int | None
    description: str


@dataclass(frozen=True)
class _ModelSpec:
    info: ModelInfo
    module: str
    class_name: str
    defaults: dict[str, Any] | None = None
    builder: Callable[[_ModelSpec, int, dict[str, Any]], nn.Module] | None = None


def _standard_builder(spec: _ModelSpec, num_classes: int, kwargs: dict[str, Any]) -> nn.Module:
    module = importlib.import_module(spec.module)
    model_class = getattr(module, spec.class_name)
    options = dict(spec.defaults or {})
    options.update(kwargs)
    return model_class(num_classes=num_classes, **options)


def _efficientnet_builder(spec: _ModelSpec, num_classes: int, kwargs: dict[str, Any]) -> nn.Module:
    variant = str(kwargs.pop("variant", "b0")).lower()
    if variant.startswith("b"):
        variant = variant[1:]
    try:
        phi = int(kwargs.pop("phi", variant))
    except ValueError as exc:
        raise ValueError("variant debe estar entre 'b0' y 'b7'.") from exc
    if phi not in range(8):
        raise ValueError("EfficientNet admite variant='b0' ... 'b7'.")

    module = importlib.import_module(spec.module)
    scaler_module = importlib.import_module("efficient-net.model.compuder_scaler")
    kwargs["scaler"] = scaler_module.CompoundScaler(phi=phi)
    return getattr(module, spec.class_name)(num_classes=num_classes, **kwargs)


_SPECS: dict[str, _ModelSpec] = {
    "lenet5": _ModelSpec(
        ModelInfo("lenet5", "classification", 1, 32, "LeNet-5"),
        "lenet.src.model",
        "LeNet5",
    ),
    "alexnet": _ModelSpec(
        ModelInfo("alexnet", "classification", 3, 224, "AlexNet"),
        "alexnet.src.model",
        "AlexNet",
    ),
    "vgg16": _ModelSpec(
        ModelInfo("vgg16", "classification", 3, 224, "VGG-16"),
        "vgg.src.model.vgg16",
        "VGG16",
    ),
    "inception_v1": _ModelSpec(
        ModelInfo("inception_v1", "classification", 3, 96, "GoogLeNet / Inception v1"),
        "incpetion.model.incpetionv1",
        "GoogLeNetV1",
    ),
    "resnet50": _ModelSpec(
        ModelInfo("resnet50", "classification", 3, 224, "ResNet-50"),
        "resnet.ResNet50.src.model",
        "ResNet50",
    ),
    "resnet101": _ModelSpec(
        ModelInfo("resnet101", "classification", 3, 224, "ResNet-101"),
        "resnet.ResNet101.src.model.restnet",
        "ResNet",
        defaults={"blocks_per_stage": (3, 4, 23, 3)},
    ),
    "unet": _ModelSpec(
        ModelInfo("unet", "segmentation", 3, None, "U-Net"),
        "u-net.src.model",
        "UNet",
    ),
    "mobilenet_v1": _ModelSpec(
        ModelInfo("mobilenet_v1", "classification", 3, 224, "MobileNet v1"),
        "mobilenet.model.mobielnet",
        "MobileNet",
        defaults={"version": "v1"},
    ),
    "mobilenet_v2": _ModelSpec(
        ModelInfo("mobilenet_v2", "classification", 3, 224, "MobileNet v2"),
        "mobilenet.model.mobielnet",
        "MobileNet",
        defaults={"version": "v2"},
    ),
    "efficientnet": _ModelSpec(
        ModelInfo("efficientnet", "classification", 3, 224, "EfficientNet B0-B7"),
        "efficient-net.model.Efficent_Net",
        "EfficientNet",
        builder=_efficientnet_builder,
    ),
}

_ALIASES = {
    "lenet": "lenet5",
    "googlenet": "inception_v1",
    "inception": "inception_v1",
    "u_net": "unet",
    "mobilenet": "mobilenet_v1",
    "efficient_net": "efficientnet",
}


def _canonical_name(name: str) -> tuple[str, dict[str, Any]]:
    normalized = name.lower().strip().replace("-", "_").replace(" ", "_")
    implied: dict[str, Any] = {}
    if normalized.startswith("efficientnet_b") and normalized[-1:].isdigit():
        implied["variant"] = normalized.rsplit("_", 1)[-1]
        normalized = "efficientnet"
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _SPECS:
        choices = ", ".join(_SPECS)
        raise ValueError(f"Modelo desconocido '{name}'. Disponibles: {choices}.")
    return normalized, implied


def list_models(task: str | None = None) -> list[str]:
    """Return canonical model names, optionally filtered by task."""

    if task is None:
        return list(_SPECS)
    task = task.lower()
    return [name for name, spec in _SPECS.items() if spec.info.task == task]


def model_info(name: str) -> ModelInfo:
    """Return task and input metadata for a registered model."""

    canonical, _ = _canonical_name(name)
    return _SPECS[canonical].info


def _replace_child(parent: nn.Module, name: str, module: nn.Module) -> None:
    if isinstance(parent, nn.Sequential | nn.ModuleList):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def adapt_input_channels(model: nn.Module, in_channels: int) -> nn.Module:
    """Replace the first Conv2d so an existing architecture accepts any channel count.

    Existing weights are preserved sensibly: RGB weights are averaged for grayscale,
    or repeated and rescaled when more channels are requested.
    """

    if in_channels < 1:
        raise ValueError("in_channels debe ser un entero positivo.")

    for full_name, child in model.named_modules():
        if not isinstance(child, nn.Conv2d):
            continue
        if child.in_channels == in_channels:
            return model
        if child.groups != 1:
            raise ValueError("La primera convolución agrupada no puede adaptarse automáticamente.")

        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = nn.Conv2d(
            in_channels,
            child.out_channels,
            child.kernel_size,
            child.stride,
            child.padding,
            child.dilation,
            child.groups,
            child.bias is not None,
            child.padding_mode,
            device=child.weight.device,
            dtype=child.weight.dtype,
        )
        with torch.no_grad():
            old = child.weight
            if in_channels == 1:
                replacement.weight.copy_(old.mean(dim=1, keepdim=True))
            else:
                repeats = (in_channels + old.shape[1] - 1) // old.shape[1]
                expanded = old.repeat(1, repeats, 1, 1)[:, :in_channels]
                replacement.weight.copy_(expanded * old.shape[1] / in_channels)
            if child.bias is not None:
                replacement.bias.copy_(child.bias)
        _replace_child(parent, child_name, replacement)
        return model
    raise ValueError("No se encontró ninguna capa Conv2d para adaptar.")


def create_model(
    name: str,
    *,
    num_classes: int,
    in_channels: int | None = None,
    **model_kwargs: Any,
) -> nn.Module:
    """Build any repository architecture through one stable entry point."""

    canonical, implied = _canonical_name(name)
    spec = _SPECS[canonical]
    options = {**implied, **model_kwargs}
    builder = spec.builder or _standard_builder
    model = builder(spec, num_classes, options)
    requested_channels = spec.info.default_input_channels if in_channels is None else in_channels
    adapt_input_channels(model, requested_channels)
    model.architecture_name = canonical
    model.task = spec.info.task
    model.input_channels = requested_channels
    model.recommended_input_size = spec.info.recommended_input_size
    return model
