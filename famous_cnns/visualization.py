"""Model-agnostic visualizations used by the high-level orchestrator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn


def plot_history(history: dict[str, list[float]], *, figsize: tuple[int, int] = (11, 4)):
    """Plot loss and metric curves and return the Matplotlib figure."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(history.get("train_loss", []), label="train")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="validation")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].legend()

    axes[1].plot(history.get("train_metric", []), label="train")
    if history.get("val_metric"):
        axes[1].plot(history["val_metric"], label="validation")
    axes[1].set(xlabel="epoch", ylabel="metric", title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_predictions(
    images: Tensor,
    predictions: Tensor,
    targets: Tensor | None = None,
    *,
    class_names: Sequence[str] | None = None,
    max_images: int = 8,
):
    """Plot a compact grid of classification predictions."""

    count = min(max_images, len(images))
    cols = min(4, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    predicted_classes = predictions.argmax(dim=1) if predictions.ndim > 1 else predictions

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        image = images[index].detach().cpu()
        if image.shape[0] == 1:
            axis.imshow(image.squeeze(0), cmap="gray")
        else:
            axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        predicted = int(predicted_classes[index])
        predicted_label = class_names[predicted] if class_names else str(predicted)
        title = f"pred: {predicted_label}"
        if targets is not None:
            actual = int(targets[index])
            actual_label = class_names[actual] if class_names else str(actual)
            title += f" | true: {actual_label}"
            axis.title.set_color("green" if predicted == actual else "red")
        axis.set_title(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_feature_maps(
    model: nn.Module,
    inputs: Tensor,
    *,
    layer: str | None = None,
    max_maps: int = 16,
    forward: Any | None = None,
):
    """Capture and plot channels from a named layer (last Conv2d by default)."""

    named_modules = dict(model.named_modules())
    if layer is None:
        candidates = [(name, module) for name, module in named_modules.items() if isinstance(module, nn.Conv2d)]
        if not candidates:
            raise ValueError("El modelo no contiene capas Conv2d.")
        layer, selected = candidates[-1]
    else:
        if layer not in named_modules:
            raise ValueError(f"No existe la capa '{layer}'. Usa dict(model.named_modules()) para listarlas.")
        selected = named_modules[layer]

    captured: list[Tensor] = []
    hook = selected.register_forward_hook(lambda _module, _args, output: captured.append(output.detach()))
    was_training = model.training
    model.eval()
    try:
        (forward or model)(inputs)
    finally:
        hook.remove()
        model.train(was_training)
    if not captured or captured[0].ndim != 4:
        raise ValueError(f"La salida de '{layer}' no es un mapa NCHW.")

    maps = captured[0][0].cpu()
    count = min(max_maps, len(maps))
    cols = min(4, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            axis.imshow(maps[index], cmap="viridis")
            axis.set_title(f"{layer} [{index}]")
    fig.tight_layout()
    return fig
