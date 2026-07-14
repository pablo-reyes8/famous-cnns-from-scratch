"""Optimizer factory shared by every architecture."""

from __future__ import annotations

from typing import Any, Iterable

from torch import Tensor
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop


_OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop,
}


def create_optimizer(
    parameters: Iterable[Tensor], name: str = "adamw", *, lr: float = 1e-3, **kwargs: Any
) -> Optimizer:
    """Create Adam, AdamW, SGD or RMSprop from a common interface."""

    key = name.lower().replace("_", "")
    if key not in _OPTIMIZERS:
        choices = ", ".join(_OPTIMIZERS)
        raise ValueError(f"Optimizador desconocido '{name}'. Disponibles: {choices}.")
    return _OPTIMIZERS[key](parameters, lr=lr, **kwargs)
