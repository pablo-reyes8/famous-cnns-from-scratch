"""Training, evaluation and visualization facade for all registered CNNs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from .factory import create_model, model_info
from .optimizers import create_optimizer


class CNNOrchestrator:
    """A single high-level interface for classification and segmentation CNNs."""

    def __init__(
        self,
        architecture: str,
        *,
        num_classes: int,
        in_channels: int | None = None,
        optimizer: str | Optimizer = "adamw",
        lr: float = 1e-3,
        optimizer_kwargs: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        device: str | torch.device | None = None,
        auto_resize: bool = False,
        **model_kwargs: Any,
    ):
        self.info = model_info(architecture)
        self.num_classes = num_classes
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.auto_resize = auto_resize
        self.model = create_model(
            architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            **model_kwargs,
        ).to(self.device)
        self.criterion = criterion or self._default_criterion()
        self.optimizer = (
            optimizer
            if isinstance(optimizer, Optimizer)
            else create_optimizer(
                self.model.parameters(),
                optimizer,
                lr=lr,
                **(optimizer_kwargs or {}),
            )
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_metric": [],
            "val_loss": [],
            "val_metric": [],
        }

    def _default_criterion(self) -> nn.Module:
        if self.info.task == "segmentation" and self.num_classes == 1:
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _prepare_inputs(self, inputs: Tensor) -> Tensor:
        inputs = inputs.to(self.device, non_blocking=True)
        target_size = self.info.recommended_input_size
        if self.auto_resize and target_size and inputs.shape[-2:] != (target_size, target_size):
            inputs = F.interpolate(inputs, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return inputs

    def _prepare_targets(self, targets: Tensor) -> Tensor:
        targets = targets.to(self.device, non_blocking=True)
        if self.info.task == "segmentation" and self.num_classes == 1:
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)
            return targets.float()
        return targets.long()

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
        if isinstance(batch, dict):
            image_key = "images" if "images" in batch else "image"
            target_key = "targets" if "targets" in batch else "target"
            return batch[image_key], batch[target_key]
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Cada batch debe ser (images, targets) o un diccionario equivalente.")

    @staticmethod
    def _main_output(output: Tensor | Sequence[Tensor]) -> Tensor:
        return output[0] if isinstance(output, (tuple, list)) else output

    def _loss(self, output: Tensor | Sequence[Tensor], targets: Tensor) -> Tensor:
        if isinstance(output, (tuple, list)):
            main, *auxiliary = output
            loss = self.criterion(main, targets)
            return loss + 0.3 * sum(self.criterion(aux, targets) for aux in auxiliary if aux is not None)
        return self.criterion(output, targets)

    def _metric_counts(self, logits: Tensor, targets: Tensor) -> tuple[int, int]:
        if self.info.task == "segmentation":
            predicted = (logits >= 0) if self.num_classes == 1 else logits.argmax(dim=1)
            expected = (targets >= 0.5) if self.num_classes == 1 else targets
            if expected.ndim == 4 and self.num_classes > 1:
                expected = expected.squeeze(1)
            return int((predicted == expected).sum()), expected.numel()
        predicted = logits.argmax(dim=1)
        return int((predicted == targets).sum()), targets.numel()

    def _run_epoch(self, loader: Iterable[Any], *, training: bool) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        observations = 0
        batches = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                inputs, targets = self._unpack_batch(batch)
                inputs = self._prepare_inputs(inputs)
                targets = self._prepare_targets(targets)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                output = self.model(inputs)
                loss = self._loss(output, targets)
                if training:
                    loss.backward()
                    self.optimizer.step()
                logits = self._main_output(output)
                batch_correct, batch_observations = self._metric_counts(logits, targets)
                total_loss += float(loss.detach())
                correct += batch_correct
                observations += batch_observations
                batches += 1
        if batches == 0:
            raise ValueError("El DataLoader está vacío.")
        return {"loss": total_loss / batches, "metric": correct / observations}

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        epochs: int,
        val_loader: Iterable[Any] | None = None,
        scheduler: Any | None = None,
        callback: Callable[[int, dict[str, float]], None] | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the configured model and return its persistent history."""

        for epoch in range(1, epochs + 1):
            train = self._run_epoch(train_loader, training=True)
            self.history["train_loss"].append(train["loss"])
            self.history["train_metric"].append(train["metric"])
            logs = {"train_loss": train["loss"], "train_metric": train["metric"]}
            if val_loader is not None:
                validation = self._run_epoch(val_loader, training=False)
                self.history["val_loss"].append(validation["loss"])
                self.history["val_metric"].append(validation["metric"])
                logs.update(val_loss=validation["loss"], val_metric=validation["metric"])
            if scheduler is not None:
                scheduler.step()
            if callback is not None:
                callback(epoch, logs)
            if verbose:
                rendered = " - ".join(f"{key}: {value:.4f}" for key, value in logs.items())
                print(f"Epoch {epoch}/{epochs} - {rendered}")
        return self.history

    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        """Return loss plus accuracy (or pixel accuracy for segmentation)."""

        return self._run_epoch(loader, training=False)

    @torch.no_grad()
    def predict(self, inputs: Tensor, *, probabilities: bool = True) -> Tensor:
        """Run inference; return probabilities by default."""

        self.model.eval()
        logits = self._main_output(self.model(self._prepare_inputs(inputs)))
        if not probabilities:
            return logits
        if self.info.task == "segmentation" and self.num_classes == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)

    def summary(self) -> dict[str, Any]:
        """Return architecture metadata and trainable parameter counts."""

        return {
            "architecture": self.model.architecture_name,
            "task": self.info.task,
            "device": str(self.device),
            "input_channels": self.model.input_channels,
            "recommended_input_size": self.info.recommended_input_size,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
            ),
            "optimizer": type(self.optimizer).__name__,
        }

    def save(self, path: str | Path) -> None:
        """Save a portable state-dict checkpoint."""

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "summary": self.summary(),
            },
            Path(path),
        )

    def plot_history(self, **kwargs: Any):
        from .visualization import plot_history

        return plot_history(self.history, **kwargs)

    def plot_predictions(
        self,
        images: Tensor,
        targets: Tensor | None = None,
        *,
        class_names: Sequence[str] | None = None,
        max_images: int = 8,
    ):
        from .visualization import plot_predictions

        if self.info.task != "classification":
            raise ValueError("plot_predictions está diseñado para clasificación.")
        probabilities = self.predict(images).cpu()
        return plot_predictions(
            images.cpu(), probabilities, targets.cpu() if targets is not None else None,
            class_names=class_names, max_images=max_images,
        )

    def plot_feature_maps(self, inputs: Tensor, *, layer: str | None = None, max_maps: int = 16):
        from .visualization import plot_feature_maps

        prepared = self._prepare_inputs(inputs)
        return plot_feature_maps(self.model, prepared, layer=layer, max_maps=max_maps)
