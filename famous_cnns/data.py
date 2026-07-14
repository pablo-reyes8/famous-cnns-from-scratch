"""Small, predictable datasets used by the unified command-line interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .factory import model_info

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255)


class ImageFolderDataset(Dataset):
    """Minimal ImageFolder-compatible dataset without an eager torchvision import."""

    def __init__(self, root: str | Path, *, transform: Callable):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        self.classes = sorted(path.name for path in root.iterdir() if path.is_dir())
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = [
            (path, self.class_to_idx[class_name])
            for class_name in self.classes
            for path in sorted((root / class_name).iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not self.samples:
            raise ValueError(f"No se encontraron imágenes organizadas por clase en {root}.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), target


class SegmentationFolderDataset(Dataset):
    """Paired segmentation dataset with ``images/`` and ``masks/`` directories."""

    def __init__(self, root: str | Path, *, image_size: int, num_classes: int, in_channels: int):
        root = Path(root)
        self.image_dir = root / "images"
        self.mask_dir = root / "masks"
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Se esperaban {self.image_dir} y {self.mask_dir}.")
        masks = {
            path.stem: path
            for path in self.mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        self.pairs = [
            (image, masks[image.stem])
            for image in sorted(self.image_dir.iterdir())
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES and image.stem in masks
        ]
        if not self.pairs:
            raise ValueError(f"No se encontraron pares de imagen/máscara en {root}.")
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("L" if self.in_channels == 1 else "RGB")
        mask = Image.open(mask_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_tensor = _match_channels(_pil_to_tensor(image), self.in_channels)
        mask_tensor = torch.from_numpy(np.array(mask, copy=True)).to(torch.uint8)
        if self.num_classes == 1:
            mask_tensor = (mask_tensor > 0).float().unsqueeze(0)
        else:
            mask_tensor = mask_tensor.long()
        return image_tensor, mask_tensor


def _match_channels(image: torch.Tensor, in_channels: int) -> torch.Tensor:
    if image.shape[0] == in_channels:
        return image
    if in_channels == 1:
        return image.mean(dim=0, keepdim=True)
    repeats = (in_channels + image.shape[0] - 1) // image.shape[0]
    return image.repeat(repeats, 1, 1)[:in_channels]


def image_transform(image_size: int, in_channels: int) -> Callable:
    """Return the deterministic preprocessing used by CLI training and inference."""

    def transform(image: Image.Image) -> torch.Tensor:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return _match_channels(_pil_to_tensor(image), in_channels)

    return transform


def build_loaders(
    *,
    model_name: str,
    data_dir: str | Path | None,
    num_classes: int,
    in_channels: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    smoke_test: bool,
) -> tuple[DataLoader, DataLoader | None, list[str] | None]:
    """Build synthetic, ImageFolder, or paired-segmentation loaders."""

    info = model_info(model_name)
    if smoke_test:
        count = max(2, batch_size)
        images = torch.randn(count, in_channels, image_size, image_size)
        if info.task == "classification":
            targets = torch.arange(count) % num_classes
        elif num_classes == 1:
            targets = torch.randint(0, 2, (count, 1, image_size, image_size)).float()
        else:
            targets = torch.randint(0, num_classes, (count, image_size, image_size))
        loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size)
        return loader, loader, None

    if data_dir is None:
        raise ValueError("Usa --data-dir para datos reales o --smoke-test para datos sintéticos.")
    data_dir = Path(data_dir)
    train_root = data_dir / "train"
    val_root = data_dir / "val"
    if info.task == "classification":
        transform = image_transform(image_size, in_channels)
        train_dataset = ImageFolderDataset(train_root, transform=transform)
        validation_dataset = (
            ImageFolderDataset(val_root, transform=transform) if val_root.is_dir() else None
        )
        class_names = train_dataset.classes
        if len(class_names) != num_classes:
            raise ValueError(
                f"ImageFolder encontró {len(class_names)} clases, pero --num-classes={num_classes}."
            )
    else:
        train_dataset = SegmentationFolderDataset(
            train_root, image_size=image_size, num_classes=num_classes, in_channels=in_channels
        )
        validation_dataset = (
            SegmentationFolderDataset(
                val_root, image_size=image_size, num_classes=num_classes, in_channels=in_channels
            )
            if val_root.is_dir()
            else None
        )
        class_names = None

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
        if validation_dataset is not None
        else None
    )
    return train_loader, validation_loader, class_names
