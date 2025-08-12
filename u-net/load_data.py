from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict
import torch
import torchvision as tv
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split


ROOT = "./data"     # carpeta donde se descargará el dataset
IMG_SIZE = (256, 256)

# --- Transforms ---
img_tf = T.Compose([
    T.Resize(IMG_SIZE, interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),])


# Para visualizar, también quiero la imagen "no normalizada":
img_tf_unnorm = T.Resize(IMG_SIZE, interpolation=Image.BILINEAR)

mask_resize = T.Resize(IMG_SIZE, interpolation=Image.NEAREST)

def mask_decode_to_rgb(mask_np):
    """
    Oxford-IIIT Pet (segmentation trimaps):
      1 = fondo, 2 = borde, 3 = mascota
    Devuelve una imagen RGB colorizada para visualizar.
    """
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Colores (BGR/BRG? No, usamos RGB)
    # fondo: negro, borde: amarillo, mascota: verde
    rgb[mask_np == 1] = (0, 0, 0)
    rgb[mask_np == 2] = (255, 255, 0)
    rgb[mask_np == 3] = (0, 200, 0)
    return rgb

class OxfordPetsSeg(Dataset):
    """
    Dataset para Oxford-IIIT Pet con segmentación.
    - binary=True: devuelve máscara binaria (fondo=0, {borde|mascota}=1)
    - binary=False: devuelve máscara multiclase remapeada a {0,1,2}={fondo,borde,mascota}
    Augment opcional (flip horizontal coherente imagen/máscara).
    """
    def __init__(self,root: str = "./data",
                 split: str = "trainval",img_size: Tuple[int,int] = (256,256),
                 binary: bool = True,augment: bool = False,download: bool = True):

        assert split in {"trainval", "test"}

        self.base = tv.datasets.OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=download)


        self.img_size = img_size
        self.binary = binary
        self.augment = augment

        self.img_resize = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.mask_resize = T.Resize(img_size, interpolation=Image.NEAREST)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    def _process_mask(self, mask_pil: Image.Image):
        m = self.mask_resize(mask_pil)
        m = np.array(m, dtype=np.uint8)  # valores originales: {1=fondo, 2=borde, 3=mascota}
        if self.binary:
            m = ((m == 2) | (m == 3)).astype(np.float32)  # foreground=1
            return torch.from_numpy(m).unsqueeze(0)       # [1,H,W] para BCE/Dice binario
        else:
            # Remap a {0,1,2} -> {fondo,borde,mascota} para CrossEntropy
            remap = np.zeros_like(m, dtype=np.int64)
            remap[m == 1] = 0
            remap[m == 2] = 1
            remap[m == 3] = 2
            return torch.from_numpy(remap)                # [H,W] long

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.base[idx]
        img = self.img_resize(img_pil)
        mask_t = self._process_mask(mask_pil)

        # Augment simple: flip horizontal coherente
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                if mask_t.ndim == 3:  # binario [1,H,W]
                    mask_t = TF.hflip(mask_t)
                else:                  # multiclase [H,W]
                    mask_t = torch.flip(mask_t, dims=[1])  # W dimension

        img = self.to_tensor(img)
        img = self.normalize(img)
        return img, mask_t


def denorm_for_show(t):
    """
    Quita la normalización para mostrar con matplotlib.
    t: tensor [3,H,W] normalizado.
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = t * std + mean
    return x.clamp(0,1)


def overlay_mask_on_image(img_np, mask_rgb, alpha=0.35):
    """
    Combina imagen (H,W,3) y máscara RGB (H,W,3) con transparencia alpha.
    img_np debe estar en [0,1].
    """
    overlay = (1 - alpha) * img_np + alpha * (mask_rgb.astype(np.float32)/255.0)
    return overlay.clip(0,1)


def show_samples(dataset, n=6):
    """
    Muestra n ejemplos: imagen, máscara y overlay.
    """
    n = min(n, len(dataset))
    cols = 3
    rows = int(np.ceil(n))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # uniformizar

    for i in range(n):
        img_t, mask_t, img_viz = dataset[i]
        # imagen no normalizada para ver
        img_show = np.array(img_viz) / 255.0     # [H,W,3] en [0,1]
        # máscara decodificada a rgb
        mask_np = mask_t.numpy()
        mask_rgb = mask_decode_to_rgb(mask_np)

        # overlay
        overlay = overlay_mask_on_image(img_show, mask_rgb, alpha=0.4)

        r = i
        # Columna 1: imagen
        axes[r, 0].imshow(img_show)
        axes[r, 0].set_title("Imagen")
        axes[r, 0].axis("off")

        # Columna 2: máscara color
        axes[r, 1].imshow(mask_rgb)
        axes[r, 1].set_title("Máscara (1=fondo, 2=borde, 3=mascota)")
        axes[r, 1].axis("off")

        # Columna 3: overlay
        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title("Overlay")
        axes[r, 2].axis("off")

    plt.tight_layout()
    plt.show()




def create_pets_loaders(root: str = "./data",
                        img_size: Tuple[int,int] = (256,256),
                        batch_size: int = 8,
                        num_workers: int = 2,
                        pin_memory: bool = True,
                        binary: bool = True,
                        augment: bool = True,
                        val_split: float = 0.15,
                        seed: int = 42) -> Dict[str, DataLoader]:
    
    """
    Crea DataLoaders para Oxford-IIIT Pet (segmentación).

    Args:
        batch_size: tamaño del batch (puedes seleccionarlo libremente)
        binary: True -> máscara binaria; False -> multiclase {0,1,2}
        augment: flips horizontales en train
        val_split: proporción de validación sacada del split 'trainval'
    """
    # Dataset base (trainval -> lo dividimos en train/val)
    full_train = OxfordPetsSeg(root=root, split="trainval",img_size=img_size, binary=binary,
                               augment=False, download=True)

    n = len(full_train)
    n_val = int(n * val_split)
    n_train = n - n_val

    # Reproducibilidad en el split
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, lengths=[n_train, n_val], generator=g)

    # Activar augment solo en el subset de train (envolvemos con mismo dataset pero augment=True)
    # Usamos indices del subset para acceder al dataset base con augment deseado.

    class _SubsetWithAug(Dataset):
        def __init__(self, base_ds, indices, use_augment):
            self.base = base_ds
            self.indices = indices
            self.use_augment = use_augment

        def __len__(self): return len(self.indices)


        def __getitem__(self, i):
            idx = self.indices[i]
            # toggle augment dinámicamente
            prev = self.base.augment
            self.base.augment = self.use_augment
            out = self.base[idx]
            self.base.augment = prev
            return out

    train_ds = _SubsetWithAug(full_train, train_subset.indices, use_augment=augment)
    val_ds   = _SubsetWithAug(full_train, val_subset.indices,   use_augment=False)

    # Test oficial del dataset
    test_ds = OxfordPetsSeg(root=root, split="test",
                            img_size=img_size, binary=binary,
                            augment=False, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return {"train": train_loader, "val": val_loader, "test": test_loader}






