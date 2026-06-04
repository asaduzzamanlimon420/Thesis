"""
Dataset loading, non-IID partitioning, and augmentation.

Supports two modes:
  1. Pre-split: images already in client_0/ … client_2/ folders.
  2. Combined + Dirichlet re-split (for ablation / non-IID experiments).
"""

import os, random, math, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config as C


# ──────────────────────── transforms ─────────────────────────

def get_train_transform(img_size=C.IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=C.MEAN, std=C.STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size=C.IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=C.MEAN, std=C.STD),
        ToTensorV2(),
    ])


# ──────────────────────── core dataset ───────────────────────

class FundusDataset(Dataset):
    """Load fundus images from folder structure: root/<class_name>/*.{jpg,png,…}"""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root, transform=None, class_names=None):
        self.root = Path(root)
        self.transform = transform
        self.class_names = class_names or C.CLASS_NAMES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples = []  # list of (path_str, label_int)
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                # try case-insensitive match
                matches = [d for d in self.root.iterdir() if d.is_dir() and d.name.lower() == cls.lower()]
                if matches:
                    cls_dir = matches[0]
                else:
                    print(f"  [WARN] class folder not found: {cls_dir}")
                    continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in self.EXTS:
                    self.samples.append((str(f), self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {root}. "
                f"Expected sub-folders: {self.class_names}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def label_distribution(self):
        labels = [s[1] for s in self.samples]
        return dict(Counter(labels))


# ──────────────────── per-client loaders ─────────────────────

def load_client_data(client_id: int):
    """
    Load train / val / test DataLoaders for one client
    from the pre-split folder structure.
    """
    client_dir = os.path.join(C.DATA_ROOT, C.CLIENT_DIRS[client_id])
    full_ds = FundusDataset(client_dir, transform=None)   # no transform yet

    n = len(full_ds)
    n_test = int(n * C.TEST_SPLIT)
    n_val  = int(n * C.VAL_SPLIT)
    n_train = n - n_test - n_val

    gen = torch.Generator().manual_seed(C.RANDOM_SEED)
    train_sub, val_sub, test_sub = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    train_ds = _TransformSubset(train_sub, get_train_transform())
    val_ds   = _TransformSubset(val_sub,   get_val_transform())
    test_ds  = _TransformSubset(test_sub,  get_val_transform())

    kw = dict(batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, test_loader


def load_combined_data():
    """Load the combined (centralised) dataset for baseline training."""
    combined_dir = os.path.join(C.DATA_ROOT, C.COMBINED_DIR)
    full_ds = FundusDataset(combined_dir, transform=None)

    n = len(full_ds)
    n_test = int(n * C.TEST_SPLIT)
    n_val  = int(n * C.VAL_SPLIT)
    n_train = n - n_test - n_val

    gen = torch.Generator().manual_seed(C.RANDOM_SEED)
    train_sub, val_sub, test_sub = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    train_ds = _TransformSubset(train_sub, get_train_transform())
    val_ds   = _TransformSubset(val_sub,   get_val_transform())
    test_ds  = _TransformSubset(test_sub,  get_val_transform())

    kw = dict(batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kw),
        DataLoader(test_ds,  shuffle=False, drop_last=False, **kw),
    )


# ──────────────── Dirichlet non-IID re-split ─────────────────

def dirichlet_split(dataset: FundusDataset, alpha=C.DIRICHLET_ALPHA, n_clients=C.NUM_CLIENTS):
    """
    Re-partition a single dataset into n_clients using Dirichlet(alpha).
    Returns list of index-lists, one per client.
    """
    np.random.seed(C.RANDOM_SEED)
    label_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_indices[label].append(idx)

    client_indices = [[] for _ in range(n_clients)]
    for cls_id in sorted(label_indices.keys()):
        indices = np.array(label_indices[cls_id])
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        splits = (np.cumsum(proportions) * len(indices)).astype(int)
        prev = 0
        for cid in range(n_clients):
            end = splits[cid] if cid < n_clients - 1 else len(indices)
            client_indices[cid].extend(indices[prev:end].tolist())
            prev = end

    for ci in client_indices:
        random.shuffle(ci)

    return client_indices


def print_distribution(client_indices, dataset):
    """Pretty-print label distribution per client."""
    print("\n" + "=" * 65)
    print("  NON-IID DATA DISTRIBUTION (Dirichlet α = {:.2f})".format(C.DIRICHLET_ALPHA))
    print("=" * 65)
    for cid, indices in enumerate(client_indices):
        labels = [dataset.samples[i][1] for i in indices]
        counts = Counter(labels)
        total = len(labels)
        print(f"\n  Client {cid}  ({total} images)")
        for cls_id, name in enumerate(C.CLASS_NAMES):
            cnt = counts.get(cls_id, 0)
            bar = "█" * int(cnt / total * 40)
            print(f"    {name:25s}: {cnt:5d}  ({cnt/total*100:5.1f}%)  {bar}")
    print("=" * 65 + "\n")


# ──────────── non-IID feature shift (quality jitter) ─────────

def apply_feature_shift(image_np, client_id):
    """
    Simulate cross-institution feature shift by applying client-specific
    colour / brightness / blur perturbations. This creates *feature-level*
    non-IID on top of the label-level Dirichlet skew.
    """
    if client_id == 0:
        # Hospital A: slightly darker, warmer tint
        image_np = A.RandomBrightnessContrast(
            brightness_limit=(-0.15, -0.05), contrast_limit=0, p=1.0
        )(image=image_np)["image"]
        image_np = A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=0, p=0.8
        )(image=image_np)["image"]
    elif client_id == 1:
        # Hospital B: noisier sensor
        image_np = A.GaussNoise(var_limit=(15, 40), p=0.7)(image=image_np)["image"]
    else:
        # Hospital C: slight blur from older lens
        image_np = A.GaussianBlur(blur_limit=(3, 5), p=0.5)(image=image_np)["image"]
    return image_np


# ──────────────────── helper classes ─────────────────────────

class _TransformSubset(Dataset):
    """Wrap a Subset so that a transform is applied at __getitem__ time."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


class ClientSubsetDataset(Dataset):
    """Dataset for a specific client after Dirichlet split, with optional feature shift."""

    def __init__(self, base_dataset, indices, transform, client_id=None, feature_shift=False):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform
        self.client_id = client_id
        self.feature_shift = feature_shift

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.base.samples[real_idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.feature_shift and self.client_id is not None:
            img = apply_feature_shift(img, self.client_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


# ──────────────────── class-weight helper ────────────────────

def compute_class_weights(loader):
    """Inverse-frequency weights for CrossEntropyLoss."""
    counts = torch.zeros(C.NUM_CLASSES)
    for _, labels in loader:
        for lbl in labels:
            counts[lbl] += 1
    total = counts.sum()
    weights = total / (C.NUM_CLASSES * counts.clamp(min=1))
    return weights
