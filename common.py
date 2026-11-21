"""Shared helpers for the PCam tumor-patch classifier (data, model, weights I/O).

FOR RESEARCH AND EDUCATION ONLY. NOT FOR MEDICAL DIAGNOSIS.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms as T

DATASET_ID = "1aurent/PatchCamelyon"           # CC0 1.0, mirror of the official PCam data
HF_SPLIT = {"train": "train", "val": "valid", "test": "test"}   # official splits - never reshuffle
IMG_SIZE = 96
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DISCLAIMER = (
    "FOR RESEARCH AND EDUCATION ONLY - NOT FOR MEDICAL DIAGNOSIS. "
    "This model was trained on one public dataset (PatchCamelyon) and may perform worse on "
    "images from other hospitals or scanners. It does not detect cancer in real patients."
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- model
def build_model(arch: str = "resnet18", weights: str = "imagenet", dropout: float = 0.2) -> nn.Module:
    """Small pretrained CNN with a single-logit head (tumor vs. normal)."""
    use_imagenet = weights == "imagenet"
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet else None
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_imagenet else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, 1))
    else:
        raise ValueError(f"unknown arch {arch!r} (use resnet18 or efficientnet_b0)")
    return m


def load_model(path, device="cpu"):
    """Load weights from a training checkpoint (.pt) or a released .safetensors file.

    Returns (model_in_eval_mode, config_dict).
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        state = load_file(str(path))
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt["model"]
        cfg = {"arch": ckpt["arch"], "threshold_val_tuned": ckpt.get("val_threshold")}
    model = build_model(cfg["arch"], weights="none")
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


# ------------------------------------------------------------------------- transforms
def train_transform():
    """Light augmentation. Histology has no 'up', so flips and 90-degree turns are safe."""
    rot = [T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomChoice(rot),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


# ---------------------------------------------------------------------------- data
class PCamDataset(Dataset):
    """Wraps a Hugging Face dataset (or a plain list of {'image','label'} dicts)."""

    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = r["image"].convert("RGB")
        return self.transform(img), torch.tensor(float(r["label"]))


def load_split(split: str, limit: int = None, seed: int = 42):
    """Official split from Hugging Face. With `limit`, stream a small random subset
    (used by --quick-test) so nothing big gets downloaded."""
    from datasets import load_dataset
    name = HF_SPLIT[split]
    if limit is None:
        return load_dataset(DATASET_ID, split=name)
    ds = load_dataset(DATASET_ID, split=name, streaming=True)
    return list(ds.shuffle(seed=seed, buffer_size=2000).take(limit))


def labels_of(rows) -> np.ndarray:
    if hasattr(rows, "column_names"):          # HF Dataset: read just the label column
        return np.asarray(rows["label"], dtype=np.int64)
    return np.asarray([int(r["label"]) for r in rows], dtype=np.int64)


def safe_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))
