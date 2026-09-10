"""Dataset and data-loader helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import Config, char_to_idx

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class CaptchaDataset(Dataset):
    def __init__(self, data_dir: str | Path, transform=None) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{self.data_dir}")
        self.transform = transform or get_transforms(False)
        self.samples = [
            (path, path.stem.split("_", 1)[0])
            for path in sorted(self.data_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        unknown = set(label) - set(char_to_idx)
        if unknown:
            raise ValueError(f"{path.name} 包含字符集外字符：{sorted(unknown)}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图像：{path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        tensor = self.transform(Image.fromarray(image))
        target = torch.tensor([char_to_idx[c] for c in label], dtype=torch.long)
        return tensor, target, label


def collate_fn(batch: Sequence):
    images, labels, raw_labels = zip(*batch)
    return (
        torch.stack(images),
        torch.cat(labels),
        torch.tensor([len(label) for label in labels], dtype=torch.long),
        raw_labels,
    )


def get_transforms(is_training: bool = True):
    steps = []
    if is_training:
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)


def create_data_loaders(
    train_dir: str | Path = Config.TRAIN_DATA_DIR,
    val_dir: str | Path = Config.VAL_DATA_DIR,
    batch_size: int = Config.BATCH_SIZE,
):
    train = CaptchaDataset(train_dir, get_transforms(True))
    val = CaptchaDataset(val_dir, get_transforms(False))
    common = dict(batch_size=batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    return DataLoader(train, shuffle=True, **common), DataLoader(val, shuffle=False, **common)
