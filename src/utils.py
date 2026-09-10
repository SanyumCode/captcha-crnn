"""Preprocessing, CTC decoding, and metrics."""
from __future__ import annotations

from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from .config import Config, idx_to_char


def decode_predictions(outputs: torch.Tensor) -> list[str]:
    indexes = outputs.argmax(dim=2).transpose(0, 1)
    results = []
    blank = Config.NUM_CLASSES - 1
    for sequence in indexes:
        decoded: list[str] = []
        previous = None
        for value in sequence.detach().cpu().tolist():
            if value == blank:
                previous = None
            elif value != previous and value in idx_to_char:
                decoded.append(idx_to_char[value])
                previous = value
        results.append("".join(decoded))
    return results


def calculate_accuracy(predictions: list[str], targets: list[str]) -> float:
    if not targets or len(predictions) != len(targets):
        return 0.0
    return sum(pred == target for pred, target in zip(predictions, targets)) / len(targets)


def preprocess_image(image_path: str | Path) -> torch.Tensor:
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(Image.fromarray(image)).unsqueeze(0)
