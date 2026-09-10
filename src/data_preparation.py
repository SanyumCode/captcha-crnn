"""Deterministic 8:1:1 dataset splitting and validation."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from .config import Config, char_to_idx

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def create_directory_structure(data_root: str | Path = Config.DATA_PATH) -> None:
    root = Path(data_root)
    for name in ("train", "val", "test"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for path in (Config.MODEL_SAVE_PATH, Config.CHECKPOINT_PATH, Config.LOG_PATH, Config.ONNX_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def split_dataset(source_dir: str | Path, output_dir: str | Path = Config.DATA_PATH, seed: int = 42) -> dict[str, int]:
    source, output = Path(source_dir), Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"源数据目录不存在：{source}")
    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    random.Random(seed).shuffle(images)
    train_end = int(len(images) * 0.8)
    val_end = train_end + int(len(images) * 0.1)
    groups = {"train": images[:train_end], "val": images[train_end:val_end], "test": images[val_end:]}
    for split, paths in groups.items():
        destination = output / split
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, destination / path.name)
    return {name: len(paths) for name, paths in groups.items()}


def validate_dataset(data_root: str | Path = Config.DATA_PATH) -> list[str]:
    errors = []
    for split in ("train", "val", "test"):
        folder = Path(data_root) / split
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = path.stem.split("_", 1)[0]
            unknown = set(label) - set(char_to_idx)
            if "_" not in path.stem or not label or unknown:
                errors.append(f"{path}: 文件名或标签无效")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="按 8:1:1 划分验证码数据")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Config.DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(split_dataset(args.source, args.output, args.seed))
    errors = validate_dataset(args.output)
    if errors:
        raise SystemExit("\n".join(errors[:20]))


if __name__ == "__main__":
    main()
