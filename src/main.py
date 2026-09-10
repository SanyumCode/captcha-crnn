"""Unified command entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config, validate_config
from .data_preparation import create_directory_structure, split_dataset, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="CAPTCHA CRNN 工作流")
    parser.add_argument("--mode", choices=["setup", "prepare", "validate"], default="setup")
    parser.add_argument("--data-dir", type=Path, help="prepare 模式的原始图片目录")
    parser.add_argument("--output-dir", type=Path, default=Config.DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_config()
    create_directory_structure(args.output_dir)
    if args.mode == "prepare":
        if args.data_dir is None:
            parser.error("--mode prepare 需要 --data-dir")
        print(split_dataset(args.data_dir, args.output_dir, args.seed))
    if args.mode in {"prepare", "validate"}:
        errors = validate_dataset(args.output_dir)
        if errors:
            raise SystemExit("\n".join(errors[:20]))
        print("数据集校验通过")


if __name__ == "__main__":
    main()
