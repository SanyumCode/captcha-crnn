"""TorchScript inference example."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.utils import decode_predictions, preprocess_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("exports/model.pt"))
    args = parser.parse_args()
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    with torch.no_grad():
        print(decode_predictions(model(preprocess_image(args.image)))[0])


if __name__ == "__main__":
    main()
