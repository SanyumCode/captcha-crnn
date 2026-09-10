"""Checkpoint inference command."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import Config
from .model import create_model
from .utils import decode_predictions, preprocess_image


class CaptchaPredictor:
    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or Config.DEVICE)
        self.model = create_model()
        checkpoint = torch.load(Path(model_path), map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict(self, image_path: str | Path) -> str:
        with torch.no_grad():
            output = self.model(preprocess_image(image_path).to(self.device))
        return decode_predictions(output)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="识别单张验证码")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Config.MODEL_SAVE_PATH / "best_model.pth")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    print(CaptchaPredictor(args.model, args.device).predict(args.image))


if __name__ == "__main__":
    main()
