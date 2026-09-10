"""Export a checkpoint to TorchScript and ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import Config
from .model import create_model


def load_checkpoint(path: Path):
    model = create_model()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model.eval()


def export(model, torchscript_path: Path, onnx_path: Path) -> None:
    torchscript_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, Config.IMG_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    torch.jit.script(model).save(str(torchscript_path))
    torch.onnx.export(
        model, example, str(onnx_path), opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {1: "batch"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 CRNN 模型")
    parser.add_argument("--checkpoint", type=Path, default=Config.MODEL_SAVE_PATH / "best_model.pth")
    parser.add_argument("--torchscript", type=Path, default=Config.TORCHSCRIPT_PATH)
    parser.add_argument("--onnx", type=Path, default=Config.ONNX_PATH)
    args = parser.parse_args()
    export(load_checkpoint(args.checkpoint), args.torchscript, args.onnx)
    print(f"TorchScript: {args.torchscript}\nONNX: {args.onnx}")


if __name__ == "__main__":
    main()
