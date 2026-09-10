"""ONNX Runtime inference example."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.config import Config, idx_to_char
from src.utils import preprocess_image


def decode(output: np.ndarray) -> str:
    indexes = output[:, 0, :].argmax(axis=1)
    blank, previous, chars = Config.NUM_CLASSES - 1, None, []
    for index in indexes:
        if index == blank:
            previous = None
        elif index != previous:
            chars.append(idx_to_char[int(index)])
            previous = index
    return "".join(chars)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("exports/model.onnx"))
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    image = preprocess_image(args.image).numpy()
    output = session.run(None, {session.get_inputs()[0].name: image})[0]
    print(decode(output))


if __name__ == "__main__":
    main()
