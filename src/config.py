"""Project configuration with portable paths."""
from __future__ import annotations

import os
import string
from pathlib import Path

import torch

PROJECT_ROOT = Path(os.getenv("CAPTCHA_CRNN_ROOT", Path(__file__).resolve().parents[1])).resolve()


class Config:
    DATA_PATH = Path(os.getenv("CAPTCHA_DATA_DIR", PROJECT_ROOT / "data")).resolve()
    TRAIN_DATA_DIR = DATA_PATH / "train"
    VAL_DATA_DIR = DATA_PATH / "val"
    TEST_DATA_DIR = DATA_PATH / "test"

    IMG_WIDTH = 216
    IMG_HEIGHT = 96
    IMG_CHANNELS = 3
    CHARACTERS = string.digits + string.ascii_letters
    NUM_CLASSES = len(CHARACTERS) + 1
    MAX_TEXT_LENGTH = 10
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2

    BATCH_SIZE = int(os.getenv("CAPTCHA_BATCH_SIZE", "32"))
    LEARNING_RATE = float(os.getenv("CAPTCHA_LEARNING_RATE", "0.001"))
    NUM_EPOCHS = int(os.getenv("CAPTCHA_EPOCHS", "100"))
    DEVICE = os.getenv("CAPTCHA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    MODEL_SAVE_PATH = Path(os.getenv("CAPTCHA_MODEL_DIR", PROJECT_ROOT / "models")).resolve()
    CHECKPOINT_PATH = Path(os.getenv("CAPTCHA_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")).resolve()
    LOG_PATH = Path(os.getenv("CAPTCHA_LOG_DIR", PROJECT_ROOT / "logs")).resolve()
    TORCHSCRIPT_PATH = Path(os.getenv("CAPTCHA_TORCHSCRIPT_PATH", PROJECT_ROOT / "exports" / "model.pt")).resolve()
    ONNX_PATH = Path(os.getenv("CAPTCHA_ONNX_PATH", PROJECT_ROOT / "exports" / "model.onnx")).resolve()


char_to_idx = {char: idx for idx, char in enumerate(Config.CHARACTERS)}
idx_to_char = {idx: char for idx, char in enumerate(Config.CHARACTERS)}


def validate_config() -> None:
    assert Config.NUM_CLASSES == len(Config.CHARACTERS) + 1
    assert Config.IMG_WIDTH > 0 and Config.IMG_HEIGHT > 0
    print(f"配置有效：{len(Config.CHARACTERS)} 字符，设备 {Config.DEVICE}")
