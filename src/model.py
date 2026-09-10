"""CNN + BiLSTM + CTC model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (5, 1), padding=(2, 0)), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3, 1), padding=(1, 0)), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(2)
        return x.permute(0, 2, 1)


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.cnn = CNNFeatureExtractor(input_channels)
        self.lstm = nn.LSTM(
            512, lstm_hidden_size, lstm_num_layers, batch_first=True,
            bidirectional=True, dropout=0.1 if lstm_num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        sequence, _ = self.lstm(features)
        logits = self.classifier(self.dropout(sequence)).permute(1, 0, 2)
        return F.log_softmax(logits, dim=2)


def create_model() -> CRNN:
    return CRNN(
        Config.NUM_CLASSES,
        Config.IMG_CHANNELS,
        Config.LSTM_HIDDEN_SIZE,
        Config.LSTM_NUM_LAYERS,
    )
