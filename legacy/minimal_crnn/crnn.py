# crnn.py
import torch
import torch.nn as nn

from .dataset_and_dataloader import CHARS


class CRNN(nn.Module):
    def __init__(self, imgH=32, num_channels=1, num_classes=len(CHARS)+1, lstm_hidden=256, lstm_layers=2):
        super().__init__()
        assert imgH % 1 == 0, "imgH should be integer"
        self.num_classes = num_classes
        # CNN backbone (可按需加深/改)
        self.cnn = nn.Sequential(
            # conv1
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/4, W/4
            # conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            # conv4
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),  # H/8, W/4 (height down)
            # conv5
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            # conv6
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),  # H/16, W/4
            # conv7
            nn.Conv2d(512, 512, kernel_size=2, padding=0), nn.ReLU(True),
            # at this point height should be small; we will adaptive pool to H=1
            nn.AdaptiveAvgPool2d((1, None))
        )
        self.imgH = imgH
        # RNN
        self.rnn_hidden = lstm_hidden
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # bidirectional

    def forward(self, x):
        # x: B x C x H x W
        conv = self.cnn(x)  # B x C' x 1 x W'
        assert conv.size(2) == 1, "the height after conv is expected to be 1"
        conv = conv.squeeze(2)  # B x C' x W'
        conv = conv.permute(0, 2, 1)  # B x W' x C'  (sequence length = W')
        # RNN expects (B, T, feature)
        rnn_out, _ = self.rnn(conv)  # B x T x 2*hidden
        logits = self.fc(rnn_out)  # B x T x num_classes
        # For CTCLoss we need shape (T, N, C) and log probs
        logits = logits.permute(1, 0, 2)  # T x B x C
        log_probs = nn.functional.log_softmax(logits, dim=2)
        return log_probs  # T x B x C
