from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Convolutional-Recurrent network for plate OCR (CNN -> BiLSTM -> CTC).

    Input:  grayscale image tensor (B, 1, H, W), H must be a multiple of 16.
    Output: log-its (B, T, num_classes) where T = W / 4 timesteps.
    """

    def __init__(self, num_classes: int, in_ch: int = 1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)            # (B, 512, H', W')
        f = self.pool(f)           # (B, 512, 1, W')
        f = f.squeeze(2)           # (B, 512, W')
        f = f.permute(0, 2, 1)     # (B, W', 512)
        out, _ = self.rnn(f)       # (B, W', 512)
        return self.fc(out)        # (B, W', num_classes)
