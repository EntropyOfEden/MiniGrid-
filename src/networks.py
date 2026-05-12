from __future__ import annotations

import torch
import torch.nn as nn


class MiniGridCNN(nn.Module):
    """专门适配 MiniGrid 符号输入的小型 CNN"""
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # MiniGrid 8x8 经过以下两层卷积，空间维度不会断崖式丢失
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # 自动计算展平后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 8, 8) # 这里用你真实的地图尺寸 8x8
            n_flat = self.features(dummy).shape[1]
            
        self.head = nn.Sequential(
            nn.Linear(n_flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
