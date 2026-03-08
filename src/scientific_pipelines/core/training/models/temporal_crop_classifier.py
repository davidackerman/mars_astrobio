"""
Lightweight 3D CNN classifier for cropped temporal sequences.
"""

import torch
from torch import nn


class TemporalCropClassifier(nn.Module):
    """
    Classify whether a crop contains a mover and/or dipole.

    Expects input shape: (B, T, C, H, W).
    """

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.features(x)
        return self.classifier(x)
