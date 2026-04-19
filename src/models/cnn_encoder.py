"""ResNet-18 spatial encoder for Sentinel-2 (B4, B8, B11) tiles."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """Outputs a 512-dim feature vector per tile.

    Args:
        in_channels: number of spectral bands fed in (default 3 for B4/B8/B11).
        pretrained: load ImageNet weights for the backbone.
    """

    OUT_DIM = 512

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Swap first conv to accept arbitrary band count. When pretrained and
        # in_channels==3 we keep the original weights; otherwise re-initialize.
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Drop the classification head; keep up through global avg-pool.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  ->  (B, 512)
        return self.encoder(x).flatten(1)
