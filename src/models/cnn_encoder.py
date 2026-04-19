"""ResNet-18 spatial encoder for multi-band Sentinel-2 patches."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Reshape `conv1` from 3-band to `in_channels`-band while keeping pretrained weights.

    Standard satellite-CV trick: average the pretrained RGB filters across the
    input-channel dim, then tile to the new band count and rescale so the
    output activation magnitude is preserved. Much better than re-init from
    scratch when in_channels > 3.
    """
    if in_channels == 3:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        # (out, 3, k, k) -> (out, 1, k, k) by averaging input dim
        avg = conv.weight.mean(dim=1, keepdim=True)
        # tile to in_channels and renormalize so magnitude matches
        new_conv.weight.copy_(avg.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class CNNEncoder(nn.Module):
    """Outputs a 512-dim feature vector per tile.

    Args:
        in_channels: number of spectral bands (default 11 for full MARIDA stack;
            3 if you're using just B4/B8/B11).
        pretrained: load ImageNet weights for the backbone.
    """

    OUT_DIM = 512

    def __init__(self, in_channels: int = 11, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        resnet.conv1 = _adapt_first_conv(resnet.conv1, in_channels)
        # Drop the classification head; keep up through global avg-pool.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  ->  (B, 512)
        return self.encoder(x).flatten(1)
