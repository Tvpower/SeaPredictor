"""ResNet-18 encoder + U-Net decoder for pixel-wise marine debris segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FocalLoss(nn.Module):
    """Multi-class focal loss: down-weights easy negatives via (1-pt)^gamma."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)   targets: (B, H, W) long
        weight = getattr(self, "weight", None)
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class _FinalUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class DebrisSegmenter(nn.Module):
    """ResNet-18 encoder + U-Net decoder.

    Input:  (B, in_channels, 256, 256)
    Output: (B, num_classes, 256, 256)  — raw logits, no softmax

    Skip connections follow the standard U-Net pattern:
        enc4 (512, /32) -> dec4 -> (256, /16) + skip enc3
        dec3 -> (128, /8)  + skip enc2
        dec2 -> (64,  /4)  + skip enc1 (layer1 out)
        dec1 -> (64,  /2)  + skip enc0 (after first conv, before maxpool)
        final_up -> (32, /1)
        head  -> (num_classes, /1)
    """

    NUM_CLASSES = 12

    def __init__(
        self,
        in_channels: int = 19,
        num_classes: int = 12,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Encoder — expose intermediate feature maps for skip connections
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64ch
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1   # /4,  64ch
        self.enc2 = backbone.layer2   # /8,  128ch
        self.enc3 = backbone.layer3   # /16, 256ch
        self.enc4 = backbone.layer4   # /32, 512ch

        # Decoder
        self.dec4 = _DecoderBlock(512, 256, 256)   # up + skip enc3
        self.dec3 = _DecoderBlock(256, 128, 128)   # up + skip enc2
        self.dec2 = _DecoderBlock(128, 64, 64)     # up + skip enc1
        self.dec1 = _DecoderBlock(64, 64, 64)      # up + skip enc0
        self.up_final = _FinalUp(64, 32)           # up to full res, no skip
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)               # (B, 64,  H/2,  W/2)
        s1 = self.enc1(self.pool(s0))   # (B, 64,  H/4,  W/4)
        s2 = self.enc2(s1)              # (B, 128, H/8,  W/8)
        s3 = self.enc3(s2)              # (B, 256, H/16, W/16)
        s4 = self.enc4(s3)              # (B, 512, H/32, W/32)

        x = self.dec4(s4, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = self.up_final(x)
        return self.head(x)
