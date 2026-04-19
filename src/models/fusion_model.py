"""Fusion model: CNN spatial features + LSTM temporal features -> per-class logits.

Output is raw logits with shape `(B, num_classes)`. Pair with
`BCEWithLogitsLoss(pos_weight=...)` for multi-label tile classification, or
slice to `(B, 1)` for the binary baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .lstm_encoder import LSTMEncoder


class DebrisPredictor(nn.Module):
    """Multi-label tile-level debris classifier.

    Args:
        in_channels: Sentinel-2 bands fed to the CNN (11 for full MARIDA).
        seq_features: features per timestep in the LSTM input.
        num_classes: head output dim. 15 for full MARIDA multi-label, 11 for
            collapsed scheme, 1 for binary.
        cnn_pretrained: load ImageNet weights for ResNet-18.
        use_temporal: if False, ignores the sequence input (CNN-only baseline).
        head_dropout: dropout in the fusion head.
    """

    def __init__(
        self,
        in_channels: int = 11,
        seq_features: int = 4,
        num_classes: int = 15,
        cnn_pretrained: bool = True,
        use_temporal: bool = True,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.num_classes = num_classes

        self.cnn = CNNEncoder(in_channels=in_channels, pretrained=cnn_pretrained)

        if use_temporal:
            self.lstm = LSTMEncoder(input_size=seq_features)
            fused_dim = self.cnn.OUT_DIM + self.lstm.out_dim
        else:
            self.lstm = None
            fused_dim = self.cnn.OUT_DIM

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, image: torch.Tensor, currents: torch.Tensor | None = None
    ) -> torch.Tensor:
        spatial = self.cnn(image)
        if self.use_temporal:
            assert currents is not None, "currents tensor required when use_temporal=True"
            temporal = self.lstm(currents)
            fused = torch.cat([spatial, temporal], dim=1)
        else:
            fused = spatial
        return self.head(fused)  # (B, num_classes)
