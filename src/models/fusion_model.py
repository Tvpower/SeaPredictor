"""Fusion model: CNN spatial features + LSTM temporal features -> debris logit."""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .lstm_encoder import LSTMEncoder


class DebrisPredictor(nn.Module):
    """Per-tile debris probability.

    Output is raw logits (no sigmoid) so it pairs with `BCEWithLogitsLoss`.
    Apply `torch.sigmoid` at inference time.

    Args:
        in_channels: Sentinel-2 bands fed to the CNN.
        seq_features: features per timestep in the LSTM input.
        cnn_pretrained: load ImageNet weights for ResNet-18.
        use_temporal: if False, ignores the sequence input (CNN-only baseline).
    """

    def __init__(
        self,
        in_channels: int = 3,
        seq_features: int = 4,
        cnn_pretrained: bool = True,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
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
            nn.Dropout(0.3),
            nn.Linear(256, 1),
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
        return self.head(fused).squeeze(-1)  # (B,)
