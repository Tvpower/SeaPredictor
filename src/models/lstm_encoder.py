"""LSTM temporal encoder for OSCAR (+ optional HYCOM) sequences."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Encodes a (B, T, F) ocean-physics sequence to a fixed vector.

    Default feature layout per timestep (F=4): [u_current, v_current, lat, lon].
    With HYCOM enabled (F=6) it becomes: [u, v, sst, salinity, lat, lon].
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def out_dim(self) -> int:
        return self.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> last layer's final hidden state (B, hidden)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]
