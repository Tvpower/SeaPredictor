"""Training configuration. CLI flags in `train.py` override these defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    # data
    data_root: Path = Path("data/raw")
    use_hycom: bool = False
    seq_length: int = 30
    seq_features: int = 4  # 6 if use_hycom

    # model
    in_channels: int = 3
    cnn_pretrained: bool = True
    use_temporal: bool = True

    # optimization
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    pos_weight: float = 10.0  # debris : non-debris ratio
    grad_clip: float = 1.0

    # runtime
    num_workers: int = 4
    device: str = field(default_factory=default_device)
    seed: int = 42

    # io
    ckpt_dir: Path = Path("checkpoints")
    log_every: int = 20

    def derive_seq_features(self) -> None:
        """Keep seq_features in sync with use_hycom."""
        self.seq_features = 6 if self.use_hycom else 4
