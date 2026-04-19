"""Training configuration for segmentation. CLI flags in train.py override these defaults."""
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
    marida_root: Path = Path("data/raw/MARIDA")

    # model
    in_channels: int = 19           # 11 raw bands + 8 spectral indices
    num_classes: int = 12           # classes 0–11 after aggregate_classes
    cnn_pretrained: bool = True

    # loss
    focal_gamma: float = 2.0        # Focal Loss gamma; 0.0 falls back to plain CE

    # optimization
    epochs: int = 30
    batch_size: int = 8             # segmentation is memory-heavier than tile-level
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # runtime
    num_workers: int = 4
    device: str = field(default_factory=default_device)
    seed: int = 42

    # io
    ckpt_dir: Path = Path("checkpoints")
    log_every: int = 20

    @property
    def split_dir(self) -> Path:
        return self.marida_root / "splits"

    @property
    def patches_dir(self) -> Path:
        return self.marida_root / "patches"

    @property
    def norm_stats_path(self) -> Path:
        return self.marida_root / "norm_stats.json"

    @property
    def class_weights_path(self) -> Path:
        return self.marida_root / "class_weights.json"
