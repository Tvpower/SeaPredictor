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


def default_num_workers(device: str) -> int:
    """macOS + MPS forks badly with DataLoader workers; force 0.

    On CUDA boxes the usual 4 workers is fine.
    """
    if device == "mps":
        return 0
    if device == "cuda":
        return 4
    return 2


@dataclass
class TrainConfig:
    # ---- data ------------------------------------------------------------ #
    # None -> auto-detect MARIDA root via marida_loader.default_marida_root()
    data_root: Path | None = None
    seq_length: int = 30
    seq_features: int = 4  # 6 if HYCOM is added later
    bands: list[int] | None = None  # None = all 11; e.g. [4, 8, 11] for B4/B8/B11

    # ---- model ----------------------------------------------------------- #
    in_channels: int = 11
    num_classes: int = 15  # MARIDA labels_mapping is 15-dim multi-label
    cnn_pretrained: bool = True
    use_temporal: bool = True
    head_dropout: float = 0.3

    # ---- optimization ---------------------------------------------------- #
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # If True, compute per-class pos_weight from the training-set label matrix.
    auto_pos_weight: bool = True
    pos_weight_clip: float = 50.0
    # Early stopping: number of epochs without macro-F1 improvement before
    # bailing. 0 disables (run the full epoch budget).
    early_stopping_patience: int = 0

    # ---- runtime --------------------------------------------------------- #
    device: str = field(default_factory=default_device)
    num_workers: int = -1
    seed: int = 42

    # ---- io -------------------------------------------------------------- #
    ckpt_dir: Path = Path("checkpoints")
    log_every: int = 20

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            self.num_workers = default_num_workers(self.device)
