"""Training loop for pixel-wise marine debris segmentation (Option A).

Usage:
    # smoke test on synthetic data (no real data needed)
    python -m src.training.train --synthetic --epochs 2 --batch-size 2

    # real training
    python -m src.training.train --epochs 30 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from src.dataset.marida_dataset import MARIDADataset
from src.dataset.normalization import load_stats
from src.models.segmentation_model import DebrisSegmenter, FocalLoss
from src.training.config import TrainConfig
from src.training.evaluate import evaluate


# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _SyntheticSegDataset(Dataset):
    """Tiny in-memory dataset for smoke-testing the training loop."""

    def __init__(self, n: int, in_channels: int, num_classes: int, size: int = 64) -> None:
        self.images = torch.randn(n, in_channels, size, size)
        self.masks = torch.randint(0, num_classes, (n, size, size))
        self.confs = torch.ones(n, size, size, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx], self.confs[idx]


def _load_class_weights(cfg: TrainConfig, device: str) -> torch.Tensor:
    """Load per-class weights from JSON and prepend background weight.

    class_weights.json holds 11 floats for classes 1–11 (index 0 = class ID 1).
    We prepend a small weight for class 0 (background, ~99% of pixels).
    """
    with open(cfg.class_weights_path) as f:
        weights = json.load(f)  # 11 floats, index 0 = class ID 1
    bg_weight = 0.1  # background strongly down-weighted; Focal Loss handles the rest
    full_weights = [bg_weight] + weights  # 12 floats, index i = class ID i
    return torch.tensor(full_weights, dtype=torch.float32, device=device)


def _build_datasets(cfg: TrainConfig, synthetic: bool) -> tuple[Dataset, Dataset]:
    if synthetic:
        return (
            _SyntheticSegDataset(128, cfg.in_channels, cfg.num_classes),
            _SyntheticSegDataset(32, cfg.in_channels, cfg.num_classes),
        )

    norm_stats = load_stats(str(cfg.norm_stats_path))
    train_ds = MARIDADataset(
        split_file=str(cfg.split_dir / "train_X.txt"),
        patches_dir=str(cfg.patches_dir),
        augment=True,
        add_indices=True,
        aggregate=True,
        norm_stats=norm_stats,
    )
    val_ds = MARIDADataset(
        split_file=str(cfg.split_dir / "val_X.txt"),
        patches_dir=str(cfg.patches_dir),
        augment=False,
        add_indices=True,
        aggregate=True,
        norm_stats=norm_stats,
    )
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# Single epoch                                                                #
# --------------------------------------------------------------------------- #
def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
    log_every: int,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    n_seen = 0
    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)

    for step, (image, mask, _conf) in enumerate(pbar):
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, mask)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = mask.size(0)
        running += loss.item() * bs
        n_seen += bs
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.4f}")

    return running / max(n_seen, 1)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig, synthetic: bool = False) -> Path:
    _seed_everything(cfg.seed)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={cfg.device}  synthetic={synthetic}")
    print(f"[train] in_channels={cfg.in_channels}  num_classes={cfg.num_classes}  focal_gamma={cfg.focal_gamma}")

    train_ds, val_ds = _build_datasets(cfg, synthetic)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    model = DebrisSegmenter(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        pretrained=cfg.cnn_pretrained,
    ).to(cfg.device)

    if synthetic:
        criterion: nn.Module = FocalLoss(gamma=cfg.focal_gamma)
    else:
        class_weights = _load_class_weights(cfg, cfg.device)
        criterion = FocalLoss(gamma=cfg.focal_gamma, weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_debris_iou = -1.0
    best_path = cfg.ckpt_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer,
            cfg.device, cfg.grad_clip, cfg.log_every, epoch,
        )
        metrics = evaluate(model, val_loader, criterion, cfg.device, cfg.num_classes)
        scheduler.step()
        dt = time.time() - t0

        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  "
            f"val[{metrics}]  lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)"
        )

        if metrics.debris_iou > best_debris_iou:
            best_debris_iou = metrics.debris_iou
            cfg_safe = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in cfg.__dict__.items()
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics.as_dict(),
                    "cfg": cfg_safe,
                },
                best_path,
            )
            print(f"[epoch {epoch:02d}] new best debris IoU={best_debris_iou:.3f} -> {best_path}")

    return best_path


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--marida-root", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    return p


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_pretrained:
        cfg.cnn_pretrained = False
    if args.focal_gamma is not None:
        cfg.focal_gamma = args.focal_gamma
    if args.marida_root is not None:
        cfg.marida_root = args.marida_root
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.device is not None:
        cfg.device = args.device
    return cfg


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _apply_overrides(TrainConfig(), args)
    train(cfg, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
