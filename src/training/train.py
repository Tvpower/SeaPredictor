"""Main training loop for the debris predictor.

Usage:
    # smoke test on synthetic data
    python -m src.training.train --synthetic --epochs 2 --batch-size 4

    # real training (requires data/raw/ populated)
    python -m src.training.train --epochs 30 --batch-size 16
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import DebrisDataset, SyntheticDebrisDataset
from src.models import DebrisPredictor
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


def _build_datasets(cfg: TrainConfig, synthetic: bool) -> tuple[Dataset, Dataset]:
    if synthetic:
        train_ds = SyntheticDebrisDataset(
            n_samples=256,
            seq_length=cfg.seq_length,
            seq_features=cfg.seq_features,
            in_channels=cfg.in_channels,
            seed=0,
        )
        val_ds = SyntheticDebrisDataset(
            n_samples=64,
            seq_length=cfg.seq_length,
            seq_features=cfg.seq_features,
            in_channels=cfg.in_channels,
            seed=1,
        )
        return train_ds, val_ds

    train_ds = DebrisDataset(
        data_root=cfg.data_root,
        split="train",
        use_hycom=cfg.use_hycom,
        seq_length=cfg.seq_length,
    )
    val_ds = DebrisDataset(
        data_root=cfg.data_root,
        split="val",
        use_hycom=cfg.use_hycom,
        seq_length=cfg.seq_length,
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

    for step, (image, seq, label) in enumerate(pbar):
        image = image.to(device, non_blocking=True)
        seq = seq.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image, seq)
        loss = criterion(logits, label)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = label.size(0)
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
    cfg.derive_seq_features()
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={cfg.device}  synthetic={synthetic}")
    print(f"[train] seq_features={cfg.seq_features}  use_temporal={cfg.use_temporal}")

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

    model = DebrisPredictor(
        in_channels=cfg.in_channels,
        seq_features=cfg.seq_features,
        cnn_pretrained=cfg.cnn_pretrained,
        use_temporal=cfg.use_temporal,
    ).to(cfg.device)

    pos_weight = torch.tensor([cfg.pos_weight], device=cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1 = -1.0
    best_path = cfg.ckpt_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            cfg.device,
            cfg.grad_clip,
            cfg.log_every,
            epoch,
        )
        metrics = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step()
        dt = time.time() - t0

        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  "
            f"val[{metrics}]  lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)"
        )

        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            cfg_safe = {
                k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
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
            print(f"[epoch {epoch:02d}] new best F1={best_f1:.3f} -> {best_path}")

    return best_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="use SyntheticDebrisDataset")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--cnn-only", action="store_true", help="disable LSTM branch")
    p.add_argument("--use-hycom", action="store_true")
    p.add_argument("--data-root", type=Path, default=None)
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
    if args.cnn_only:
        cfg.use_temporal = False
    if args.use_hycom:
        cfg.use_hycom = True
    if args.data_root is not None:
        cfg.data_root = args.data_root
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
