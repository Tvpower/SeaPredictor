"""Main training loop for the multi-label debris predictor.

Usage:
    # smoke test on synthetic data
    python -m src.training.train --synthetic --epochs 2 --batch-size 4

    # real MARIDA training (auto-detects data/data/raw/MARIDA)
    python -m src.training.train --epochs 30 --batch-size 16

    # CNN-only ablation (skip the LSTM branch entirely)
    python -m src.training.train --cnn-only --epochs 30
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import DebrisDataset, SyntheticDebrisDataset
from src.dataset.marida_loader import compute_pos_weight
from src.models import DebrisPredictor
from src.training.config import TrainConfig
from src.training.evaluate import evaluate


# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #
def _configure_device(device: str) -> None:
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _build_datasets(cfg: TrainConfig, synthetic: bool) -> tuple[Dataset, Dataset]:
    if synthetic:
        train_ds = SyntheticDebrisDataset(
            n_samples=256,
            seq_length=cfg.seq_length,
            seq_features=cfg.seq_features,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            seed=0,
        )
        val_ds = SyntheticDebrisDataset(
            n_samples=64,
            seq_length=cfg.seq_length,
            seq_features=cfg.seq_features,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            seed=1,
        )
        return train_ds, val_ds

    train_ds = DebrisDataset(
        data_root=cfg.data_root,
        split="train",
        seq_length=cfg.seq_length,
        seq_features=cfg.seq_features,
        bands=cfg.bands,
    )
    val_ds = DebrisDataset(
        data_root=cfg.data_root,
        split="val",
        seq_length=cfg.seq_length,
        seq_features=cfg.seq_features,
        bands=cfg.bands,
    )
    return train_ds, val_ds


def _resolve_pos_weight(
    cfg: TrainConfig, train_ds: Dataset, synthetic: bool, device: str
) -> torch.Tensor | None:
    if not cfg.auto_pos_weight:
        return None
    if synthetic:
        return torch.ones(cfg.num_classes, device=device)
    # Real MARIDA dataset exposes its index for cheap label-matrix access.
    assert isinstance(train_ds, DebrisDataset)
    label_matrix = np.stack([r.label for r in train_ds.records], axis=0)
    weight = compute_pos_weight(label_matrix, clip=cfg.pos_weight_clip)
    print(
        f"[train] auto pos_weight (per class): "
        f"min={weight.min():.2f}  max={weight.max():.2f}  mean={weight.mean():.2f}"
    )
    return torch.from_numpy(weight).to(device)


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
        # `non_blocking` is a CUDA pinned-memory thing; on MPS it's a no-op
        # that makes downstream .cpu() chains race. Just don't use it.
        image = image.to(device)
        seq = seq.to(device)
        label = label.to(device)

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
    _configure_device(cfg.device)
    _seed_everything(cfg.seed)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train] device={cfg.device}  workers={cfg.num_workers}  "
        f"synthetic={synthetic}  num_classes={cfg.num_classes}"
    )
    print(
        f"[train] in_channels={cfg.in_channels}  seq_features={cfg.seq_features}  "
        f"use_temporal={cfg.use_temporal}"
    )

    train_ds, val_ds = _build_datasets(cfg, synthetic)
    print(f"[train] train tiles={len(train_ds)}  val tiles={len(val_ds)}")
    if not synthetic and isinstance(train_ds, DebrisDataset):
        if train_ds.oscar is not None:
            o = train_ds.oscar
            print(
                f"[train] OSCAR enabled: {len(o.available_dates)} daily files "
                f"({o.min_date}..{o.max_date})"
            )
        else:
            print("[train] OSCAR not found -> LSTM input zero-filled")

    pin_memory = cfg.device == "cuda"
    persistent_workers = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = DebrisPredictor(
        in_channels=cfg.in_channels,
        seq_features=cfg.seq_features,
        num_classes=cfg.num_classes,
        cnn_pretrained=cfg.cnn_pretrained,
        use_temporal=cfg.use_temporal,
        head_dropout=cfg.head_dropout,
    ).to(cfg.device)

    pos_weight = _resolve_pos_weight(cfg, train_ds, synthetic, cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1 = -1.0
    best_path = cfg.ckpt_dir / "best.pt"
    epochs_since_best = 0

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
        metrics = evaluate(model, val_loader, criterion, cfg.device, cfg.num_classes)
        scheduler.step()
        dt = time.time() - t0

        cov_msg = ""
        if isinstance(train_ds, DebrisDataset) and train_ds.oscar is not None:
            cov_msg = f"  oscar_cov={train_ds.mean_oscar_coverage:.2f}"
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  "
            f"val[{metrics}]  lr={scheduler.get_last_lr()[0]:.2e}{cov_msg}  ({dt:.1f}s)"
        )

        if metrics.macro_f1 > best_f1:
            best_f1 = metrics.macro_f1
            epochs_since_best = 0
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
        else:
            epochs_since_best += 1
            if (
                cfg.early_stopping_patience > 0
                and epochs_since_best >= cfg.early_stopping_patience
            ):
                print(
                    f"[epoch {epoch:02d}] early stopping: no F1 improvement in "
                    f"{cfg.early_stopping_patience} epochs (best={best_f1:.3f})"
                )
                break

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
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--bands", type=int, nargs="+", default=None,
                   help="1-indexed band subset; e.g. --bands 4 8 11 for B4 B8 B11")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--head-dropout", type=float, default=None,
                   help="Dropout in the fusion head (default 0.3)")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None,
                   help="Stop if val macro-F1 doesn't improve in N epochs (0=off)")
    p.add_argument("--pos-weight-clip", type=float, default=None)
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
    if args.num_classes is not None:
        cfg.num_classes = args.num_classes
    if args.in_channels is not None:
        cfg.in_channels = args.in_channels
    if args.bands is not None:
        cfg.bands = args.bands
        cfg.in_channels = len(args.bands)
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.device is not None:
        cfg.device = args.device
    if args.head_dropout is not None:
        cfg.head_dropout = args.head_dropout
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.early_stopping_patience is not None:
        cfg.early_stopping_patience = args.early_stopping_patience
    if args.pos_weight_clip is not None:
        cfg.pos_weight_clip = args.pos_weight_clip
    return cfg


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _apply_overrides(TrainConfig(), args)
    train(cfg, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
