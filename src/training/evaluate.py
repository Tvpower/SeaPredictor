"""Validation / evaluation utilities for multi-label tile classification."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from src.dataset import DebrisDataset, SyntheticDebrisDataset
from src.models import DebrisPredictor
from src.training.config import TrainConfig


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float           # exact-match across all classes (strict)
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} acc={self.accuracy:.3f} "
            f"P={self.macro_precision:.3f} R={self.macro_recall:.3f} "
            f"F1={self.macro_f1:.3f}"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
) -> EvalMetrics:
    model.eval()

    # Metrics live on CPU — torchmetrics on MPS still has gaps and the per-batch
    # transfer is negligible.
    acc = MultilabelAccuracy(num_labels=num_classes, average="macro")
    prec = MultilabelPrecision(num_labels=num_classes, average="macro")
    rec = MultilabelRecall(num_labels=num_classes, average="macro")
    f1 = MultilabelF1Score(num_labels=num_classes, average="macro")

    total_loss = 0.0
    total_n = 0
    for image, seq, label_cpu in loader:
        # `non_blocking` does nothing on MPS but ENABLES races when chained with
        # later .cpu()/.int() calls — drop it. Keep the label on CPU for metrics
        # (the MPS int32 cast is buggy on some PyTorch builds and saturates to
        # INT32_MAX, which torchmetrics rejects). Only the device copy goes to
        # the model + loss.
        image = image.to(device)
        seq = seq.to(device)
        label_dev = label_cpu.to(device)

        logits = model(image, seq)
        loss = criterion(logits, label_dev)
        total_loss += loss.item() * label_cpu.size(0)
        total_n += label_cpu.size(0)

        probs = torch.sigmoid(logits.detach()).cpu()
        preds = (probs >= 0.5).to(torch.long)
        # label_cpu never touched MPS -> safe int cast on CPU
        target = (label_cpu > 0.5).to(torch.long)
        acc.update(preds, target)
        prec.update(preds, target)
        rec.update(preds, target)
        f1.update(preds, target)

    return EvalMetrics(
        loss=total_loss / max(total_n, 1),
        accuracy=acc.compute().item(),
        macro_precision=prec.compute().item(),
        macro_recall=rec.compute().item(),
        macro_f1=f1.compute().item(),
    )


def _load_eval_dataset(cfg: TrainConfig, synthetic: bool):
    if synthetic:
        return SyntheticDebrisDataset(
            n_samples=64,
            seq_length=cfg.seq_length,
            seq_features=cfg.seq_features,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
        )
    return DebrisDataset(
        data_root=cfg.data_root,
        split="val",
        seq_length=cfg.seq_length,
        seq_features=cfg.seq_features,
        bands=cfg.bands,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = TrainConfig(batch_size=args.batch_size)

    dataset = _load_eval_dataset(cfg, args.synthetic)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    model = DebrisPredictor(
        in_channels=cfg.in_channels,
        seq_features=cfg.seq_features,
        num_classes=cfg.num_classes,
        cnn_pretrained=False,
        use_temporal=cfg.use_temporal,
    ).to(cfg.device)
    state = torch.load(args.ckpt, map_location=cfg.device, weights_only=False)
    model.load_state_dict(state["model"])

    criterion = nn.BCEWithLogitsLoss()
    metrics = evaluate(model, loader, criterion, cfg.device, cfg.num_classes)
    print(f"[eval] {metrics}")


if __name__ == "__main__":
    main()
