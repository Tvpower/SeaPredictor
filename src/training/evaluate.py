"""Validation / evaluation utilities."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from src.dataset import DebrisDataset, SyntheticDebrisDataset
from src.models import DebrisPredictor
from src.training.config import TrainConfig


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} acc={self.accuracy:.3f} "
            f"P={self.precision:.3f} R={self.recall:.3f} "
            f"F1={self.f1:.3f} IoU={self.iou:.3f}"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> EvalMetrics:
    model.eval()

    acc = BinaryAccuracy().to(device)
    prec = BinaryPrecision().to(device)
    rec = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    iou = BinaryJaccardIndex().to(device)

    total_loss = 0.0
    total_n = 0
    for image, seq, label in loader:
        image = image.to(device, non_blocking=True)
        seq = seq.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(image, seq)
        loss = criterion(logits, label)
        total_loss += loss.item() * label.size(0)
        total_n += label.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        target = label.int()
        acc.update(preds, target)
        prec.update(preds, target)
        rec.update(preds, target)
        f1.update(preds, target)
        iou.update(preds, target)

    return EvalMetrics(
        loss=total_loss / max(total_n, 1),
        accuracy=acc.compute().item(),
        precision=prec.compute().item(),
        recall=rec.compute().item(),
        f1=f1.compute().item(),
        iou=iou.compute().item(),
    )


def _load_eval_dataset(cfg: TrainConfig, synthetic: bool) -> torch.utils.data.Dataset:
    if synthetic:
        return SyntheticDebrisDataset(
            n_samples=64,
            seq_length=cfg.seq_length,
            seq_features=cfg.seq_features,
            in_channels=cfg.in_channels,
        )
    return DebrisDataset(
        data_root=cfg.data_root,
        split="val",
        use_hycom=cfg.use_hycom,
        seq_length=cfg.seq_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = TrainConfig(batch_size=args.batch_size)
    cfg.derive_seq_features()

    dataset = _load_eval_dataset(cfg, args.synthetic)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    model = DebrisPredictor(
        in_channels=cfg.in_channels,
        seq_features=cfg.seq_features,
        cnn_pretrained=False,
        use_temporal=cfg.use_temporal,
    ).to(cfg.device)
    state = torch.load(args.ckpt, map_location=cfg.device)
    model.load_state_dict(state["model"])

    pos_weight = torch.tensor([cfg.pos_weight], device=cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    metrics = evaluate(model, loader, criterion, cfg.device)
    print(f"[eval] {metrics}")


if __name__ == "__main__":
    main()
