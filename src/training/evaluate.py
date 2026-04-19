"""Validation / evaluation utilities for pixel-wise segmentation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.models.segmentation_model import DebrisSegmenter, FocalLoss
from src.training.config import TrainConfig


DEBRIS_CLASS_ID = 1  # class index for Marine Debris after aggregate_classes


@dataclass
class EvalMetrics:
    loss: float
    pixel_acc: float       # overall pixel accuracy
    mean_iou: float        # mean IoU across all 12 classes
    debris_iou: float      # IoU for Marine Debris (class 1) — primary metric
    debris_f1: float       # F1  for Marine Debris (class 1)
    debris_recall: float   # recall for Marine Debris — prioritise not missing debris

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} px_acc={self.pixel_acc:.3f} "
            f"mIoU={self.mean_iou:.3f} "
            f"debris[IoU={self.debris_iou:.3f} F1={self.debris_f1:.3f} R={self.debris_recall:.3f}]"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int = 12,
) -> EvalMetrics:
    model.eval()

    acc = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    miou = MulticlassJaccardIndex(num_classes=num_classes, average="macro").to(device)
    # Per-class metrics for debris: index = DEBRIS_CLASS_ID
    iou_per = MulticlassJaccardIndex(num_classes=num_classes, average="none").to(device)
    f1_per = MulticlassF1Score(num_classes=num_classes, average="none").to(device)
    rec_per = MulticlassRecall(num_classes=num_classes, average="none").to(device)

    total_loss = 0.0
    total_n = 0

    for image, mask, _conf in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(image)                       # (B, C, H, W)
        loss = criterion(logits, mask)
        total_loss += loss.item() * mask.size(0)
        total_n += mask.size(0)

        preds = logits.argmax(dim=1)               # (B, H, W)
        acc.update(preds, mask)
        miou.update(preds, mask)
        iou_per.update(preds, mask)
        f1_per.update(preds, mask)
        rec_per.update(preds, mask)

    iou_classes = iou_per.compute()
    f1_classes = f1_per.compute()
    rec_classes = rec_per.compute()

    return EvalMetrics(
        loss=total_loss / max(total_n, 1),
        pixel_acc=acc.compute().item(),
        mean_iou=miou.compute().item(),
        debris_iou=iou_classes[DEBRIS_CLASS_ID].item(),
        debris_f1=f1_classes[DEBRIS_CLASS_ID].item(),
        debris_recall=rec_classes[DEBRIS_CLASS_ID].item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = TrainConfig(batch_size=args.batch_size)

    if args.synthetic:
        from src.training.train import _SyntheticSegDataset
        dataset = _SyntheticSegDataset(32, cfg.in_channels, cfg.num_classes)
    else:
        from src.dataset.marida_dataset import MARIDADataset
        from src.dataset.normalization import load_stats
        norm_stats = load_stats(str(cfg.norm_stats_path))
        dataset = MARIDADataset(
            split_file=str(cfg.split_dir / "val_X.txt"),
            patches_dir=str(cfg.patches_dir),
            augment=False,
            add_indices=True,
            aggregate=True,
            norm_stats=norm_stats,
        )

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    model = DebrisSegmenter(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        pretrained=False,
    ).to(cfg.device)
    state = torch.load(args.ckpt, map_location=cfg.device)
    model.load_state_dict(state["model"])

    criterion = FocalLoss(gamma=cfg.focal_gamma)
    metrics = evaluate(model, loader, criterion, cfg.device, cfg.num_classes)
    print(f"[eval] {metrics}")


if __name__ == "__main__":
    main()
