"""Locked-in test-set evaluation.

Use this once per checkpoint to produce the final number you report. Loads:
  - a trained checkpoint (`best.pt`)
  - optional `thresholds.json` produced by `tune_thresholds.py`

and prints the headline macro metrics + per-class F1 on the held-out test split.
Optionally writes a JSON report next to the checkpoint.

Usage:
    python -m src.training.eval_test \
        --ckpt checkpoints/cnn_only/best.pt \
        --thresholds checkpoints/cnn_only/thresholds.json \
        --report checkpoints/cnn_only/test_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DebrisDataset
from src.models import DebrisPredictor
from src.training.config import TrainConfig
from src.training.tune_thresholds import collect_probs, f1_at, macro_f1


def per_class_stats(
    probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> list[dict]:
    out = []
    for c in range(probs.shape[1]):
        thr = float(thresholds[c])
        preds = (probs[:, c] >= thr).astype(np.int32)
        tp = int(((preds == 1) & (labels[:, c] == 1)).sum())
        fp = int(((preds == 1) & (labels[:, c] == 0)).sum())
        fn = int(((preds == 0) & (labels[:, c] == 1)).sum())
        tn = int(((preds == 0) & (labels[:, c] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(labels[:, c].sum())
        out.append({
            "class": c,
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="Optional thresholds.json from tune_thresholds.py")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_cfg = state.get("cfg", {})

    cfg = TrainConfig(
        in_channels=saved_cfg.get("in_channels", 11),
        num_classes=saved_cfg.get("num_classes", 15),
        seq_features=saved_cfg.get("seq_features", 4),
        seq_length=saved_cfg.get("seq_length", 30),
        use_temporal=saved_cfg.get("use_temporal", True),
        bands=saved_cfg.get("bands"),
        batch_size=args.batch_size,
    )
    data_root = saved_cfg.get("data_root")
    if isinstance(data_root, str):
        cfg.data_root = Path(data_root)
    elif data_root is not None:
        cfg.data_root = data_root

    device = cfg.device
    print(f"[test] device={device}  ckpt={args.ckpt}  split={args.split}")

    if args.thresholds is not None and args.thresholds.exists():
        thr_payload = json.loads(args.thresholds.read_text())
        thresholds = np.asarray(thr_payload["thresholds"], dtype=np.float32)
        print(f"[test] using tuned thresholds from {args.thresholds}")
    else:
        thresholds = np.full(cfg.num_classes, 0.5, dtype=np.float32)
        print("[test] no thresholds.json supplied -> using 0.50 for all classes")

    dataset = DebrisDataset(
        data_root=cfg.data_root,
        split=args.split,
        seq_length=cfg.seq_length,
        seq_features=cfg.seq_features,
        bands=cfg.bands,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    model = DebrisPredictor(
        in_channels=cfg.in_channels,
        seq_features=cfg.seq_features,
        num_classes=cfg.num_classes,
        cnn_pretrained=False,
        use_temporal=cfg.use_temporal,
    ).to(device)
    model.load_state_dict(state["model"])

    probs, labels = collect_probs(model, loader, device)
    n = labels.shape[0]

    macro_default = macro_f1(probs, labels, np.full(cfg.num_classes, 0.5, dtype=np.float32))
    macro_tuned = macro_f1(probs, labels, thresholds)

    per_class = per_class_stats(probs, labels, thresholds)
    macro_p = float(np.mean([c["precision"] for c in per_class]))
    macro_r = float(np.mean([c["recall"] for c in per_class]))

    print(f"[test] tiles={n}  classes={cfg.num_classes}")
    print(f"[test] macro-F1 @0.50  : {macro_default:.4f}")
    print(f"[test] macro-F1 @tuned : {macro_tuned:.4f}")
    print(f"[test] macro-P @tuned  : {macro_p:.4f}")
    print(f"[test] macro-R @tuned  : {macro_r:.4f}")

    print("[test] per-class:")
    print(f"  {'class':>5}  {'thr':>5}  {'P':>5}  {'R':>5}  {'F1':>5}  {'sup':>5}")
    for row in per_class:
        print(
            f"  {row['class']:5d}  {row['threshold']:5.2f}  "
            f"{row['precision']:5.3f}  {row['recall']:5.3f}  "
            f"{row['f1']:5.3f}  {row['support']:5d}"
        )

    if args.report is not None:
        report = {
            "ckpt": str(args.ckpt),
            "split": args.split,
            "tiles": n,
            "num_classes": int(cfg.num_classes),
            "thresholds_used": [float(t) for t in thresholds],
            "macro_f1_default": macro_default,
            "macro_f1_tuned": macro_tuned,
            "macro_precision_tuned": macro_p,
            "macro_recall_tuned": macro_r,
            "per_class": per_class,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"[test] wrote {args.report}")


if __name__ == "__main__":
    main()
