"""Per-class decision-threshold tuning on the validation set.

The training loop picks the best checkpoint by macro-F1 at threshold 0.5. That's
fine for early stopping, but multi-label models on imbalanced data almost always
have a per-class optimal threshold somewhere else (often 0.2-0.4 for rare
classes, ~0.5 for common ones).

This script:
  1. Loads a trained checkpoint.
  2. Runs inference on the val split, collecting raw sigmoid probabilities.
  3. For each class, sweeps thresholds in [0.05, 0.95] and picks the one that
     maximizes per-class F1.
  4. Reports macro-F1 before/after and writes `thresholds.json` next to the ckpt.

Usage:
    python -m src.training.tune_thresholds \
        --ckpt checkpoints/cnn_only/best.pt \
        --batch-size 16
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


@torch.no_grad()
def collect_probs(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference, return (probs, labels) as (N, C) numpy arrays."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for image, seq, label_cpu in loader:
        image = image.to(device)
        seq = seq.to(device)
        logits = model(image, seq)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label_cpu.numpy())
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)


def f1_at(probs: np.ndarray, labels: np.ndarray, thr: float) -> float:
    """Single-class F1 given a threshold."""
    preds = (probs >= thr).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def macro_f1(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
    """Macro-F1 across classes, each class evaluated at its own threshold."""
    per_class = [
        f1_at(probs[:, c], labels[:, c], float(thresholds[c]))
        for c in range(probs.shape[1])
    ]
    return float(np.mean(per_class))


def tune_per_class(
    probs: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray,
    min_positives: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """For each class, return (best_threshold, best_f1).

    Classes with fewer than `min_positives` positive samples in val are kept at
    the default 0.5 (no signal to tune on).
    """
    n_classes = probs.shape[1]
    best_thr = np.full(n_classes, 0.5, dtype=np.float32)
    best_f1 = np.zeros(n_classes, dtype=np.float32)

    for c in range(n_classes):
        n_pos = int(labels[:, c].sum())
        if n_pos < min_positives:
            best_f1[c] = f1_at(probs[:, c], labels[:, c], 0.5)
            continue
        scores = np.array([f1_at(probs[:, c], labels[:, c], float(t)) for t in grid])
        idx = int(scores.argmax())
        best_thr[c] = float(grid[idx])
        best_f1[c] = float(scores[idx])

    return best_thr, best_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write thresholds.json (default: alongside ckpt)")
    parser.add_argument(
        "--grid-step", type=float, default=0.02,
        help="Threshold grid resolution (default 0.02 -> 46 candidates in [0.05, 0.95])."
    )
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_cfg = state.get("cfg", {})

    # Rehydrate cfg from the checkpoint so we use the right model shape + bands.
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
    print(f"[tune] device={device}  num_classes={cfg.num_classes}  "
          f"in_channels={cfg.in_channels}  use_temporal={cfg.use_temporal}")

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

    print(f"[tune] collecting probabilities on split={args.split} "
          f"({len(dataset)} tiles)...")
    probs, labels = collect_probs(model, loader, device)
    print(f"[tune] probs shape {probs.shape}, label positives per class: "
          f"{labels.sum(0).astype(int).tolist()}")

    default_thr = np.full(cfg.num_classes, 0.5, dtype=np.float32)
    macro_default = macro_f1(probs, labels, default_thr)

    grid = np.arange(0.05, 0.95 + args.grid_step / 2, args.grid_step, dtype=np.float32)
    best_thr, best_f1 = tune_per_class(probs, labels, grid)
    macro_tuned = macro_f1(probs, labels, best_thr)

    print(f"[tune] macro-F1 @0.50 (default): {macro_default:.4f}")
    print(f"[tune] macro-F1 @tuned        : {macro_tuned:.4f}  "
          f"(delta {macro_tuned - macro_default:+.4f})")

    print("[tune] per-class results:")
    print(f"  {'class':>5}  {'thr':>5}  {'F1':>6}  {'pos':>5}")
    for c in range(cfg.num_classes):
        print(f"  {c:5d}  {best_thr[c]:5.2f}  {best_f1[c]:6.3f}  "
              f"{int(labels[:, c].sum()):5d}")

    out_path = args.out or args.ckpt.parent / "thresholds.json"
    payload = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "num_classes": int(cfg.num_classes),
        "thresholds": [float(t) for t in best_thr],
        "per_class_f1": [float(f) for f in best_f1],
        "per_class_positives": [int(p) for p in labels.sum(0)],
        "macro_f1_default": float(macro_default),
        "macro_f1_tuned": float(macro_tuned),
        "grid_step": float(args.grid_step),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[tune] wrote {out_path}")


if __name__ == "__main__":
    main()
