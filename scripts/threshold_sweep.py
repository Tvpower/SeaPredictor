"""Sweep the Marine Debris (class 0) threshold on the test split.

Holds every other class's threshold at its tuned value, varies only class 0
across [0.30 .. 0.80] in 0.05 steps. For each step, reports:
  - overall test precision/recall/F1 for class 0
  - same metrics restricted to "detectable" tiles (>= N GT debris pixels)
  - macro-F1 across all 15 classes (so you see the cost to other classes)

Usage:
    python -m scripts.threshold_sweep \
        --ckpt checkpoints/cnn_only_v3/best.pt \
        --thresholds checkpoints/cnn_only_v3/thresholds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import DebrisDataset
from src.models import DebrisPredictor
from src.training.tune_thresholds import collect_probs, macro_f1


CLASS = 0
DEBRIS_MASK_VALUE = 1
DETECTABLE_MIN_PIXELS = 10


def _f1(probs_c, labels_c, thr):
    pred = (probs_c >= thr).astype(int)
    tp = int(((pred == 1) & (labels_c == 1)).sum())
    fp = int(((pred == 1) & (labels_c == 0)).sum())
    fn = int(((pred == 0) & (labels_c == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f, tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True,
                        help="Tuned thresholds.json — only class 0 is overridden in the sweep.")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    model = DebrisPredictor(
        in_channels=cfg["in_channels"],
        seq_features=cfg["seq_features"],
        num_classes=cfg["num_classes"],
        cnn_pretrained=False,
        use_temporal=cfg["use_temporal"],
    ).to(args.device)
    model.load_state_dict(state["model"])

    ds = DebrisDataset(split="test")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"[sweep] collecting probs on test ({len(ds)} tiles)...")
    probs, labels = collect_probs(model, loader, args.device)

    thr_full = np.array(json.loads(args.thresholds.read_text())["thresholds"],
                        dtype=np.float32)
    current_thr = float(thr_full[CLASS])
    print(f"[sweep] current tuned class-{CLASS} threshold = {current_thr:.3f}")

    print(f"[sweep] counting GT debris pixels per tile...")
    n_debris = np.zeros(len(ds.records), dtype=np.int32)
    for i, rec in enumerate(ds.records):
        with rasterio.open(rec.mask_path) as src:
            cl = src.read(1)
        n_debris[i] = int((cl == DEBRIS_MASK_VALUE).sum())
    detectable = n_debris >= DETECTABLE_MIN_PIXELS
    print(f"[sweep] detectable positives (>= {DETECTABLE_MIN_PIXELS} px): "
          f"{int(detectable.sum())} / {int((labels[:, CLASS] == 1).sum())}")

    print()
    print(f"{'thr':>5}  {'P':>5}  {'R':>5}  {'F1':>5}  "
          f"{'P_det':>6}  {'R_det':>6}  {'F1_det':>6}  "
          f"{'macroF1':>7}  {'tp':>3}  {'fp':>3}  {'fn':>3}  note")
    print("-" * 96)
    steps = []
    thr = args.low
    while thr <= args.high + 1e-9:
        steps.append(round(thr, 4))
        thr += args.step

    best_f1 = best_f1_thr = -1.0
    best_macro = best_macro_thr = -1.0
    rows = []
    for thr_c in steps:
        p, r, f, tp, fp, fn = _f1(probs[:, CLASS], labels[:, CLASS], thr_c)
        det_mask = detectable | (labels[:, CLASS] == 0)  # all negatives + detectable positives
        p_d, r_d, f_d, _, _, _ = _f1(
            probs[det_mask, CLASS], labels[det_mask, CLASS], thr_c
        )
        thr_full_swept = thr_full.copy()
        thr_full_swept[CLASS] = thr_c
        m_f1 = macro_f1(probs, labels, thr_full_swept)

        notes = []
        if abs(thr_c - current_thr) < 0.025:
            notes.append("<-- current")
        if f > best_f1:
            best_f1 = f
            best_f1_thr = thr_c
        if m_f1 > best_macro:
            best_macro = m_f1
            best_macro_thr = thr_c

        rows.append((thr_c, p, r, f, p_d, r_d, f_d, m_f1, tp, fp, fn, ", ".join(notes)))

    # Add best-marker notes after the loop now that we know which is best.
    out_rows = []
    for row in rows:
        thr_c, p, r, f, p_d, r_d, f_d, m_f1, tp, fp, fn, note = row
        extra = []
        if note:
            extra.append(note)
        if abs(thr_c - best_f1_thr) < 1e-6:
            extra.append("BEST class-0 F1")
        if abs(thr_c - best_macro_thr) < 1e-6:
            extra.append("BEST macro-F1")
        out_rows.append((thr_c, p, r, f, p_d, r_d, f_d, m_f1, tp, fp, fn,
                         "  ".join(extra)))

    for thr_c, p, r, f, p_d, r_d, f_d, m_f1, tp, fp, fn, note in out_rows:
        print(f"{thr_c:5.2f}  {p:5.3f}  {r:5.3f}  {f:5.3f}  "
              f"{p_d:6.3f}  {r_d:6.3f}  {f_d:6.3f}  "
              f"{m_f1:7.4f}  {tp:3d}  {fp:3d}  {fn:3d}  {note}")

    print()
    print(f"[sweep] best class-0 F1 = {best_f1:.4f} at thr={best_f1_thr:.2f}")
    print(f"[sweep] best macro F1   = {best_macro:.4f} at thr={best_macro_thr:.2f}")


if __name__ == "__main__":
    main()
