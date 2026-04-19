"""Find all MARIDA tiles with Marine Debris (class 0) and optionally render
them so you can compare ground-truth pixel locations against model output.

The MARIDA `_cl.tif` mask uses pixel values 1..15 (with 0 = background).
Marine Debris is class index 1 in that mask scheme — which corresponds to
`label[0]` in the tile-level multi-label vector from `labels_mapping.txt`.

Usage:

    # Just list the debris-positive tiles in the test split
    python -m scripts.list_debris_tiles --split test

    # Same, all splits
    python -m scripts.list_debris_tiles --split all

    # Render up to 30 tiles with mask + model overlay to /tmp/marida_debris/
    python -m scripts.list_debris_tiles --split test --render --limit 30 \
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import DebrisDataset
from src.dataset.marida_loader import MaridaIndex
from src.models import DebrisPredictor
from src.training.tune_thresholds import collect_probs


CLASS_IDX = 0          # tile-level vector index for Marine Debris
MASK_PIXEL_VALUE = 1   # _cl.tif uses 1..15 (Marine Debris = 1, 0 = background)
DETECTABLE_MIN_PIXELS = 10  # below this, the GT label is essentially noise
OUT_DIR = Path("/tmp/marida_debris")
LABEL_BAR_H = 60


def count_debris_pixels(mask_path: Path) -> int:
    with rasterio.open(mask_path) as src:
        cl = src.read(1)
    return int((cl == MASK_PIXEL_VALUE).sum())


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Monaco.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def list_debris_tiles(splits: list[str]) -> dict[str, list]:
    """Return {split: [TileRecord, ...]} for tiles where label[CLASS_IDX] == 1."""
    idx = MaridaIndex.from_root(None)
    out: dict[str, list] = {}
    for split in splits:
        recs = idx.split_records(split)
        debris = [r for r in recs if int(r.label[CLASS_IDX]) == 1]
        out[split] = debris
    return out


def render_tile(
    tif_path: Path,
    mask_path: Path,
    out_path: Path,
    *,
    title_suffix: str,
    model_prob: float | None,
    model_pred: int | None,
    threshold: float | None,
    font: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
) -> None:
    """3-panel preview: RGB | per-pixel debris mask | FDI heatmap."""
    with rasterio.open(tif_path) as src:
        rgb = src.read([4, 3, 2]).astype(np.float32)  # B04, B03, B02
        b6, b8, b11 = src.read([6, 8, 10]).astype(np.float32)
    with rasterio.open(mask_path) as src:
        cl = src.read(1).astype(np.int32)

    rgb = np.clip(rgb / max(np.percentile(rgb, 99), 1e-6), 0, 1)
    rgb_img = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)

    h, w = rgb_img.shape[:2]
    debris_mask = (cl == MASK_PIXEL_VALUE)
    n_debris_px = int(debris_mask.sum())

    mask_panel = rgb_img.copy()
    overlay = np.zeros_like(mask_panel)
    overlay[debris_mask] = (0, 255, 0)
    mask_panel = ((mask_panel * 0.55) + (overlay * 0.45)).astype(np.uint8)

    fdi = b8 - (b6 + (b11 - b6) * (834 - 665) / (1610 - 665))
    fdi_n = np.clip((fdi - np.percentile(fdi, 5))
                    / (np.percentile(fdi, 99) - np.percentile(fdi, 5) + 1e-6), 0, 1)
    fdi_panel = np.zeros_like(rgb_img)
    fdi_panel[..., 0] = (fdi_n * 255).astype(np.uint8)

    canvas = np.zeros((h + LABEL_BAR_H, w * 3, 3), dtype=np.uint8)
    canvas[LABEL_BAR_H:, :w] = rgb_img
    canvas[LABEL_BAR_H:, w:2 * w] = mask_panel
    canvas[LABEL_BAR_H:, 2 * w:] = fdi_panel

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    GREEN = (0, 220, 0)
    RED = (255, 70, 70)
    YELLOW = (255, 220, 80)

    line1 = f"{tif_path.name}    {title_suffix}"
    line2 = f"GT debris pixels: {n_debris_px:5d} / {h * w}"
    if model_prob is not None and model_pred is not None and threshold is not None:
        verdict = "FOUND" if model_pred == 1 else "MISSED"
        color = GREEN if model_pred == 1 else RED
        line3 = (f"v3: {'YES' if model_pred else 'NO ':<3}  p={model_prob:.2f}  "
                 f"thr={threshold:.2f}    [{verdict}]")
    else:
        line3 = "(no model overlay)"
        color = (200, 200, 200)

    draw.text((6, 4), line1, fill=YELLOW, font=font)
    draw.text((6, 22), line2, fill=YELLOW, font=font)
    draw.text((6, 42), line3, fill=color, font=font_small)

    # Tiny panel labels
    draw.text((6, LABEL_BAR_H + 4), "RGB", fill=YELLOW, font=font_small)
    draw.text((w + 6, LABEL_BAR_H + 4), "GT debris (green)", fill=YELLOW, font=font_small)
    draw.text((2 * w + 6, LABEL_BAR_H + 4), "FDI", fill=YELLOW, font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _model_probs_for(records, ckpt_path: Path, thresholds_path: Path | None,
                     split: str, device: str = "mps"):
    """Run the model on `split` and return {tile_id: (prob, pred)} for class 0."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    model = DebrisPredictor(
        in_channels=cfg["in_channels"],
        seq_features=cfg["seq_features"],
        num_classes=cfg["num_classes"],
        cnn_pretrained=False,
        use_temporal=cfg["use_temporal"],
    ).to(device)
    model.load_state_dict(state["model"])

    ds = DebrisDataset(split=split)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    probs, _ = collect_probs(model, loader, device)

    if thresholds_path and thresholds_path.exists():
        thr = np.array(json.loads(thresholds_path.read_text())["thresholds"])
    else:
        thr = np.full(probs.shape[1], 0.5)

    out = {}
    for i, rec in enumerate(ds.records):
        p = float(probs[i, CLASS_IDX])
        out[rec.tile_id] = (p, int(p >= thr[CLASS_IDX]), float(thr[CLASS_IDX]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="test")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max tiles to render (only used with --render).")
    parser.add_argument("--render", action="store_true",
                        help="Also render PNG previews to /tmp/marida_debris/")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Optional model checkpoint to overlay predictions.")
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    debris = list_debris_tiles(splits)

    total = sum(len(v) for v in debris.values())
    print(f"\nMarine Debris (class {CLASS_IDX}) tile counts:")
    for split, recs in debris.items():
        print(f"  {split:5s}: {len(recs):4d} tile(s)")
    print(f"  total: {total}")

    # Pixel-count breakdown so the user knows how many tiles are sparse-label.
    print(f"\nDetectability breakdown (>= {DETECTABLE_MIN_PIXELS} GT debris pixels):")
    pixel_counts: dict[str, dict[str, int]] = {}
    for split, recs in debris.items():
        counts = [count_debris_pixels(r.mask_path) for r in recs]
        det = sum(1 for c in counts if c >= DETECTABLE_MIN_PIXELS)
        sparse = sum(1 for c in counts if 0 < c < DETECTABLE_MIN_PIXELS)
        empty = sum(1 for c in counts if c == 0)
        print(f"  {split:5s}: detectable={det:4d}  sparse(<{DETECTABLE_MIN_PIXELS}px)={sparse:4d}  "
              f"label-only(0px)={empty:4d}")
        pixel_counts[split] = {r.tile_id: c for r, c in zip(recs, counts)}

    if not args.render:
        for split, recs in debris.items():
            print(f"\n--- {split} ({len(recs)} tiles) ---")
            for r in recs[:50]:
                npx = pixel_counts[split][r.tile_id]
                tag = " (sparse)" if 0 < npx < DETECTABLE_MIN_PIXELS else \
                      " (label-only)" if npx == 0 else ""
                print(f"  {r.tile_id:38s}  gt_px={npx:5d}{tag}")
            if len(recs) > 50:
                print(f"  ... ({len(recs) - 50} more)")
        print("\n(use --render to also write PNG previews)")
        return

    model_lookup = None
    if args.ckpt is not None:
        if args.split == "all":
            print("Note: --ckpt + --split all only overlays test/val/train one at a time;")
            print("      collecting probs per split now.")
        model_lookup = {}
        for split in splits:
            model_lookup[split] = _model_probs_for(
                debris[split], args.ckpt, args.thresholds, split, args.device,
            )

    font = _load_font(14)
    font_small = _load_font(12)

    for split, recs in debris.items():
        chosen = recs[: args.limit]
        print(f"\nRendering {len(chosen)} {split} tile(s) -> {args.out / split}/")
        for r in chosen:
            prob = pred = thr = None
            if model_lookup is not None:
                prob, pred, thr = model_lookup[split].get(r.tile_id,
                                                          (None, None, None))
            out_path = args.out / split / (r.image_path.stem + ".png")
            render_tile(
                r.image_path,
                r.mask_path,
                out_path,
                title_suffix=f"split={split}",
                model_prob=prob,
                model_pred=pred,
                threshold=thr,
                font=font,
                font_small=font_small,
            )
    print(f"\nDone. Open {args.out}/")


if __name__ == "__main__":
    main()
