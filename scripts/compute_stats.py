"""Compute and save normalization stats and class weights from the training split.

Run from repo root:
    python scripts/compute_stats.py
"""
import sys
import json
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        print("(tqdm not installed — no progress bar)")
        return it

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.marida_dataset import load_patch, load_mask, aggregate_classes
from src.dataset.spectral_indices import stack_indices
from src.dataset.normalization import compute_band_stats, compute_class_weights, save_stats

DATA_ROOT   = Path("data/raw/MARIDA")
PATCHES_DIR = DATA_ROOT / "patches"
SPLITS_DIR  = DATA_ROOT / "splits"
TRAIN_SPLIT = SPLITS_DIR / "train_X.txt"

NORM_STATS_PATH    = DATA_ROOT / "norm_stats.json"
CLASS_WEIGHTS_PATH = DATA_ROOT / "class_weights.json"

with open(TRAIN_SPLIT) as f:
    patch_ids = [l.strip() for l in f if l.strip()]

print(f"Loading {len(patch_ids)} training patches...")

image_stacks = []
agg_masks    = []
skipped      = 0

for pid in tqdm(patch_ids, desc="Loading patches"):
    parts  = pid.rsplit('_', 1)
    folder = 'S2_' + parts[0]
    stem   = 'S2_' + pid
    base   = PATCHES_DIR / folder / stem
    tif    = str(base) + '.tif'
    cl     = str(base) + '_cl.tif'

    try:
        data, _ = load_patch(tif)
        idx_stack = stack_indices(data)
        idx_stack = np.nan_to_num(idx_stack, nan=0.0, posinf=0.0, neginf=0.0)
        full = np.concatenate([data, idx_stack], axis=0).astype(np.float32)  # (19,H,W)
        image_stacks.append(full)

        mask = load_mask(cl)
        agg_masks.append(aggregate_classes(mask))
    except Exception as e:
        print(f"  SKIP {pid}: {e}")
        skipped += 1

print(f"\nLoaded {len(image_stacks)} patches ({skipped} skipped)")

# --- normalization stats ---
print("\nComputing per-band mean/std...")
stats = compute_band_stats(image_stacks)
save_stats(stats, NORM_STATS_PATH)
print(f"Saved -> {NORM_STATS_PATH}")

print("\nBand stats (19 channels = 11 raw + 8 indices):")
band_labels = [
    "B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
    "NDVI","NDWI","FDI","FAI","NDMI","BSI","NRD","SI",
]
for i, label in enumerate(band_labels):
    print(f"  {label:<6s}: mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}")

# --- class weights ---
print("\nComputing class weights...")
weights = compute_class_weights(agg_masks, num_classes=11)
weights_list = weights.tolist()
with open(CLASS_WEIGHTS_PATH, 'w') as f:
    json.dump(weights_list, f, indent=2)
print(f"Saved -> {CLASS_WEIGHTS_PATH}")

CLASS_NAMES = {
    0: "Marine Debris", 1: "Dense Sargassum", 2: "Sparse Sargassum",
    3: "Natural Organic", 4: "Ship", 5: "Clouds",
    6: "Marine Water", 7: "Sediment-Laden Water", 8: "Foam",
    9: "Turbid Water", 10: "Shallow Water",
}
print("\nClass weights (11 classes, 1-indexed in masks):")
for i, w in enumerate(weights_list):
    name = CLASS_NAMES.get(i, f"class_{i+1}")
    print(f"  [{i+1:2d}] {name:<25s}: {w:.4f}")

print("\nDone. norm_stats.json and class_weights.json are ready for Ethan's training loop.")
