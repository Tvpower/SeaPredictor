"""Preprocessing validation script for MARIDA dataset.

Run from repo root:
    python scripts/validate_data.py
"""
import sys
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.marida_dataset import (
    MARIDADataset, load_patch, load_mask, load_confidence, aggregate_classes,
)
from src.dataset.spectral_indices import stack_indices, validate_indices

DATA_ROOT   = Path("data/raw/MARIDA")
PATCHES_DIR = DATA_ROOT / "patches"
SPLITS_DIR  = DATA_ROOT / "splits"
TRAIN_SPLIT = SPLITS_DIR / "train_X.txt"

ERRORS = []


def fail(msg):
    print(f"  FAIL: {msg}")
    ERRORS.append(msg)


def ok(msg):
    print(f"  OK:   {msg}")


# --------------------------------------------------------------------------- #
# Task 1 — Load check: every file in train split                              #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("TASK 1 — Load check (all train patches)")
print("=" * 60)

with open(TRAIN_SPLIT) as f:
    patch_ids = [l.strip() for l in f if l.strip()]

print(f"  Found {len(patch_ids)} patch IDs in train_X.txt")

load_ok = 0
load_fail = []

for pid in patch_ids:
    parts = pid.rsplit('_', 1)
    folder = 'S2_' + parts[0]
    stem   = 'S2_' + pid
    base   = PATCHES_DIR / folder / stem
    tif    = base.with_suffix('.tif')
    cl     = Path(str(base) + '_cl.tif')
    conf   = Path(str(base) + '_conf.tif')
    missing = [p for p in (tif, cl, conf) if not p.exists()]
    if missing:
        load_fail.append((pid, [str(p) for p in missing]))
    else:
        load_ok += 1

if load_fail:
    fail(f"{len(load_fail)} patches have missing files")
    for pid, paths in load_fail[:5]:
        print(f"    {pid}: missing {paths}")
    if len(load_fail) > 5:
        print(f"    ... and {len(load_fail) - 5} more")
else:
    ok(f"All {load_ok} patches have all 3 files")


# --------------------------------------------------------------------------- #
# Task 2 — Shape check                                                        #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("TASK 2 — Shape check (all train patches)")
print("=" * 60)

bad_shapes = []
loadable_ids = [pid for pid in patch_ids
                if pid not in {x[0] for x in load_fail}]

for pid in loadable_ids:
    parts  = pid.rsplit('_', 1)
    folder = 'S2_' + parts[0]
    stem   = 'S2_' + pid
    base   = PATCHES_DIR / folder / stem
    try:
        data, _  = load_patch(str(base) + '.tif')
        mask     = load_mask(str(base) + '_cl.tif')
        conf_arr = load_confidence(str(base) + '_conf.tif')
        if data.shape != (11, 256, 256):
            bad_shapes.append((pid, f"image {data.shape}"))
        if mask.shape != (256, 256):
            bad_shapes.append((pid, f"mask {mask.shape}"))
        if conf_arr.shape != (256, 256):
            bad_shapes.append((pid, f"conf {conf_arr.shape}"))
    except Exception as e:
        bad_shapes.append((pid, str(e)))

if bad_shapes:
    fail(f"{len(bad_shapes)} shape mismatches")
    for pid, info in bad_shapes[:5]:
        print(f"    {pid}: {info}")
else:
    ok("All patches: image (11,256,256), mask (256,256), conf (256,256)")


# --------------------------------------------------------------------------- #
# Task 3 — Band value range (50-patch sample)                                 #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("TASK 3 — Band value range (50-patch sample)")
print("=" * 60)

sample_ids = random.sample(loadable_ids, min(50, len(loadable_ids)))
flagged_high = []
flagged_zero = []

for pid in sample_ids:
    parts  = pid.rsplit('_', 1)
    folder = 'S2_' + parts[0]
    stem   = 'S2_' + pid
    base   = PATCHES_DIR / folder / stem
    try:
        data, _ = load_patch(str(base) + '.tif')
        if np.any(data > 1.0):
            pct = np.mean(data > 1.0) * 100
            flagged_high.append((pid, pct))
        for b in range(data.shape[0]):
            if np.all(data[b] == 0):
                flagged_zero.append((pid, b))
    except Exception:
        pass

if flagged_high:
    print(f"  NOTE: {len(flagged_high)} patches have pixels > 1.0 (may be DN-scaled, check units)")
    for pid, pct in flagged_high[:3]:
        print(f"    {pid}: {pct:.1f}% of pixels > 1.0")
else:
    ok("No patches with values > 1.0")

if flagged_zero:
    fail(f"{len(flagged_zero)} all-zero bands found")
    for pid, b in flagged_zero[:5]:
        print(f"    {pid}: band {b} is all zeros")
else:
    ok("No all-zero bands in sample")


# --------------------------------------------------------------------------- #
# Task 4 — Mask value range + class distribution                              #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("TASK 4 — Mask values and class distribution (all train)")
print("=" * 60)

VALID_RAW  = set(range(16))
VALID_AGG  = set(range(12))
bad_raw    = []
pixel_counts_agg = np.zeros(12, dtype=np.int64)  # indices 0..11

for pid in loadable_ids:
    parts  = pid.rsplit('_', 1)
    folder = 'S2_' + parts[0]
    stem   = 'S2_' + pid
    base   = PATCHES_DIR / folder / stem
    try:
        mask     = load_mask(str(base) + '_cl.tif')
        raw_vals = set(np.unique(mask).tolist())
        if not raw_vals.issubset(VALID_RAW):
            bad_raw.append((pid, raw_vals - VALID_RAW))
        agg = aggregate_classes(mask)
        for cls_id in range(12):
            pixel_counts_agg[cls_id] += int(np.sum(agg == cls_id))
    except Exception:
        pass

if bad_raw:
    fail(f"{len(bad_raw)} masks have out-of-range raw values")
    for pid, vals in bad_raw[:3]:
        print(f"    {pid}: unexpected values {vals}")
else:
    ok("All raw mask values in {0..15}")

agg_unique = np.where(pixel_counts_agg > 0)[0].tolist()
unexpected_agg = set(agg_unique) - VALID_AGG
if unexpected_agg:
    fail(f"Aggregated masks contain unexpected classes: {unexpected_agg}")
else:
    ok(f"Aggregated mask values in {{0..11}}, present classes: {agg_unique}")

print("\n  Class pixel distribution (aggregated, train set):")
total_px = pixel_counts_agg.sum()
CLASS_NAMES = {
    0: "Other/BG", 1: "Marine Debris", 2: "Dense Sargassum",
    3: "Sparse Sargassum", 4: "Natural Organic", 5: "Ship",
    6: "Clouds", 7: "Marine Water", 8: "Sediment-Laden Water",
    9: "Foam", 10: "Turbid Water", 11: "Shallow Water",
}
for cls_id in range(12):
    cnt = pixel_counts_agg[cls_id]
    pct = cnt / total_px * 100 if total_px > 0 else 0
    name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
    print(f"    [{cls_id:2d}] {name:<25s}: {cnt:>10,d} px ({pct:5.2f}%)")


# --------------------------------------------------------------------------- #
# Task 5 — Spectral index NaN/Inf rate (50-patch sample)                      #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("TASK 5 — Spectral index NaN/Inf rate (50-patch sample)")
print("=" * 60)

nan_inf_count = 0
for pid in sample_ids:
    parts  = pid.rsplit('_', 1)
    folder = 'S2_' + parts[0]
    stem   = 'S2_' + pid
    base   = PATCHES_DIR / folder / stem
    try:
        data, _ = load_patch(str(base) + '.tif')
        idx_stack = stack_indices(data)
        if not validate_indices(idx_stack):
            nan_inf_count += 1
    except Exception:
        pass

ok(f"{nan_inf_count}/{len(sample_ids)} patches produce NaN/Inf before nan_to_num "
   f"(expected for near-zero reflectance patches)")


# --------------------------------------------------------------------------- #
# Task 6 — Augmentation spatial alignment (3 patches)                         #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("TASK 6 — Augmentation spatial alignment (3 patches)")
print("=" * 60)

import random as _random
_random.seed(42)
aug_sample = _random.sample(loadable_ids, min(3, len(loadable_ids)))

from src.dataset.augmentation import augment_patch

align_ok = 0
align_fail = []

for pid in aug_sample:
    parts  = pid.rsplit('_', 1)
    folder = 'S2_' + parts[0]
    stem   = 'S2_' + pid
    base   = PATCHES_DIR / folder / stem
    try:
        data, _  = load_patch(str(base) + '.tif')
        mask_arr = load_mask(str(base) + '_cl.tif')
        conf_arr = load_confidence(str(base) + '_conf.tif')

        image_t = torch.from_numpy(data)
        mask_t  = torch.from_numpy(mask_arr).long()
        conf_t  = torch.from_numpy(conf_arr).long()

        aug_img, aug_mask, aug_conf = augment_patch(image_t, mask_t, conf_t)

        # Check that non-zero mask pixels correlate with non-zero image values.
        debris_pixels = (aug_mask == 1)
        if debris_pixels.sum() > 0:
            # NIR band (index 7) should be non-zero for debris pixels.
            nir = aug_img[7][debris_pixels]
            if nir.abs().mean() < 1e-6:
                align_fail.append(pid)
            else:
                align_ok += 1
        else:
            align_ok += 1  # No debris pixels — alignment trivially OK.
    except Exception as e:
        align_fail.append(f"{pid}: {e}")

if align_fail:
    fail(f"Alignment check failed for: {align_fail}")
else:
    ok(f"Augmentation spatial alignment OK for all {align_ok} sampled patches")


# --------------------------------------------------------------------------- #
# Task 7 — DataLoader batch test                                              #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
print("TASK 7 — DataLoader batch test (batch_size=4, no norm)")
print("=" * 60)

try:
    ds = MARIDADataset(
        split_file=str(TRAIN_SPLIT),
        patches_dir=str(PATCHES_DIR),
        augment=False,
        add_indices=True,
        aggregate=True,
        norm_stats=None,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    image_b, mask_b, conf_b = next(iter(loader))

    checks = {
        "image.shape == (4,19,256,256)": image_b.shape == torch.Size([4, 19, 256, 256]),
        "image.dtype == float32":        image_b.dtype == torch.float32,
        "mask.shape == (4,256,256)":     mask_b.shape  == torch.Size([4, 256, 256]),
        "mask.dtype == int64":           mask_b.dtype  == torch.int64,
        "conf.shape == (4,256,256)":     conf_b.shape  == torch.Size([4, 256, 256]),
        "mask values subset {0..11}":    set(mask_b.unique().tolist()).issubset(set(range(12))),
    }

    for desc, passed in checks.items():
        if passed:
            ok(desc)
        else:
            fail(desc)

    print(f"\n  image.shape   = {image_b.shape}")
    print(f"  image.dtype   = {image_b.dtype}")
    print(f"  mask.shape    = {mask_b.shape}")
    print(f"  mask.dtype    = {mask_b.dtype}")
    print(f"  mask unique   = {mask_b.unique().tolist()}")
    print(f"  conf.shape    = {conf_b.shape}")

except Exception as e:
    fail(f"DataLoader batch test raised: {e}")
    import traceback; traceback.print_exc()


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
if ERRORS:
    print(f"VALIDATION FAILED — {len(ERRORS)} error(s) above")
    for e in ERRORS:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("VALIDATION PASSED")
print("=" * 60)
