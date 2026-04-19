import json
from pathlib import Path
import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from src.dataset import DebrisDataset
from src.models import DebrisPredictor
from src.training.tune_thresholds import collect_probs

PREVIEW_DIR = Path("/tmp/v2v3_diff")
LABEL_BAR_H = 60  # px reserved at the top for the text overlay


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    # Try a few mac/linux fonts; fall back to PIL default if none exist.
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Monaco.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT = _load_font(14)
_FONT_SMALL = _load_font(12)


DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_path):
    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = s["cfg"]
    m = DebrisPredictor(in_channels=cfg["in_channels"], seq_features=cfg["seq_features"],
                        num_classes=cfg["num_classes"], cnn_pretrained=False,
                        use_temporal=cfg["use_temporal"]).to(DEVICE)
    m.load_state_dict(s["model"])
    return m, cfg

ds = DebrisDataset(split="test")
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

m2, _ = load("checkpoints/cnn_only_v2/best.pt")
m3, _ = load("checkpoints/cnn_only_v3/best.pt")

p2, y = collect_probs(m2, loader, DEVICE)
p3, _ = collect_probs(m3, loader, DEVICE)

t2 = np.array(json.load(open("checkpoints/cnn_only_v2/thresholds.json"))["thresholds"])
t3 = np.array(json.load(open("checkpoints/cnn_only_v3/thresholds.json"))["thresholds"])

pred2 = (p2 >= t2).astype(int)
pred3 = (p3 >= t3).astype(int)

CLASS = 0  # Marine Debris
DEBRIS_MASK_VALUE = 1  # _cl.tif uses 1..15; 1 = Marine Debris
DETECTABLE_MIN_PIXELS = 10  # below this, the GT label is essentially noise

print(f"\n[diff] counting GT debris pixels per tile...")
n_debris_pixels = np.zeros(len(ds.records), dtype=np.int32)
for i, rec in enumerate(ds.records):
    with rasterio.open(rec.mask_path) as src:
        cl = src.read(1)
    n_debris_pixels[i] = int((cl == DEBRIS_MASK_VALUE).sum())
fp_fixed = np.where((pred2[:, CLASS] == 1) & (y[:, CLASS] == 0)
                    & (pred3[:, CLASS] == 0))[0]
fp_introduced = np.where((pred2[:, CLASS] == 0) & (y[:, CLASS] == 0)
                         & (pred3[:, CLASS] == 1))[0]
fn_fixed = np.where((pred2[:, CLASS] == 0) & (y[:, CLASS] == 1)
                    & (pred3[:, CLASS] == 1))[0]
fn_introduced = np.where((pred2[:, CLASS] == 1) & (y[:, CLASS] == 1)
                         & (pred3[:, CLASS] == 0))[0]

# Ground-truth-positive tiles: did v3 catch them? (sorted by v3 confidence)
gt_pos_idx = np.where(y[:, CLASS] == 1)[0]
tp_v3 = gt_pos_idx[(pred3[gt_pos_idx, CLASS] == 1)]
tp_v3 = tp_v3[np.argsort(-p3[tp_v3, CLASS])]   # most-confident first
fn_v3 = gt_pos_idx[(pred3[gt_pos_idx, CLASS] == 0)]
fn_v3 = fn_v3[np.argsort(-p3[fn_v3, CLASS])]   # closest-misses first

print(f"\nMarine Debris on test split:")
print(f"  ground-truth positives:        {len(gt_pos_idx):3d}")
print(f"  v3 true positives (caught):    {len(tp_v3):3d}  recall={len(tp_v3)/max(len(gt_pos_idx),1):.3f}")
print(f"  v3 false negatives (missed):   {len(fn_v3):3d}")

# Detectability breakdown — separates "real misses" from "label-noise misses".
detectable = n_debris_pixels >= DETECTABLE_MIN_PIXELS
det_pos = gt_pos_idx[detectable[gt_pos_idx]]
det_tp = det_pos[(pred3[det_pos, CLASS] == 1)]
det_fn = det_pos[(pred3[det_pos, CLASS] == 0)]
sparse_pos = gt_pos_idx[~detectable[gt_pos_idx]]
sparse_tp = sparse_pos[(pred3[sparse_pos, CLASS] == 1)]
sparse_fn = sparse_pos[(pred3[sparse_pos, CLASS] == 0)]
print()
print(f"  --- detectability breakdown (>= {DETECTABLE_MIN_PIXELS} GT debris pixels) ---")
print(f"  detectable positives:          {len(det_pos):3d}")
print(f"    v3 caught:                   {len(det_tp):3d}  recall={len(det_tp)/max(len(det_pos),1):.3f}")
print(f"    v3 missed:                   {len(det_fn):3d}")
print(f"  sparse-label positives (<{DETECTABLE_MIN_PIXELS}px): {len(sparse_pos):3d}")
print(f"    v3 caught (lucky):           {len(sparse_tp):3d}  recall={len(sparse_tp)/max(len(sparse_pos),1):.3f}")
print(f"    v3 missed (likely noise):    {len(sparse_fn):3d}")

print()
print(f"  v3 fixed FPs:     {len(fp_fixed):3d}  (v2 said debris, v3 didn't, no debris there)")
print(f"  v3 fixed FNs:     {len(fn_fixed):3d}  (v3 found debris v2 missed)")
print(f"  v3 introduced FP: {len(fp_introduced):3d}  (regression)")
print(f"  v3 introduced FN: {len(fn_introduced):3d}  (regression)")

# Print the actual tile names so you can eyeball them
for label, idxs in [("V3 TP (top 10 most confident)", tp_v3),
                    ("V3 FN (top 10 closest misses)", fn_v3),
                    ("FIXED FP", fp_fixed), ("FIXED FN", fn_fixed),
                    ("NEW FP", fp_introduced), ("NEW FN", fn_introduced)]:
    print(f"\n{label} ({len(idxs)} tiles):")
    for i in idxs[:10]:
        rec = ds.records[i]
        npx = n_debris_pixels[i]
        tag = "  (sparse)" if 0 < npx < DETECTABLE_MIN_PIXELS else ""
        print(f"  {rec.image_path.name}  v2={p2[i,CLASS]:.2f} "
              f"v3={p3[i,CLASS]:.2f}  gt_px={npx}{tag}")


def render(
    tif_path: Path,
    label_dir: str,
    gt: int,
    v2_prob: float,
    v2_pred: int,
    v3_prob: float,
    v3_pred: int,
    v2_thr: float,
    v3_thr: float,
    n_debris_px: int = 0,
    out_path_override: Path | None = None,
) -> None:
    """Two-panel preview (RGB | FDI) with a text bar overlaying GT + v2 + v3."""
    with rasterio.open(tif_path) as src:
        rgb = src.read([4, 3, 2]).astype(np.float32)  # B04, B03, B02
        b6, b8, b11 = src.read([6, 8, 10]).astype(np.float32)
    rgb = np.clip(rgb / max(np.percentile(rgb, 99), 1e-6), 0, 1)
    rgb_img = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
    fdi = b8 - (b6 + (b11 - b6) * (834 - 665) / (1610 - 665))
    fdi_n = np.clip((fdi - np.percentile(fdi, 5))
                    / (np.percentile(fdi, 99) - np.percentile(fdi, 5) + 1e-6), 0, 1)

    h, w = rgb_img.shape[:2]
    canvas = np.zeros((h + LABEL_BAR_H, w * 2, 3), dtype=np.uint8)
    canvas[LABEL_BAR_H:, :w] = rgb_img
    canvas[LABEL_BAR_H:, w:, 0] = (fdi_n * 255).astype(np.uint8)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    GREEN = (0, 220, 0)
    RED = (255, 70, 70)
    GREY = (200, 200, 200)
    YELLOW = (255, 220, 80)

    v2_color = GREEN if v2_pred == gt else RED
    v3_color = GREEN if v3_pred == gt else RED
    gt_label = "DEBRIS" if gt == 1 else "no debris"
    if gt == 1:
        sparse_tag = " [SPARSE]" if 0 < n_debris_px < DETECTABLE_MIN_PIXELS else ""
        gt_label = f"{gt_label} ({n_debris_px}px){sparse_tag}"

    line1 = f"{tif_path.name}    GT: {gt_label}"
    verdict = {
        "fixed_fp": "[v3 FIXED FP]",
        "fixed_fn": "[v3 FIXED FN]",
        "new_fp": "[v3 NEW FP - regression]",
        "new_fn": "[v3 NEW FN - regression]",
        "tp_v3": "[v3 TRUE POSITIVE - debris correctly found]",
        "fn_v3": "[v3 FALSE NEGATIVE - debris missed]",
    }.get(label_dir, "")

    draw.text((6, 4), line1, fill=YELLOW, font=_FONT)
    # Render the v2/v3 segments separately so each can be color-coded.
    x = 6
    y = 22
    seg_v2 = f"v2: {'YES' if v2_pred else 'NO':<3}  p={v2_prob:.2f}  thr={v2_thr:.2f}"
    seg_sep = "     "
    seg_v3 = f"v3: {'YES' if v3_pred else 'NO':<3}  p={v3_prob:.2f}  thr={v3_thr:.2f}"
    draw.text((x, y), seg_v2, fill=v2_color, font=_FONT)
    bbox = draw.textbbox((x, y), seg_v2, font=_FONT)
    x = bbox[2]
    draw.text((x, y), seg_sep, fill=GREY, font=_FONT)
    bbox = draw.textbbox((x, y), seg_sep, font=_FONT)
    x = bbox[2]
    draw.text((x, y), seg_v3, fill=v3_color, font=_FONT)

    if verdict:
        draw.text((6, 42), verdict, fill=GREY, font=_FONT_SMALL)

    out = out_path_override or (
        PREVIEW_DIR / label_dir / tif_path.name.replace(".tif", ".png")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


for label, idxs in [("tp_v3", tp_v3), ("fn_v3", fn_v3),
                    ("fixed_fp", fp_fixed), ("fixed_fn", fn_fixed),
                    ("new_fp", fp_introduced), ("new_fn", fn_introduced)]:
    for rank, i in enumerate(idxs):
        # Prefix tp_v3 / fn_v3 with rank so file ordering matches confidence.
        prefix = f"{rank:03d}_" if label in ("tp_v3", "fn_v3") else ""
        out_name = (PREVIEW_DIR / label / (prefix + ds.records[i].image_path.name)
                    .replace(".tif", ".png"))
        render(
            ds.records[i].image_path,
            label,
            gt=int(y[i, CLASS]),
            v2_prob=float(p2[i, CLASS]),
            v2_pred=int(pred2[i, CLASS]),
            v3_prob=float(p3[i, CLASS]),
            v3_pred=int(pred3[i, CLASS]),
            v2_thr=float(t2[CLASS]),
            v3_thr=float(t3[CLASS]),
            n_debris_px=int(n_debris_pixels[i]),
            out_path_override=out_name,
        )
print(f"\nPreviews written to {PREVIEW_DIR}/")