# SeaPredictor — AGENTS.md

Project for the FullyHacks hackathon. Predicts ocean garbage patch location/movement
from Sentinel-2 satellite imagery fused with NOAA ocean current sequences.

---

## Team ownership

| Area | Owner |
|------|-------|
| Preprocessing / DataLoader | Yves |
| CNN model (spatial encoder) | Ethan |
| LSTM + fusion model, training loop | TBD |
| Human Delta agent integration | TBD |
| Frontend (Leaflet.js heatmap) | TBD |

---

## Conda environment

```bash
conda activate debris_env   # Python 3.10
pip install -r requirements.txt
```

---

## Data

MARIDA dataset (5 GB) lives at `data/raw/MARIDA/` — gitignored. Dataset is hosted on
HuggingFace at `HallowsYves/SeaPredictor` (private). Teammates get it via:

```bash
python scripts/download_marida.py   # pulls from HuggingFace, requires HF_TOKEN in .env
```

You need `HF_TOKEN=<read-only token>` in your `.env`. Get the token from Yves.
See `.env.example` for the required variable name.

### Layout after download

```
data/raw/MARIDA/
├── patches/
│   └── S2_<DD-M-YY>_<TILE>/
│       ├── S2_<patch_id>.tif        # 11-band (256×256×11), Rayleigh reflectance
│       ├── S2_<patch_id>_cl.tif     # class mask, values 1–15
│       └── S2_<patch_id>_conf.tif   # confidence mask, values 1–3
├── splits/
│   ├── train_X.txt   # 694 patches (50%)
│   ├── val_X.txt     # 328 patches (25%)
│   └── test_X.txt    # 359 patches (25%)
├── norm_stats.json     ← committed to git, computed from train split
└── class_weights.json  ← committed to git
```

**Critical path quirk:** Split files contain IDs *without* the `S2_` prefix
(e.g. `1-12-19_48MYU_0`), but folders and files on disk are named with `S2_`
prepended. `MARIDADataset._resolve_paths()` handles this mapping — don't change it.

Splits are scene-aware (no spatial leakage between train/val/test). Never reshuffle them.

---

## Open decisions — resolve before Phase 2 starts

1. **Segmentation vs. tile-level classification** — the most important decision right now.
   - **Option A — Segmentation:** Replace the fusion head with a U-Net decoder, use
     `CrossEntropyLoss` with class weights. More faithful to MARIDA's pixel labels,
     gives spatial heatmaps for the frontend. Ethan owns the decoder.

2. **train.py must be updated** — currently wired to the `DebrisDataset` stub. Once the
   segmentation/classification decision is made, swap to `MARIDADataset`.

3. **CNNEncoder input channels** — must be instantiated as `CNNEncoder(in_channels=19)`,
   not the default `in_channels=3`.

4. **Focal Loss** — Marine Debris is 0.004% of pixels. Plain CrossEntropyLoss with weights
   may still underfit debris. Consider Focal Loss (`gamma=2.0`) in the training loop.

5. **OSCAR/HYCOM download** — temporal data not yet acquired. Phase 3 is blocked on this.

6. **MADOS dataset** — optional augmentation source, not yet downloaded.
   Zenodo record: https://zenodo.org/records/10664073

---

## Source modules

### `src/dataset/` — preprocessing (Yves, DONE)

| File | Purpose | Status |
|------|---------|--------|
| `marida_dataset.py` | `MARIDADataset` — the working PyTorch Dataset class | **Done** |
| `spectral_indices.py` | 8 spectral indices (NDVI, NDWI, FDI, FAI, NDMI, BSI, NRD, SI) from 11 raw bands | **Done** |
| `normalization.py` | Per-band z-score normalization; `compute_band_stats`, `compute_class_weights` | **Done** |
| `augmentation.py` | Random rotation + hflip + vflip, mask-aligned | **Done** |
| `debris_dataset.py` | Older stub — `DebrisDataset` (tile-level, not yet wired to real data) and `SyntheticDebrisDataset` | Stub |

Use `MARIDADataset`, not `DebrisDataset`, for real training.

**`MARIDADataset` output per sample:**

```
image : torch.float32  (19, 256, 256)  — 11 raw bands + 8 spectral indices, z-score normalized
mask  : torch.int64    (256, 256)      — aggregated class IDs 0–11
conf  : torch.int64    (256, 256)      — confidence 1–3 (0 = no annotation)
```

**Loading with normalization:**

```python
from src.dataset.marida_dataset import MARIDADataset
from src.dataset.normalization import load_stats

stats = load_stats("data/raw/MARIDA/norm_stats.json")

train_ds = MARIDADataset(
    split_file="data/raw/MARIDA/splits/train_X.txt",
    patches_dir="data/raw/MARIDA/patches",
    augment=True,
    add_indices=True,
    aggregate=True,
    norm_stats=stats,
)
```

**Class mapping (after `aggregate_classes`):**

| ID | Class | Train px | Weight |
|----|-------|----------|--------|
| 0  | Other/Background | 45,052,572 (99.06%) | — |
| 1  | Marine Debris | 1,943 | 0.869 |
| 2  | Dense Sargassum | 870 | 1.941 |
| 3  | Sparse Sargassum | 1,091 | 1.548 |
| 4  | Natural Organic | 723 | 2.335 |
| 5  | Ship | 3,289 | 0.513 |
| 6  | Clouds | 65,295 | 0.026 |
| 7  | Marine Water | 100,725 | 0.017 |
| 8  | Sediment-Laden Water | 154,335 | 0.011 |
| 9  | Foam | 469 | 3.600 |
| 10 | Turbid Water | 86,820 | 0.019 |
| 11 | Shallow Water | 13,852 | 0.122 |

Class imbalance is severe — Marine Debris is 0.004% of pixels. Load class weights:

```python
import json
with open("data/raw/MARIDA/class_weights.json") as f:
    class_weights = json.load(f)   # list of 11 floats, index 0 = class ID 1
```

### `src/models/` — model architecture (Ethan)

| File | Purpose | Status |
|------|---------|--------|
| `cnn_encoder.py` | `CNNEncoder` — ResNet-18 backbone, removes classification head, outputs (B, 512) | Ready |
| `lstm_encoder.py` | `LSTMEncoder` — 2-layer LSTM for OSCAR sequences, outputs (B, 128) | Ready |
| `fusion_model.py` | `DebrisPredictor` — concatenates CNN+LSTM features → scalar logit | Ready (tile-level only) |

### `src/training/`

| File | Purpose | Status |
|------|---------|--------|
| `config.py` | `TrainConfig` dataclass — all hyperparameters | Ready |
| `train.py` | Training loop — currently wired to `DebrisDataset` stub | **Needs update to `MARIDADataset`** |
| `evaluate.py` | Evaluation metrics (accuracy, precision, recall, F1, IoU) | Ready |

### `src/utils/`

| File | Purpose | Status |
|------|---------|--------|
| `export.py` | TorchScript export | Not started |

---

## Scripts

```bash
python scripts/download_marida.py  # pull dataset from HuggingFace (requires HF_TOKEN)
python scripts/validate_data.py    # 7-point preprocessing validation — all pass
python scripts/compute_stats.py    # recomputes norm_stats.json + class_weights.json (5 min)
```

Smoke test the training loop on synthetic data (no real data needed):

```bash
python -m src.training.train --synthetic --epochs 2 --batch-size 4
```

---

## Build phases vs. current state

| Phase | Deliverable | Status |
|-------|------------|--------|
| 1 — Data Pipeline | `MARIDADataset`, norm stats, HuggingFace upload | **Done** |
| 2 — CNN Baseline | ResNet-18 trained on MARIDA (>75% accuracy) | **Blocked** — needs segmentation/classification decision |
| 3 — Hybrid Model | CNN + LSTM fusion with OSCAR temporal input | Not started — OSCAR data not downloaded |
| 4 — Export | TorchScript `.pt` for C++ inference server | Not started |
| 5 — Agent | Human Delta API enrichment at inference | Not started |
| 6 — Frontend | Leaflet.js heatmap + citation sidebar | Not started |

---

## Key invariants — do not break these

- Never reshuffle train/val/test splits — they are scene-aware.
- Always apply `nan_to_num` to spectral indices before concatenating to image tensor.
- Compute normalization stats from training split only.
- `MARIDADataset` expects `patches_dir` pointing to `data/raw/MARIDA/patches/` and
  split files from `data/raw/MARIDA/splits/`.
