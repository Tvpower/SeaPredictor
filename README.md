# SeaPredictor (OceanWatch)

Hybrid CNN + LSTM model that fuses Sentinel-2 imagery with NOAA OSCAR/HYCOM ocean
physics to predict marine debris accumulation zones. See
`garbage_patch_predictor_overview.md` for the full project overview.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Smoke-test the training loop with synthetic data (no downloads needed)
python -m src.training.train --synthetic --epochs 2 --batch-size 4
```

## Layout

```
src/
  dataset/    # Per-source loaders + master DebrisDataset
  models/     # CNN / LSTM / Fusion modules
  training/   # config, train.py, evaluate.py
  utils/      # FDI, geo helpers, TorchScript export
data/         # Download scripts; raw/ is gitignored
inference/    # C++ LibTorch server (later)
agent/        # Human Delta integration (later)
```

## Build order

1. Data download scripts (`data/download_*.{sh,py}`)
2. Real loaders inside `src/dataset/` (currently stubbed; synthetic mode works today)
3. Train CNN-only baseline → add LSTM → fusion
4. Export to TorchScript via `src/utils/export.py`

## Training entry points

```bash
# Full training (requires real data under data/raw/)
python -m src.training.train --epochs 30 --batch-size 16

# Eval a checkpoint
python -m src.training.evaluate --ckpt checkpoints/best.pt

# Export TorchScript
python -m src.utils.export --ckpt checkpoints/best.pt --out inference/debris_predictor.pt
```
