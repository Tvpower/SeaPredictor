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

### Training on Apple Silicon (MPS / Metal)

The training stack auto-detects MPS on Apple Silicon Macs — no flags required.
What's wired up for you:

- `default_device()` returns `"mps"` when available.
- `num_workers` defaults to `0` on MPS (PyTorch DataLoader workers fork badly on
  macOS and routinely crash the run).
- `pin_memory` is disabled on MPS (only useful for CUDA's pinned host transfers).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so any op Metal hasn't
  implemented yet falls back to CPU instead of erroring.
- Metrics (torchmetrics) are computed on CPU, since Metal coverage there is uneven.
- `torch.mps.manual_seed` is called alongside `torch.manual_seed` for reproducibility.

```bash
# Smoke test on MPS (auto-detected)
python -m src.training.train --synthetic --epochs 2 --batch-size 8

# Force CPU if you need to debug
python -m src.training.train --synthetic --device cpu

# Real training on M-series with full pretrained ResNet-18
python -m src.training.train --epochs 30 --batch-size 16
```

Notes:

- Keep batch size modest (8–16 on M1/M2 8 GB, 16–32 on 16 GB+). Activations for
  256×256 ResNet-18 + a 30-step LSTM add up.
- Mixed precision (`autocast`) on MPS is still flaky in PyTorch 2.x; the loop
  runs in fp32. If you want to experiment, wrap the forward in
  `torch.autocast(device_type="mps", dtype=torch.float16)` — expect occasional
  numerical issues with `BCEWithLogitsLoss` + `pos_weight`.
- First run downloads the pretrained ResNet-18 weights to `~/.cache/torch/hub`.

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
