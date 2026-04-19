# SeaPredictor

**Detect marine debris in Sentinel-2 satellite imagery and forecast where it will drift, end-to-end.**

SeaPredictor is a two-stage system built for the FullyHacks hackathon:

1. **Detection** — a ResNet-18 classifier trained on MARIDA tells you which 256×256 m ocean tiles contain debris (and what class: plastic, sargassum, foam, etc.).
2. **Forecasting** — OpenDrift takes those detections and propagates particles forward through NOAA OSCAR ocean currents to show where the debris is likely to go over the next 1–30 days.

A web app wraps it in a 3D globe (CesiumJS) with a scene picker and an on-demand forecast form. Forecast runs are cached by parameter hash so repeat demos are instant.

---

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data layout (already on disk if you've been using this repo)

```
data/data/raw/MARIDA/patches/<scene>/*.tif     # Sentinel-2 patches + masks
data/data/raw/MARIDA/splits/{train,val,test}_X.txt
data/data/raw/MARIDA/tile_index.csv            # maps tile -> date, lat, lon
data/data/raw/MARIDA/norm_stats.json           # per-band mean/std
data/data/raw/oscar/oscar_currents_interim_YYYYMMDD.nc   # 231 daily files
```

OSCAR coverage window: **2020-02-06 → 2021-01-23** (231 days). MARIDA scenes outside this window can be detected but not forecasted until more OSCAR data is downloaded.

### 3. Train (already done — you can skip if you have a checkpoint)

```bash
# CNN-only (the production config — OSCAR-as-feature regressed, see notes).
python -m src.training.train --cnn-only \
  --epochs 25 --batch-size 16 \
  --head-dropout 0.5 --weight-decay 0.01 --early-stopping-patience 6 \
  --ckpt-dir checkpoints/cnn_only_v2

# Per-class threshold tuning on the val split (+4 macro-F1 points).
python -m src.training.tune_thresholds --ckpt checkpoints/cnn_only_v2/best.pt

# Lock in numbers on the held-out test split.
python -m src.training.eval_test \
  --ckpt checkpoints/cnn_only_v2/best.pt \
  --thresholds checkpoints/cnn_only_v2/thresholds.json \
  --report checkpoints/cnn_only_v2/test_report.json
```

Test-set macro-F1 with tuned thresholds: **0.60** (Marine Debris alone: F1=0.85).

### 4. Build the scene cache (one-time, ~5–15 min on CPU)

```bash
python -m src.pipeline.build_scenes \
  --ckpt checkpoints/cnn_only_v2/best.pt \
  --thresholds checkpoints/cnn_only_v2/thresholds.json \
  --out web/scenes
```

This runs the detector on every MARIDA scene that falls inside the OSCAR window and writes a web-friendly cache:

```
web/scenes/
  index.json                           # manifest (centroid, date, count per scene)
  <scene_id>/
    predictions.json                   # raw detector output
    detections.geojson                 # map-ready polygons (WGS84)
    meta.json                          # summary stats
```

### 5. Boot the API + frontend

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` → 3D globe loads with red markers at every scene hotspot. Click a marker, tweak the forecast params, hit **Run Forecast**. First run takes 20–60 s (OpenDrift); second run of the same params is instant (cache hit).

---

## Architecture snapshot

```
┌──────────────────────────┐        ┌──────────────────────────┐
│  STAGE 1: DETECTION       │        │  STAGE 2: FORECAST        │
│  "What's in this image?"  │  ───►  │  "Where will it drift?"   │
│                           │        │                           │
│  ResNet-18 CNN            │        │  OpenDrift OceanDrift     │
│  (MARIDA-trained, 15 cls) │        │  (Lagrangian physics      │
│                           │        │   on OSCAR u/v fields)    │
└──────────────────────────┘        └──────────────────────────┘
         ▲                                       ▲
  Sentinel-2 11-band tiles               NOAA OSCAR daily NetCDFs
  (via src.inference.predict)            (via src.forecast.oscar_concat)
                  │                                   │
                  └──── predictions.json ─────┐       │
                                              ▼       ▼
                            ┌──────────────────────────────┐
                            │  src.forecast.seed           │
                            │  filter + reproject + dates  │
                            └──────────────────────────────┘
                                          │
                                          ▼
                            ┌──────────────────────────────┐
                            │  src.forecast.drift          │
                            │  Monte Carlo particle sim    │
                            └──────────────────────────────┘
                                          │
                                          ▼
                   paths.geojson · final.geojson · trajectory.nc
```

Two things to internalize:

- **Detection is pure ML, forecasting is pure physics.** They communicate through one JSON file.
- **OSCAR is only used by the forecaster.** The CNN doesn't read currents (the original LSTM branch regressed during training, so `--cnn-only` is the production config).

See `Updated_process.md` for a step-by-step walkthrough with data-flow diagrams and `garbage_patch_predictor_overview.md` for the pitch.

---

## What ships vs. what was originally planned

| Component | Original plan | Status |
|---|---|---|
| ResNet-18 CNN detector | ✅ | Shipped |
| Per-class threshold tuning | Added late | Shipped |
| Test-set evaluation + report | Added late | Shipped |
| TorchScript export | ✅ | Shipped (unused; Python inference is fast enough) |
| OpenDrift forecasting | Phase 2 | Shipped |
| Tier 1 + Tier 2 validation | Added late | Shipped |
| Scene pre-cache pipeline | Not in plan | Shipped |
| FastAPI backend | Not in plan | Shipped |
| CesiumJS 3D globe UI | Not in plan | Shipped |
| CNN + LSTM fusion (OSCAR as feature) | ✅ | Implemented, **regressed F1 → deprecated** |
| HYCOM SST/salinity | Phase 2 | Deferred |
| 15 → 11 class collapse | Planned | Deferred |
| C++ LibTorch server | Phase 4 | Deferred (Python is fine) |
| Human Delta enrichment | Phase 5 | Deferred |
| MADOS / NASA-IMPACT datasets | Planned | Not integrated |
| U-Net per-pixel segmentation | Discussed | Skipped for MVP |
| Time-animated drift playback | Nice-to-have | Deferred (paths are static LineStrings) |

---

## Training on Apple Silicon (MPS / Metal)

Auto-detected — no flags needed. The training stack was hardened for MPS:

- `default_device()` returns `"mps"` when available.
- `num_workers=0` on MPS (forking DataLoader workers crashes macOS).
- `pin_memory` disabled on MPS (CUDA-only).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set so unimplemented ops fall back to CPU.
- `non_blocking=True` removed from tensor transfers — it's a no-op on MPS and causes a known race with `torch.int32` casts that produces garbage label values.
- Labels are kept on CPU for metrics; only the model-forward copy goes to MPS.

Keep batch size modest (16 on M1/M2 8 GB, 16–32 on 16 GB+).

---

## Entry points cheat sheet

| Command | What it does |
|---|---|
| `python -m src.training.train --cnn-only ...` | Train the detector |
| `python -m src.training.tune_thresholds --ckpt ...` | Per-class F1-max thresholds |
| `python -m src.training.eval_test --ckpt ... --thresholds ...` | Locked-in test metrics |
| `python -m src.inference.predict --ckpt ... --tiles <dir>` | Detect on a single scene or tile dir |
| `python -m src.inference.export --ckpt ...` | Export TorchScript `.ts.pt` |
| `python -m src.forecast.drift --predictions ... --days N` | Run OpenDrift forecast |
| `python -m src.forecast.validate --traj ... [--scene-b ...]` | Tier 1 + Tier 2 validation |
| `python -m src.pipeline.build_scenes --ckpt ... --out web/scenes` | Build the web cache |
| `uvicorn src.api.server:app` | Boot the demo backend + frontend |

---

## Known limitations

1. **OSCAR window**: 2020-02-06 → 2021-01-23 only. 23 of MARIDA's 63 scenes are inside this window.
2. **Currents-only forecast**: no wind / Stokes drift. Adds a systematic bias in windy regions.
3. **Cloud false positives**: FDI lights up for thin clouds. Current model handles OK but it's the dominant failure mode.
4. **Sub-pixel debris**: even our highest-scoring tiles have <0.1 % debris pixels — Sentinel-2 just doesn't resolve individual items.
5. **OpenDrift isn't thread-safe**: the API serializes forecast runs behind a lock (fine for single-user demo).
6. **First forecast is slow**: 20–60 s for typical params. Pre-warm the cache before demoing.

See `garbage_patch_predictor_overview.md` for fix paths.
