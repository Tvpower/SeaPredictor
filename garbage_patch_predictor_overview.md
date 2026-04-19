# SeaPredictor: Marine Debris Detection + Drift Forecasting

## Project Overview

SeaPredictor is a hackathon-scale system that answers two questions about marine debris:

1. **"Is debris visible in this satellite image?"** — answered by a ResNet-18 CNN trained on MARIDA-labeled Sentinel-2 tiles.
2. **"Where will that debris drift over the next 1–30 days?"** — answered by OpenDrift, a physics-based Lagrangian particle simulator forced by NOAA OSCAR ocean currents.

These two stages are glued together by a single JSON file of detections. A FastAPI backend serves pre-cached detection results and runs the drift simulator on demand; a CesiumJS single-page app provides an interactive 3D globe where the user picks a scene, tweaks forecast parameters, and sees the predicted particle cloud.

This document describes **what actually shipped**. Sections flagged as *deferred* describe items that were in the original plan but were intentionally scoped out for the hackathon MVP.

---

## The Problem

Floating marine debris is dispersed rather than island-sized. Even in high-concentration zones it typically occupies < 0.1 % of a given Sentinel-2 pixel. Identifying it manually at satellite scale is infeasible; forecasting where drifting debris will concentrate is currently dominated by physical oceanographic modeling with heavy human intervention.

The goal is to combine automated detection (machine learning) with physically grounded forecasting (Lagrangian simulation) so that an operator can point at a satellite scene and ask "where does this go next?" without needing to run either a full segmentation model or a custom integrator.

---

## System Overview

```
┌──────────────────────────┐        ┌──────────────────────────┐
│  STAGE 1 — DETECTION     │        │  STAGE 2 — FORECAST      │
│  What is in the image?   │  ───►  │  Where will it drift?    │
│                          │        │                          │
│  ResNet-18 CNN           │        │  OpenDrift OceanDrift    │
│  (MARIDA-trained, 15     │        │  (Lagrangian Monte Carlo │
│   multi-label classes)   │        │   on OSCAR u/v fields)   │
└──────────────────────────┘        └──────────────────────────┘
         ▲                                       ▲
  Sentinel-2 11-band tiles               NOAA OSCAR daily NetCDFs
         │                                       │
         └───────────── predictions.json ────────┘
                              │
                              ▼
                      Web backend (FastAPI)
                              │
                              ▼
              CesiumJS 3D globe frontend
```

**Key architectural insight**: Stage 1 is pure machine learning; Stage 2 is pure numerical integration. They share no parameters and no state. The only contract is a JSON list of `(tile_id, predicted_classes, geo_bounds)` records. This separation is what makes each stage independently testable and what lets the web UI cache detections offline and only invoke the expensive physics solver on demand.

---

## Stage 1 — Detection

### Data

Training uses **MARIDA (Marine Debris Archive)** — 1,381 Sentinel-2 tiles with pixel-level multi-label annotations across 15 classes (Marine Debris, Dense/Sparse Sargassum, Ship, Clouds, various water types, Foam, etc.). Tiles are 256×256 at 10 m spatial resolution, 11 spectral bands.

Official splits give 694 train / 328 val / 359 test tiles. We use the MARIDA-provided per-tile multi-label vector (one 15-dim 0/1 target per tile) rather than the per-pixel mask; this reduces the problem to multi-label classification, which is what the current model head produces.

*Deferred*: MADOS (174 additional scenes) and NASA-IMPACT (PlanetScope cross-sensor) were planned as augmentation and cross-validation datasets. Not integrated in the MVP.

### Architecture

```
Sentinel-2 tile (11, 256, 256)
        │
        ▼
ResNet-18 encoder
  - First conv adapted from 3-band ImageNet pretrain
    to 11-band by tiling+rescaling pretrained RGB filters
  - Global-avg-pool output: 512-dim feature vector
        │
        ▼
Multi-label head
  - Linear(512 -> 256) -> ReLU -> Dropout(0.5) -> Linear(256 -> 15)
        │
        ▼
15 logits  →  Sigmoid  →  15 per-class probabilities
        │
        ▼
Per-class thresholds (from thresholds.json)
        │
        ▼
Binary predictions (one per class)
```

*Deferred*: A parallel LSTM branch processing 30-day OSCAR ocean-current sequences was implemented and trained (`src/models/lstm_encoder.py`, `--use-temporal` flag). It regressed macro-F1 by 2 points because OSCAR coverage of MARIDA's training scenes was only 26 %. We keep the code for future use but run `--cnn-only` in production.

### Training

| Hyperparameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` with per-class `pos_weight` auto-derived from train-set positive rates, clipped to 50 |
| Optimizer | AdamW, lr = 1e-4, weight_decay = 0.01 |
| Scheduler | Cosine annealing |
| Batch size | 16 |
| Epochs | 25 with early stopping (patience = 6) |
| Head dropout | 0.5 |
| Device | Apple MPS (CUDA-ready; CPU fallback for unsupported ops) |

MPS-specific hardening is in place: `non_blocking=True` removed from tensor transfers (caused `int32` cast races on MPS), labels kept on CPU for metric updates, `PYTORCH_ENABLE_MPS_FALLBACK=1` set automatically.

### Evaluation

Three phases:

1. **Per-epoch validation** — macro-accuracy, macro-precision/recall/F1.
2. **Threshold tuning** (`src.training.tune_thresholds`) — per-class sweep of thresholds 0.05–0.95 on val, picking the value that maximizes per-class F1. Gives us a `thresholds.json` that replaces the default 0.5 at inference time.
3. **Locked-in test** (`src.training.eval_test`) — single run on the held-out test split with the tuned thresholds.

**Headline test-set numbers** (CNN-only, `checkpoints/cnn_only_v2`):

| Metric | Value |
|---|---|
| Macro-F1 @ default 0.5 | 0.59 |
| Macro-F1 @ tuned thresholds | 0.60 |
| Marine Debris (class 0) F1 | 0.85 |
| Marine Water (class 6) F1 | 0.85 |
| Sparse Sargassum F1 | 0.67 |
| Dense Sargassum F1 | 0.48 |
| Foam F1 | 0.69 |

The strong performers are the high-prevalence classes with distinct spectral signatures (water, debris, sargassum). The weak performers (Natural Organic Material F1 = 0.20, Cloud Shadows = 0.42) have few positive examples in training.

---

## Stage 2 — Forecasting

### Data

NOAA OSCAR (Ocean Surface Current Analysis Real-time) provides daily surface current vectors (u, v in m/s) on a 0.25° grid. We have 231 daily files covering **2020-02-06 through 2021-01-23**. 23 of MARIDA's 63 scenes fall inside this window and are demo-able end-to-end; the remaining 40 can be detected but not forecasted without downloading more OSCAR data.

OSCAR's file format requires preprocessing before OpenDrift can consume it:

- Time coord is `cftime.DatetimeJulian` (not numpy-compatible)
- Longitude runs 0–360 (MARIDA uses ±180)
- Dimension order is `(time, lon, lat)` (OpenDrift expects `(time, lat, lon)`)

`src/forecast/oscar_concat.py` builds a CF-compliant concatenated NetCDF that fixes all three, caching the result on disk so the next simulation over an overlapping window is instant.

*Deferred*: Wind forcing (GFS or ERA5) would add Stokes drift and windage. Currents-only forecasts systematically underestimate motion by roughly the wind contribution (30–50 % for floating plastic in windy regions).

### Pipeline

```
predictions.json
     │
     ▼
┌──────────────────────────────┐
│ src.forecast.seed            │
│  - keep tiles with debris    │
│    class flags == 1          │
│  - compute WGS84 centroid    │
│    from UTM tile bounds      │
│  - resolve obs_date per tile │
│    (tile_index.csv lookup)   │
└──────────────────────────────┘
     │
     ▼  list[Seed(tile_id, lat, lon, obs_date, classes, prob)]
     │
┌──────────────────────────────┐
│ src.forecast.drift           │
│  - build/read OSCAR concat    │
│  - OceanDrift model setup    │
│    (diffusivity 10 m²/s,      │
│     coastline: freeze)       │
│  - Monte Carlo seed:          │
│    N particles around each   │
│    detection, Gaussian-     │
│    distributed in radius R   │
│  - dt = 30 min, duration = D │
│  - integrate (u, v) field    │
└──────────────────────────────┘
     │
     ▼
Three outputs:
  - run.nc               OpenDrift native trajectory file (truth source)
  - paths.geojson        per-particle LineStrings (map overlay)
  - final.geojson        final-position Points (heatmap input)
```

The simulator is currents-only, 30-minute internal timestep, hourly output. Default demo params: 100 particles per seed, 1000 m seed radius, 7-day horizon.

### Validation

`src/forecast/validate.py` computes two tiers:

**Tier 1 — Plausibility** (from the trajectory alone, no external data):

- Mean / median / max displacement over the run
- Mean drift bearing (circular mean)
- Mean drift speed
- Beached fraction (particles frozen against coastline)
- Diffusion growth exponent (how particle cloud spreads over time)

**Tier 2 — Temporal cross-validation** (requires a later satellite detection of the same region):

- Forecast at time T+N is compared to a *fresh* detector run on a second Sentinel-2 scene from day T+N
- **Hit rate**: percentage of forecast particles within 5 km of any observed debris in scene B
- Centroid drift error (km)
- Density IoU

**Honduras demo result** (seed scene 2020-09-18, evaluation scene 2020-09-23, 5 days forward):

| Tier 2 metric | Value |
|---|---|
| Hit rate @ 5 km | 72 % |
| Centroid drift error | 7.4 km |
| Density IoU | 0.31 |

Translation: 5 days into the future, our physics-grounded forecast put 72 % of its particles within 5 km of where the detector independently spotted debris. That's Tier 2 evidence that the end-to-end system works, not just each stage in isolation.

---

## The Web Application

The backend and frontend were not in the original plan — they were added because a demo benefits more from an interactive globe than from a CLI.

### Backend (`src/api/server.py`)

A FastAPI app exposing these routes, all under `/api`:

| Endpoint | Purpose |
|---|---|
| `GET /scenes` | Manifest of every cached scene (centroid, date, detection counts) |
| `GET /scenes/{id}` | One scene's metadata |
| `GET /scenes/{id}/detections` | Map-ready GeoJSON polygons of debris-positive tiles |
| `GET /scenes/{id}/predictions` | Full predictions.json for inspection |
| `POST /forecast` | Run OpenDrift with custom params; cached by parameter hash |
| `GET /forecast/{key}/paths` | Per-particle trajectory GeoJSON |
| `GET /forecast/{key}/final` | Final-position Point GeoJSON |

Forecast requests are serialized behind a module-level `threading.Lock` (OpenDrift is not thread-safe). Each unique parameter combination hashes to a stable 16-character cache key — identical requests return instantly after the first run.

### Frontend (`web/app/index.html`)

Single HTML file, no build step, CesiumJS loaded from jsDelivr CDN, OpenStreetMap tiles as the imagery provider (no Ion token required).

User experience:

1. Page loads the 3D globe. On first render, GET `/api/scenes` returns the manifest; the app drops a red marker at each scene centroid, sized by detection count.
2. User clicks a marker (on the globe) or an entry in the sidebar list. Camera flies to the scene's bounding box; detection polygons load as a red overlay.
3. A forecast form appears in the sidebar: days, particles per detection, seed radius, diffusivity, and class checkboxes.
4. "Run Forecast" POSTs the params. For ~30 seconds a spinner shows; then the response arrives with URLs to the path and final-position GeoJSONs. The frontend fetches and renders them: blue translucent lines for trajectories, orange dots for final positions.
5. A stats panel displays particle count, compute time, wall time, and whether the result was served from cache.

### Pre-cache pipeline (`src/pipeline/build_scenes.py`)

Runs offline, typically once after training. For every MARIDA scene inside the OSCAR window:

- Runs the detector on every tile in `data/data/raw/MARIDA/patches/<scene>/`
- Writes `web/scenes/<scene_id>/{predictions,detections,meta}.json`
- Builds a top-level `web/scenes/index.json` manifest

CPU runtime: ~5–15 minutes for all 23 OSCAR-covered scenes. The API serves these files statically.

---

## Repository Structure

See `project_structure.md` for the full tree. Short version:

```
src/
  dataset/     MARIDA + OSCAR loaders, augmentation, normalization
  models/      CNN encoder, (unused) LSTM encoder, fused head
  training/    train.py, evaluate.py, tune_thresholds.py, eval_test.py
  inference/   predict.py (run the model on new data), export.py (TorchScript)
  forecast/    seed.py, oscar_concat.py, drift.py, validate.py
  pipeline/    build_scenes.py (offline web cache builder)
  api/         server.py (FastAPI backend + static mount)
  utils/       preview_tile.py (RGB/FDI/mask visualization)
web/
  app/         index.html (CesiumJS single-page demo)
  scenes/      pre-cached detection artifacts (gitignored)
  forecast_cache/  lazily populated per request (gitignored)
```

---

## Build Phases (reality check)

| Phase | Original plan | Actual outcome |
|---|---|---|
| 1. Data pipeline | MARIDA + MADOS + OSCAR + HYCOM dataset | MARIDA + OSCAR only; MADOS/HYCOM deferred |
| 2. CNN baseline | ResNet-18 on MARIDA | ✅ F1 ≈ 0.60 macro, 0.85 Marine Debris |
| 3. Hybrid CNN+LSTM | OSCAR-as-feature for detection | Built, **regressed F1**, deprecated to `--cnn-only` |
| 4. TorchScript export | For C++ LibTorch server | Exports correctly; C++ server not built |
| 5. Human Delta agent | Scientific-source enrichment | Deferred |
| 6. Frontend | Leaflet 2D map | **Upgraded to CesiumJS 3D globe** |
| (new) Threshold tuning | Not in plan | Added; +1 test macro-F1, better recall |
| (new) Test-set locked eval | Not in plan | Added; defensible benchmark numbers |
| (new) OpenDrift forecasting | Vague "phase 2" | ✅ Shipped with Tier 1 + Tier 2 validation |
| (new) Web backend + cache | Not in plan | ✅ FastAPI + param-hash cache |

---

## Significant Constraints

These are real limitations, not roadmap items — flag them to anyone evaluating the system:

1. **OSCAR temporal coverage**: 2020-02-06 → 2021-01-23. Only 23/63 MARIDA scenes are forecastable. Expanding requires more OSCAR downloads or switching to a real-time source like HYCOM / Copernicus Marine.

2. **Currents-only physics**: no wind, no Stokes drift. Forecasts will underestimate motion in windy regions by ~30 %. Adding a GFS/ERA5 reader is ~half a day of work but not done.

3. **Cloud false positives**: thin clouds share spectral behavior with some debris classes. Our highest-confidence false positives in the test set are clouds misclassified as Marine Debris. Future work: add S2 QA60 or Fmask cloud masking as preprocessing.

4. **Sub-pixel debris**: Sentinel-2's 10 m resolution means individual plastic items are invisible; the model is really learning to detect spectrally anomalous water, not physical objects. Claims about debris are always statistical.

5. **Geographic generalization**: MARIDA is biased toward tropical/subtropical scenes (Caribbean, Mediterranean, Indonesia, Indian Ocean). Performance in Arctic, Antarctic, or temperate North Atlantic waters is unknown.

6. **Single-user backend**: forecast runs serialize behind a `threading.Lock`. Multi-user concurrent usage would need a Celery queue.

7. **No live satellite ingestion**: the system cannot forecast "today's debris." All scenes come from MARIDA's 2019–2021 snapshot. Adding Copernicus Hub integration is a multi-day task.

---

## References

- **MARIDA dataset** — https://zenodo.org/record/5151941
- **NOAA OSCAR currents** — https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_third-deg
- **OpenDrift** — https://opendrift.github.io/
- **Sentinel-2 Floating Debris Index (FDI)** — Biermann et al., *Sci Rep* 2020
- **Sentinel-2 Copernicus Browser** — https://browser.dataspace.copernicus.eu
- **CesiumJS** — https://cesium.com/platform/cesiumjs/
