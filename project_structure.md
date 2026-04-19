# Project Structure

Actual on-disk layout as of the current build. Each path below is a real file or directory in the repo.

```
SeaPredictor/
│
├── README.md                            # entry point + quick start
├── garbage_patch_predictor_overview.md  # product pitch, shipped vs planned
├── Updated_process.md                   # end-to-end technical walkthrough
├── project_structure.md                 # this file
│
├── requirements.txt                     # torch, opendrift, fastapi, cesium clients, ...
├── .env.example                         # (unused currently)
├── .gitignore                           # excludes checkpoints/, data/raw/, web/scenes/, ...
│
├── checkpoints/                         # gitignored — trained model artifacts
│   ├── cnn_only_v2/
│   │   ├── best.pt                      # best-F1 checkpoint
│   │   ├── thresholds.json              # per-class tuned decision thresholds
│   │   └── test_report.json             # locked test-set metrics
│   └── fused/                           # legacy CNN+LSTM (regressed, not used)
│
├── data/                                # gitignored except for norm_stats.json + class_weights.json
│   ├── data/raw/MARIDA/
│   │   ├── patches/<scene>/*.tif        # 11-band Sentinel-2 + _cl.tif + _conf.tif
│   │   ├── splits/{train,val,test}_X.txt
│   │   ├── tile_index.csv               # tile_id -> (date, lat, lon, scene, path)
│   │   ├── labels_mapping.txt           # tile -> 15-elem multi-label vector
│   │   ├── norm_stats.json              # per-band mean/std (11 bands + 8 derived indices)
│   │   └── class_weights.json           # for the 11-class collapsed scheme (unused)
│   ├── data/raw/oscar/*.nc              # 231 daily OSCAR NetCDFs (2020-02-06..2021-01-23)
│   ├── forecast/                        # cached concat OSCAR files (auto-populated)
│   ├── download_oscar.py                # OSCAR grabber (NASA Earthdata)
│   └── test.py                          # MARIDA tile_index.csv generator
│
├── predictions/                         # gitignored — ad-hoc detector outputs
│   └── honduras_sep18.json              # etc.
│
├── forecast/                            # gitignored — ad-hoc forecast outputs
│   ├── honduras_sep18.nc                # OpenDrift trajectory NetCDF
│   ├── honduras_sep18.paths.geojson     # per-particle LineStrings
│   ├── honduras_sep18.final.geojson     # final-position Points
│   └── honduras_sep18.validation.json   # Tier 1 + Tier 2 metrics
│
├── web/                                 # gitignored — demo artifacts + frontend
│   ├── app/
│   │   └── index.html                   # CesiumJS 3D globe UI (single file, no build step)
│   ├── scenes/                          # built by src.pipeline.build_scenes
│   │   ├── index.json                   # scene manifest (served at /api/scenes)
│   │   └── <scene_id>/
│   │       ├── predictions.json         # raw detector output
│   │       ├── detections.geojson       # map-ready polygons
│   │       └── meta.json                # centroid, bbox, date, counts
│   └── forecast_cache/                  # populated by POST /api/forecast
│       └── <hash16>/
│           ├── run.nc                   # OpenDrift trajectory NetCDF
│           ├── paths.geojson
│           ├── final.geojson
│           ├── params.json              # echo of the request body
│           └── stats.json               # n_particles, elapsed_s, ...
│
└── src/
    │
    ├── dataset/
    │   ├── marida_loader.py             # MaridaIndex, TileRecord, default_marida_root
    │   ├── marida_dataset.py            # raw MARIDA per-tile Dataset (returns masks too)
    │   ├── debris_dataset.py            # master multi-label Dataset (image, seq, label)
    │   ├── oscar_loader.py              # OSCARLoader — daily files, 0..360 lon fix, julian cal
    │   ├── normalization.py             # normalize_bands()
    │   ├── spectral_indices.py          # FDI / NDVI / NDWI stack (unused in production model)
    │   └── augmentation.py              # flip/rotate for mask-aware segmentation training
    │
    ├── models/
    │   ├── cnn_encoder.py               # ResNet-18 with 11-band first-conv adapter
    │   ├── lstm_encoder.py              # 2-layer LSTM (present but unused in production)
    │   └── fusion_model.py              # DebrisPredictor: CNN [+ LSTM] -> 15 multi-label head
    │
    ├── training/
    │   ├── config.py                    # TrainConfig dataclass (dropout, weight_decay, ...)
    │   ├── train.py                     # main loop + early stopping + auto pos_weight
    │   ├── evaluate.py                  # val-loop with MPS-safe label handling
    │   ├── tune_thresholds.py           # per-class F1-max threshold sweep on val
    │   └── eval_test.py                 # final test-set evaluation with tuned thresholds
    │
    ├── inference/
    │   ├── predict.py                   # run detector on a tile dir or full scene
    │   └── export.py                    # TorchScript jit.trace + sanity-check + .meta.json
    │
    ├── forecast/
    │   ├── seed.py                      # predictions.json -> list[Seed(lat, lon, date, ...)]
    │   ├── oscar_concat.py              # build CF-compliant concat NetCDF for OpenDrift
    │   ├── drift.py                     # OpenDrift orchestrator, writes .nc + 2 GeoJSONs
    │   └── validate.py                  # Tier 1 plausibility + Tier 2 cross-detection hit rate
    │
    ├── pipeline/
    │   └── build_scenes.py              # offline cache builder consumed by the web backend
    │
    ├── api/
    │   └── server.py                    # FastAPI app: /api/scenes, /api/forecast, + static mount
    │
    └── utils/
        ├── export.py                    # (duplicate of src/inference/export.py; pre-refactor)
        └── preview_tile.py              # RGB + FDI + GT mask side-by-side viz
```

---

## Module responsibilities at a glance

| Package | Responsibility | Dependencies |
|---|---|---|
| `src.dataset` | Read MARIDA tiles + OSCAR sequences off disk, normalize, split | rasterio, xarray |
| `src.models` | Network definitions (CNN, optional LSTM, fused head) | torch, torchvision |
| `src.training` | Training loop, metrics, threshold tuning, test eval | torchmetrics |
| `src.inference` | Run a trained checkpoint on new tiles; export TorchScript | torch |
| `src.forecast` | Turn predictions into drift seeds, simulate, validate | opendrift, pyproj |
| `src.pipeline` | Offline batch processing that produces the web scene cache | — (composes the above) |
| `src.api` | FastAPI backend + static frontend mount | fastapi, uvicorn, pydantic |
| `web/app` | CesiumJS single-page demo UI (no build step) | — (CDN only) |

---

## Data-flow recap

```
MARIDA .tif  ─► src.inference.predict ─► predictions.json
                                               │
                                               ▼
                                  src.forecast.seed (filter + reproject)
                                               │
OSCAR .nc ─► src.forecast.oscar_concat ───────►│ src.forecast.drift
                                               │     (OpenDrift integration)
                                               ▼
                              paths.geojson · final.geojson · run.nc
                                               │
                    Scene B predictions ──────►│ src.forecast.validate
                                               ▼
                                     tier1_*.json · tier2_*.json
```

The same chain runs twice in different orchestration harnesses:

- **CLI scripts** (for one-off analysis): `predict.py` → `drift.py` → `validate.py`.
- **Web pipeline**: `src.pipeline.build_scenes` pre-caches all detections offline; `src.api.server` runs OpenDrift on demand per user request and caches by parameter hash.

---

## Gitignored outputs

Everything listed under `forecast/`, `predictions/`, `web/scenes/`, `web/forecast_cache/`, `data/raw/`, `data/data/`, `data/forecast/`, and `checkpoints/` is gitignored. These are generated artifacts — reproducible from the MARIDA raw data + OSCAR files + a trained checkpoint.

What IS committed:

- Source code under `src/`
- Frontend single-file under `web/app/index.html`
- `data/data/raw/MARIDA/norm_stats.json` and `class_weights.json` (tiny, dataset-specific)
- `requirements.txt`, `.gitignore`, docs
