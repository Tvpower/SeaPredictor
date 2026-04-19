# The Full Pipeline — End to End

A step-by-step walkthrough of how SeaPredictor goes from raw Sentinel-2 pixels to a 3D globe showing where debris is headed. Read this after `README.md` if you want to understand the code, not just run it.

---

## The one-line summary

> Use a CNN to tell you **where to start** a physics simulation. Let the physics do the time-travel.

The machine-learning part is a seeing machine that looks at pixels and says "debris." It doesn't understand currents, wind, or time. OpenDrift is a time-machine for particles. Given a starting point and a velocity field, it projects forward — it doesn't understand what debris is. The combined system is stronger than either alone.

---

## System overview

```
┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
│  STAGE 1: DETECTOR  │        │  STAGE 2: FORECAST  │        │  STAGE 3: WEB UI    │
│  "Is debris here    │   ───► │  "Where will it go  │   ───► │  "Show me + let me  │
│   NOW?"             │        │   over N days?"     │        │   replay any param" │
│                     │        │                     │        │                     │
│  ResNet-18 CNN      │        │  OpenDrift physics  │        │  FastAPI + Cesium   │
│  (MARIDA-trained)   │        │  simulator          │        │  3D globe           │
└─────────────────────┘        └─────────────────────┘        └─────────────────────┘
         ▲                              ▲                              ▲
  Sentinel-2 imagery            OSCAR ocean currents           User clicks + forecast
  (11-band, 256x256 tiles)      (NOAA daily NetCDFs)           form in browser
```

Three important things to internalize:

1. **Stage 1 is pure ML. Stage 2 is pure physics.** They share zero parameters, zero state. The only thing connecting them is a JSON file (`predictions.json`).
2. **OSCAR is only used by Stage 2.** The CNN doesn't read currents (the LSTM branch regressed during training; we deprecated it).
3. **Stage 3 exists to demo Stages 1 and 2.** It pre-caches detection results offline and runs the physics simulator on demand, caching each unique parameter combination.

---

## Stage 1 — What the trained model does

**Role**: Look at a satellite image, decide what's in it. That's the entire job.

**Input**: a 256×256 pixel tile of 11-band Sentinel-2 imagery (B1 through B12, skipping B9/B10 which aren't useful for water).

**Model**: a ResNet-18 CNN adapted to 11 input channels (via averaging-and-tiling the pretrained RGB first-conv weights). The head produces 15 logits, one per MARIDA class: Marine Debris, Dense Sargassum, Sparse Sargassum, Natural Organic Material, Ship, Clouds, Marine Water, Sediment-Laden Water, Foam, Turbid Water, Shallow Water, Waves, Cloud Shadows, Wakes, Mixed Water. A sigmoid gives a probability per class; a per-class threshold from `thresholds.json` converts those to 0/1 predictions.

**Training**: 694 tiles from MARIDA with multi-label annotations, `BCEWithLogitsLoss` with per-class `pos_weight` auto-derived from positive rates, AdamW + cosine LR, 25 epochs with early stopping (patience 6), head dropout 0.5. Test-set macro-F1 with tuned thresholds ≈ 0.60 (Marine Debris F1 = 0.85).

**Output**: one JSON record per tile:

```json
{
  "tile_id": "S2_18-9-20_16PCC_0",
  "probs":  [0.87, 0.02, ..., 0.91, ...],   // 15 values, one per class
  "preds":  [1, 0, ..., 1, ...],             // class 0 (Marine Debris) = YES
  "predicted_classes": [0, 6],
  "geo": {
    "crs": "EPSG:32616",                     // UTM zone 16N in this example
    "bounds": [minx, miny, maxx, maxy]       // where on Earth this tile is
  }
}
```

Run it with:

```bash
python -m src.inference.predict \
  --ckpt checkpoints/cnn_only_v2/best.pt \
  --thresholds checkpoints/cnn_only_v2/thresholds.json \
  --tiles data/data/raw/MARIDA/patches/S2_18-9-20_16PCC \
  --out predictions/honduras_sep18.json
```

**Key insight**: Stage 1 does **no prediction of the future**. It's a pure "what does this image show right now" classifier. The model doesn't even know what a current is.

---

## Stage 2 — What OpenDrift does

**Role**: Given a list of locations where debris currently is, simulate where it will drift to over the next N days.

**Inputs**:

- **Seed points** — list of `(lat, lon, date)` where debris exists today (produced by Stage 1 + the handoff below).
- **Ocean current field** — OSCAR daily NetCDFs. At `(lat, lon, day)` the water is flowing at `(u, v)` m/s.

**What OpenDrift is**:

- A Lagrangian physics simulator, not an ML model
- Zero learned parameters
- Just numerical integration of `dx/dt = u(x, y, t)`, `dy/dt = v(x, y, t)`
- Adds random Gaussian kicks for eddy diffusion (sub-grid turbulence)
- Freezes particles that hit a coastline

**The physics in plain English**: take a tiny imaginary debris particle, ask OSCAR "what's the water doing at this exact lat/lon on this date?", move the particle a tiny step in that direction, advance the clock by 30 minutes, ask OSCAR again. Repeat for 7 days. Do that 100 times per seed (Monte Carlo) so you get an uncertainty spread instead of a single line.

**Key insight**: OpenDrift doesn't care about images, pixels, CNNs, or debris at all. It just moves points around in a velocity field. You could seed it with duck decoys or oil droplets — it'd run the same way.

---

## The handoff — how Stage 1 feeds Stage 2

This is where `src/forecast/seed.py` does the magic. It reads Stage 1's output JSON and does three things:

### 1. Filter to debris-only tiles

```python
# Keep only tiles where the model predicted one of the trackable classes.
# Defaults to class 0 (Marine Debris); configurable via --debris-classes.
debris_set = {0, 1, 2, 8}  # Marine Debris, Sargassum x2, Foam
if not any(preds[c] == 1 for c in debris_set):
    skip
```

### 2. Convert pixel-space tile bounds → real-world lat/lon

Stage 1 outputs bounds in UTM (the satellite's native projection). OpenDrift wants WGS84. `pyproj` reprojects the centroid:

```python
transformer = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
lon, lat = transformer.transform(cx, cy)
```

### 3. Resolve each tile's observation date

OpenDrift needs a start time for each particle. The script:

- Looks up the date from MARIDA's `tile_index.csv` (built once from scene filenames), OR
- Uses the `--default-date` CLI arg as a fallback, OR
- Is forced to a specific date with `--override-date` (useful when MARIDA scenes fall outside the OSCAR window and you're faking a demo date inside it)

### Output of this handoff

A list of `Seed` objects:

```python
Seed(tile_id="S2_18-9-20_16PCC_0", lat=16.09, lon=-88.32, obs_date=date(2020, 9, 18),
     matched_classes=(0,), max_prob=0.91)
```

Everything else from Stage 1 — the image, the neural network, the pixel-level data — is discarded. Only `(lat, lon, date)` survives into Stage 2.

---

## How OSCAR data is prepared

OpenDrift needs a CF-compliant NetCDF it can read via its generic reader. OSCAR's files are **not** CF-compliant out of the box:

| Issue | Fix in `src/forecast/oscar_concat.py` |
|---|---|
| Time coord is `cftime.DatetimeJulian`, not numpy-compatible | Re-encode to `seconds since 1970-01-01` |
| Longitude is 0..360, MARIDA uses ±180 | Roll longitude dimension to −180..180 |
| Dimension order is `(time, lon, lat)` | Transpose to `(time, lat, lon)` |
| Each day is a separate file | Concatenate into one NetCDF per simulation window |

The output is cached under `data/forecast/oscar_concat_YYYYMMDD_YYYYMMDD.nc`. Subsequent simulations covering the same window reuse the cached file.

**Important distinction**: OSCAR appears in two unrelated places in this system —

| Role | When | Used? |
|---|---|---|
| Feature for the detector (via LSTM) | Training the original CNN+LSTM model | ❌ **Deprecated.** OSCAR coverage of MARIDA is only 26 %; the LSTM regressed macro-F1. Production uses `--cnn-only`. |
| Velocity field for the simulator | Forecasting | ✅ The actual u/v currents that move particles. |

In the current MVP, OSCAR only matters for Stage 2. The detector doesn't touch it.

---

## End-to-end trace: the Honduras demo

What actually happened on your laptop when you ran the canonical demo:

### Step 1 — `predict.py` on `S2_18-9-20_16PCC`

- Load `best.pt` (the ResNet-18)
- Load 79 tiles from `data/data/raw/MARIDA/patches/S2_18-9-20_16PCC/`
- For each tile:
  - Read 11 bands with rasterio
  - Normalize using MARIDA's per-band mean/std
  - Push through the model → 15 sigmoid probabilities
  - Apply per-class threshold from `thresholds.json`
  - Record preds + geo bounds
- Write `predictions/honduras_sep18.json` with 79 records

### Step 2 — `drift.py`

- Read `predictions/honduras_sep18.json`
- For each of the 79 tiles, filter by `preds[0] == 1` (Marine Debris) → **35 tiles** kept
- Reproject 35 tile centroids from UTM 16N → WGS84 → 35 `(lat, lon)` pairs
- Look up each tile's date from `tile_index.csv` → all say 2020-09-18
- Build/read OSCAR concat NetCDF covering 2020-09-17 through 2020-09-26
- Create `OpenDrift.OceanDrift`, attach the OSCAR reader, set coastline action to "previous" (freeze on hit), horizontal diffusivity 10 m²/s
- Seed 200 particles around each of the 35 locations → **7000 particles** total
- For 7 days × 48 steps/day = 336 iterations:
  - Ask OSCAR what `(u, v)` is at each particle's current location
  - Move each particle by `(u·dt, v·dt)` plus a small random perturbation
  - Freeze any particle that crosses into land
- Save trajectories to `forecast/honduras_sep18.nc`
- Export 7000 particle paths as LineStrings → `forecast/honduras_sep18.paths.geojson`
- Export final-position Points → `forecast/honduras_sep18.final.geojson`

### Step 3 — `validate.py`

- Load the 7000-particle trajectory
- **Tier 1**: compute mean displacement, beaching fraction, diffusion growth, circular-mean bearing
- **Tier 2**: load `predictions/honduras_sep23.json` (detector run on a second Sentinel-2 scene 5 days later), find its 19 debris centroids, ask "how many of my 7000 forecast particles at Sep 23 are within 5 km of any of those 19 points?"
- Result: **72 % hit rate** at 5 km, 7.4 km centroid drift error

---

## Stage 3 — The web application

Stages 1 and 2 are the science. Stage 3 is the demo apparatus that makes them clickable.

### Offline: scene pre-caching

`src/pipeline/build_scenes.py` runs once (after training). It iterates every MARIDA scene inside the OSCAR window, runs Stage 1 on every tile, and writes a web-friendly directory:

```
web/scenes/
    index.json                  # manifest (scene_id, centroid, date, counts)
    <scene_id>/
        predictions.json        # raw Stage 1 output
        detections.geojson      # WGS84 polygons of positive tiles (map-ready)
        meta.json               # summary (bbox, per-class counts, etc.)
```

~5–15 minutes on CPU for all 23 OSCAR-covered scenes.

### Online: the API

`src/api/server.py` is a FastAPI app. Its endpoints fall into two groups:

**Static reads** (pre-built by the pipeline):

- `GET /api/scenes` → the index manifest
- `GET /api/scenes/{id}/detections` → the polygon GeoJSON
- `GET /api/scenes/{id}/predictions` → the raw JSON (debugging)

**Dynamic writes** (runs OpenDrift per request):

- `POST /api/forecast` with a `ForecastRequest` body
  - `{ scene_id, days, n_per_seed, seed_radius_m, horizontal_diffusivity, debris_classes, min_prob, timestep_minutes }`
  - The server hashes a normalized copy of these params → 16-char cache key
  - If the cache directory exists, returns `cached: true` with the same URLs
  - Otherwise runs `src.forecast.drift.run_drift()` inside a module-level threading lock (OpenDrift is not thread-safe), saves `paths.geojson` + `final.geojson` + `stats.json`, returns

- `GET /api/forecast/{key}/paths` → cached LineString GeoJSON
- `GET /api/forecast/{key}/final` → cached Point GeoJSON
- `GET /api/forecast` → list of all cached runs (with their params and stats)
- `GET /api/forecast/{key}` → stats + params for one run

Everything is served from the same FastAPI process. `/` falls through to a `StaticFiles` mount on `web/app/`, so a single origin serves both the API and the frontend.

### Frontend: Cesium 3D globe

`web/app/index.html` is a single self-contained HTML file. It loads CesiumJS 1.120 from jsDelivr and uses OpenStreetMap tiles as the imagery provider (no Cesium Ion token needed).

The flow:

```
1. Page load
     │
     ▼  GET /api/scenes
   ┌─────────────────────────────────┐
   │ Globe rotates to show Earth.    │
   │ Red markers appear at every     │
   │ cached scene centroid. Marker   │
   │ size scales with detection      │
   │ count. Side panel lists scenes. │
   └─────────────────────────────────┘
     │
     ▼  User clicks a marker or list entry
   ┌─────────────────────────────────┐
   │ Camera flies to scene bbox.     │
   │ GET /api/scenes/{id}/detections │
   │ → red polygon overlay on map.   │
   │ Forecast form appears in panel. │
   └─────────────────────────────────┘
     │
     ▼  User tweaks days/particles/radius/classes, clicks "Run Forecast"
   ┌─────────────────────────────────┐
   │ Spinner.                        │
   │ POST /api/forecast              │
   │ (blocks 20–60s if uncached,      │
   │  instant if cached)             │
   │                                 │
   │ Response carries paths_url +    │
   │ final_url.                      │
   │                                 │
   │ GET each URL → Cesium renders:  │
   │   - blue translucent polylines  │
   │     (particle paths)            │
   │   - orange dots (final pos)     │
   │                                 │
   │ Stats panel fills in:           │
   │   n_particles, elapsed_s,       │
   │   cached status, wall time.     │
   └─────────────────────────────────┘
```

Same-origin means no CORS issues. No bundler, no build step, no npm — just edit the HTML and refresh.

---

## Data-flow summary (complete)

```
MARIDA Sentinel-2 .tif tiles
         │
         ▼
[src.inference.predict / src.pipeline.build_scenes]     (Stage 1 — ML)
         │
         ▼
predictions.json   (per-tile: probs, preds, geo bounds)
         │
         ├────────────► web/scenes/<id>/{predictions,detections,meta}.{json,geojson}
         │                (served statically by FastAPI)
         │
         ▼
[src.forecast.seed]      (filter by class, reproject to WGS84, resolve date)
         │
         ▼
list[Seed(lat, lon, obs_date)]
         │
OSCAR .nc files  ──►  [src.forecast.oscar_concat]   (build CF-compliant NetCDF)
         │                     │
         │                     ▼
         │            data/forecast/oscar_concat_*.nc
         │                     │
         ▼                     ▼
[src.forecast.drift]                                       (Stage 2 — physics)
  OpenDrift OceanDrift: Monte Carlo Lagrangian integration
         │
         ▼
{run.nc, paths.geojson, final.geojson}
         │
         ├──── web/forecast_cache/<hash16>/              (cached per param set)
         │         │
         │         ▼
         │    POST /api/forecast  ──►  returns URLs to the geojsons
         │                                    │
         │                                    ▼
         │                         Cesium renders them on the globe
         │
         ▼
predictions_sceneB.json  ──► [src.forecast.validate]      (Tier 2 cross-detection)
                                    │
                                    ▼
                         { hit_rate, centroid_error, density_iou }
```

---

## What was originally in the plan but isn't here

For completeness — these were documented in earlier plans and intentionally cut for the MVP:

- **CNN + LSTM fusion** — implemented in `src/models/lstm_encoder.py` and `fusion_model.py`; trained as `--use-temporal`. Macro-F1 dropped ~2 points vs. `--cnn-only` because OSCAR only covered 26 % of MARIDA scenes. Kept as code but not used in production.
- **HYCOM SST / salinity** — deferred. Currents alone do the physics work; SST and salinity don't directly drive surface drift.
- **15 → 11 class collapse** — marginal F1 gain for significant engineering effort. Not done.
- **MADOS + NASA-IMPACT datasets** — planned for augmentation and cross-sensor validation. Not integrated.
- **C++ LibTorch inference server** — TorchScript export (`src/inference/export.py`) works and is tested, but we serve inference from Python/FastAPI. For a single-user demo, Python is plenty fast (~100 ms per tile on MPS).
- **Human Delta agent enrichment** — out of scope for the hackathon MVP.
- **U-Net per-pixel segmentation** — would give sharper map overlays but take 2–3 days of focused work. Skipped.
- **Time-animated drift playback** — paths are currently static LineStrings. Adding a Cesium `Clock` + CZML output is ~1 day of work. Deferred.

---

## The mental model to keep

- **Stage 1 = "see."** The CNN looks at pixels and says "debris." It has no concept of time, currents, or motion.
- **Stage 2 = "extrapolate."** OpenDrift takes a point and a velocity field and evolves it forward. It has no concept of what debris is.
- **Stage 3 = "explore."** The web app lets a user sample the product of the first two without re-running detection and with transparent caching of physics runs.

That's the current MVP. The CNN detects, OpenDrift forecasts, the validator proves it works, and the web app makes it clickable.
