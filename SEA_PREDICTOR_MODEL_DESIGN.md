# SeaPredictor Model And UI Design Doc

## Purpose

SeaPredictor combines a visual detector with a physics simulator. The detector answers "where is debris visible now?" from Sentinel-2 tiles. The simulator answers "where will those detected points drift?" using OSCAR current fields and OpenDrift. The web UI should make that handoff obvious: detections are seeds, forecasts are particle trajectories, and cached API runs make the demo repeatable.

## Mental Model

Stage 1 is seeing. A ResNet-18 CNN reads 11-band Sentinel-2 MARIDA tiles and emits multi-label probabilities for 15 surface classes. It has no memory of time and no current-field awareness.

Stage 2 is extrapolation. OpenDrift receives only seed points, dates, and velocity data. It has no image understanding and no learned parameters. It numerically integrates particles through OSCAR currents with Monte Carlo spread and coastline behavior.

Stage 3 is exploration. The web app lets users pick cached scenes, inspect detection polygons, run or load forecasts, and visualize drift paths and final particle positions on a 3D globe.

## Data Contracts

### Stage 1 Prediction Record

Each prediction record should preserve the model output and tile geometry:

```json
{
  "tile_id": "S2_18-9-20_16PCC_0",
  "probs": [0.87, 0.02],
  "preds": [1, 0],
  "predicted_classes": [0, 6],
  "geo": {
    "crs": "EPSG:32616",
    "bounds": [0, 0, 1, 1]
  }
}
```

### Stage 2 Seed

Only the minimal forecast handoff survives:

```python
Seed(
    tile_id="S2_18-9-20_16PCC_0",
    lat=16.09,
    lon=-88.32,
    obs_date=date(2020, 9, 18),
    matched_classes=(0,),
    max_prob=0.91,
)
```

### Forecast Request

The API should normalize and hash this request shape for cache keys:

```json
{
  "scene_id": "honduras_sep18",
  "days": 7,
  "n_per_seed": 200,
  "seed_radius_m": 1000,
  "horizontal_diffusivity": 10,
  "debris_classes": [0],
  "min_prob": 0.5,
  "timestep_minutes": 30
}
```

## Backend Responsibilities

The detector pipeline should produce `web/scenes/index.json`, scene-level `predictions.json`, `detections.geojson`, and `meta.json`. Those artifacts must be static reads so the UI can load scene options instantly.

The forecast API should expose static scene endpoints and dynamic forecast endpoints. OpenDrift execution must remain serialized behind a module-level lock because it is not thread-safe. Every unique parameter set should write a cache directory containing `paths.geojson`, `final.geojson`, `stats.json`, and the normalized params used for hashing.

OSCAR preparation must stay isolated in `src/forecast/oscar_concat.py`. The detector should not depend on OSCAR for the MVP; OSCAR is only the velocity field for OpenDrift.

## Frontend Responsibilities

The UI should show three layers of truth:

1. Scene detections: red or coral seed markers/polygons from cached detector outputs.
2. Forecast paths: blue or cyan translucent drift paths from OpenDrift.
3. Final positions and uncertainty: warm final dots plus spread/rings or density styling.

The first screen should be the working globe, not a landing page. The user should immediately see available scenes, selected parameters, pipeline status, and map overlays. Copy should reinforce the model boundary: CNN detects, OpenDrift forecasts, FastAPI caches.

## Initial MVP UI Behavior

On page load, fetch `/api/scenes`, render scene centroids, and populate the scene list. If the API is not available, keep a polished static demo state so the interface remains presentable.

When a user selects a scene, fly the camera to its centroid or bbox, load `/api/scenes/{id}/detections`, and draw the detection overlay.

When a user runs a forecast, POST `/api/forecast` with the selected parameters. While waiting, show a running state and preserve the previous forecast layer. When complete, fetch `paths_url` and `final_url`, render the new layers, and update stats including particle count, elapsed seconds, cached status, and cache key.

## Visual Direction

Use an ocean operations palette: deep navy base, cyan current paths, coral detection seeds, warm final particles, and restrained green success states. Avoid a one-hue dashboard. The globe should feel like the main instrument: full-bleed, animated, and layered with readable overlays. Panels should be compact, glassy, and operational, with clear hierarchy and no marketing copy.

## Acceptance Criteria

- The frontend builds with `npm run build`.
- The first viewport shows the globe as the primary experience.
- The UI explains the Stage 1 to Stage 2 handoff through labels and data panels.
- The globe renders seed markers, drift paths, and final/uncertainty indicators.
- The design remains usable when API data is unavailable by falling back to demo data.
- Future API wiring can replace static demo data without redesigning the screen.
