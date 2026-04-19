"""FastAPI server for the SeaPredictor demo.

Endpoints (all under /api):

    GET  /scenes                      -> manifest of cached scenes
    GET  /scenes/{scene_id}           -> meta.json for one scene
    GET  /scenes/{scene_id}/detections -> detections.geojson (map-ready polygons)
    GET  /scenes/{scene_id}/predictions -> full predictions.json (debug / inspection)

    POST /forecast                    -> run OpenDrift on a cached scene.
                                         Body: see ForecastRequest.
                                         Response: see ForecastResponse.

    GET  /forecast/{cache_key}/paths   -> cached per-particle paths GeoJSON
    GET  /forecast/{cache_key}/final   -> cached final positions GeoJSON
    GET  /forecast/{cache_key}         -> stats + param echo

The server keeps *everything* on the local filesystem:

    web/scenes/           pre-built by src.pipeline.build_scenes
    web/forecast_cache/   lazily populated by the /forecast endpoint, keyed
                          by a hash of (scene_id, run params).

OpenDrift is not thread-safe; we serialize forecast runs with a module-level
lock. For a single-user hackathon demo this is fine.

Run:
    uvicorn src.api.server:app --reload
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.forecast.drift import run_drift
from src.forecast.seed import DEFAULT_DEBRIS_CLASSES


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = REPO_ROOT / "web" / "scenes"
FORECAST_CACHE = REPO_ROOT / "web" / "forecast_cache"
WEB_APP_DIR = REPO_ROOT / "web" / "app"
FORECAST_CACHE.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #
app = FastAPI(title="SeaPredictor API", version="0.1.0")

# Wide open during dev; tighten before prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenDrift isn't thread-safe; serialize forecast runs.
_drift_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Request / response models                                                   #
# --------------------------------------------------------------------------- #
class ForecastRequest(BaseModel):
    scene_id: str
    days: int = Field(7, ge=1, le=30)
    n_per_seed: int = Field(100, ge=10, le=500)
    seed_radius_m: float = Field(1000.0, ge=10.0, le=50000.0)
    horizontal_diffusivity: float = Field(10.0, ge=0.0, le=500.0)
    debris_classes: list[int] = Field(default_factory=lambda: list(DEFAULT_DEBRIS_CLASSES))
    min_prob: float = Field(0.0, ge=0.0, le=1.0)
    timestep_minutes: int = Field(30, ge=5, le=180)


class ForecastStats(BaseModel):
    cache_key: str
    cached: bool
    n_particles: int
    n_features_paths: int
    n_features_final: int
    elapsed_s: float


class ForecastResponse(BaseModel):
    cache_key: str
    cached: bool
    stats: ForecastStats
    params: ForecastRequest
    paths_url: str
    final_url: str


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _scene_dir(scene_id: str) -> Path:
    d = SCENES_DIR / scene_id
    if not d.is_dir() or not (d / "predictions.json").exists():
        raise HTTPException(
            status_code=404,
            detail=f"scene '{scene_id}' not found. Build the cache with "
                   f"`python -m src.pipeline.build_scenes ...` first.",
        )
    return d


def _cache_key(req: ForecastRequest) -> str:
    """Stable hash over normalized params. Same inputs -> same cache dir."""
    payload = {
        "scene_id": req.scene_id,
        "days": req.days,
        "n_per_seed": req.n_per_seed,
        "seed_radius_m": round(req.seed_radius_m, 3),
        "horizontal_diffusivity": round(req.horizontal_diffusivity, 3),
        "debris_classes": sorted(req.debris_classes),
        "min_prob": round(req.min_prob, 4),
        "timestep_minutes": req.timestep_minutes,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _count_features(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return len(data.get("features", []))
    except json.JSONDecodeError:
        return 0


# --------------------------------------------------------------------------- #
# Routes: scenes                                                              #
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "scenes_dir": str(SCENES_DIR), "cache_dir": str(FORECAST_CACHE)}


@app.get("/api/scenes")
def list_scenes() -> JSONResponse:
    idx = SCENES_DIR / "index.json"
    if not idx.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Scene index missing at {idx}. Build it with "
                   f"`python -m src.pipeline.build_scenes ...`.",
        )
    return JSONResponse(json.loads(idx.read_text()))


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str) -> JSONResponse:
    meta = _scene_dir(scene_id) / "meta.json"
    if not meta.exists():
        raise HTTPException(404, f"meta.json missing for scene '{scene_id}'")
    return JSONResponse(json.loads(meta.read_text()))


@app.get("/api/scenes/{scene_id}/detections")
def get_scene_detections(scene_id: str) -> FileResponse:
    path = _scene_dir(scene_id) / "detections.geojson"
    if not path.exists():
        raise HTTPException(404, f"detections.geojson missing for scene '{scene_id}'")
    return FileResponse(path, media_type="application/geo+json")


@app.get("/api/scenes/{scene_id}/predictions")
def get_scene_predictions(scene_id: str) -> FileResponse:
    path = _scene_dir(scene_id) / "predictions.json"
    return FileResponse(path, media_type="application/json")


# --------------------------------------------------------------------------- #
# Routes: forecast                                                            #
# --------------------------------------------------------------------------- #
def _paths_url(cache_key: str) -> str:
    return f"/api/forecast/{cache_key}/paths"


def _final_url(cache_key: str) -> str:
    return f"/api/forecast/{cache_key}/final"


def _load_cached_stats(cache_dir: Path, req: ForecastRequest, cache_key: str) -> ForecastResponse:
    stats_raw = json.loads((cache_dir / "stats.json").read_text())
    stats = ForecastStats(**stats_raw)
    return ForecastResponse(
        cache_key=cache_key,
        cached=True,
        stats=stats,
        params=req,
        paths_url=_paths_url(cache_key),
        final_url=_final_url(cache_key),
    )


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    scene_dir = _scene_dir(req.scene_id)
    predictions_path = scene_dir / "predictions.json"

    cache_key = _cache_key(req)
    cache_dir = FORECAST_CACHE / cache_key
    paths_path = cache_dir / "paths.geojson"
    final_path = cache_dir / "final.geojson"
    stats_path = cache_dir / "stats.json"

    if paths_path.exists() and final_path.exists() and stats_path.exists():
        return _load_cached_stats(cache_dir, req, cache_key)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "params.json").write_text(req.model_dump_json(indent=2))

    out_stem = cache_dir / "run"

    t0 = time.perf_counter()
    with _drift_lock:
        try:
            run_drift(
                predictions_path=predictions_path,
                out_stem=out_stem,
                days=req.days,
                timestep_minutes=req.timestep_minutes,
                n_per_seed=req.n_per_seed,
                seed_radius_m=req.seed_radius_m,
                horizontal_diffusivity=req.horizontal_diffusivity,
                debris_classes=tuple(req.debris_classes),
                min_prob=req.min_prob,
            )
        except RuntimeError as e:
            # e.g. no seeds matched the class filter
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001 - surface drift errors to client
            raise HTTPException(status_code=500, detail=f"drift failed: {e}") from e
    elapsed = time.perf_counter() - t0

    # run_drift writes `<stem>.paths.geojson` / `<stem>.final.geojson`; rename
    # to canonical names so the static URL paths are stable.
    src_paths = out_stem.with_suffix(".paths.geojson")
    src_final = out_stem.with_suffix(".final.geojson")
    if src_paths.exists():
        src_paths.replace(paths_path)
    if src_final.exists():
        src_final.replace(final_path)

    n_paths = _count_features(paths_path)
    n_final = _count_features(final_path)

    stats = ForecastStats(
        cache_key=cache_key,
        cached=False,
        n_particles=n_final,
        n_features_paths=n_paths,
        n_features_final=n_final,
        elapsed_s=round(elapsed, 2),
    )
    stats_path.write_text(stats.model_dump_json(indent=2))

    return ForecastResponse(
        cache_key=cache_key,
        cached=False,
        stats=stats,
        params=req,
        paths_url=_paths_url(cache_key),
        final_url=_final_url(cache_key),
    )


@app.get("/api/forecast/{cache_key}")
def get_forecast_stats(cache_key: str) -> JSONResponse:
    cache_dir = FORECAST_CACHE / cache_key
    stats_path = cache_dir / "stats.json"
    params_path = cache_dir / "params.json"
    if not stats_path.exists() or not params_path.exists():
        raise HTTPException(404, f"forecast '{cache_key}' not found")
    return JSONResponse({
        "cache_key": cache_key,
        "stats": json.loads(stats_path.read_text()),
        "params": json.loads(params_path.read_text()),
        "paths_url": _paths_url(cache_key),
        "final_url": _final_url(cache_key),
    })


@app.get("/api/forecast/{cache_key}/paths")
def get_forecast_paths(cache_key: str) -> FileResponse:
    path = FORECAST_CACHE / cache_key / "paths.geojson"
    if not path.exists():
        raise HTTPException(404, f"paths for forecast '{cache_key}' not found")
    return FileResponse(path, media_type="application/geo+json")


@app.get("/api/forecast/{cache_key}/final")
def get_forecast_final(cache_key: str) -> FileResponse:
    path = FORECAST_CACHE / cache_key / "final.geojson"
    if not path.exists():
        raise HTTPException(404, f"final positions for forecast '{cache_key}' not found")
    return FileResponse(path, media_type="application/geo+json")


# --------------------------------------------------------------------------- #
# Diagnostics: list cached forecasts                                          #
# --------------------------------------------------------------------------- #
@app.get("/api/forecast")
def list_cached_forecasts() -> JSONResponse:
    out: list[dict] = []
    for cache_dir in sorted(FORECAST_CACHE.iterdir()):
        if not cache_dir.is_dir():
            continue
        stats_path = cache_dir / "stats.json"
        params_path = cache_dir / "params.json"
        if not stats_path.exists() or not params_path.exists():
            continue
        out.append({
            "cache_key": cache_dir.name,
            "stats": json.loads(stats_path.read_text()),
            "params": json.loads(params_path.read_text()),
        })
    return JSONResponse({"n": len(out), "forecasts": out})


# --------------------------------------------------------------------------- #
# Static frontend (must be mounted LAST so /api/* routes win)                 #
# --------------------------------------------------------------------------- #
if WEB_APP_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_APP_DIR), html=True), name="app")


def _maybe_run_as_script() -> None:
    # Convenience: `python -m src.api.server` boots uvicorn on :8000.
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    _maybe_run_as_script()
