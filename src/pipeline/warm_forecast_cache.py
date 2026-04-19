"""Pre-warm the forecast cache for demo scenes.

Runs OpenDrift offline using the same cache-key scheme as src.api.server,
so the warmed entries are picked up on the first user request as
`cached: true`. Use before a demo so judges never see the 30-60s spinner.

Usage:
    python -m src.pipeline.warm_forecast_cache                     # default top scenes
    python -m src.pipeline.warm_forecast_cache --scenes S2_18-9-20_16PCC ...
    python -m src.pipeline.warm_forecast_cache --top 5             # top-5 by detections
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

# Re-use the server's request schema + cache key so warmed entries hit the
# cache lookup in src.api.server.forecast() exactly.
from src.api.server import FORECAST_CACHE, ForecastRequest, ForecastStats, _cache_key
from src.forecast.drift import run_drift


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = REPO_ROOT / "web" / "scenes"


# Default warm-up param sets. Mirrors the UI defaults plus a couple of
# alternate horizons so common "what about 14 days?" demos are also cached.
DEFAULT_PARAM_SETS: tuple[dict, ...] = (
    # UI defaults
    {"days": 7,  "n_per_seed": 100, "seed_radius_m": 1000.0,
     "horizontal_diffusivity": 10.0, "timestep_minutes": 30},
    # Longer horizon
    {"days": 14, "n_per_seed": 100, "seed_radius_m": 1000.0,
     "horizontal_diffusivity": 10.0, "timestep_minutes": 30},
    # Higher particle density
    {"days": 7,  "n_per_seed": 200, "seed_radius_m": 1000.0,
     "horizontal_diffusivity": 10.0, "timestep_minutes": 30},
)


def _count_features(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text()).get("features", []))
    except json.JSONDecodeError:
        return 0


def _load_index() -> dict:
    return json.loads((SCENES_DIR / "index.json").read_text())


def _select_scenes(explicit: list[str] | None, top: int) -> list[str]:
    """Resolve which scenes to warm. Explicit list wins; else top-N by detections."""
    idx = _load_index()
    scenes = idx["scenes"]
    if explicit:
        ids = {s["scene_id"]: s for s in scenes}
        missing = [sid for sid in explicit if sid not in ids]
        if missing:
            raise SystemExit(f"unknown scene_id(s): {missing}")
        return list(explicit)
    ranked = sorted(scenes, key=lambda s: -(s.get("n_detections") or 0))
    return [s["scene_id"] for s in ranked[:top] if (s.get("n_detections") or 0) > 0]


def _ui_default_debris_classes() -> list[int]:
    """Pull the same default class list the UI sends (from the scene index).

    The UI reads `default_debris_classes` from /api/scenes; that field comes
    from src.pipeline.build_scenes.DEFAULT_DEBRIS_CLASSES = (0, 1, 2, 8).
    NOTE: this differs from src.forecast.seed.DEFAULT_DEBRIS_CLASSES = (0,)
    which is what ForecastRequest defaults to — if we relied on the model
    default we'd warm a cache key the UI never hits.
    """
    return list(_load_index().get("default_debris_classes", [0, 1, 2, 8]))


def warm_one(
    scene_id: str,
    params: dict,
    force: bool = False,
    debris_classes: list[int] | None = None,
) -> dict:
    """Run a single forecast and write the cache files using the server's schema."""
    if debris_classes is not None:
        params = {**params, "debris_classes": debris_classes}
    req = ForecastRequest(scene_id=scene_id, **params)
    cache_key = _cache_key(req)
    cache_dir = FORECAST_CACHE / cache_key
    paths_path = cache_dir / "paths.geojson"
    final_path = cache_dir / "final.geojson"
    stats_path = cache_dir / "stats.json"

    if (not force) and paths_path.exists() and final_path.exists() and stats_path.exists():
        return {"scene_id": scene_id, "cache_key": cache_key, "status": "already-cached"}

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "params.json").write_text(req.model_dump_json(indent=2))

    predictions_path = SCENES_DIR / scene_id / "predictions.json"
    if not predictions_path.exists():
        return {"scene_id": scene_id, "cache_key": cache_key,
                "status": "skip-missing-predictions"}

    out_stem = cache_dir / "run"
    t0 = time.perf_counter()
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
        return {"scene_id": scene_id, "cache_key": cache_key,
                "status": f"skip: {e}"}
    elapsed = time.perf_counter() - t0

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

    return {
        "scene_id": scene_id,
        "cache_key": cache_key,
        "status": "warmed",
        "n_particles": n_final,
        "elapsed_s": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Explicit scene_ids to warm (overrides --top).")
    parser.add_argument("--top", type=int, default=5,
                        help="Warm the top-N scenes by detection count (default: 5).")
    parser.add_argument("--params", choices=("default", "all"), default="default",
                        help="'default' = UI default params only; "
                             "'all' = also longer horizon + higher particle count.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cached.")
    args = parser.parse_args()

    scenes = _select_scenes(args.scenes, args.top)
    param_sets = DEFAULT_PARAM_SETS if args.params == "all" else DEFAULT_PARAM_SETS[:1]
    ui_classes = _ui_default_debris_classes()

    print(f"[warm] {len(scenes)} scene(s) x {len(param_sets)} param set(s) "
          f"= {len(scenes) * len(param_sets)} run(s)")
    print(f"[warm] debris_classes (UI default): {ui_classes}")
    print(f"[warm] cache root: {FORECAST_CACHE}")
    for sid in scenes:
        for params in param_sets:
            result = warm_one(sid, params, force=args.force, debris_classes=ui_classes)
            tag = result["status"]
            extra = ""
            if "n_particles" in result:
                extra = f"  particles={result['n_particles']}  elapsed={result['elapsed_s']}s"
            print(f"  [{params['days']}d/{params['n_per_seed']}p] "
                  f"{sid:30s} -> {result['cache_key']}  {tag}{extra}")


if __name__ == "__main__":
    main()
