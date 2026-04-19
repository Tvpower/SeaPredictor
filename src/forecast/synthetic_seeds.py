"""Generate a synthetic predictions.json for drift forecasting demos.

Use this when you want to seed OpenDrift somewhere that isn't a MARIDA tile,
e.g. inside a known oceanic garbage patch / gyre center, without re-running
the CNN inference pipeline.

The output schema matches what `src.inference.predict` writes, so
`src.forecast.drift.run_drift` can consume it unchanged.

Usage (single point, North Atlantic gyre):
    python -m src.forecast.synthetic_seeds \\
        --out data/forecast/synthetic/natlantic.json \\
        --date 2020-09-15 \\
        --point 30 -50

Usage (multiple seeds, named preset):
    python -m src.forecast.synthetic_seeds \\
        --out data/forecast/synthetic/grid.json \\
        --date 2020-09-15 \\
        --preset north_atlantic_grid

Then drive a forecast:
    python -m src.forecast.drift \\
        --predictions data/forecast/synthetic/natlantic.json \\
        --out data/forecast/runs/natlantic_glorys/run \\
        --days 30 --n-per-seed 200 --seed-radius-m 25000 \\
        --current-source glorys \\
        --default-date 2020-09-15 --override-date \\
        --debris-classes 0
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


# Hand-picked centers of major oceanic accumulation zones (approximate, but
# inside the gyre on most days — fine for visual demos, not for ops).
PRESETS: dict[str, list[tuple[float, float, str]]] = {
    "north_atlantic": [
        (30.0, -50.0, "Sargasso/NAtl gyre center"),
    ],
    "north_pacific": [
        (32.0, -145.0, "Great Pacific Garbage Patch"),
    ],
    "south_atlantic": [
        (-30.0, -25.0, "South Atlantic gyre"),
    ],
    "south_pacific": [
        (-32.0, -100.0, "South Pacific gyre"),
    ],
    "indian": [
        (-32.0, 85.0, "Indian Ocean gyre"),
    ],
    # 5x5 grid spanning the North Atlantic gyre (~25-35N, -60..-40W).
    "north_atlantic_grid": [
        (lat, lon, f"natl_grid_{lat:+.0f}_{lon:+.0f}")
        for lat in (25.0, 27.5, 30.0, 32.5, 35.0)
        for lon in (-60.0, -55.0, -50.0, -45.0, -40.0)
    ],
    "north_pacific_grid": [
        (lat, lon, f"npac_grid_{lat:+.0f}_{lon:+.0f}")
        for lat in (28.0, 31.0, 34.0, 37.0, 40.0)
        for lon in (-155.0, -150.0, -145.0, -140.0, -135.0)
    ],
    # Florida Current / Gulf Stream entry — gives dramatic NE drift.
    "gulf_stream": [
        (26.0, -79.0, "Florida Current entry"),
        (28.0, -78.0, "Gulf Stream south"),
        (32.0, -76.0, "Cape Hatteras"),
    ],
    # Strong westward Caribbean Current jet north of Honduras/Jamaica.
    "caribbean_jet": [
        (16.5, -82.0, "Caribbean jet east"),
        (17.0, -85.0, "Caribbean jet mid"),
        (17.5, -88.0, "Caribbean jet west"),
    ],
}


def _record_for_point(
    tile_id: str, lat: float, lon: float, half_extent_deg: float = 0.05
) -> dict:
    """Build a single predictions.json record at (lat, lon).

    `half_extent_deg` controls the synthetic 'tile' size around the point.
    0.05 deg ~= 5.5 km square — small enough that the centroid is the point
    and large enough to read like a real Sentinel-2 tile.
    """
    minx = lon - half_extent_deg
    maxx = lon + half_extent_deg
    miny = lat - half_extent_deg
    maxy = lat + half_extent_deg
    # Class layout follows MARIDA: 15 classes, index 0 = Marine Debris.
    n_classes = 15
    preds = [0] * n_classes
    probs = [0.0] * n_classes
    preds[0] = 1
    probs[0] = 0.99
    return {
        "tile_id": tile_id,
        "preds": preds,
        "probs": probs,
        "geo": {
            "bounds": [minx, miny, maxx, maxy],
            "crs": "EPSG:4326",
        },
    }


def build_payload(seeds: list[tuple[float, float, str]]) -> dict:
    """Wrap synthesized records in the exact envelope `extract_seeds` expects."""
    records = [
        _record_for_point(name or f"synthetic_{i:04d}", lat, lon)
        for i, (lat, lon, name) in enumerate(seeds)
    ]
    return {
        "schema": "predictions/synthetic-v1",
        "n_records": len(records),
        "records": records,
    }


def write_predictions(out_path: Path, seeds: list[tuple[float, float, str]]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(seeds)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _detections_geojson(seeds: list[tuple[float, float, str]]) -> dict:
    """Point FeatureCollection mirroring `<scene>/detections.geojson` shape."""
    features = []
    for lat, lon, name in seeds:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "tile_id": name,
                "predicted_classes": [0],
                "max_prob": 0.99,
                "synthetic": True,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _scene_meta_entry(
    scene_id: str,
    obs_date: str,
    seeds: list[tuple[float, float, str]],
    debris_classes: list[int],
) -> dict:
    """Build a SceneIndexEntry-compatible dict for `web/scenes/index.json`."""
    lats = [s[0] for s in seeds]
    lons = [s[1] for s in seeds]
    pad = 0.1
    bbox = [
        min(lons) - pad,
        min(lats) - pad,
        max(lons) + pad,
        max(lats) + pad,
    ]
    return {
        "scene_id": scene_id,
        "obs_date": obs_date,
        "has_oscar_coverage": True,
        "n_tiles": len(seeds),
        "n_detections": len(seeds),
        "n_cloud_suppressed": 0,
        "per_class_detections": {"Marine Debris": len(seeds)},
        "centroid_lat": sum(lats) / len(lats),
        "centroid_lon": sum(lons) / len(lons),
        "bbox_wgs84": bbox,
        "debris_classes_considered": debris_classes,
        "synthetic": True,
    }


def register_scene(
    scene_id: str,
    obs_date: str,
    seeds: list[tuple[float, float, str]],
    scenes_root: Path = Path("web/scenes"),
    debris_classes: tuple[int, ...] = (0,),
) -> Path:
    """Drop a synthetic scene into `web/scenes/` so the UI lists it.

    Writes:
        web/scenes/<scene_id>/predictions.json
        web/scenes/<scene_id>/detections.geojson
        web/scenes/<scene_id>/meta.json
    and upserts the entry in web/scenes/index.json.
    """
    scene_dir = scenes_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    write_predictions(scene_dir / "predictions.json", seeds)
    (scene_dir / "detections.geojson").write_text(
        json.dumps(_detections_geojson(seeds))
    )
    entry = _scene_meta_entry(scene_id, obs_date, seeds, list(debris_classes))
    (scene_dir / "meta.json").write_text(json.dumps(entry, indent=2))

    idx_path = scenes_root / "index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    else:
        idx = {
            "generated_from_ckpt": "synthetic",
            "class_names": [
                "Marine Debris", "Dense Sargassum", "Sparse Sargassum",
                "Natural Organic Material", "Ship", "Clouds", "Marine Water",
                "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water",
                "Waves", "Cloud Shadows", "Wakes", "Mixed Water",
            ],
            "default_debris_classes": [0],
            "oscar_coverage": {"start": "1993-01-01", "end": "2021-06-30", "n_dates": 0},
            "n_scenes": 0,
            "scenes": [],
        }
    scenes = [s for s in idx.get("scenes", []) if s.get("scene_id") != scene_id]
    scenes.append(entry)
    scenes.sort(key=lambda s: s.get("obs_date", ""), reverse=True)
    idx["scenes"] = scenes
    idx["n_scenes"] = len(scenes)
    idx_path.write_text(json.dumps(idx, indent=2))
    return scene_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to write the synthetic predictions.json")
    parser.add_argument("--preset", type=str, default=None,
                        choices=sorted(PRESETS.keys()),
                        help="Use a built-in preset of seed locations")
    parser.add_argument("--point", type=float, nargs=2, action="append",
                        metavar=("LAT", "LON"), default=None,
                        help="Add a single seed point. Repeatable.")
    parser.add_argument("--date", type=str, default=None,
                        help="Observation date (YYYY-MM-DD). Required with "
                             "--register; otherwise just a reminder for "
                             "`drift --default-date --override-date`.")
    parser.add_argument("--register", type=str, default=None, metavar="SCENE_ID",
                        help="Also drop a fake scene into web/scenes/<SCENE_ID>/ "
                             "and update web/scenes/index.json so the UI lists it.")
    parser.add_argument("--scenes-root", type=Path, default=Path("web/scenes"),
                        help="Where to register the scene (default: web/scenes).")
    args = parser.parse_args()

    seeds: list[tuple[float, float, str]] = []
    if args.preset:
        seeds.extend(PRESETS[args.preset])
    if args.point:
        for i, (lat, lon) in enumerate(args.point):
            seeds.append((lat, lon, f"cli_point_{i:03d}"))
    if not seeds:
        parser.error("Provide --preset and/or --point at least once.")

    out = write_predictions(args.out, seeds)
    print(f"[synthetic] wrote {len(seeds)} seeds -> {out}")
    for lat, lon, name in seeds:
        print(f"  {name:40s}  lat={lat:+.3f}  lon={lon:+.3f}")

    if args.register:
        if not args.date:
            parser.error("--register requires --date YYYY-MM-DD")
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            parser.error(f"--date {args.date!r} is not YYYY-MM-DD")
        scene_dir = register_scene(
            scene_id=args.register,
            obs_date=args.date,
            seeds=seeds,
            scenes_root=args.scenes_root,
        )
        print(f"[synthetic] registered scene -> {scene_dir}")
        print(f"[synthetic] index updated -> {args.scenes_root / 'index.json'}")
        print(f"[synthetic] UI should list scene_id='{args.register}' on next refresh")
    elif args.date:
        print(f"[synthetic] hint: pass `--default-date {args.date} --override-date` to drift")


if __name__ == "__main__":
    main()
