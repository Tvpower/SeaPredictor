"""Build the web scene cache from existing prediction JSON files.

This bridge is for the demo path where the trained detector has already
produced ``predictions/*.json`` but the raw MARIDA tile directory is not
present locally. It preserves the same ``web/scenes`` contract consumed by
``src.api.server`` and the Cesium frontend.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

CLASS_NAMES: tuple[str, ...] = (
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
    "Waves",
    "Cloud Shadows",
    "Wakes",
    "Mixed Water",
)

DEFAULT_DEBRIS_CLASSES: tuple[int, ...] = (0, 1, 2, 8)


def _scene_id_from_predictions(path: Path, payload: dict) -> str:
    if payload.get("scene_id"):
        return str(payload["scene_id"])
    records = payload.get("records", [])
    if records:
        return str(records[0]["tile_id"]).rsplit("_", 1)[0]
    return path.stem


def _tile_meta(repo_root: Path) -> dict[str, dict]:
    path = repo_root / "data" / "tile_index.csv"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                out[row["tile_id"]] = {
                    "date": date.fromisoformat(row["date"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
            except (KeyError, ValueError):
                continue
    return out


def _scene_date(scene_id: str, records: list[dict], tile_meta: dict[str, dict]) -> date | None:
    for rec in records:
        meta = tile_meta.get(rec.get("tile_id"))
        if meta is not None:
            return meta["date"]
    try:
        raw = scene_id.split("_", 2)[1]
        return datetime.strptime(raw, "%d-%m-%y").date()
    except (IndexError, ValueError):
        return None


def _oscar_dates() -> set[date]:
    out: set[date] = set()
    for root in (Path("data/data/raw/oscar"), Path("data/raw/oscar")):
        if not root.is_dir():
            continue
        for path in root.glob("oscar_currents_interim_*.nc"):
            try:
                out.add(datetime.strptime(path.stem.rsplit("_", 1)[1], "%Y%m%d").date())
            except (IndexError, ValueError):
                continue
    return out


def _centroid_wgs84(bounds: list[float], src_crs: str | None) -> tuple[float, float] | None:
    if src_crs is None or len(bounds) != 4:
        return None
    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    if src_crs.upper() in {"EPSG:4326", "WGS84"}:
        return float(cy), float(cx)
    try:
        from pyproj import Transformer
    except ModuleNotFoundError:
        return None
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return float(lat), float(lon)


def _polygon_wgs84(bounds: list[float], src_crs: str | None) -> list[list[float]] | None:
    if src_crs is None or len(bounds) != 4:
        return None
    minx, miny, maxx, maxy = bounds
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    if src_crs.upper() in {"EPSG:4326", "WGS84"}:
        return [[float(x), float(y)] for x, y in corners]
    try:
        from pyproj import Transformer
    except ModuleNotFoundError:
        return None
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    return [[float(lon), float(lat)] for lon, lat in (transformer.transform(x, y) for x, y in corners)]


def _centroid_for_record(rec: dict, tile_meta: dict[str, dict]) -> tuple[float, float] | None:
    geo = rec.get("geo", {})
    centroid = _centroid_wgs84(geo.get("bounds", []), geo.get("crs"))
    if centroid is not None:
        return centroid
    meta = tile_meta.get(rec.get("tile_id"))
    if meta is None:
        return None
    return float(meta["lat"]), float(meta["lon"])


def _fallback_ring(centroid: tuple[float, float]) -> list[list[float]]:
    lat, lon = centroid
    delta = 0.012
    return [
        [lon - delta, lat - delta],
        [lon + delta, lat - delta],
        [lon + delta, lat + delta],
        [lon - delta, lat + delta],
        [lon - delta, lat - delta],
    ]


def _build_detections_geojson(
    records: list[dict],
    debris_classes: tuple[int, ...],
    tile_meta: dict[str, dict],
) -> dict:
    features: list[dict] = []
    debris_set = set(debris_classes)
    for rec in records:
        preds = rec.get("preds", [])
        probs = rec.get("probs", [])
        hits = [i for i in sorted(debris_set) if i < len(preds) and preds[i] == 1]
        if not hits:
            continue
        centroid = _centroid_for_record(rec, tile_meta)
        if centroid is None:
            continue
        geo = rec.get("geo", {})
        ring = _polygon_wgs84(geo.get("bounds", []), geo.get("crs")) or _fallback_ring(centroid)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "tile_id": rec.get("tile_id"),
                "matched_classes": hits,
                "class_names": [CLASS_NAMES[i] for i in hits],
                "max_prob": max((float(probs[i]) for i in hits if i < len(probs)), default=0.0),
                "centroid_lat": centroid[0],
                "centroid_lon": centroid[1],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _compute_scene_meta(
    scene_id: str,
    records: list[dict],
    detections: dict,
    tile_meta: dict[str, dict],
    obs_date: date | None,
    has_oscar: bool,
) -> dict:
    centroids = [c for rec in records if (c := _centroid_for_record(rec, tile_meta)) is not None]
    lats = [c[0] for c in centroids]
    lons = [c[1] for c in centroids]

    per_class = {CLASS_NAMES[i]: 0 for i in DEFAULT_DEBRIS_CLASSES}
    for feature in detections.get("features", []):
        for cls in feature.get("properties", {}).get("matched_classes", []):
            if cls < len(CLASS_NAMES):
                per_class[CLASS_NAMES[cls]] = per_class.get(CLASS_NAMES[cls], 0) + 1

    return {
        "scene_id": scene_id,
        "obs_date": obs_date.isoformat() if obs_date else None,
        "has_oscar_coverage": has_oscar,
        "n_tiles": len(records),
        "n_detections": len(detections.get("features", [])),
        "n_cloud_suppressed": 0,
        "per_class_detections": per_class,
        "centroid_lat": sum(lats) / len(lats) if lats else None,
        "centroid_lon": sum(lons) / len(lons) if lons else None,
        "bbox_wgs84": [min(lons), min(lats), max(lons), max(lats)] if lats and lons else None,
        "debris_classes_considered": list(DEFAULT_DEBRIS_CLASSES),
    }


def ensure_scene_cache_from_predictions(repo_root: Path) -> bool:
    """Create ``web/scenes`` from ``predictions/*.json`` if no scene index exists."""
    scenes_dir = repo_root / "web" / "scenes"
    index_path = scenes_dir / "index.json"
    if index_path.exists():
        return False

    prediction_files = sorted((repo_root / "predictions").glob("*.json"))
    if not prediction_files:
        return False

    scenes_dir.mkdir(parents=True, exist_ok=True)
    tile_meta = _tile_meta(repo_root)
    oscar_dates = _oscar_dates()
    manifest: list[dict] = []

    for path in prediction_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            continue
        scene_id = _scene_id_from_predictions(path, payload)
        obs_date = _scene_date(scene_id, records, tile_meta)
        has_oscar = bool(
            obs_date is not None
            and oscar_dates
            and min(oscar_dates) <= obs_date <= max(oscar_dates)
        )

        scene_dir = scenes_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        detections = _build_detections_geojson(records, DEFAULT_DEBRIS_CLASSES, tile_meta)
        meta = _compute_scene_meta(scene_id, records, detections, tile_meta, obs_date, has_oscar)

        (scene_dir / "predictions.json").write_text(json.dumps(payload), encoding="utf-8")
        (scene_dir / "detections.geojson").write_text(json.dumps(detections), encoding="utf-8")
        (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.append(meta)

    index = {
        "generated_from_ckpt": "predictions/*.json",
        "class_names": list(CLASS_NAMES),
        "default_debris_classes": list(DEFAULT_DEBRIS_CLASSES),
        "oscar_coverage": {
            "start": min(oscar_dates).isoformat() if oscar_dates else None,
            "end": max(oscar_dates).isoformat() if oscar_dates else None,
            "n_dates": len(oscar_dates),
        },
        "n_scenes": len(manifest),
        "scenes": sorted(
            manifest,
            key=lambda item: (item.get("obs_date") or "", -(item.get("n_detections") or 0)),
        ),
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return True
