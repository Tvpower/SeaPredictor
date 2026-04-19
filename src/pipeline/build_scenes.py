"""Build the demo scene cache consumed by the web backend.

For each MARIDA scene that falls inside our OSCAR coverage window, run the
trained detector across every 256x256 patch and dump a stable, web-friendly
directory layout:

    web/scenes/
        index.json                          # top-level manifest (all scenes)
        <scene_id>/
            predictions.json                # raw detector output (same as src.inference.predict)
            detections.geojson              # polygons of positive tiles only (map-ready)
            meta.json                       # centroid, bbox, date, counts, coverage flags

The FastAPI backend (src.api.server) serves these directly. OpenDrift is *not*
run here — that happens on demand per user request from the frontend.

Usage:
    python -m src.pipeline.build_scenes \\
        --ckpt checkpoints/cnn_only_v2/best.pt \\
        --thresholds checkpoints/cnn_only_v2/thresholds.json \\
        --out web/scenes
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np

from src.dataset.marida_loader import MaridaIndex, default_marida_root
from src.dataset.oscar_loader import OSCARLoader, default_oscar_root
from src.inference.predict import (
    build_sequence,
    iter_dir_tiles,
    load_model,
    load_norm_stats,
    normalize_tile,
    run_inference,
)


# MARIDA's 15-class scheme. Indices match `labels_mapping.txt` / `_cl.tif` (1-indexed there,
# but our label vectors and predictions use 0-indexed positions).
CLASS_NAMES: tuple[str, ...] = (
    "Marine Debris",        # 0
    "Dense Sargassum",      # 1
    "Sparse Sargassum",     # 2
    "Natural Organic Material",  # 3
    "Ship",                 # 4
    "Clouds",               # 5
    "Marine Water",         # 6
    "Sediment-Laden Water", # 7
    "Foam",                 # 8
    "Turbid Water",         # 9
    "Shallow Water",        # 10
    "Waves",                # 11
    "Cloud Shadows",        # 12
    "Wakes",                # 13
    "Mixed Water",          # 14
)

# Classes we surface in the UI as "trackable floating material".
DEFAULT_DEBRIS_CLASSES: tuple[int, ...] = (0, 1, 2, 8)  # debris, sargassum x2, foam


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _bounds_centroid_wgs84(bounds: list[float], src_crs: str | None) -> tuple[float, float] | None:
    """(minx, miny, maxx, maxy) in src_crs -> (lat, lon). None on unknown CRS."""
    if src_crs is None or len(bounds) != 4:
        return None
    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    if src_crs.upper() in {"EPSG:4326", "WGS84"}:
        return float(cy), float(cx)
    from pyproj import Transformer
    tr = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(cx, cy)
    return float(lat), float(lon)


def _bounds_polygon_wgs84(bounds: list[float], src_crs: str | None) -> list[list[float]] | None:
    """Return a closed 5-point ring in (lon, lat) suitable for a GeoJSON Polygon."""
    if src_crs is None or len(bounds) != 4:
        return None
    minx, miny, maxx, maxy = bounds
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    if src_crs.upper() in {"EPSG:4326", "WGS84"}:
        return [[float(x), float(y)] for x, y in corners]
    from pyproj import Transformer
    tr = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    ring: list[list[float]] = []
    for x, y in corners:
        lon, lat = tr.transform(x, y)
        ring.append([float(lon), float(lat)])
    return ring


def _scene_date(scene_id: str, tile_meta: dict[str, dict]) -> date | None:
    """Find an observation date for a scene by looking up any of its tiles."""
    for tile_id, meta in tile_meta.items():
        if tile_id.startswith(scene_id + "_"):
            d = meta.get("date")
            if d is not None:
                return d
    return None


def _scene_inside_oscar(scene_date: date | None, oscar_dates: set[date]) -> bool:
    if scene_date is None or not oscar_dates:
        return False
    lo, hi = min(oscar_dates), max(oscar_dates)
    return lo <= scene_date <= hi


# --------------------------------------------------------------------------- #
# Per-scene detection                                                         #
# --------------------------------------------------------------------------- #
def detect_scene(
    scene_dir: Path,
    model,
    cfg: dict,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: np.ndarray,
    device: str,
    oscar: OSCARLoader | None,
    obs_date: date | None,
    bands: list[int] | None,
) -> list[dict]:
    """Run inference over every .tif in a scene directory. Returns tile records."""
    tile_size = int(cfg.get("tile_size", 256))
    records: list[dict] = []
    for tile_id, arr, geo in iter_dir_tiles(scene_dir, bands):
        if arr.shape[1:] != (tile_size, tile_size):
            continue
        norm = normalize_tile(arr, mean, std)
        # For OSCAR sequence we want the actual tile centroid, not a single lat/lon.
        latlon = _bounds_centroid_wgs84(geo.get("bounds"), geo.get("crs"))
        lat = latlon[0] if latlon else None
        lon = latlon[1] if latlon else None
        seq = build_sequence(cfg, oscar, obs_date, lat, lon)
        probs = run_inference(model, norm, seq, device)
        preds = (probs >= thresholds).astype(np.int32)
        records.append({
            "tile_id": tile_id,
            "probs": [float(p) for p in probs],
            "preds": [int(p) for p in preds],
            "predicted_classes": [i for i, p in enumerate(preds) if p == 1],
            "geo": {
                "bounds": geo.get("bounds"),
                "crs": geo.get("crs"),
            },
        })
    return records


def build_detections_geojson(records: list[dict], debris_classes: Iterable[int]) -> dict:
    """Emit polygons for tiles that predicted at least one debris-like class.

    Geometry is a WGS84 polygon derived from the tile's bounding box so the
    frontend can drop it straight onto Leaflet.
    """
    debris_set = set(debris_classes)
    features: list[dict] = []
    for rec in records:
        preds = rec["preds"]
        hits = [i for i in debris_set if i < len(preds) and preds[i] == 1]
        if not hits:
            continue
        bounds = rec["geo"].get("bounds")
        crs = rec["geo"].get("crs")
        ring = _bounds_polygon_wgs84(bounds, crs)
        centroid = _bounds_centroid_wgs84(bounds, crs)
        if ring is None or centroid is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "tile_id": rec["tile_id"],
                "matched_classes": hits,
                "class_names": [CLASS_NAMES[i] for i in hits],
                "max_prob": max(rec["probs"][i] for i in hits),
                "probs": rec["probs"],
                "preds": rec["preds"],
                "centroid_lat": centroid[0],
                "centroid_lon": centroid[1],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def compute_scene_meta(
    scene_id: str,
    records: list[dict],
    debris_classes: Iterable[int],
    obs_date: date | None,
    has_oscar: bool,
) -> dict:
    """Aggregate stats + map hints for the UI scene picker."""
    debris_set = set(debris_classes)
    centroids: list[tuple[float, float]] = []
    lons: list[float] = []
    lats: list[float] = []
    n_detections = 0
    per_class_hits = {i: 0 for i in debris_set}
    for rec in records:
        bounds = rec["geo"].get("bounds")
        crs = rec["geo"].get("crs")
        c = _bounds_centroid_wgs84(bounds, crs)
        if c is not None:
            centroids.append(c)
            lats.append(c[0])
            lons.append(c[1])
        preds = rec["preds"]
        hits = [i for i in debris_set if i < len(preds) and preds[i] == 1]
        if hits:
            n_detections += 1
            for i in hits:
                per_class_hits[i] += 1

    scene_centroid = None
    bbox = None
    if centroids:
        scene_centroid = (float(np.mean(lats)), float(np.mean(lons)))
        bbox = [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))]

    return {
        "scene_id": scene_id,
        "obs_date": obs_date.isoformat() if obs_date else None,
        "has_oscar_coverage": has_oscar,
        "n_tiles": len(records),
        "n_detections": n_detections,
        "per_class_detections": {
            CLASS_NAMES[i]: per_class_hits[i] for i in sorted(per_class_hits)
        },
        "centroid_lat": scene_centroid[0] if scene_centroid else None,
        "centroid_lon": scene_centroid[1] if scene_centroid else None,
        "bbox_wgs84": bbox,
        "debris_classes_considered": sorted(debris_set),
    }


# --------------------------------------------------------------------------- #
# Top-level orchestration                                                     #
# --------------------------------------------------------------------------- #
def build(
    ckpt: Path,
    thresholds_path: Path | None,
    out_dir: Path,
    debris_classes: tuple[int, ...] = DEFAULT_DEBRIS_CLASSES,
    limit: int | None = None,
    only_with_detections: bool = True,
    require_oscar: bool = True,
    scene_allowlist: tuple[str, ...] | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, device = load_model(ckpt)
    print(f"[build] device={device}  in_channels={cfg.get('in_channels')}  "
          f"use_temporal={cfg.get('use_temporal')}  num_classes={cfg.get('num_classes')}")

    mean, std = load_norm_stats(cfg)
    bands = cfg.get("bands")

    num_classes = int(cfg.get("num_classes", 15))
    if thresholds_path is not None and thresholds_path.exists():
        thr_payload = json.loads(thresholds_path.read_text())
        thresholds = np.asarray(thr_payload["thresholds"], dtype=np.float32)
        print(f"[build] thresholds <- {thresholds_path}")
    else:
        thresholds = np.full(num_classes, 0.5, dtype=np.float32)
        print("[build] thresholds=0.5 (no tuned file provided)")

    # OSCAR (only used for the temporal branch when cfg.use_temporal).
    oscar: OSCARLoader | None = None
    oscar_dates: set[date] = set()
    root = default_oscar_root()
    if root is not None:
        try:
            oscar = OSCARLoader(root)
            oscar_dates = set(oscar.available_dates)
            print(f"[build] OSCAR: {len(oscar_dates)} dates "
                  f"({min(oscar_dates)}..{max(oscar_dates)})")
        except FileNotFoundError:
            pass

    # MARIDA scenes.
    marida_root = default_marida_root()
    try:
        idx = MaridaIndex.from_root(marida_root)
        tile_meta = idx._tile_meta  # noqa: SLF001 - internal on purpose
    except FileNotFoundError:
        tile_meta = {}
    patches_root = marida_root / "patches"

    all_scene_dirs = sorted(p for p in patches_root.iterdir() if p.is_dir())
    if scene_allowlist:
        allow = set(scene_allowlist)
        all_scene_dirs = [p for p in all_scene_dirs if p.name in allow]
    print(f"[build] {len(all_scene_dirs)} candidate scenes in {patches_root}")

    manifest: list[dict] = []
    skipped: list[tuple[str, str]] = []
    n_kept = 0
    for scene_dir in all_scene_dirs:
        scene_id = scene_dir.name
        d = _scene_date(scene_id, tile_meta)
        has_oscar = _scene_inside_oscar(d, oscar_dates)
        if require_oscar and not has_oscar:
            skipped.append((scene_id, "no OSCAR coverage"))
            continue

        print(f"[build] -> {scene_id}  date={d}  oscar={has_oscar}")
        records = detect_scene(
            scene_dir=scene_dir,
            model=model,
            cfg=cfg,
            mean=mean,
            std=std,
            thresholds=thresholds,
            device=device,
            oscar=oscar if cfg.get("use_temporal") else None,
            obs_date=d,
            bands=bands,
        )

        meta = compute_scene_meta(
            scene_id=scene_id,
            records=records,
            debris_classes=debris_classes,
            obs_date=d,
            has_oscar=has_oscar,
        )

        if only_with_detections and meta["n_detections"] == 0:
            skipped.append((scene_id, "no debris detections"))
            continue

        scene_out = out_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        (scene_out / "predictions.json").write_text(json.dumps({
            "scene_id": scene_id,
            "ckpt": str(ckpt),
            "num_classes": num_classes,
            "thresholds": [float(t) for t in thresholds],
            "n_tiles": len(records),
            "records": records,
        }))

        detections_fc = build_detections_geojson(records, debris_classes)
        (scene_out / "detections.geojson").write_text(json.dumps(detections_fc))

        (scene_out / "meta.json").write_text(json.dumps(meta, indent=2))
        manifest.append(meta)
        n_kept += 1
        print(f"           tiles={meta['n_tiles']}  detections={meta['n_detections']}  "
              f"classes={meta['per_class_detections']}")

        if limit is not None and n_kept >= limit:
            print(f"[build] hit --limit={limit}, stopping")
            break

    # Manifest / index for the UI scene picker.
    manifest_sorted = sorted(
        manifest, key=lambda m: (m["obs_date"] or "", -m["n_detections"])
    )
    index_payload = {
        "generated_from_ckpt": str(ckpt),
        "class_names": list(CLASS_NAMES),
        "default_debris_classes": list(debris_classes),
        "oscar_coverage": {
            "start": min(oscar_dates).isoformat() if oscar_dates else None,
            "end": max(oscar_dates).isoformat() if oscar_dates else None,
            "n_dates": len(oscar_dates),
        },
        "n_scenes": len(manifest_sorted),
        "scenes": manifest_sorted,
    }
    (out_dir / "index.json").write_text(json.dumps(index_payload, indent=2))
    print(f"[build] wrote {len(manifest_sorted)} scenes -> {out_dir/'index.json'}")
    if skipped:
        print(f"[build] skipped {len(skipped)} scene(s):")
        for sid, reason in skipped[:20]:
            print(f"    - {sid}: {reason}")
        if len(skipped) > 20:
            print(f"    ... ({len(skipped) - 20} more)")
    return manifest_sorted


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("web/scenes"))
    parser.add_argument(
        "--debris-classes", type=int, nargs="+", default=list(DEFAULT_DEBRIS_CLASSES),
        help="Classes that count as floating trackable material (default: 0,1,2,8).",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after building N scenes (smoke test).")
    parser.add_argument("--include-empty", action="store_true",
                        help="Keep scenes that produced 0 debris detections.")
    parser.add_argument("--ignore-oscar-window", action="store_true",
                        help="Also include scenes outside the OSCAR date window. "
                             "They'll work for detection display but can't be "
                             "forecasted unless more OSCAR data is downloaded.")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Optional allowlist of scene_ids (e.g. S2_18-9-20_16PCC).")
    args = parser.parse_args()

    build(
        ckpt=args.ckpt,
        thresholds_path=args.thresholds,
        out_dir=args.out,
        debris_classes=tuple(args.debris_classes),
        limit=args.limit,
        only_with_detections=not args.include_empty,
        require_oscar=not args.ignore_oscar_window,
        scene_allowlist=tuple(args.scenes) if args.scenes else None,
    )


if __name__ == "__main__":
    main()
