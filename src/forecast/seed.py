"""Turn predictions.json from src.inference.predict into drift seeds.

For each tile in the predictions file:
  * compute the centroid of `geo.bounds` in the tile's CRS
  * reproject to WGS84 lat/lon
  * keep only tiles whose `predicted_classes` intersect `--debris-classes`
  * resolve an observation date for each tile (tile_index.csv lookup, or CLI default)

Output: list of `Seed(tile_id, lat, lon, obs_date, classes, prob_max)`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from src.dataset.marida_loader import MaridaIndex


# Default "things worth tracking" for MARIDA's 15-class scheme. Class 0 is
# Marine Debris itself; 1/2 are Sargassum (also floating); 8 is Foam.
DEFAULT_DEBRIS_CLASSES: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class Seed:
    tile_id: str
    lat: float
    lon: float
    obs_date: date
    matched_classes: tuple[int, ...]
    max_prob: float


def _bounds_centroid_to_wgs84(
    bounds: list[float], src_crs: str | None
) -> tuple[float, float] | None:
    """`bounds` = [minx, miny, maxx, maxy] in src_crs. Returns (lat, lon) or None."""
    if src_crs is None or len(bounds) != 4:
        return None
    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    if src_crs.upper() in {"EPSG:4326", "WGS84"}:
        return float(cy), float(cx)
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError("pyproj is required for CRS reprojection") from e
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return float(lat), float(lon)


def _resolve_dates(
    tile_ids: Iterable[str],
    default_date: date | None,
    override: bool = False,
) -> dict[str, date]:
    """Look up obs_date per tile_id.

    If `override=True` and `default_date` is set, every tile gets `default_date`.
    Otherwise we first try MARIDA's tile_index.csv, then fall back to
    `default_date`, then drop the tile.
    """
    if override and default_date is not None:
        return {tid: default_date for tid in tile_ids}
    out: dict[str, date] = {}
    try:
        idx = MaridaIndex.from_root()
    except FileNotFoundError:
        idx = None
    for tid in tile_ids:
        d: date | None = None
        if idx is not None:
            meta = idx._tile_meta.get(tid)  # noqa: SLF001 (intentional internal access)
            if meta:
                d = meta.get("date")
        if d is None:
            d = default_date
        if d is None:
            continue
        out[tid] = d
    return out


def extract_seeds(
    predictions_path: Path,
    debris_classes: Iterable[int] = DEFAULT_DEBRIS_CLASSES,
    default_date: date | None = None,
    min_prob: float = 0.0,
    override_date: bool = False,
) -> list[Seed]:
    """Build Seed list from a predictions.json on disk.

    Args:
        predictions_path: file produced by `src.inference.predict`.
        debris_classes: only tiles with at least one of these classes are kept.
        default_date: fallback obs_date when tile_index.csv has no entry.
        min_prob: also require the max-prob over `debris_classes` to be at
            least this. Useful for filtering low-confidence detections that
            slipped past the per-class threshold.
        override_date: if True, force `default_date` onto every tile (ignore
            the tile_index lookup). Demo-friendly when MARIDA scenes fall
            outside your OSCAR window.
    """
    payload = json.loads(predictions_path.read_text())
    debris_set = set(debris_classes)
    records = payload.get("records", [])
    if not records:
        return []

    dates = _resolve_dates(
        (r["tile_id"] for r in records), default_date, override=override_date
    )
    seeds: list[Seed] = []
    for rec in records:
        tile_id = rec["tile_id"]
        preds = rec.get("preds", [])
        probs = rec.get("probs", [])
        matched = tuple(c for c in debris_set if c < len(preds) and preds[c] == 1)
        if not matched:
            continue
        max_prob = max((probs[c] for c in matched), default=0.0)
        if max_prob < min_prob:
            continue
        geo = rec.get("geo", {})
        bounds = geo.get("bounds")
        crs = geo.get("crs")
        if bounds is None or crs is None:
            continue
        latlon = _bounds_centroid_to_wgs84(bounds, crs)
        if latlon is None:
            continue
        obs_date = dates.get(tile_id)
        if obs_date is None:
            continue
        seeds.append(Seed(
            tile_id=tile_id,
            lat=latlon[0],
            lon=latlon[1],
            obs_date=obs_date,
            matched_classes=matched,
            max_prob=float(max_prob),
        ))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--debris-classes", type=int, nargs="+", default=list(DEFAULT_DEBRIS_CLASSES),
        help="Class indices counted as floating debris (default: 0 = Marine Debris)",
    )
    parser.add_argument("--default-date", type=str, default=None,
                        help="YYYY-MM-DD fallback date when tile_index.csv lacks the tile")
    parser.add_argument("--override-date", action="store_true",
                        help="Force --default-date onto ALL tiles (demo mode)")
    parser.add_argument("--min-prob", type=float, default=0.0)
    args = parser.parse_args()

    default_date = (
        datetime.strptime(args.default_date, "%Y-%m-%d").date()
        if args.default_date else None
    )
    seeds = extract_seeds(
        args.predictions,
        debris_classes=args.debris_classes,
        default_date=default_date,
        min_prob=args.min_prob,
        override_date=args.override_date,
    )
    print(f"[seed] {len(seeds)} seeds extracted "
          f"(debris_classes={args.debris_classes})")
    for s in seeds[:20]:
        print(
            f"  {s.tile_id}  lat={s.lat:.4f}  lon={s.lon:.4f}  "
            f"date={s.obs_date}  classes={s.matched_classes}  p={s.max_prob:.3f}"
        )
    if len(seeds) > 20:
        print(f"  ... ({len(seeds) - 20} more)")


if __name__ == "__main__":
    main()
