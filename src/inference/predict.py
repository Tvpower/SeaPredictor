"""Run trained model on Sentinel-2 tiles and emit per-tile predictions.

Two input modes:

  --tiles <dir>     Process every .tif in <dir> as an already-cut 256x256 patch
                    (matches MARIDA's `patches/<scene>/<tile>.tif` layout).

  --scene <path>    Process a single multi-band GeoTIFF by tiling it into
                    256x256 chips with stride `--stride` (default 256, no overlap).

Outputs:

  --out <path.json>      JSON list of per-tile records (always written).
  --geojson <path>       Optional GeoJSON FeatureCollection — one polygon per
                         tile, properties carry per-class probabilities and
                         predicted labels (only meaningful with --scene since
                         we have geocoords there).

Optional knobs:
  --thresholds <json>    Use tuned thresholds (from tune_thresholds.py) instead
                         of 0.5 across the board.
  --currents-date YYYY-MM-DD
                         Date to fetch OSCAR currents for (only used if the
                         model was trained with --use-temporal). If omitted,
                         currents are zero-filled.

Usage:
    python -m src.inference.predict \
        --ckpt checkpoints/cnn_only/best.pt \
        --thresholds checkpoints/cnn_only/thresholds.json \
        --tiles data/data/raw/MARIDA/patches/S2_1-12-19_48MYU \
        --out predictions.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch

from src.dataset.marida_loader import MaridaIndex
from src.dataset.oscar_loader import OSCARLoader, default_oscar_root
from src.inference.cloud_mask import CloudFilterConfig, apply_cloud_filter
from src.models import DebrisPredictor


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def load_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict, str]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg", {})
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = DebrisPredictor(
        in_channels=cfg.get("in_channels", 11),
        seq_features=cfg.get("seq_features", 4),
        num_classes=cfg.get("num_classes", 15),
        cnn_pretrained=False,
        use_temporal=cfg.get("use_temporal", True),
        head_dropout=0.0,
    ).to(device).eval()
    model.load_state_dict(state["model"])
    return model, cfg, device


def load_norm_stats(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull MARIDA per-band mean/std (the training stats)."""
    data_root = cfg.get("data_root")
    idx = MaridaIndex.from_root(Path(data_root) if data_root else None)
    mean = idx.norm_mean.copy()
    std = idx.norm_std.copy()
    bands = cfg.get("bands")
    if bands is not None:
        sel = np.asarray(bands, dtype=int) - 1
        mean = mean[sel]
        std = std[sel]
    return mean, std


def normalize_tile(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (arr - mean[:, None, None]) / std[:, None, None]


# --------------------------------------------------------------------------- #
# Tile sources                                                                #
# --------------------------------------------------------------------------- #
def iter_dir_tiles(directory: Path, bands: list[int] | None):
    """Yield (tile_id, image_array, geotransform) for each .tif in `directory`.

    Skips files ending in `_cl.tif` / `_conf.tif` (MARIDA mask siblings).
    """
    import rasterio
    for path in sorted(directory.glob("*.tif")):
        if path.stem.endswith("_cl") or path.stem.endswith("_conf"):
            continue
        with rasterio.open(path) as src:
            if bands is None:
                arr = src.read().astype(np.float32)
            else:
                arr = src.read(bands).astype(np.float32)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
        yield path.stem, arr, {"transform": transform, "crs": str(crs) if crs else None,
                               "bounds": list(bounds)}


def iter_scene_tiles(
    scene_path: Path, bands: list[int] | None, tile_size: int, stride: int
):
    """Tile a full Sentinel-2 scene into windows. Yields (tile_id, arr, geo)."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(scene_path) as src:
        H, W = src.height, src.width
        for row in range(0, H - tile_size + 1, stride):
            for col in range(0, W - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)
                if bands is None:
                    arr = src.read(window=window).astype(np.float32)
                else:
                    arr = src.read(bands, window=window).astype(np.float32)
                # Compute polygon for this window in scene CRS
                tx = src.window_transform(window)
                xs = [0, tile_size, tile_size, 0, 0]
                ys = [0, 0, tile_size, tile_size, 0]
                coords = [tx * (x, y) for x, y in zip(xs, ys)]
                tile_id = f"{scene_path.stem}_r{row}_c{col}"
                yield tile_id, arr, {
                    "transform": tx,
                    "crs": str(src.crs) if src.crs else None,
                    "polygon_scene_crs": coords,
                }


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    seq: np.ndarray,
    device: str,
) -> np.ndarray:
    img_t = torch.from_numpy(image).unsqueeze(0).to(device)
    seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
    logits = model(img_t, seq_t)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def build_sequence(
    cfg: dict,
    oscar: OSCARLoader | None,
    obs_date: date | None,
    lat: float | None,
    lon: float | None,
) -> np.ndarray:
    seq_length = cfg.get("seq_length", 30)
    seq_features = cfg.get("seq_features", 4)
    zeros = np.zeros((seq_length, seq_features), dtype=np.float32)
    if oscar is None or obs_date is None or lat is None or lon is None:
        return zeros
    seq, _cov = oscar.get_sequence(lat=lat, lon=lon, end_date=obs_date, window=seq_length)
    if seq.shape[1] == seq_features:
        return seq
    if seq.shape[1] < seq_features:
        out = np.zeros((seq_length, seq_features), dtype=np.float32)
        out[:, : seq.shape[1]] = seq
        return out
    return seq[:, :seq_features]


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=None)
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--tiles", type=Path, help="Directory of pre-cut .tif tiles")
    src_group.add_argument("--scene", type=Path, help="Single full-scene multi-band GeoTIFF")
    parser.add_argument("--stride", type=int, default=256, help="Stride for --scene tiling")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--currents-date", type=str, default=None,
                        help="YYYY-MM-DD, used for OSCAR sequence lookup")
    parser.add_argument("--lat", type=float, default=None,
                        help="Center latitude for OSCAR lookup (single value, used for all tiles)")
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument("--geojson", type=Path, default=None)
    # Cloud filter knobs (mirror src.pipeline.build_scenes).
    parser.add_argument("--no-cloud-filter", action="store_true",
                        help="Disable post-hoc cloud suppression.")
    parser.add_argument("--cloud-vis-reflectance", type=float, default=0.18)
    parser.add_argument("--cloud-nir-reflectance", type=float, default=0.08)
    parser.add_argument("--cloud-max-frac", type=float, default=0.50)
    parser.add_argument("--debris-classes", type=int, nargs="+", default=[0, 1, 2, 8],
                        help="Classes suppressed when a tile is cloud-dominated.")
    args = parser.parse_args()

    cloud_cfg: CloudFilterConfig | None = None
    if not args.no_cloud_filter:
        cloud_cfg = CloudFilterConfig(
            vis_reflectance=args.cloud_vis_reflectance,
            nir_reflectance=args.cloud_nir_reflectance,
            max_cloud_frac=args.cloud_max_frac,
            debris_classes=tuple(args.debris_classes),
        )

    model, cfg, device = load_model(args.ckpt)
    print(f"[predict] device={device}  num_classes={cfg.get('num_classes', 15)}  "
          f"in_channels={cfg.get('in_channels', 11)}  use_temporal={cfg.get('use_temporal')}")

    bands = cfg.get("bands")
    mean, std = load_norm_stats(cfg)

    # Thresholds
    num_classes = int(cfg.get("num_classes", 15))
    if args.thresholds is not None and args.thresholds.exists():
        thr_payload = json.loads(args.thresholds.read_text())
        thresholds = np.asarray(thr_payload["thresholds"], dtype=np.float32)
        print(f"[predict] using tuned thresholds from {args.thresholds}")
    else:
        thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    # OSCAR (only if model wants temporal)
    oscar = None
    obs_date = None
    if cfg.get("use_temporal", True):
        root = default_oscar_root()
        if root is not None:
            try:
                oscar = OSCARLoader(root)
                print(f"[predict] OSCAR enabled: {len(oscar.available_dates)} files")
            except FileNotFoundError:
                oscar = None
        if args.currents_date:
            obs_date = datetime.strptime(args.currents_date, "%Y-%m-%d").date()

    # Iterate tiles
    if args.tiles is not None:
        tile_iter = iter_dir_tiles(args.tiles, bands)
    else:
        tile_iter = iter_scene_tiles(args.scene, bands, args.tile_size, args.stride)

    records = []
    features = []
    n = 0
    n_cloud_suppressed = 0
    for tile_id, arr, geo in tile_iter:
        if arr.shape[1:] != (args.tile_size, args.tile_size):
            # Skip edge fragments shorter than tile_size
            continue
        norm = normalize_tile(arr, mean, std)
        seq = build_sequence(cfg, oscar, obs_date, args.lat, args.lon)
        probs = run_inference(model, norm, seq, device)
        preds = (probs >= thresholds).astype(np.int32)

        cloud_frac = 0.0
        cloud_suppressed = False
        cloud_reason: str | None = None
        if cloud_cfg is not None:
            probs, preds, cloud_frac, cloud_suppressed, cloud_reason = apply_cloud_filter(
                probs, preds, arr, cloud_cfg
            )
            if cloud_suppressed:
                n_cloud_suppressed += 1

        rec = {
            "tile_id": tile_id,
            "probs": [float(p) for p in probs],
            "preds": [int(p) for p in preds],
            "predicted_classes": [i for i, p in enumerate(preds) if p == 1],
            "cloud_fraction": cloud_frac,
            "cloud_suppressed": cloud_suppressed,
            "cloud_suppressed_reason": cloud_reason,
            "geo": {k: v for k, v in geo.items() if k != "transform"},
        }
        records.append(rec)

        if args.geojson is not None and "polygon_scene_crs" in geo:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[float(x), float(y)] for x, y in geo["polygon_scene_crs"]]],
                },
                "properties": {
                    "tile_id": tile_id,
                    "probs": rec["probs"],
                    "preds": rec["preds"],
                    "predicted_classes": rec["predicted_classes"],
                    "crs": geo.get("crs"),
                },
            })

        n += 1
        if n % 50 == 0:
            print(f"[predict] processed {n} tiles...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "n_tiles": n,
        "n_cloud_suppressed": n_cloud_suppressed,
        "num_classes": num_classes,
        "thresholds": [float(t) for t in thresholds],
        "records": records,
    }, indent=2))
    print(f"[predict] wrote {n} tile records ({n_cloud_suppressed} cloud-suppressed) "
          f"-> {args.out}")

    if args.geojson is not None and features:
        args.geojson.parent.mkdir(parents=True, exist_ok=True)
        args.geojson.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": features,
        }, indent=2))
        print(f"[predict] wrote {len(features)} polygons -> {args.geojson}")


if __name__ == "__main__":
    main()
