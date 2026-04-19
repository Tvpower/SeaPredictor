"""MARIDA disk-layout helpers.

The MARIDA distribution provides:

    <root>/
      patches/<scene>/<tile>.tif         11-band Sentinel-2 patch (256x256, float32)
      patches/<scene>/<tile>_cl.tif      per-pixel class mask (values 0..15)
      patches/<scene>/<tile>_conf.tif    per-pixel confidence mask (0/1)
      splits/{train,val,test}_X.txt      tile IDs WITHOUT the `S2_` prefix or `.tif`
      labels_mapping.txt                 JSON {"<tile.tif>": [15-elem 0/1 vector]}
      class_weights.json                 11-elem (collapsed scheme; see notes)
      norm_stats.json                    {"mean":[19], "std":[19]}  (11 bands + 8 indices)

Two layouts are common in this repo:
    1) data/raw/marida/...        (gitignored placeholder)
    2) data/data/raw/MARIDA/...   (where the user actually unpacked it)

`MaridaIndex.from_root` accepts either; `default_marida_root()` finds the right
one if you don't pass one explicitly.

NOTE on class counts:
    `labels_mapping.txt` uses the original 15-class scheme (matches `_cl.tif`
    pixel values 1..15, with 0 = background). `class_weights.json` corresponds
    to the official 11-class collapsed scheme used for trained baselines. We
    have NOT implemented the 15->11 merge yet, so multi-label training currently
    runs with 15 outputs and no per-class weighting (all classes are roughly
    O(50-900) positive samples, so uniform pos_weight is acceptable to start).
    TODO: pull the official 15->11 mapping from the marine-debris repo and
    expose `merge_15_to_11()` here.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


NUM_CLASSES = 15  # length of labels_mapping vectors / unique non-zero values in _cl.tif
NUM_BANDS = 11    # Sentinel-2 bands per patch (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
TILE_SIZE = 256

Split = Literal["train", "val", "test"]


# --------------------------------------------------------------------------- #
# Path discovery                                                              #
# --------------------------------------------------------------------------- #
_CANDIDATE_ROOTS = (
    "data/data/raw/MARIDA",  # observed layout in this repo
    "data/raw/marida",       # original spec layout
)


def default_marida_root() -> Path:
    """Return the first existing MARIDA root, or raise."""
    env = os.environ.get("MARIDA_ROOT")
    if env:
        p = Path(env)
        if (p / "patches").is_dir():
            return p
        raise FileNotFoundError(f"MARIDA_ROOT={env} has no `patches/` dir")

    for c in _CANDIDATE_ROOTS:
        p = Path(c)
        if (p / "patches").is_dir() and (p / "splits").is_dir():
            return p

    raise FileNotFoundError(
        "Could not locate MARIDA. Looked in: "
        + ", ".join(_CANDIDATE_ROOTS)
        + ". Set MARIDA_ROOT or pass --data-root."
    )


# --------------------------------------------------------------------------- #
# Tile records + index                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TileRecord:
    tile_id: str          # e.g. "S2_1-12-19_48MYU_0"
    scene_id: str         # e.g. "S2_1-12-19_48MYU"  (= tile minus trailing _<n>)
    image_path: Path
    mask_path: Path
    conf_path: Path
    label: np.ndarray     # shape (NUM_CLASSES,), float32 in {0,1}
    obs_date: date | None = None  # tile observation date, from tile_index.csv
    lat: float | None = None      # WGS84 center latitude
    lon: float | None = None      # WGS84 center longitude (-180..180)


def _resolve_tile_id(raw: str) -> str:
    """Splits store IDs like '12-12-20_16PCC_0' (no 'S2_' prefix)."""
    raw = raw.strip()
    if not raw:
        return raw
    return raw if raw.startswith("S2_") else f"S2_{raw}"


def _scene_of(tile_id: str) -> str:
    # Strip the trailing "_<patch_index>"
    return tile_id.rsplit("_", 1)[0]


def _read_split_ids(splits_dir: Path, split: Split) -> list[str]:
    fname = {"train": "train_X.txt", "val": "val_X.txt", "test": "test_X.txt"}[split]
    path = splits_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing MARIDA split file: {path}")
    with path.open() as f:
        return [_resolve_tile_id(line) for line in f if line.strip()]


class MaridaIndex:
    """In-memory index of MARIDA tiles with their official splits + labels."""

    def __init__(
        self,
        root: Path,
        labels: dict[str, list[int]],
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        class_weights: np.ndarray | None,
        tile_meta: dict[str, dict] | None = None,
    ) -> None:
        self.root = root
        self._labels = labels
        # We only use the first NUM_BANDS entries (raw Sentinel-2 bands); the
        # remaining 8 entries in norm_stats are derived indices we don't load.
        self.norm_mean = norm_mean[:NUM_BANDS].astype(np.float32)
        self.norm_std = norm_std[:NUM_BANDS].astype(np.float32)
        # Avoid divide-by-zero downstream
        self.norm_std = np.where(self.norm_std < 1e-8, 1.0, self.norm_std)
        self.class_weights = class_weights  # may be None or length-11 (collapsed scheme)
        self._tile_meta = tile_meta or {}

    # ---- factory ---------------------------------------------------------- #
    @classmethod
    def from_root(cls, root: Path | str | None = None) -> "MaridaIndex":
        root = Path(root) if root is not None else default_marida_root()
        if not (root / "patches").is_dir():
            raise FileNotFoundError(f"{root}/patches does not exist")

        with (root / "labels_mapping.txt").open() as f:
            labels = json.load(f)

        with (root / "norm_stats.json").open() as f:
            stats = json.load(f)
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

        weights_path = root / "class_weights.json"
        if weights_path.exists():
            with weights_path.open() as f:
                class_weights = np.asarray(json.load(f), dtype=np.float32)
        else:
            class_weights = None

        tile_meta = _load_tile_index(root / "tile_index.csv")

        return cls(root, labels, mean, std, class_weights, tile_meta)

    # ---- queries ---------------------------------------------------------- #
    def split_records(self, split: Split) -> list[TileRecord]:
        ids = _read_split_ids(self.root / "splits", split)
        out: list[TileRecord] = []
        for tile_id in ids:
            scene_id = _scene_of(tile_id)
            scene_dir = self.root / "patches" / scene_id
            image = scene_dir / f"{tile_id}.tif"
            mask = scene_dir / f"{tile_id}_cl.tif"
            conf = scene_dir / f"{tile_id}_conf.tif"
            if not image.exists():
                # Some splits reference tiles whose patches weren't downloaded
                # (or were filtered). Skip silently rather than crash training.
                continue
            label_vec = self._labels.get(image.name)
            if label_vec is None:
                continue
            label = np.asarray(label_vec, dtype=np.float32)
            if label.shape != (NUM_CLASSES,):
                raise ValueError(
                    f"Unexpected label shape {label.shape} for {image.name}"
                )
            meta = self._tile_meta.get(tile_id, {})
            out.append(
                TileRecord(
                    tile_id=tile_id,
                    scene_id=scene_id,
                    image_path=image,
                    mask_path=mask,
                    conf_path=conf,
                    label=label,
                    obs_date=meta.get("date"),
                    lat=meta.get("lat"),
                    lon=meta.get("lon"),
                )
            )
        return out

    def split_label_matrix(self, split: Split) -> np.ndarray:
        """(N, 15) binary matrix of labels for a split. Useful for pos_weight."""
        return np.stack([r.label for r in self.split_records(split)], axis=0)


# --------------------------------------------------------------------------- #
# tile_index.csv helper                                                       #
# --------------------------------------------------------------------------- #
def _load_tile_index(path: Path) -> dict[str, dict]:
    """Read MARIDA's tile_index.csv into {tile_id: {date, lat, lon}}.

    The CSV is generated by `data/test.py` with columns:
        tile_id, date, lat, lon, scene, path
    Returns an empty dict if the CSV is absent (loader still works without
    OSCAR temporal context).
    """
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
                out[row["tile_id"]] = {
                    "date": d,
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
            except (ValueError, KeyError):
                continue
    return out


# --------------------------------------------------------------------------- #
# pos_weight helper                                                           #
# --------------------------------------------------------------------------- #
def compute_pos_weight(label_matrix: np.ndarray, clip: float = 50.0) -> np.ndarray:
    """For BCEWithLogitsLoss(pos_weight=...) on multi-label targets.

    pos_weight = #negatives / #positives, per class. Clipped to avoid extreme
    values for ultra-rare classes.
    """
    n_pos = label_matrix.sum(axis=0)
    n_neg = label_matrix.shape[0] - n_pos
    # Avoid div-by-zero for classes with 0 positives in a given split.
    weight = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), 1.0)
    return np.clip(weight, 1.0, clip).astype(np.float32)
