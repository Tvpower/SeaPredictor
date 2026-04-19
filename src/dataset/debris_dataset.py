"""Master Dataset for the debris predictor.

Two modes:

* `DebrisDataset` — real MARIDA / MADOS tiles paired with OSCAR (and optional HYCOM)
  sequences. The per-source loaders are intentionally stubbed so the training loop
  can be wired up first; fill them in during Phase 1.

* `SyntheticDebrisDataset` — pure-Python random tensors with shapes that match
  the real pipeline. Used for smoke-testing the training loop and CI.

Both yield the same triple:
    (image: (3, 256, 256), seq: (T, F), label: scalar 0/1)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]


# --------------------------------------------------------------------------- #
# Tile metadata + loader stubs                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TileRecord:
    scene_id: str
    image_path: Path
    mask_path: Path
    lat: float
    lon: float
    date: str  # ISO yyyy-mm-dd


def _load_marida_index(root: Path) -> list[TileRecord]:
    """TODO: parse MARIDA splits/*.txt + GeoTIFF headers into TileRecords."""
    return []


def _load_mados_index(root: Path) -> list[TileRecord]:
    """TODO: parse MADOS index into TileRecords."""
    return []


def _split_by_scene(tiles: list[TileRecord], split: Split, seed: int = 42) -> list[TileRecord]:
    """Group by scene_id, then deterministically allocate scenes 70/15/15."""
    if not tiles:
        return []
    scenes = sorted({t.scene_id for t in tiles})
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = set(scenes[:n_train])
    val = set(scenes[n_train : n_train + n_val])
    test = set(scenes[n_train + n_val :])
    bucket = {"train": train, "val": val, "test": test}[split]
    return [t for t in tiles if t.scene_id in bucket]


# --------------------------------------------------------------------------- #
# Real dataset                                                                #
# --------------------------------------------------------------------------- #
class DebrisDataset(Dataset):
    """Reads MARIDA/MADOS tiles + OSCAR sequences from `data/raw/`.

    Currently the per-source loaders are stubs; instantiating this on an empty
    `data/raw/` will raise. Use `SyntheticDebrisDataset` for smoke testing.
    """

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        split: Split = "train",
        use_hycom: bool = False,
        seq_length: int = 30,
        label_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.split = split
        self.use_hycom = use_hycom
        self.seq_length = seq_length
        self.label_threshold = label_threshold

        marida = _load_marida_index(self.root / "marida")
        mados = _load_mados_index(self.root / "mados")
        all_tiles = marida + mados
        self.tiles = _split_by_scene(all_tiles, split)

        if not self.tiles:
            raise RuntimeError(
                f"No tiles found under {self.root!s}. Either run the data download "
                f"scripts or use SyntheticDebrisDataset for testing."
            )

        # TODO: wire OSCAR / HYCOM loaders here.
        # self.oscar = OSCARLoader(self.root / "oscar")
        # self.hycom = HYCOMLoader(self.root / "hycom") if use_hycom else None

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Hook this up to the real loaders (rasterio + xarray) in Phase 1."
        )


# --------------------------------------------------------------------------- #
# Synthetic dataset (works today, no downloads required)                      #
# --------------------------------------------------------------------------- #
class SyntheticDebrisDataset(Dataset):
    """Random-tensor stand-in matching the real pipeline's shapes.

    Generates a debris label that is correlated with a known feature so the
    smoke-test sees loss actually decrease.
    """

    def __init__(
        self,
        n_samples: int = 256,
        image_size: int = 256,
        in_channels: int = 3,
        seq_length: int = 30,
        seq_features: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n = n_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.seq_features = seq_features
        self._rng = np.random.default_rng(seed)

        # Pre-generate so the same sample is returned for the same idx (epoch-stable).
        self._labels = self._rng.integers(0, 2, size=n_samples).astype(np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label = float(self._labels[idx])
        rng = np.random.default_rng(idx + 1)

        # Image: positive samples get a brighter NIR-like channel signature.
        image = rng.standard_normal(
            (self.in_channels, self.image_size, self.image_size)
        ).astype(np.float32)
        if label > 0.5:
            image[1] += 0.5  # bias band B8 (NIR) upward, mimicking debris reflectance

        # Sequence: positive samples get a DC offset on u-current.
        seq = rng.standard_normal((self.seq_length, self.seq_features)).astype(np.float32)
        if label > 0.5:
            seq[:, 0] += 0.3

        return (
            torch.from_numpy(image),
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.float32),
        )
