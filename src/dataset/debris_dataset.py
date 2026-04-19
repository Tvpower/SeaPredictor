"""Master Dataset for the debris predictor.

Two modes:

* `DebrisDataset` — real MARIDA tiles via `MaridaIndex`, with norm_stats
  normalization, official splits, multi-label tile-level targets (15-dim).
  When OSCAR sequences land, swap the zero-fill in `_load_sequence` for the
  real loader.

* `SyntheticDebrisDataset` — random tensors with shapes that match the real
  pipeline. Used for smoke-testing the training loop.

Both yield:
    image : (NUM_BANDS, 256, 256) float32
    seq   : (T, F)               float32
    label : (NUM_CLASSES,)       float32 in {0, 1}
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import augment_image_only
from .marida_loader import (
    NUM_BANDS,
    NUM_CLASSES,
    TILE_SIZE,
    MaridaIndex,
    TileRecord,
)
from .oscar_loader import OSCARLoader, default_oscar_root


Split = Literal["train", "val", "test"]


# --------------------------------------------------------------------------- #
# Real dataset                                                                #
# --------------------------------------------------------------------------- #
class DebrisDataset(Dataset):
    """Reads MARIDA tiles + (eventually) OSCAR sequences from disk.

    Args:
        data_root: MARIDA root (e.g. `data/data/raw/MARIDA`). If None, auto-detected.
        split: "train" | "val" | "test" — uses MARIDA's official splits.
        seq_length: temporal window length to emit (currently zero-filled).
        seq_features: features per timestep (4 OSCAR-only, 6 with HYCOM).
        bands: optional subset of bands to load (1-indexed, rasterio-style).
            Default loads all 11 bands.
        return_mask: if True, also returns the per-pixel `_cl.tif` mask. Used
            once the U-Net decoder lands.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: Split = "train",
        seq_length: int = 30,
        seq_features: int = 4,
        bands: list[int] | None = None,
        return_mask: bool = False,
        oscar_root: str | Path | None | bool = None,
        augment: bool = False,
        augment_noise_std: float = 0.0,
    ) -> None:
        """
        Args:
            ...
            oscar_root: where OSCAR daily NetCDFs live.
                * None (default) — auto-detect; if not found, zero-fill sequences.
                * Path/str       — use this root explicitly (raises if missing).
                * False          — disable OSCAR even if data exists; pure zero-fill.
        """
        super().__init__()
        # Lazy import so synthetic-only runs don't need rasterio.
        try:
            import rasterio  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "rasterio is required for real MARIDA loading. "
                "Run: pip install rasterio"
            ) from e

        self.index = MaridaIndex.from_root(data_root)
        self.split = split
        self.seq_length = seq_length
        self.seq_features = seq_features
        self.bands = bands  # None = all bands
        self.return_mask = return_mask
        self.augment = augment
        self.augment_noise_std = augment_noise_std
        if return_mask and augment:
            # Tile-level augment isn't aligned with the per-pixel mask.
            # Fall back to no-op + warn rather than silently corrupting masks.
            raise ValueError(
                "augment=True is image-only; combine with return_mask=False or "
                "use augment_patch() from src.dataset.augmentation for segmentation."
            )

        self.records: list[TileRecord] = self.index.split_records(split)
        if not self.records:
            raise RuntimeError(
                f"No tiles found for split={split!r} under {self.index.root}. "
                f"Check that splits/ and patches/ are populated."
            )

        # Wire up OSCAR if available / requested.
        self.oscar: OSCARLoader | None = None
        if oscar_root is False:
            pass
        else:
            try:
                self.oscar = OSCARLoader(
                    None if oscar_root is None else oscar_root
                )
            except FileNotFoundError:
                if oscar_root is not None:
                    raise
                # Default behavior: silently fall back to zero-fill.
                self.oscar = None
        # Rolling coverage stats so train.py can log how often OSCAR fired.
        self._n_seq_calls = 0
        self._coverage_sum = 0.0

    def __len__(self) -> int:
        return len(self.records)

    # ---- helpers ---------------------------------------------------------- #
    def _load_image(self, path: Path) -> np.ndarray:
        import rasterio
        with rasterio.open(path) as src:
            if self.bands is None:
                arr = src.read().astype(np.float32)  # (C, H, W)
            else:
                arr = src.read(self.bands).astype(np.float32)

        # Replace NaNs/Infs (some MARIDA patches have masked nodata)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize per-band using norm_stats (only first NUM_BANDS entries used)
        # If a band subset was requested, slice the stats to match.
        mean = self.index.norm_mean
        std = self.index.norm_std
        if self.bands is not None:
            idx = np.array(self.bands, dtype=int) - 1  # rasterio is 1-indexed
            mean = mean[idx]
            std = std[idx]
        arr = (arr - mean[:, None, None]) / std[:, None, None]
        return arr

    def _load_mask(self, path: Path) -> np.ndarray:
        import rasterio
        with rasterio.open(path) as src:
            m = src.read(1).astype(np.int64)
        return m  # (H, W), values in 0..NUM_CLASSES

    def _load_sequence(self, record: TileRecord) -> np.ndarray:
        """Pull OSCAR sequence if available; fall back to zeros otherwise.

        Tracks coverage stats on the dataset for logging.
        """
        zeros = np.zeros((self.seq_length, self.seq_features), dtype=np.float32)

        if self.oscar is None or record.obs_date is None or record.lat is None:
            self._n_seq_calls += 1
            return zeros

        seq, coverage = self.oscar.get_sequence(
            lat=record.lat,
            lon=record.lon,  # type: ignore[arg-type]
            end_date=record.obs_date,
            window=self.seq_length,
        )
        self._n_seq_calls += 1
        self._coverage_sum += coverage

        # OSCAR currently emits 4 features [u, v, lat, lon].
        # If a caller wants 6 features (HYCOM SST+salinity), pad with zeros.
        if seq.shape[1] == self.seq_features:
            return seq
        if seq.shape[1] < self.seq_features:
            padded = np.zeros((self.seq_length, self.seq_features), dtype=np.float32)
            padded[:, : seq.shape[1]] = seq
            return padded
        return seq[:, : self.seq_features]

    @property
    def mean_oscar_coverage(self) -> float:
        """Fraction of timesteps backed by real OSCAR data, averaged across calls."""
        if self._n_seq_calls == 0:
            return 0.0
        return self._coverage_sum / self._n_seq_calls

    # ---- protocol --------------------------------------------------------- #
    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = self._load_image(rec.image_path)
        seq = self._load_sequence(rec)
        label = rec.label  # (NUM_CLASSES,)

        image_t = torch.from_numpy(image)
        seq_t = torch.from_numpy(seq)
        label_t = torch.from_numpy(label)

        if self.augment:
            image_t = augment_image_only(image_t, noise_std=self.augment_noise_std)

        if self.return_mask:
            mask = self._load_mask(rec.mask_path)
            return image_t, seq_t, label_t, torch.from_numpy(mask)
        return image_t, seq_t, label_t


# --------------------------------------------------------------------------- #
# Synthetic dataset (no I/O, no rasterio)                                     #
# --------------------------------------------------------------------------- #
class SyntheticDebrisDataset(Dataset):
    """Random-tensor stand-in matching the real pipeline's shapes.

    Generates a multi-label vector that's correlated with the input so loss
    actually decreases during smoke tests.
    """

    def __init__(
        self,
        n_samples: int = 256,
        image_size: int = TILE_SIZE,
        in_channels: int = NUM_BANDS,
        seq_length: int = 30,
        seq_features: int = 4,
        num_classes: int = NUM_CLASSES,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n = n_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.seq_features = seq_features
        self.num_classes = num_classes
        self._rng = np.random.default_rng(seed)
        self._labels = self._rng.integers(0, 2, size=(n_samples, num_classes)).astype(
            np.float32
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label = self._labels[idx]
        rng = np.random.default_rng(idx + 1)

        image = rng.standard_normal(
            (self.in_channels, self.image_size, self.image_size)
        ).astype(np.float32)
        # Per-channel bias proportional to label[i] gives the model a learnable signal.
        bias_channels = min(self.in_channels, self.num_classes)
        image[:bias_channels] += label[:bias_channels, None, None] * 0.5

        seq = rng.standard_normal((self.seq_length, self.seq_features)).astype(np.float32)
        seq[:, 0] += float(label[0]) * 0.3

        return (
            torch.from_numpy(image),
            torch.from_numpy(seq),
            torch.from_numpy(label),
        )
