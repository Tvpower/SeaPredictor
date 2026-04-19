"""NOAA OSCAR daily current loader.

Layout on disk:

    <root>/oscar_currents_interim_YYYYMMDD.nc
        Coords:  lat (-89.75..89.75, 0.25 deg), lon (0..359.75, 0.25 deg)
        Vars:    u, v   (total surface current, m/s)
                 ug, vg (geostrophic component; we don't use these)

Two awkward facts the loader handles for you:

1. The `time` coord is `cftime.DatetimeJulian`, which doesn't convert cleanly
   to numpy/pandas datetimes. We sidestep it entirely by keying off the
   filename (`...YYYYMMDD.nc`) and opening files with `decode_times=False`.

2. OSCAR uses 0..360 longitude; MARIDA stores -180..180. We convert at lookup
   time, not on the dataset.

Per-tile sequence shape:  (window, 4) = [u, v, lat_norm, lon_norm]
where lat_norm = lat/90, lon_norm = ((lon + 180) % 360 - 180)/180.
lat/lon are constant across timesteps for a given tile (anchors the LSTM).
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np


_OSCAR_NAME_RE = re.compile(r"oscar_currents_interim_(\d{8})\.nc$")

_CANDIDATE_ROOTS = (
    "data/data/raw/oscar",
    "data/raw/oscar",
)


def default_oscar_root() -> Path | None:
    """Return the first existing OSCAR directory with at least one .nc file, or None."""
    env = os.environ.get("OSCAR_ROOT")
    if env:
        p = Path(env)
        if any(p.glob("oscar_currents_interim_*.nc")):
            return p
        return None
    for c in _CANDIDATE_ROOTS:
        p = Path(c)
        if p.is_dir() and any(p.glob("oscar_currents_interim_*.nc")):
            return p
    return None


def _date_from_filename(path: Path) -> date | None:
    m = _OSCAR_NAME_RE.search(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()


# --------------------------------------------------------------------------- #
# Loader                                                                      #
# --------------------------------------------------------------------------- #
class OSCARLoader:
    """Lazy, single-process loader for OSCAR daily NetCDFs.

    Use one instance per DataLoader worker. With `num_workers=0` (the default
    on MPS), one instance lives for the full training run and the LRU cache
    on day-file reads gives you near-free repeats across overlapping windows.
    """

    GRID_RES_DEG = 0.25
    FEATURE_DIM = 4  # [u, v, lat_norm, lon_norm]

    def __init__(self, root: Path | str | None = None, cache_size: int = 64) -> None:
        try:
            import xarray  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "xarray + netCDF4 required for OSCAR. Run: "
                "pip install xarray netCDF4"
            ) from e

        if root is None:
            root_p = default_oscar_root()
            if root_p is None:
                raise FileNotFoundError(
                    "Could not locate OSCAR data. Set OSCAR_ROOT or place files in "
                    + " or ".join(_CANDIDATE_ROOTS)
                )
            root = root_p
        self.root = Path(root)

        self.date_to_path: dict[date, Path] = {}
        for p in self.root.glob("oscar_currents_interim_*.nc"):
            d = _date_from_filename(p)
            if d is not None:
                self.date_to_path[d] = p

        if not self.date_to_path:
            raise FileNotFoundError(f"No OSCAR daily files in {self.root}")

        self.available_dates = sorted(self.date_to_path)
        self.min_date = self.available_dates[0]
        self.max_date = self.available_dates[-1]

        # Grid is identical across files; cache lat/lon vectors once.
        first_path = self.date_to_path[self.min_date]
        self._lat_grid, self._lon_grid = self._read_grid(first_path)

        # Per-instance LRU on file-read so the LSTM gets fast windows.
        self._read_uv_cached = lru_cache(maxsize=cache_size)(self._read_uv_uncached)

    # ---- low-level reads ------------------------------------------------- #
    @staticmethod
    def _read_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
        import xarray as xr
        with xr.open_dataset(path, decode_times=False) as ds:
            lat = np.asarray(ds["lat"].values, dtype=np.float64)
            lon = np.asarray(ds["lon"].values, dtype=np.float64)
        return lat, lon

    def _read_uv_uncached(self, day: date) -> np.ndarray | None:
        """Return (2, n_lon, n_lat) array of [u, v] for `day`, or None if missing."""
        path = self.date_to_path.get(day)
        if path is None:
            return None
        import xarray as xr
        with xr.open_dataset(path, decode_times=False) as ds:
            u = np.asarray(ds["u"].values, dtype=np.float32)
            v = np.asarray(ds["v"].values, dtype=np.float32)
        # File shape: (time=1, longitude, latitude); drop the time dim.
        u = u[0]
        v = v[0]
        out = np.stack([u, v], axis=0)
        # OSCAR has NaN over land; replace with 0 so the LSTM doesn't ingest NaN.
        return np.nan_to_num(out, nan=0.0)

    # ---- coord conversion ------------------------------------------------ #
    def _grid_index(self, lat: float, lon: float) -> tuple[int, int]:
        """Nearest grid index for (lat, lon). lon is converted from -180..180 to 0..360."""
        lon360 = lon % 360.0
        lat_idx = int(np.argmin(np.abs(self._lat_grid - lat)))
        lon_idx = int(np.argmin(np.abs(self._lon_grid - lon360)))
        return lon_idx, lat_idx

    # ---- public API ------------------------------------------------------ #
    def get_sequence(
        self,
        lat: float,
        lon: float,
        end_date: date,
        window: int = 30,
    ) -> tuple[np.ndarray, float]:
        """Return (sequence, coverage) for a tile.

        Args:
            lat: tile center latitude in -90..90.
            lon: tile center longitude in -180..180 (or 0..360 — we mod it).
            end_date: tile observation date (inclusive end of the window).
            window: number of days back from end_date.

        Returns:
            sequence: (window, 4) float32 = [u, v, lat/90, lon_signed/180]
                      Days outside OSCAR coverage are zero-filled (u=v=0).
                      Missing days INSIDE coverage are forward-filled from the
                      most recent available day in the window.
            coverage: fraction of timesteps that came from a real OSCAR file
                      (vs zero-fill). Useful for monitoring the train/val split.
        """
        lon_idx, lat_idx = self._grid_index(lat, lon)
        seq = np.zeros((window, self.FEATURE_DIM), dtype=np.float32)

        # Constant lat/lon channels (LSTM positional anchor).
        lon_signed = ((lon + 180.0) % 360.0) - 180.0
        seq[:, 2] = lat / 90.0
        seq[:, 3] = lon_signed / 180.0

        n_real = 0
        last_uv: np.ndarray | None = None
        for i in range(window):
            day = end_date - timedelta(days=window - 1 - i)
            uv_grid = self._read_uv_cached(day)
            if uv_grid is not None:
                u = float(uv_grid[0, lon_idx, lat_idx])
                v = float(uv_grid[1, lon_idx, lat_idx])
                last_uv = np.array([u, v], dtype=np.float32)
                seq[i, 0] = u
                seq[i, 1] = v
                n_real += 1
            elif last_uv is not None:
                # Forward-fill missing day from previous valid day in this window.
                seq[i, 0] = last_uv[0]
                seq[i, 1] = last_uv[1]
            # else: leave zeros (no prior data yet in this window)

        coverage = n_real / window
        return seq, coverage

    def __repr__(self) -> str:
        return (
            f"OSCARLoader(root={self.root!s}, "
            f"days={len(self.available_dates)}, "
            f"range={self.min_date}..{self.max_date})"
        )
