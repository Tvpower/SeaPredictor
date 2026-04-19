"""Download a regional GLORYS12V1 surface-current subset for OpenDrift.

GLORYS12V1 = Mercator Ocean global ocean reanalysis at 1/12 deg, daily means.
Unlike OSCAR, the `uo`/`vo` fields are *total* surface currents (geostrophic +
Ekman + tide residual), so floating-debris drift no longer needs the wind
leeway term to be doing all the work.

Auth: requires Copernicus Marine credentials saved via
    `copernicusmarine login`
or env vars `COPERNICUSMARINE_SERVICE_USERNAME` / `..._PASSWORD`.

Output schema mirrors `src/forecast/oscar_concat.py`:
    dims    = (time, latitude, longitude)
    vars    = u (eastward_sea_water_velocity), v (northward_sea_water_velocity)
    units   = m s-1
    lon     = -180..180 ascending
    lat     = -90..90 ascending
so it is a drop-in for OpenDrift's `reader_netCDF_CF_generic`.

Usage:
    python -m src.forecast.glorys_fetch \\
        --bbox -10 35 5 45 \\
        --start 2018-04-01 --end 2018-04-15
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np


# GLORYS12V1 reanalysis: daily means, 1/12 deg, 1993..2021-06.
# Surface drift only needs the topmost depth level (~0.494 m).
GLORYS_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
GLORYS_VARIABLES = ("uo", "vo")
SURFACE_DEPTH_M = 0.494  # GLORYS12 top level


DEFAULT_CACHE_DIR = Path("data/forecast/glorys")


@dataclass(frozen=True)
class Bbox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def buffered(self, deg: float) -> "Bbox":
        return Bbox(
            min_lon=max(-180.0, self.min_lon - deg),
            min_lat=max(-90.0, self.min_lat - deg),
            max_lon=min(180.0, self.max_lon + deg),
            max_lat=min(90.0, self.max_lat + deg),
        )

    def cache_token(self) -> str:
        return (
            f"{self.min_lon:+07.2f}_{self.max_lon:+07.2f}"
            f"_{self.min_lat:+06.2f}_{self.max_lat:+06.2f}"
        )


def bbox_from_seeds(
    lats: list[float], lons: list[float], buffer_deg: float = 5.0
) -> Bbox:
    """Tight bbox around seed positions, padded by `buffer_deg` for drift room."""
    if not lats or not lons:
        raise ValueError("Need at least one seed lat/lon to build a bbox")
    raw = Bbox(
        min_lon=float(min(lons)),
        min_lat=float(min(lats)),
        max_lon=float(max(lons)),
        max_lat=float(max(lats)),
    )
    return raw.buffered(buffer_deg)


def _expected_path(bbox: Bbox, start: date, end: date, cache_dir: Path) -> Path:
    name = (
        f"glorys12_{bbox.cache_token()}_"
        f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.nc"
    )
    return cache_dir / name


def _normalize_for_opendrift(raw_path: Path, out_path: Path) -> None:
    """Re-emit the downloaded file with OSCAR-style schema OpenDrift likes.

    The Copernicus subset comes with dims (time, depth, latitude, longitude)
    and vars `uo`/`vo`. We squeeze depth, rename to `u`/`v`, and ensure
    monotonically-ascending lat/lon plus a clean time encoding.
    """
    import xarray as xr

    with xr.open_dataset(raw_path) as ds_in:
        ds = ds_in.load()

    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)
    elif "depth" in ds.coords:
        ds = ds.reset_coords("depth", drop=True)

    rename_map = {}
    if "uo" in ds.variables:
        rename_map["uo"] = "u"
    if "vo" in ds.variables:
        rename_map["vo"] = "v"
    if "longitude" not in ds.dims and "lon" in ds.dims:
        rename_map["lon"] = "longitude"
    if "latitude" not in ds.dims and "lat" in ds.dims:
        rename_map["lat"] = "latitude"
    ds = ds.rename(rename_map)

    if float(ds["latitude"].values[0]) > float(ds["latitude"].values[-1]):
        ds = ds.isel(latitude=slice(None, None, -1))
    if float(ds["longitude"].values[0]) > float(ds["longitude"].values[-1]):
        ds = ds.isel(longitude=slice(None, None, -1))

    ds["u"].attrs.update(
        standard_name="eastward_sea_water_velocity",
        long_name="Eastward surface current",
        units="m s-1",
    )
    ds["v"].attrs.update(
        standard_name="northward_sea_water_velocity",
        long_name="Northward surface current",
        units="m s-1",
    )
    ds["latitude"].attrs.update(
        standard_name="latitude", units="degrees_north", axis="Y"
    )
    ds["longitude"].attrs.update(
        standard_name="longitude", units="degrees_east", axis="X"
    )
    ds.attrs.update(
        Conventions="CF-1.8",
        title="GLORYS12V1 surface currents (regional subset)",
        source=f"Copernicus Marine {GLORYS_DATASET_ID}",
    )

    encoding = {
        "u": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "v": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "time": {"units": "seconds since 1970-01-01 00:00:00", "calendar": "standard"},
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        ds.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=encoding)
    finally:
        ds.close()
    tmp.replace(out_path)


def fetch_glorys(
    bbox: Bbox,
    start: date,
    end: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
) -> Path:
    """Download (or reuse) a GLORYS12V1 surface-current subset.

    Returns the path to a NetCDF file ready for `reader_netCDF_CF_generic`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _expected_path(bbox, start, end, cache_dir)

    if out_path.exists() and not overwrite:
        if out_path.stat().st_size > 0:
            print(f"[glorys] reusing existing {out_path}")
            return out_path
        print(f"[glorys] removing empty cache file {out_path}")
        out_path.unlink()

    try:
        import copernicusmarine
    except ImportError as e:
        raise ImportError(
            "copernicusmarine is required. Install with `pip install copernicusmarine`."
        ) from e

    raw_path = out_path.with_suffix(".raw.nc")
    print(
        f"[glorys] downloading {GLORYS_DATASET_ID}  "
        f"bbox=({bbox.min_lon:+.2f},{bbox.min_lat:+.2f})..({bbox.max_lon:+.2f},{bbox.max_lat:+.2f})  "
        f"time={start}..{end}"
    )
    # The subset call streams a NetCDF for the requested window. It throws
    # if credentials are missing (tells the user to run `copernicusmarine login`).
    copernicusmarine.subset(
        dataset_id=GLORYS_DATASET_ID,
        variables=list(GLORYS_VARIABLES),
        minimum_longitude=bbox.min_lon,
        maximum_longitude=bbox.max_lon,
        minimum_latitude=bbox.min_lat,
        maximum_latitude=bbox.max_lat,
        start_datetime=f"{start.isoformat()}T00:00:00",
        end_datetime=f"{end.isoformat()}T23:59:59",
        minimum_depth=0.0,
        maximum_depth=1.0,
        output_directory=str(raw_path.parent),
        output_filename=raw_path.name,
        overwrite=True,
    )
    if not raw_path.exists():
        # Some toolbox versions sanitize the filename; pick the newest .nc
        # that is not our final cached file.
        candidates = sorted(
            (p for p in raw_path.parent.glob("*.nc") if p != out_path),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"copernicusmarine.subset produced no NetCDF in {raw_path.parent}"
            )
        raw_path = candidates[0]

    print(f"[glorys] normalizing -> {out_path}")
    _normalize_for_opendrift(raw_path, out_path)
    try:
        os.remove(raw_path)
    except OSError:
        pass
    print(f"[glorys] ready: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bbox", type=float, nargs=4, required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bbox = Bbox(
        min_lon=args.bbox[0], min_lat=args.bbox[1],
        max_lon=args.bbox[2], max_lat=args.bbox[3],
    )
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    fetch_glorys(bbox, start, end, cache_dir=args.cache_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()


# Re-export `timedelta` so callers building windows don't need their own import.
__all__ = ["Bbox", "bbox_from_seeds", "fetch_glorys", "timedelta"]
