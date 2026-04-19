"""Concatenate OSCAR daily NetCDFs into a single OpenDrift-friendly file.

OSCAR daily files are CF-ish but have three quirks OpenDrift's generic CF reader
chokes on:

  1. Calendar is `julian` and time units are `days since 1990-1-1` — fine on
     paper but cftime decoding to numpy datetime64 needs `use_cftime=True`,
     which then breaks downstream pandas conversion. We sidestep it by
     re-encoding time as `seconds since 1970-01-01 00:00:00` (standard).
  2. Longitude is in 0..360. OpenDrift's grid lookups work in either, but the
     standard convention for global fields used with most coastline products
     is -180..180. We roll the grid so it's monotonic in -180..180.
  3. Dim order is (time, longitude, latitude). The CF generic reader prefers
     (time, latitude, longitude). We transpose.

The output is one NetCDF with shape (T, n_lat, n_lon) for `u` and `v`,
covering every available day in `[start, end]` (or all days if you don't
specify). Cached on disk by date range — repeat calls are no-ops.

Usage:
    python -m src.forecast.oscar_concat \\
        --start 2020-02-06 --end 2020-02-20 \\
        --out data/forecast/oscar_concat_20200206_20200220.nc
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from src.dataset.oscar_loader import (
    _date_from_filename,
    default_oscar_root,
)


def _daily_files_in_range(
    root: Path, start: date | None, end: date | None
) -> list[tuple[date, Path]]:
    files: list[tuple[date, Path]] = []
    for p in root.glob("oscar_currents_interim_*.nc"):
        d = _date_from_filename(p)
        if d is None:
            continue
        if start and d < start:
            continue
        if end and d > end:
            continue
        files.append((d, p))
    files.sort()
    return files


def concat_oscar(
    start: date | None = None,
    end: date | None = None,
    root: Path | None = None,
    out_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Build (or reuse) a single concatenated OSCAR NetCDF.

    Returns the path to the output file.
    """
    import xarray as xr

    if root is None:
        r = default_oscar_root()
        if r is None:
            raise FileNotFoundError("No OSCAR root found.")
        root = r
    root = Path(root)

    files = _daily_files_in_range(root, start, end)
    if not files:
        raise FileNotFoundError(
            f"No OSCAR files in {root} within "
            f"{start or 'beginning'}..{end or 'end'}"
        )

    if out_path is None:
        s = files[0][0].strftime("%Y%m%d")
        e = files[-1][0].strftime("%Y%m%d")
        out_path = Path(f"data/forecast/oscar_concat_{s}_{e}.nc")

    if out_path.exists() and not overwrite:
        print(f"[oscar_concat] reusing existing {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[oscar_concat] reading {len(files)} daily files "
          f"({files[0][0]}..{files[-1][0]})")

    times: list[np.datetime64] = []
    u_stack: list[np.ndarray] = []
    v_stack: list[np.ndarray] = []
    lat = lon = None

    for d, path in files:
        with xr.open_dataset(path, decode_times=False) as ds:
            if lat is None:
                lat = np.asarray(ds["lat"].values, dtype=np.float64)
                lon = np.asarray(ds["lon"].values, dtype=np.float64)
            # File shape: (time=1, longitude, latitude). Drop time, transpose
            # to (latitude, longitude) for the CF reader.
            u_i = np.asarray(ds["u"].values, dtype=np.float32)[0].T  # (lat, lon)
            v_i = np.asarray(ds["v"].values, dtype=np.float32)[0].T
            u_stack.append(u_i)
            v_stack.append(v_i)
            t = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=timezone.utc)
            times.append(np.datetime64(t.replace(tzinfo=None), "s"))

    u = np.stack(u_stack, axis=0)  # (T, lat, lon)
    v = np.stack(v_stack, axis=0)
    time_arr = np.asarray(times, dtype="datetime64[s]")

    # Roll lon from 0..360 to -180..180 and sort.
    assert lon is not None and lat is not None
    lon_signed = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_signed)
    lon_signed = lon_signed[order]
    u = u[:, :, order]
    v = v[:, :, order]

    # Sort latitude ascending too (some readers assume it).
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]

    ds_out = xr.Dataset(
        data_vars=dict(
            u=(("time", "latitude", "longitude"), u, {
                "standard_name": "eastward_sea_water_velocity",
                "long_name": "Eastward surface current",
                "units": "m s-1",
            }),
            v=(("time", "latitude", "longitude"), v, {
                "standard_name": "northward_sea_water_velocity",
                "long_name": "Northward surface current",
                "units": "m s-1",
            }),
        ),
        coords=dict(
            time=("time", time_arr, {
                "standard_name": "time",
                "long_name": "time",
                "axis": "T",
            }),
            latitude=("latitude", lat, {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
                "axis": "Y",
            }),
            longitude=("longitude", lon_signed, {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
                "axis": "X",
            }),
        ),
        attrs=dict(
            Conventions="CF-1.8",
            title="OSCAR surface currents (concatenated)",
            source=f"NOAA OSCAR daily files in {root}",
        ),
    )

    encoding = {
        "u": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "v": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "time": {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "standard",
        },
    }
    ds_out.to_netcdf(out_path, format="NETCDF4", engine="netcdf4", encoding=encoding)
    ds_out.close()

    print(f"[oscar_concat] wrote {out_path}  shape u={u.shape}  "
          f"time_range={time_arr[0]}..{time_arr[-1]}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--root", type=Path, default=None,
                        help="OSCAR daily-files directory; auto-detected if omitted")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    concat_oscar(start, end, args.root, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
