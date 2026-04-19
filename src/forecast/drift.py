"""Drift forecasting on top of OpenDrift's OceanDrift, forced by OSCAR.

End-to-end:
  1. Read predictions.json, extract seed (lat, lon, obs_date) per debris tile.
  2. Build (or reuse) a concatenated OSCAR NetCDF covering the simulation window.
  3. Configure OpenDrift OceanDrift with that as a current reader.
  4. Seed N particles around each detection (Monte-Carlo for uncertainty).
  5. Run forward `--days` simulation, write:
        <out>.nc       OpenDrift's native trajectory file (full data)
        <out>.geojson  per-particle paths as polylines (for the demo map)

Currents only — wind/Stokes drift are intentionally left off until you wire in
a wind reader. With windage 0 and only currents, OpenDrift behaves like a
straight Lagrangian integrator on (u, v) with diffusion.

Usage:
    python -m src.forecast.drift \\
        --predictions predictions.json \\
        --days 7 \\
        --out forecast/run1
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.forecast.oscar_concat import concat_oscar
from src.forecast.seed import (
    DEFAULT_DEBRIS_CLASSES,
    Seed,
    extract_seeds,
)


def _suppress_opendrift_logs() -> None:
    """OpenDrift is *very* chatty by default. Only show warnings+."""
    for name in ("opendrift", "opendrift.models", "opendrift.readers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _seed_particles(model, seeds: list[Seed], n_per_seed: int, radius_m: float) -> None:
    """Add `n_per_seed` particles per detection within `radius_m` (gaussian)."""
    for s in seeds:
        # Earliest start time wins; OpenDrift handles per-particle start times.
        start = datetime(s.obs_date.year, s.obs_date.month, s.obs_date.day, 12, 0, 0)
        model.seed_elements(
            lon=s.lon,
            lat=s.lat,
            radius=radius_m,
            number=n_per_seed,
            time=start,
            origin_marker=hash(s.tile_id) & 0xFFFF,
        )


def _read_trajectory_arrays(nc_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (lon, lat, status) arrays out of an OpenDrift trajectory NetCDF.

    Each is shape (n_particles, n_steps). Inactive timesteps are NaN.
    """
    import xarray as xr
    with xr.open_dataset(nc_path) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        status = (
            np.asarray(ds["status"].values) if "status" in ds.variables else np.zeros_like(lon)
        )
    return lon, lat, status


def _trajectories_to_geojson(nc_path: Path, out_path: Path) -> int:
    """Dump per-particle trajectories as a GeoJSON FeatureCollection of LineStrings."""
    lon, lat, _ = _read_trajectory_arrays(nc_path)
    n_particles, n_steps = lon.shape
    features: list[dict] = []
    for p in range(n_particles):
        coords: list[list[float]] = []
        for t in range(n_steps):
            x = float(lon[p, t])
            y = float(lat[p, t])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            coords.append([x, y])
        if len(coords) < 2:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"particle_id": int(p), "n_steps": len(coords)},
        })

    out_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features,
    }))
    return len(features)


def _final_positions_geojson(nc_path: Path, out_path: Path) -> int:
    """Dump final-step (last finite) particle positions as a Point FeatureCollection."""
    lon, lat, _ = _read_trajectory_arrays(nc_path)
    features: list[dict] = []
    for p in range(lon.shape[0]):
        finite_mask = np.isfinite(lon[p]) & np.isfinite(lat[p])
        if not finite_mask.any():
            continue
        last = np.where(finite_mask)[0][-1]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon[p, last]), float(lat[p, last])],
            },
            "properties": {"particle_id": int(p), "step_index": int(last)},
        })
    out_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features,
    }))
    return len(features)


def run_drift(
    predictions_path: Path,
    out_stem: Path,
    days: int = 7,
    timestep_minutes: int = 30,
    n_per_seed: int = 100,
    seed_radius_m: float = 1000.0,
    horizontal_diffusivity: float = 10.0,
    debris_classes: tuple[int, ...] = DEFAULT_DEBRIS_CLASSES,
    default_date: str | None = None,
    override_date: bool = False,
    min_prob: float = 0.0,
    oscar_concat_path: Path | None = None,
) -> Path:
    """Main entry: turn detections into a drift forecast. Returns out_stem."""
    from datetime import date as _date
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.readers import reader_netCDF_CF_generic

    out_stem.parent.mkdir(parents=True, exist_ok=True)

    parsed_default = (
        datetime.strptime(default_date, "%Y-%m-%d").date() if default_date else None
    )
    seeds = extract_seeds(
        predictions_path,
        debris_classes=debris_classes,
        default_date=parsed_default,
        min_prob=min_prob,
        override_date=override_date,
    )
    if not seeds:
        raise RuntimeError(
            f"No seeds extracted from {predictions_path}. Either no tiles match "
            f"debris_classes={list(debris_classes)} or geo metadata is missing."
        )
    print(f"[drift] {len(seeds)} debris seeds  "
          f"({n_per_seed} particles each = {len(seeds) * n_per_seed} total)")

    # Determine simulation window. Earliest obs_date is the start; end is +days.
    start_date: _date = min(s.obs_date for s in seeds)
    end_date: _date = max(s.obs_date for s in seeds) + timedelta(days=days)
    print(f"[drift] simulation window {start_date}..{end_date}  "
          f"(forward {days} days)")

    # OSCAR window must cover the full sim. Pad by 1 day each side for safety.
    oscar_path = oscar_concat_path or concat_oscar(
        start=start_date - timedelta(days=1),
        end=end_date + timedelta(days=1),
    )

    _suppress_opendrift_logs()
    model = OceanDrift(loglevel=30)  # 30 = WARNING
    reader = reader_netCDF_CF_generic.Reader(str(oscar_path))
    model.add_reader(reader)

    # Currents-only forecast; no wind, no waves, no Stokes for now.
    model.set_config("environment:fallback:x_wind", 0.0)
    model.set_config("environment:fallback:y_wind", 0.0)
    model.set_config("environment:fallback:x_sea_water_velocity", 0.0)
    model.set_config("environment:fallback:y_sea_water_velocity", 0.0)
    model.set_config("drift:horizontal_diffusivity", horizontal_diffusivity)
    model.set_config("general:coastline_action", "previous")

    _seed_particles(model, seeds, n_per_seed=n_per_seed, radius_m=seed_radius_m)

    nc_out = out_stem.with_suffix(".nc")
    duration = timedelta(days=days)
    print(f"[drift] running OpenDrift  dt={timestep_minutes} min  "
          f"duration={duration}")
    model.run(
        time_step=timedelta(minutes=timestep_minutes),
        time_step_output=timedelta(hours=1),
        duration=duration,
        outfile=str(nc_out),
        export_variables=["lon", "lat", "status", "origin_marker"],
    )
    print(f"[drift] wrote trajectory NetCDF -> {nc_out}")

    paths_out = out_stem.with_suffix(".paths.geojson")
    n_paths = _trajectories_to_geojson(nc_out, paths_out)
    print(f"[drift] wrote {n_paths} particle paths -> {paths_out}")

    final_out = out_stem.with_suffix(".final.geojson")
    n_final = _final_positions_geojson(nc_out, final_out)
    print(f"[drift] wrote {n_final} final positions -> {final_out}")

    return out_stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output stem (extensions added: .nc, .paths.geojson, .final.geojson)")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--timestep-minutes", type=int, default=30)
    parser.add_argument("--n-per-seed", type=int, default=100,
                        help="Particles per detection (Monte Carlo for uncertainty)")
    parser.add_argument("--seed-radius-m", type=float, default=1000.0,
                        help="Gaussian seed radius around each detection")
    parser.add_argument("--horizontal-diffusivity", type=float, default=10.0,
                        help="Eddy diffusivity m^2/s for sub-grid spread")
    parser.add_argument(
        "--debris-classes", type=int, nargs="+", default=list(DEFAULT_DEBRIS_CLASSES),
    )
    parser.add_argument("--default-date", type=str, default=None,
                        help="Fallback YYYY-MM-DD when tile_index.csv has no entry")
    parser.add_argument("--override-date", action="store_true",
                        help="Force --default-date onto every tile (demo mode "
                             "when MARIDA scenes fall outside your OSCAR window)")
    parser.add_argument("--min-prob", type=float, default=0.0)
    parser.add_argument("--oscar-nc", type=Path, default=None,
                        help="Pre-built concatenated OSCAR NetCDF. "
                             "If omitted, one is built from default OSCAR root.")
    args = parser.parse_args()

    run_drift(
        predictions_path=args.predictions,
        out_stem=args.out,
        days=args.days,
        timestep_minutes=args.timestep_minutes,
        n_per_seed=args.n_per_seed,
        seed_radius_m=args.seed_radius_m,
        horizontal_diffusivity=args.horizontal_diffusivity,
        debris_classes=tuple(args.debris_classes),
        default_date=args.default_date,
        override_date=args.override_date,
        min_prob=args.min_prob,
        oscar_concat_path=args.oscar_nc,
    )


if __name__ == "__main__":
    main()
