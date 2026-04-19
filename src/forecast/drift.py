"""Drift forecasting on top of OpenDrift's OceanDrift, forced by OSCAR.

End-to-end:
  1. Read predictions.json, extract seed (lat, lon, obs_date) per debris tile.
  2. Build (or reuse) a concatenated OSCAR NetCDF covering the simulation window.
  3. Configure OpenDrift OceanDrift with OSCAR as a current reader, optionally
     overlaid with a uniform wind field + leeway (windage) for floating debris.
  4. Seed N particles around each detection (Monte-Carlo for uncertainty).
  5. Run forward simulation, write:
        <out>.nc            OpenDrift's native trajectory file (full data)
        <out>.paths.geojson per-particle polylines (static demo map)
        <out>.final.geojson final particle positions as Points
        <out>.czml          time-animated entities for Cesium playback

Wind: a constant (uniform-in-space) wind vector can be supplied via
`wind_speed_ms` + `wind_dir_deg` (meteorological convention: direction the
wind is FROM, degrees clockwise from North). Floating debris is then advected
by `wind_drift_factor * wind` on top of the OSCAR currents. Industry-standard
leeway for floating plastic / SAR objects is ~0.02-0.04 (2-4%).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.forecast.glorys_fetch import bbox_from_seeds, fetch_glorys
from src.forecast.oscar_concat import concat_oscar
from src.forecast.seed import (
    DEFAULT_DEBRIS_CLASSES,
    Seed,
    extract_seeds,
)


# Which surface-current product feeds OpenDrift.
#   "oscar"  : NOAA OSCAR daily, 1/4 deg, geostrophic-only (Ekman EXCLUDED)
#   "glorys" : CMEMS GLORYS12V1, 1/12 deg, total currents (Ekman INCLUDED)
CurrentSource = str  # Literal["oscar", "glorys"] kept loose for FastAPI


def _wind_components(speed_ms: float, dir_from_deg: float) -> tuple[float, float]:
    """Convert meteorological wind (speed, FROM-direction) to (u_east, v_north).

    `dir_from_deg` is degrees CW from North that the wind is blowing FROM
    (e.g. 270 = westerly = wind blowing toward the east).
    """
    if speed_ms <= 0.0:
        return 0.0, 0.0
    rad = math.radians(dir_from_deg)
    u = -speed_ms * math.sin(rad)
    v = -speed_ms * math.cos(rad)
    return u, v


def _suppress_opendrift_logs() -> None:
    """OpenDrift is *very* chatty by default. Only show warnings+."""
    for name in ("opendrift", "opendrift.models", "opendrift.readers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _seed_particles(
    model,
    seeds: list[Seed],
    n_per_seed: int,
    radius_m: float,
    wind_drift_factor: float = 0.0,
) -> None:
    """Add `n_per_seed` particles per detection within `radius_m` (gaussian)."""
    for s in seeds:
        # Earliest start time wins; OpenDrift handles per-particle start times.
        start = datetime(s.obs_date.year, s.obs_date.month, s.obs_date.day, 12, 0, 0)
        kwargs = dict(
            lon=s.lon,
            lat=s.lat,
            radius=radius_m,
            number=n_per_seed,
            time=start,
            origin_marker=hash(s.tile_id) & 0xFFFF,
        )
        # Per-particle leeway. Set when > 0 so currents-only runs stay clean.
        if wind_drift_factor > 0.0:
            kwargs["wind_drift_factor"] = wind_drift_factor
        model.seed_elements(**kwargs)


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


def _read_trajectory_with_time(
    nc_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[datetime]]:
    """Same as `_read_trajectory_arrays` but also returns the time axis."""
    import xarray as xr
    with xr.open_dataset(nc_path) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        status = (
            np.asarray(ds["status"].values) if "status" in ds.variables else np.zeros_like(lon)
        )
        time_vals = np.asarray(ds["time"].values)
    times: list[datetime] = []
    for t in time_vals:
        try:
            dt = t.astype("datetime64[s]").astype(object)
            if isinstance(dt, datetime):
                times.append(dt.replace(tzinfo=timezone.utc))
            else:
                times.append(datetime.fromtimestamp(int(t.astype("int64")), tz=timezone.utc))
        except (AttributeError, ValueError):
            times.append(datetime.fromtimestamp(0, tz=timezone.utc))
    return lon, lat, status, times


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


def _trajectories_to_czml(
    nc_path: Path,
    out_path: Path,
    max_particles: int = 250,
    point_color_rgba: tuple[int, int, int, int] = (255, 165, 2, 220),
    point_size_px: int = 6,
) -> tuple[int, datetime, datetime]:
    """Write a CZML file with one time-varying Point per particle.

    Cesium's CzmlDataSource will then play the whole ensemble back along the
    viewer clock. We optionally subsample particles to keep file size sane;
    OpenDrift can produce thousands and the timeline doesn't benefit from
    drawing all of them simultaneously.
    """
    lon, lat, _, times = _read_trajectory_with_time(nc_path)
    n_particles, n_steps = lon.shape
    if n_steps < 2:
        raise ValueError(f"trajectory at {nc_path} has fewer than 2 timesteps")

    # Subsample particles by stride so we keep coverage of every seed.
    if n_particles > max_particles:
        stride = max(1, n_particles // max_particles)
        keep = list(range(0, n_particles, stride))[:max_particles]
    else:
        keep = list(range(n_particles))

    # Document-level clock spans the union of all valid timesteps.
    t_start = times[0]
    t_end = times[-1]
    epoch_iso = t_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = t_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    seconds_offsets = np.array(
        [(t - t_start).total_seconds() for t in times], dtype=np.float64
    )

    document = {
        "id": "document",
        "name": "SeaPredictor drift forecast",
        "version": "1.0",
        "clock": {
            "interval": f"{epoch_iso}/{end_iso}",
            "currentTime": epoch_iso,
            "multiplier": 3600,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }
    out: list[dict] = [document]

    for p in keep:
        finite = np.isfinite(lon[p]) & np.isfinite(lat[p])
        if finite.sum() < 2:
            continue
        idxs = np.where(finite)[0]
        first, last = int(idxs[0]), int(idxs[-1])

        cart: list[float] = []
        for i in idxs:
            cart.extend([
                float(seconds_offsets[i]),
                float(lon[p, i]),
                float(lat[p, i]),
                0.0,
            ])

        avail = (
            f"{times[first].strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"/{times[last].strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )

        out.append({
            "id": f"particle_{p}",
            "availability": avail,
            "position": {
                "epoch": epoch_iso,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 1,
                "cartographicDegrees": cart,
            },
            "point": {
                "color": {"rgba": list(point_color_rgba)},
                "outlineColor": {"rgba": [255, 255, 255, 200]},
                "outlineWidth": 1,
                "pixelSize": point_size_px,
                "heightReference": "CLAMP_TO_GROUND",
            },
            "path": {
                "material": {
                    "polylineGlow": {
                        "color": {"rgba": [80, 180, 255, 180]},
                        "glowPower": 0.18,
                    }
                },
                "width": 1.5,
                "leadTime": 0,
                "trailTime": 6 * 3600,
                "resolution": 600,
            },
        })

    out_path.write_text(json.dumps(out))
    return len(out) - 1, t_start, t_end


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
    wind_speed_ms: float = 0.0,
    wind_dir_deg: float = 0.0,
    wind_drift_factor: float = 0.0,
    current_source: CurrentSource = "oscar",
    glorys_nc_path: Path | None = None,
    glorys_bbox_buffer_deg: float = 5.0,
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

    start_date: _date = min(s.obs_date for s in seeds)
    end_date: _date = max(s.obs_date for s in seeds) + timedelta(days=days)
    print(f"[drift] simulation window {start_date}..{end_date}  "
          f"(forward {days} days)")

    src_norm = current_source.lower()
    if src_norm not in {"oscar", "glorys"}:
        raise ValueError(
            f"current_source must be 'oscar' or 'glorys', got {current_source!r}"
        )

    if src_norm == "glorys":
        if glorys_nc_path is None:
            bbox = bbox_from_seeds(
                lats=[s.lat for s in seeds],
                lons=[s.lon for s in seeds],
                buffer_deg=glorys_bbox_buffer_deg,
            )
            glorys_nc_path = fetch_glorys(
                bbox=bbox,
                start=start_date - timedelta(days=1),
                end=end_date + timedelta(days=1),
            )
        currents_path = glorys_nc_path
        print(f"[drift] currents: GLORYS12V1 (total surface) -> {currents_path}")
    else:
        currents_path = oscar_concat_path or concat_oscar(
            start=start_date - timedelta(days=1),
            end=end_date + timedelta(days=1),
        )
        print(f"[drift] currents: OSCAR (geostrophic only) -> {currents_path}")

    _suppress_opendrift_logs()
    model = OceanDrift(loglevel=30)
    reader = reader_netCDF_CF_generic.Reader(str(currents_path))
    model.add_reader(reader)

    u_wind, v_wind = _wind_components(wind_speed_ms, wind_dir_deg)
    if wind_speed_ms > 0.0:
        # Constant uniform-in-space wind field. OpenDrift treats `constant`
        # as a no-reader background that's always present (vs `fallback`
        # which only kicks in when no reader covers the point).
        model.set_config("environment:constant:x_wind", u_wind)
        model.set_config("environment:constant:y_wind", v_wind)
        print(
            f"[drift] wind: speed={wind_speed_ms:.1f} m/s  "
            f"from={wind_dir_deg:.0f} deg  "
            f"-> u={u_wind:+.2f}  v={v_wind:+.2f} m/s  "
            f"leeway={wind_drift_factor*100:.1f}%"
        )
    else:
        # Pure currents. Suppress any wind influence completely.
        model.set_config("environment:fallback:x_wind", 0.0)
        model.set_config("environment:fallback:y_wind", 0.0)
        print("[drift] wind: OFF (currents-only)")

    model.set_config("environment:fallback:x_sea_water_velocity", 0.0)
    model.set_config("environment:fallback:y_sea_water_velocity", 0.0)
    model.set_config("drift:horizontal_diffusivity", horizontal_diffusivity)
    model.set_config("general:coastline_action", "previous")

    _seed_particles(
        model,
        seeds,
        n_per_seed=n_per_seed,
        radius_m=seed_radius_m,
        wind_drift_factor=wind_drift_factor,
    )

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

    czml_out = out_stem.with_suffix(".czml")
    try:
        n_czml, t0, t1 = _trajectories_to_czml(nc_out, czml_out)
        print(
            f"[drift] wrote {n_czml} animated entities -> {czml_out}  "
            f"({t0.isoformat()} .. {t1.isoformat()})"
        )
    except Exception as e:  # noqa: BLE001 - never let CZML break the run
        print(f"[drift] WARN: failed to write CZML ({e}); animation disabled")

    return out_stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output stem (extensions added: .nc, .paths.geojson, .final.geojson, .czml)")
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
    parser.add_argument("--wind-speed", type=float, default=0.0,
                        help="Uniform wind speed in m/s (0 = no wind forcing)")
    parser.add_argument("--wind-dir", type=float, default=0.0,
                        help="Wind direction in deg CW from N, FROM convention "
                             "(270 = westerly, blows toward east)")
    parser.add_argument("--wind-drift-factor", type=float, default=0.0,
                        help="Per-particle leeway as fraction of wind speed "
                             "(0.02-0.04 typical for floating debris)")
    parser.add_argument("--current-source", type=str, default="oscar",
                        choices=["oscar", "glorys"],
                        help="Surface-current product. 'glorys' includes Ekman "
                             "(needs Copernicus Marine credentials).")
    parser.add_argument("--glorys-nc", type=Path, default=None,
                        help="Pre-downloaded GLORYS NetCDF; auto-fetched if omitted.")
    parser.add_argument("--glorys-buffer-deg", type=float, default=5.0,
                        help="Bbox padding around seeds when auto-downloading GLORYS.")
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
        wind_speed_ms=args.wind_speed,
        wind_dir_deg=args.wind_dir,
        wind_drift_factor=args.wind_drift_factor,
        current_source=args.current_source,
        glorys_nc_path=args.glorys_nc,
        glorys_bbox_buffer_deg=args.glorys_buffer_deg,
    )


if __name__ == "__main__":
    main()
