"""Evaluate a drift forecast.

Two tiers of metrics:

Tier 1 — Plausibility (always available, just needs the forecast NetCDF)
    - mean / median / max particle displacement
    - mean drift bearing
    - speed in km/day
    - beached fraction
    - diffusion growth exponent (should be ~0.5 for proper sqrt(t) spread)
    - backward-forward consistency is NOT computed here; would require a 2nd run

Tier 2 — Temporal cross-validation (optional, needs a "scene B" predictions.json
         from a later date over roughly the same region)
    - hit rate at K km: fraction of forecast particles within K km of any
      detected debris in scene B
    - centroid drift error: distance between forecast mean position and
      scene-B debris centroid
    - density IoU at a chosen grid resolution

Usage:
    # Tier 1 only
    python -m src.forecast.validate \\
        --trajectory forecast/honduras_sep18.nc \\
        --report forecast/honduras_sep18.validation.json

    # Tier 1 + Tier 2 (with scene B from 5 days later)
    python -m src.forecast.validate \\
        --trajectory forecast/honduras_sep18.nc \\
        --scene-b predictions/honduras_sep23.json \\
        --scene-b-date 2020-09-23 \\
        --debris-classes 0 \\
        --report forecast/honduras_sep18.validation.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np

from src.forecast.seed import DEFAULT_DEBRIS_CLASSES, extract_seeds


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #
EARTH_R_KM = 6371.0088


def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance, vectorized. Inputs in degrees."""
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def initial_bearing_deg(lat1: np.ndarray, lon1: np.ndarray,
                        lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compass bearing from (lat1, lon1) to (lat2, lon2). Degrees CW from N."""
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def circular_mean_deg(angles: np.ndarray) -> float:
    """Mean of angular values (degrees). Handles wrap-around at 360."""
    a = np.radians(angles[np.isfinite(angles)])
    if a.size == 0:
        return float("nan")
    return float((np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) + 360) % 360)


# --------------------------------------------------------------------------- #
# Trajectory ingest                                                           #
# --------------------------------------------------------------------------- #
def load_trajectory(nc_path: Path) -> dict:
    """Pull the bits we need out of an OpenDrift NetCDF."""
    import xarray as xr
    with xr.open_dataset(nc_path) as ds:
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        time_vals = np.asarray(ds["time"].values)
    # Convert datetime64 -> python datetimes for human inspection.
    times: list[datetime | None] = []
    for t in time_vals:
        if isinstance(t, np.datetime64) and np.isnat(t):
            times.append(None)
            continue
        try:
            times.append(t.astype("datetime64[s]").astype(object))
        except (AttributeError, ValueError):
            times.append(None)
    return {"lon": lon, "lat": lat, "times": times}


def first_last_finite(lon: np.ndarray, lat: np.ndarray):
    """For each particle, return (i0, iN, lon0, lat0, lonN, latN, n_finite)."""
    n_p, n_t = lon.shape
    out = []
    for p in range(n_p):
        finite = np.isfinite(lon[p]) & np.isfinite(lat[p])
        idx = np.where(finite)[0]
        if idx.size < 2:
            out.append(None)
            continue
        i0, iN = int(idx[0]), int(idx[-1])
        out.append((i0, iN,
                    float(lon[p, i0]), float(lat[p, i0]),
                    float(lon[p, iN]), float(lat[p, iN]),
                    int(idx.size)))
    return out


# --------------------------------------------------------------------------- #
# Tier 1                                                                      #
# --------------------------------------------------------------------------- #
def tier1_plausibility(traj: dict, output_step_hours: float = 1.0) -> dict:
    lon, lat = traj["lon"], traj["lat"]
    fl = first_last_finite(lon, lat)

    disp_km = []
    bearings = []
    speeds = []
    last_active_step = []

    for entry in fl:
        if entry is None:
            continue
        i0, iN, x0, y0, xN, yN, _ = entry
        d = float(haversine_km(np.array([y0]), np.array([x0]),
                               np.array([yN]), np.array([xN]))[0])
        disp_km.append(d)
        bearings.append(float(initial_bearing_deg(np.array([y0]), np.array([x0]),
                                                  np.array([yN]), np.array([xN]))[0]))
        hours = (iN - i0) * output_step_hours
        if hours > 0:
            speeds.append(d / (hours / 24.0))  # km/day
        last_active_step.append(iN)

    disp_km = np.array(disp_km)
    speeds = np.array(speeds)

    # Beached: a particle whose final position has been static for >=4 output steps.
    n_beached = 0
    for entry in fl:
        if entry is None:
            continue
        _, iN, _, _, xN, yN, _ = entry
        # walk back and count steps with the SAME (xN, yN) value -- OpenDrift's
        # coastline_action="previous" sets identical positions when parked.
        p_idx = fl.index(entry)
        repeats = 0
        for back in range(iN, max(iN - 8, -1), -1):
            if (lon[p_idx, back] == xN) and (lat[p_idx, back] == yN):
                repeats += 1
            else:
                break
        if repeats >= 4:
            n_beached += 1

    # Diffusion growth: sigma(t) of particle positions ≈ A * t^alpha
    # We compute sigma from the full ensemble at each timestep, then fit log-log.
    n_p, n_t = lon.shape
    sigma_t = []
    for t in range(n_t):
        finite = np.isfinite(lon[:, t]) & np.isfinite(lat[:, t])
        if finite.sum() < 10:
            continue
        # use planar approximation around the mean lat for sigma in km
        mean_lat = float(np.nanmean(lat[finite, t]))
        x_km = (lon[finite, t] - np.nanmean(lon[finite, t])) * 111.0 * np.cos(np.radians(mean_lat))
        y_km = (lat[finite, t] - mean_lat) * 111.0
        sigma = float(np.sqrt(np.var(x_km) + np.var(y_km)))
        sigma_t.append((t * output_step_hours / 24.0, sigma))

    diffusion_exponent = float("nan")
    if len(sigma_t) >= 5:
        ts = np.array([s[0] for s in sigma_t])
        ss = np.array([s[1] for s in sigma_t])
        good = (ts > 0) & (ss > 0)
        if good.sum() >= 5:
            # log(sigma) = alpha * log(t) + c
            slope, _ = np.polyfit(np.log(ts[good]), np.log(ss[good]), 1)
            diffusion_exponent = float(slope)

    return {
        "n_particles": int(len(fl)),
        "n_active": int(np.sum([e is not None for e in fl])),
        "n_beached": int(n_beached),
        "beached_fraction": float(n_beached / max(len(fl), 1)),
        "displacement_km": {
            "mean": float(np.mean(disp_km)) if disp_km.size else None,
            "median": float(np.median(disp_km)) if disp_km.size else None,
            "max": float(np.max(disp_km)) if disp_km.size else None,
            "p95": float(np.percentile(disp_km, 95)) if disp_km.size else None,
        },
        "speed_km_per_day": {
            "mean": float(np.mean(speeds)) if speeds.size else None,
            "median": float(np.median(speeds)) if speeds.size else None,
        },
        "mean_drift_bearing_deg": circular_mean_deg(np.array(bearings)),
        "diffusion_growth_exponent_alpha": diffusion_exponent,
        "diffusion_alpha_interpretation": (
            "~0.5 = pure diffusion (good); ~1.0 = ballistic (currents dominant); "
            "~0 = stuck (probably broken)"
        ),
    }


# --------------------------------------------------------------------------- #
# Tier 2                                                                      #
# --------------------------------------------------------------------------- #
def particles_at_time(traj: dict, target_time: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) for every particle at the snapshot closest to target_time.

    Drops particles that are NaN at that snapshot.
    """
    times = traj["times"]
    if not times:
        return np.array([]), np.array([])
    deltas = []
    for t in times:
        if t is None:
            deltas.append(float("inf"))
        else:
            deltas.append(abs((t - target_time).total_seconds()))
    t_idx = int(np.argmin(deltas))
    lon = traj["lon"][:, t_idx]
    lat = traj["lat"][:, t_idx]
    finite = np.isfinite(lon) & np.isfinite(lat)
    return lon[finite], lat[finite]


def tier2_cross_validation(
    traj: dict,
    scene_b_predictions: Path,
    scene_b_date: date,
    debris_classes: tuple[int, ...],
    hit_radii_km: tuple[float, ...] = (5.0, 10.0, 25.0),
    grid_res_deg: float = 0.05,
) -> dict:
    """Compare forecast at scene-B time vs detected debris in scene B."""
    seeds_b = extract_seeds(
        scene_b_predictions,
        debris_classes=debris_classes,
        default_date=scene_b_date,
        override_date=True,
    )
    if not seeds_b:
        return {"error": "No debris seeds in scene-B predictions"}

    obs_lat = np.array([s.lat for s in seeds_b])
    obs_lon = np.array([s.lon for s in seeds_b])

    # Forecast snapshot at scene-B local noon.
    target = datetime(scene_b_date.year, scene_b_date.month, scene_b_date.day, 12, 0, 0)
    fc_lon, fc_lat = particles_at_time(traj, target)
    if fc_lon.size == 0:
        return {"error": "No active particles at scene-B timestamp"}

    # Hit rate: distance from each forecast particle to its NEAREST observed centroid
    # (vectorized via small loop over observed centroids; obs is small).
    nearest = np.full(fc_lon.size, np.inf)
    for ob_lat, ob_lon in zip(obs_lat, obs_lon):
        d = haversine_km(np.full_like(fc_lat, ob_lat), np.full_like(fc_lon, ob_lon),
                         fc_lat, fc_lon)
        nearest = np.minimum(nearest, d)
    hit_rates = {f"hit_rate_at_{int(k)}km": float((nearest <= k).mean()) for k in hit_radii_km}

    # Centroid drift error
    fc_centroid_lat = float(np.mean(fc_lat))
    fc_centroid_lon = float(np.mean(fc_lon))
    obs_centroid_lat = float(np.mean(obs_lat))
    obs_centroid_lon = float(np.mean(obs_lon))
    centroid_err_km = float(haversine_km(
        np.array([fc_centroid_lat]), np.array([fc_centroid_lon]),
        np.array([obs_centroid_lat]), np.array([obs_centroid_lon]),
    )[0])

    # Density IoU on a common grid
    all_lon = np.concatenate([fc_lon, obs_lon])
    all_lat = np.concatenate([fc_lat, obs_lat])
    pad = 0.5  # deg pad
    lon_min, lon_max = all_lon.min() - pad, all_lon.max() + pad
    lat_min, lat_max = all_lat.min() - pad, all_lat.max() + pad
    n_x = max(int(np.ceil((lon_max - lon_min) / grid_res_deg)), 4)
    n_y = max(int(np.ceil((lat_max - lat_min) / grid_res_deg)), 4)
    fc_grid, _, _ = np.histogram2d(fc_lon, fc_lat, bins=[n_x, n_y],
                                   range=[[lon_min, lon_max], [lat_min, lat_max]])
    obs_grid, _, _ = np.histogram2d(obs_lon, obs_lat, bins=[n_x, n_y],
                                    range=[[lon_min, lon_max], [lat_min, lat_max]])
    fc_mask = fc_grid > 0
    obs_mask = obs_grid > 0
    inter = int((fc_mask & obs_mask).sum())
    union = int((fc_mask | obs_mask).sum())
    iou = float(inter / union) if union else 0.0

    return {
        "scene_b_date": scene_b_date.isoformat(),
        "scene_b_debris_count": int(len(seeds_b)),
        "n_active_forecast_particles": int(fc_lon.size),
        **hit_rates,
        "centroid_drift_error_km": centroid_err_km,
        "forecast_centroid": [fc_centroid_lat, fc_centroid_lon],
        "observed_centroid": [obs_centroid_lat, obs_centroid_lon],
        "density_iou": iou,
        "grid_resolution_deg": grid_res_deg,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True,
                        help="OpenDrift trajectory NetCDF (output of drift.py)")
    parser.add_argument("--output-step-hours", type=float, default=1.0,
                        help="OpenDrift time_step_output (drift.py default: 1.0)")
    parser.add_argument("--scene-b", type=Path, default=None,
                        help="predictions.json for a LATER scene over the same area")
    parser.add_argument("--scene-b-date", type=str, default=None,
                        help="YYYY-MM-DD of scene B observation")
    parser.add_argument("--debris-classes", type=int, nargs="+",
                        default=list(DEFAULT_DEBRIS_CLASSES))
    parser.add_argument("--report", type=Path, default=None,
                        help="Write JSON report here")
    args = parser.parse_args()

    print(f"[validate] loading trajectory {args.trajectory}")
    traj = load_trajectory(args.trajectory)
    print(f"[validate] particles={traj['lon'].shape[0]}  steps={traj['lon'].shape[1]}")

    report = {"trajectory": str(args.trajectory)}

    print("[validate] computing Tier 1 (plausibility)...")
    report["tier1_plausibility"] = tier1_plausibility(traj, args.output_step_hours)
    t1 = report["tier1_plausibility"]
    print(f"  active={t1['n_active']}  beached={t1['n_beached']} "
          f"({t1['beached_fraction']:.0%})")
    print(f"  displacement: mean={t1['displacement_km']['mean']:.1f} km  "
          f"median={t1['displacement_km']['median']:.1f}  "
          f"p95={t1['displacement_km']['p95']:.1f}  "
          f"max={t1['displacement_km']['max']:.1f}")
    print(f"  speed: mean={t1['speed_km_per_day']['mean']:.1f} km/day")
    print(f"  bearing: {t1['mean_drift_bearing_deg']:.0f}° from N")
    print(f"  diffusion growth exponent α = {t1['diffusion_growth_exponent_alpha']:.2f}  "
          f"(0.5=pure diffusion, 1.0=ballistic)")

    if args.scene_b is not None and args.scene_b_date is not None:
        print(f"[validate] computing Tier 2 (cross-validation against {args.scene_b})...")
        scene_b_date = datetime.strptime(args.scene_b_date, "%Y-%m-%d").date()
        report["tier2_cross_validation"] = tier2_cross_validation(
            traj=traj,
            scene_b_predictions=args.scene_b,
            scene_b_date=scene_b_date,
            debris_classes=tuple(args.debris_classes),
        )
        t2 = report["tier2_cross_validation"]
        if "error" in t2:
            print(f"  ERROR: {t2['error']}")
        else:
            print(f"  scene B: {t2['scene_b_date']}  "
                  f"({t2['scene_b_debris_count']} observed, "
                  f"{t2['n_active_forecast_particles']} forecast particles)")
            for k in ("hit_rate_at_5km", "hit_rate_at_10km", "hit_rate_at_25km"):
                if k in t2:
                    print(f"  {k}: {t2[k]:.0%}")
            print(f"  centroid drift error: {t2['centroid_drift_error_km']:.1f} km")
            print(f"  density IoU @{t2['grid_resolution_deg']}°: {t2['density_iou']:.3f}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str))
        print(f"[validate] wrote report -> {args.report}")


if __name__ == "__main__":
    main()
