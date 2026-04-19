"""Forecasting layer: take detector outputs, simulate drift, emit predictions.

Pipeline:
    predictions.json  ->  seed centroids (lat, lon, date)
    OSCAR daily NCs   ->  concatenated CF-compliant NetCDF (cached)
    OpenDrift OceanDrift forced by OSCAR currents
    output: trajectories.nc + GeoJSON (per-particle paths) + density GeoTIFF
"""
