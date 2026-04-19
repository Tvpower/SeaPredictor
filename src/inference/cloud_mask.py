"""Post-hoc cloud filtering for Sentinel-2 tiles.

MARIDA tiles do not ship with QA60 / SCL cloud masks, so we derive a cloud
score from two independent signals and suppress debris-class predictions
when either fires:

    1. SPECTRAL: fraction of pixels with mean(B2,B3,B4) > vis_thresh AND
       B8 > nir_thresh. Catches bright opaque clouds on arbitrary S2 input.
       MARIDA itself is already screened of bright cumulus, so this rarely
       trips on cached scenes — its main role is generalization to fresh
       Sentinel-2 acquisitions outside the training distribution.

    2. MODEL CLOUD CLASS: probability of class 5 (Clouds) and/or class 12
       (Cloud Shadows) above a confidence floor. MARIDA labels a lot of
       thin cirrus / haze as "Clouds", so the model's own class 5 head is
       actually our best signal for those subtle cases (median visible
       reflectance for MARIDA cloud tiles is ~0.03 — indistinguishable
       spectrally from open ocean).

Sentinel-2 band order in MARIDA tiles is:

    index  band   wavelength   role
    0      B1     443 nm       coastal aerosol
    1      B2     490 nm       blue          <- visible
    2      B3     560 nm       green         <- visible
    3      B4     665 nm       red           <- visible
    4      B5     705 nm       red edge 1
    5      B6     740 nm       red edge 2
    6      B7     783 nm       red edge 3
    7      B8     842 nm       NIR           <- NIR
    8      B8A    865 nm       narrow NIR
    9      B11    1610 nm      SWIR1
    10     B12    2190 nm      SWIR2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Default debris classes that we suppress on cloudy tiles. Mirrors
# DEFAULT_DEBRIS_CLASSES in src.pipeline.build_scenes / src.forecast.seed.
DEFAULT_DEBRIS_CLASSES: tuple[int, ...] = (0, 1, 2, 8)

# Spectral thresholds in TOA reflectance units (matching MARIDA's storage).
DEFAULT_VIS_REFLECTANCE = 0.18   # mean(B2,B3,B4) above this -> bright pixel
DEFAULT_NIR_REFLECTANCE = 0.08   # B8 above this  -> not water
DEFAULT_MAX_CLOUD_FRAC = 0.50    # tile-level cloud pixel fraction trigger

# Model-prediction thresholds: MARIDA's class 5 (Clouds) / class 12 (Cloud
# Shadows). If the model itself thinks the tile is cloud-dominated, kill
# any concurrent debris call. 0.6 is conservative — class 5's tuned F1-max
# threshold in checkpoints/cnn_only_v2 is 0.75.
DEFAULT_CLOUD_CLASS_PROB = 0.60
DEFAULT_CLOUD_SHADOW_PROB = 0.60
CLOUD_CLASS_IDX = 5
CLOUD_SHADOW_CLASS_IDX = 12

# Band indices (0-based) in MARIDA's 11-band stack.
B2_IDX, B3_IDX, B4_IDX, B8_IDX = 1, 2, 3, 7


@dataclass(frozen=True)
class CloudFilterConfig:
    vis_reflectance: float = DEFAULT_VIS_REFLECTANCE
    nir_reflectance: float = DEFAULT_NIR_REFLECTANCE
    max_cloud_frac: float = DEFAULT_MAX_CLOUD_FRAC
    cloud_class_prob: float = DEFAULT_CLOUD_CLASS_PROB
    cloud_shadow_prob: float = DEFAULT_CLOUD_SHADOW_PROB
    use_model_cloud_class: bool = True
    debris_classes: tuple[int, ...] = DEFAULT_DEBRIS_CLASSES


def cloud_pixel_mask(
    arr: np.ndarray,
    vis_reflectance: float = DEFAULT_VIS_REFLECTANCE,
    nir_reflectance: float = DEFAULT_NIR_REFLECTANCE,
) -> np.ndarray:
    """Boolean (H, W) mask of cloud-like pixels.

    Args:
        arr: raw (un-normalized) tile array of shape (C, H, W) where bands
             follow MARIDA's standard order (see module docstring).
        vis_reflectance: TOA reflectance threshold for the visible mean.
        nir_reflectance: TOA reflectance threshold for B8.
    """
    if arr.ndim != 3 or arr.shape[0] <= max(B2_IDX, B3_IDX, B4_IDX, B8_IDX):
        # Wrong shape / too few bands -> can't decide; assume "not cloud".
        return np.zeros(arr.shape[-2:], dtype=bool)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vis_mean = (arr[B2_IDX] + arr[B3_IDX] + arr[B4_IDX]) / 3.0
    return (vis_mean > vis_reflectance) & (arr[B8_IDX] > nir_reflectance)


def cloud_fraction(
    arr: np.ndarray,
    vis_reflectance: float = DEFAULT_VIS_REFLECTANCE,
    nir_reflectance: float = DEFAULT_NIR_REFLECTANCE,
) -> float:
    """Fraction of pixels in `arr` that look cloud-like (0.0 .. 1.0)."""
    mask = cloud_pixel_mask(arr, vis_reflectance, nir_reflectance)
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def apply_cloud_filter(
    probs: np.ndarray,
    preds: np.ndarray,
    raw_arr: np.ndarray,
    cfg: CloudFilterConfig = CloudFilterConfig(),
) -> tuple[np.ndarray, np.ndarray, float, bool, str | None]:
    """Suppress debris-class predictions on cloud-dominated tiles.

    Returns:
        probs_out: copy of `probs` with debris-class probs zeroed if the tile
                   is cloud-dominated.
        preds_out: copy of `preds` with debris-class preds zeroed if cloudy.
        cloud_frac: spectral cloud-pixel fraction in the tile (always returned).
        suppressed: True if debris classes were zeroed for this tile.
        reason: short tag for *why* it was suppressed ("spectral", "cloud_class",
                "cloud_shadow", "spectral+cloud_class", ...). None if not suppressed.
    """
    frac = cloud_fraction(raw_arr, cfg.vis_reflectance, cfg.nir_reflectance)
    reasons: list[str] = []
    if frac > cfg.max_cloud_frac:
        reasons.append("spectral")

    if cfg.use_model_cloud_class:
        if (CLOUD_CLASS_IDX < probs.shape[0]
                and probs[CLOUD_CLASS_IDX] >= cfg.cloud_class_prob):
            reasons.append("cloud_class")
        if (CLOUD_SHADOW_CLASS_IDX < probs.shape[0]
                and probs[CLOUD_SHADOW_CLASS_IDX] >= cfg.cloud_shadow_prob):
            reasons.append("cloud_shadow")

    if not reasons:
        return probs, preds, frac, False, None

    probs_out = probs.copy()
    preds_out = preds.copy()
    for c in cfg.debris_classes:
        if 0 <= c < probs_out.shape[0]:
            probs_out[c] = 0.0
            preds_out[c] = 0
    return probs_out, preds_out, frac, True, "+".join(reasons)
