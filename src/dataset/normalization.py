import numpy as np
import json


def compute_band_stats(data_list: list) -> dict:
    """Compute per-band mean/std from list of (C,H,W) arrays. Train split only."""
    n_bands = data_list[0].shape[0]
    all_means = [[] for _ in range(n_bands)]
    all_stds = [[] for _ in range(n_bands)]
    for data in data_list:
        for i in range(n_bands):
            band = data[i]
            valid = band[~np.isnan(band)]
            if len(valid) > 0:
                all_means[i].append(np.mean(valid))
                all_stds[i].append(np.std(valid))
    return {
        'mean': [float(np.mean(m)) for m in all_means],
        'std':  [float(np.mean(s)) for s in all_stds],
    }


def normalize_bands(data: np.ndarray, stats: dict) -> np.ndarray:
    C, H, W = data.shape
    out = np.zeros_like(data, dtype=np.float32)
    for i in range(C):
        mu = stats['mean'][i]
        sigma = stats['std'][i] + 1e-10
        out[i] = (data[i] - mu) / sigma
    return out


def save_stats(stats, path):
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def compute_class_weights(train_masks: list, num_classes: int = 11) -> np.ndarray:
    all_pixels = np.concatenate([m.flatten() for m in train_masks])
    counts = np.zeros(num_classes, dtype=np.float64)
    for cls_id in range(num_classes):
        counts[cls_id] = np.sum(all_pixels == (cls_id + 1))
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights.astype(np.float32)


CONFIDENCE_WEIGHTS = {1: 1.0, 2: 0.667, 3: 0.333}
