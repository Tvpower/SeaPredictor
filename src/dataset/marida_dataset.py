import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio

WATER_MERGE = {12: 7, 13: 7, 14: 7, 15: 7}


def load_patch(patch_path):
    with rasterio.open(patch_path) as src:
        data = src.read().astype(np.float32)
        bounds = src.bounds
        crs = src.crs
    return data, {'bounds': bounds, 'crs': crs}


def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1).astype(np.int64)


def load_confidence(conf_path):
    with rasterio.open(conf_path) as src:
        return src.read(1).astype(np.int64)


def aggregate_classes(mask, merge_map=WATER_MERGE):
    out = mask.copy()
    for old_val, new_val in merge_map.items():
        out[mask == old_val] = new_val
    return out


def binarize_debris(mask):
    return (mask == 1).astype(np.int64)


class MARIDADataset(Dataset):
    """MARIDA marine debris dataset loader.

    Split files contain IDs without the 'S2_' prefix (e.g. '1-12-19_48MYU_0'),
    but on disk folders and files are named with 'S2_' prepended. _resolve_paths
    handles this mapping.
    """

    def __init__(self, split_file, patches_dir, augment=False, add_indices=True,
                 aggregate=True, binary=False, norm_stats=None):
        self.patches_dir = Path(patches_dir)
        self.augment = augment
        self.add_indices = add_indices
        self.aggregate = aggregate
        self.binary = binary
        self.norm_stats = norm_stats
        with open(split_file) as f:
            self.patch_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.patch_ids)

    def _resolve_paths(self, patch_id):
        # Split IDs lack S2_ prefix; folders and files on disk have it.
        parts = patch_id.rsplit('_', 1)
        folder = 'S2_' + parts[0]
        filename_stem = 'S2_' + patch_id
        base = self.patches_dir / folder / filename_stem
        return str(base) + '.tif', str(base) + '_cl.tif', str(base) + '_conf.tif'

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        patch_path, mask_path, conf_path = self._resolve_paths(patch_id)

        image, geo_meta = load_patch(patch_path)
        mask = load_mask(mask_path)
        conf = load_confidence(conf_path)

        if self.add_indices:
            from .spectral_indices import stack_indices
            idx_stack = stack_indices(image)
            idx_stack = np.nan_to_num(idx_stack, nan=0.0, posinf=0.0, neginf=0.0)
            image = np.concatenate([image, idx_stack], axis=0)

        if self.norm_stats is not None:
            from .normalization import normalize_bands
            image = normalize_bands(image, self.norm_stats)

        if self.binary:
            mask = binarize_debris(mask)
        elif self.aggregate:
            mask = aggregate_classes(mask)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()
        conf = torch.from_numpy(conf).long()

        if self.augment:
            from .augmentation import augment_patch
            image, mask, conf = augment_patch(image, mask, conf)

        return image, mask, conf
