"""Patch-level data augmentation for MARIDA tiles.

`augment_patch` is the segmentation-aware version (image + per-pixel mask +
confidence). `augment_image_only` is the multi-label classification version —
flips and 90-degree rotations are label-preserving for tile-level multi-label
targets (rotating a tile that contains debris still contains debris), so we
can apply them freely without touching the label vector.
"""
from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF


def augment_patch(image, mask, conf):
    """Segmentation augment: keeps mask + confidence aligned with the image."""
    angle = random.choice([-90, 0, 90, 180])
    if angle != 0:
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)
        conf = TF.rotate(conf.unsqueeze(0), angle).squeeze(0)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        conf = TF.hflip(conf.unsqueeze(0)).squeeze(0)
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        conf = TF.vflip(conf.unsqueeze(0)).squeeze(0)
    return image, mask, conf


def augment_image_only(
    image: torch.Tensor,
    noise_std: float = 0.0,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    rotate_p: float = 0.75,
) -> torch.Tensor:
    """Label-preserving augment for tile-level multi-label classification.

    All ops are dihedral symmetries of a square — they don't change *what*
    is in the tile, only its orientation. Adding a small bit of Gaussian
    noise on top further regularizes against sensor banding.

    Args:
        image: (C, H, W) float tensor (already normalized).
        noise_std: stddev of additive Gaussian noise in normalized units.
                   0 disables noise. 0.05 is a reasonable starting point.
        hflip_p / vflip_p / rotate_p: probabilities for each op. The 90-degree
                   rotation picks uniformly from {90, 180, 270} when triggered.
    """
    if random.random() < rotate_p:
        k = random.choice((1, 2, 3))  # number of 90-degree rotations
        image = torch.rot90(image, k=k, dims=(-2, -1))
    if random.random() < hflip_p:
        image = TF.hflip(image)
    if random.random() < vflip_p:
        image = TF.vflip(image)
    if noise_std > 0:
        image = image + torch.randn_like(image) * noise_std
    return image