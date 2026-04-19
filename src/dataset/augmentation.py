import torch
import torchvision.transforms.functional as TF
import random


def augment_patch(image, mask, conf):
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