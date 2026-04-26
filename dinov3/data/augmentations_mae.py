# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class DataAugmentationMAE:
    """
    Single-crop augmentation for MAE pretraining.
    No colour jitter — MAE reconstructs raw pixels, so aggressive colour
    distortions would create an ill-posed reconstruction target.
    """

    def __init__(
        self,
        global_crops_size=224,
        global_crops_scale=(0.2, 1.0),
        horizontal_flips=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        transforms_list = [
            v2.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
        if horizontal_flips:
            transforms_list.append(v2.RandomHorizontalFlip())
        transforms_list += [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
        self.transform = v2.Compose(transforms_list)

    def __call__(self, image):
        crop = self.transform(image)
        return {
            "global_crops": [crop],
            "global_crops_teacher": [crop],
            "local_crops": [],
            "offsets": (),
        }
