# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

logger = logging.getLogger("dinov3")


def make_interpolation_mode(mode_str: str) -> v2.InterpolationMode:
    return {mode.value: mode for mode in v2.InterpolationMode}[mode_str]


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class ContentAwareRandomResizedCrop(v2.RandomResizedCrop):
    """
    RandomResizedCrop that retries if the cropped region is too dark.
    Designed for images with large dead zones (e.g. endoscopic ultrasound fan shape).
    When min_content_mean == 0.0 the check is skipped and behaviour is
    identical to v2.RandomResizedCrop (safe default, zero overhead).
    """

    def __init__(self, *args, min_content_mean: float = 0.0, max_retries: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_content_mean = min_content_mean
        self.max_retries = max_retries

    def forward(self, img):
        if self.min_content_mean <= 0.0:
            return super().forward(img)
        candidate = super().forward(img)
        for _ in range(self.max_retries - 1):
            if self._mean_intensity(candidate) >= self.min_content_mean:
                return candidate
            candidate = super().forward(img)
        # Exhausted retries — return last candidate regardless
        return candidate

    @staticmethod
    def _mean_intensity(img) -> float:
        if isinstance(img, Image.Image):
            return np.asarray(img, dtype=np.float32).mean() / 255.0
        # Tensor: normalise uint8 to [0, 1] if needed
        t = img.float()
        if t.max() > 1.0:
            t = t / 255.0
        return t.mean().item()


class _DepthAttenuationImpl(torch.nn.Module):
    """
    Acoustic depth-attenuation: I(r,c) = I_orig(r,c) * exp(-alpha * r / H).
    alpha is sampled uniformly from [alpha_min, alpha_max] each call.
    Operates on PIL Images; passes other types through unchanged.
    """

    def __init__(self, alpha_min: float = 0.01, alpha_max: float = 0.05):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def forward(self, img):
        if not isinstance(img, Image.Image):
            return img
        alpha = float(np.random.uniform(self.alpha_min, self.alpha_max))
        arr = np.asarray(img, dtype=np.float32)   # (H, W) or (H, W, C)
        H = arr.shape[0]
        rows = np.arange(H, dtype=np.float32)
        decay = np.exp(-alpha * rows / H)          # shape (H,)
        if arr.ndim == 3:
            decay = decay[:, np.newaxis, np.newaxis]
        else:
            decay = decay[:, np.newaxis]
        result = np.clip(arr * decay, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode=img.mode)


class DepthAttenuation(v2.RandomApply):
    """
    Randomly applies acoustic depth-attenuation to PIL Images.
    Simulates exponential signal loss with tissue depth in ultrasound.
    p=0.0 (default) means never applied, preserving existing behaviour.
    """

    def __init__(
        self,
        *,
        p: float = 0.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.05,
    ):
        transform = _DepthAttenuationImpl(alpha_min=alpha_min, alpha_max=alpha_max)
        super().__init__(transforms=[transform], p=p)


class _GaussianShadowImpl(torch.nn.Module):
    """
    Acoustic shadow dropout: I(r,c) = I_orig(r,c) * (1 - strength * G(r,c)).
    G is a 2-D Gaussian centred at a random (cx, cy), sigma = sigma_ratio * min(H, W).
    strength ~ U[intensity_min, intensity_max].
    Operates on PIL Images; passes other types through unchanged.
    """

    def __init__(
        self,
        intensity_min: float = 0.3,
        intensity_max: float = 0.8,
        sigma_ratio: float = 0.15,
    ):
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.sigma_ratio = sigma_ratio

    def forward(self, img):
        if not isinstance(img, Image.Image):
            return img
        arr = np.asarray(img, dtype=np.float32)   # (H, W) or (H, W, C)
        H, W = arr.shape[:2]
        sigma = self.sigma_ratio * min(H, W)
        cx = float(np.random.uniform(0, W))
        cy = float(np.random.uniform(0, H))
        strength = float(np.random.uniform(self.intensity_min, self.intensity_max))
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)              # (H, W)
        G = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        shadow = 1.0 - strength * G               # (H, W)
        if arr.ndim == 3:
            shadow = shadow[:, :, np.newaxis]
        result = np.clip(arr * shadow, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode=img.mode)


class GaussianShadow(v2.RandomApply):
    """
    Randomly applies a Gaussian acoustic-shadow mask to PIL Images.
    Simulates signal dropout from ribs or calcified structures in ultrasound.
    p=0.0 (default) means never applied, preserving existing behaviour.
    """

    def __init__(
        self,
        *,
        p: float = 0.0,
        intensity_min: float = 0.3,
        intensity_max: float = 0.8,
        sigma_ratio: float = 0.15,
    ):
        transform = _GaussianShadowImpl(
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            sigma_ratio=sigma_ratio,
        )
        super().__init__(transforms=[transform], p=p)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


def make_base_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=mean, std=std),
        ]
    )


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [v2.ToImage(), v2.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built classification train transform\n{transform}")
    return transform


def make_resize_transform(
    *,
    resize_size: int,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
):
    assert not (resize_square and resize_large_side), "These two options can not be set together"
    if resize_square:
        logger.info("resizing image as a square")
        size = (resize_size, resize_size)
        transform = v2.Resize(size=size, interpolation=interpolation)
        return transform
    elif resize_large_side:
        logger.info("resizing based on large side")
        transform = v2.Resize(size=None, max_size=resize_size, interpolation=interpolation)
        return transform
    else:
        transform = v2.Resize(resize_size, interpolation=interpolation)
        return transform


# Derived from make_classification_eval_transform() with more control over resize and crop
def make_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    transforms_list = [v2.ToImage()]
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=resize_square,
        resize_large_side=resize_large_side,
        interpolation=interpolation,
    )
    transforms_list.append(resize_transform)
    if crop_size:
        transforms_list.append(v2.CenterCrop(crop_size))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built eval transform\n{transform}")
    return transform


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )


def voc2007_classification_target_transform(label, n_categories=20):
    one_hot = torch.zeros(n_categories, dtype=int)
    for instance in label.instances:
        one_hot[instance.category_id] = True
    return one_hot


def imaterialist_classification_target_transform(label, n_categories=294):
    one_hot = torch.zeros(n_categories, dtype=int)
    one_hot[label.attributes] = True
    return one_hot


def get_target_transform(dataset_str):
    if "VOC2007" in dataset_str:
        return voc2007_classification_target_transform
    elif "IMaterialist" in dataset_str:
        return imaterialist_classification_target_transform
    return None
