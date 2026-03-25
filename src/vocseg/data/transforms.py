"""Paired image/mask transforms."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from vocseg.constants import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD


PairTransformOutput = tuple[Any, Any, dict[str, Any]]


def _ensure_size_tuple(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    return size


class ComposePairs:
    def __init__(self, transforms: list[Any]):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        meta: dict[str, Any] = {}
        for transform in self.transforms:
            image, mask, extra = transform(image, mask)
            meta.update(extra)
        return image, mask, meta


class RandomScale:
    def __init__(self, min_scale: float, max_scale: float):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        scale = random.uniform(self.min_scale, self.max_scale)
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))
        image = image.resize((new_width, new_height), resample=Image.BILINEAR)
        mask = mask.resize((new_width, new_height), resample=Image.NEAREST)
        return image, mask, {"scale_factor": scale}


class RandomCropOrPad:
    def __init__(self, size: int | tuple[int, int]):
        self.size = _ensure_size_tuple(size)

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        target_h, target_w = self.size
        pad_h = max(0, target_h - image.height)
        pad_w = max(0, target_w - image.width)
        if pad_h or pad_w:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=IGNORE_INDEX)

        if image.height == target_h and image.width == target_w:
            top = 0
            left = 0
        else:
            top = random.randint(0, image.height - target_h)
            left = random.randint(0, image.width - target_w)

        image = TF.crop(image, top, left, target_h, target_w)
        mask = TF.crop(mask, top, left, target_h, target_w)
        return image, mask, {"crop_top": top, "crop_left": left}


class HorizontalFlip:
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        flipped = random.random() < self.probability
        if flipped:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask, {"flipped": flipped}


class ImageOnlyColorJitter:
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        self.jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        return self.jitter(image), mask, {}


class ImageOnlyGaussianBlur:
    def __init__(self, probability: float, radius_min: float, radius_max: float):
        self.probability = probability
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        if random.random() < self.probability:
            radius = random.uniform(self.radius_min, self.radius_max)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            return image, mask, {"gaussian_blur_radius": radius}
        return image, mask, {"gaussian_blur_radius": 0.0}


class ResizeAndPad:
    def __init__(self, size: int | tuple[int, int]):
        self.size = _ensure_size_tuple(size)

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        target_h, target_w = self.size
        scale = min(target_w / image.width, target_h / image.height)
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))

        image_resized = image.resize((new_width, new_height), resample=Image.BILINEAR)
        mask_resized = mask.resize((new_width, new_height), resample=Image.NEAREST)

        pad_w = target_w - new_width
        pad_h = target_h - new_height
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        image_padded = ImageOps.expand(
            image_resized,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=0,
        )
        mask_padded = ImageOps.expand(
            mask_resized,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=IGNORE_INDEX,
        )
        meta = {
            "eval_scale_factor": scale,
            "resized_size": [new_height, new_width],
            "padding": [pad_left, pad_top, pad_right, pad_bottom],
        }
        return image_padded, mask_padded, meta


class ToTensorAndNormalize:
    def __init__(self, mean: tuple[float, float, float] = IMAGENET_MEAN, std: tuple[float, float, float] = IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image) -> PairTransformOutput:
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        mask_array = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor, {}


@dataclass
class AugmentationSpec:
    name: str
    scale_range: tuple[float, float]
    hflip: float
    color_jitter: tuple[float, float, float, float]
    blur_probability: float
    blur_radius: tuple[float, float]


def resolve_augmentation_spec(name: str) -> AugmentationSpec:
    presets = {
        "none": AugmentationSpec(
            name="none",
            scale_range=(1.0, 1.0),
            hflip=0.0,
            color_jitter=(0.0, 0.0, 0.0, 0.0),
            blur_probability=0.0,
            blur_radius=(0.0, 0.0),
        ),
        "standard": AugmentationSpec(
            name="standard",
            scale_range=(0.5, 2.0),
            hflip=0.5,
            color_jitter=(0.2, 0.2, 0.2, 0.05),
            blur_probability=0.2,
            blur_radius=(0.1, 1.0),
        ),
        "strong": AugmentationSpec(
            name="strong",
            scale_range=(0.5, 2.25),
            hflip=0.5,
            color_jitter=(0.35, 0.35, 0.35, 0.08),
            blur_probability=0.35,
            blur_radius=(0.1, 1.5),
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown augmentation preset: {name}")
    return presets[name]


def build_train_transforms(data_cfg: dict[str, Any]) -> ComposePairs:
    image_size = data_cfg.get("crop_size", 512)
    aug_name = data_cfg.get("augmentation", {}).get("name", "standard")
    aug = resolve_augmentation_spec(aug_name)
    transforms: list[Any] = []
    if not math.isclose(aug.scale_range[0], 1.0) or not math.isclose(aug.scale_range[1], 1.0):
        transforms.append(RandomScale(*aug.scale_range))
    transforms.append(RandomCropOrPad(image_size))
    if aug.hflip > 0:
        transforms.append(HorizontalFlip(aug.hflip))
    if any(value > 0 for value in aug.color_jitter):
        transforms.append(ImageOnlyColorJitter(*aug.color_jitter))
    if aug.blur_probability > 0:
        transforms.append(ImageOnlyGaussianBlur(aug.blur_probability, *aug.blur_radius))
    transforms.append(ToTensorAndNormalize())
    return ComposePairs(transforms)


def build_eval_transforms(data_cfg: dict[str, Any]) -> ComposePairs:
    image_size = data_cfg.get("eval_size", data_cfg.get("crop_size", 512))
    return ComposePairs(
        [
            ResizeAndPad(image_size),
            ToTensorAndNormalize(),
        ]
    )
