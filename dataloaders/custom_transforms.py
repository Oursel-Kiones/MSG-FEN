# dataloaders/custom_transforms.py
"""
This module provides a set of custom transformations for data augmentation in
semantic and panoptic segmentation tasks. The transformations are designed to
operate on a dictionary of samples, ensuring that geometric transformations
are applied consistently to both the image and all associated masks.
"""

import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms.functional as TF
from typing import Dict, Any, List, Tuple, Optional

# /workspace/deep参考1/dataloaders/custom_transforms.py

import torch
import random
# ... (其他 import 语句)

# =========================================================================
# <<< 在这里添加 ExtCompose 类的定义 >>>
# =========================================================================
class ExtCompose(object):
    """
    A custom compose class that applies a list of transformations to a
    sample dictionary. This is necessary because torchvision.transforms.Compose
    only works on single PIL Images or Tensors, not on dictionaries.
    """
    def __init__(self, transforms):
        """
        Args:
            transforms (list of transform objects): A list of transforms to apply.
        """
        self.transforms = transforms

    def __call__(self, sample):
        """
        Applies the transformations sequentially to the sample.

        Args:
            sample (dict): The input sample dictionary.

        Returns:
            dict: The transformed sample dictionary.
        """
        for t in self.transforms:
            sample = t(sample)
        return sample
# =========================================================================

# ... (下面是您文件中已有的其他类，如 Normalize, ToTensor 等) ...
# =========================================================================
# <<< 全局常量定义 (来自您的设计) >>>
# =========================================================================

# A list of all keys corresponding to masks that should undergo geometric transformations.
ALL_MASK_KEYS: List[str] = [
    'label',           # Main semantic label
    'stuff_gt',        # Stuff segmentation ground truth
    'object_gt',       # Thing segmentation ground truth (future use)
    'objectness_gt',   # Coarse objectness ground truth
    'boundary_gt',     # Boundary detection ground truth (future use)
"small_object_gt"  # Small object mask ground truth (future use)
]

# Default fill values for padding operations during transformations.
# Ensures that different types of masks are padded with their respective ignore/default values.
_DEFAULT_FILL_VALUES: Dict[str, int] = {
    'image': 0,
    'label': 255,          # Typically, 255 is the ignore index for semantic loss
    'stuff_gt': 255,       # Ignore index for stuff loss
    'object_gt': 255,      # Ignore index for object loss
    'objectness_gt': 255,  # Ignore index for objectness BCE loss
    'boundary_gt': 0,        # Boundary masks are often binary; 0 is a safe fill value
    "small_object_gt": 0   # Binary mask; 0 is a safe fill value
}

# =========================================================================

Sample = Dict[str, Any]

class Normalize(object):
    """
    Normalizes a tensor image with a given mean and standard deviation.
    This transform should be applied after ToTensor.
    """
    def __init__(self, mean: Tuple[float, ...] = (0., 0., 0.), std: Tuple[float, ...] = (1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Sample) -> Sample:
        """
        Args:
            sample (Dict): A sample dictionary containing the image tensor.
        Returns:
            Dict: The sample dictionary with the image normalized.
        """
        if 'image' not in sample or not isinstance(sample['image'], torch.Tensor):
            raise TypeError("Input 'image' for Normalize must be a torch.Tensor. "
                            "Ensure ToTensor is applied before Normalize.")
        sample['image'] = TF.normalize(sample['image'], self.mean, self.std)
        return sample


class ToTensor(object):
    """
    Converts PIL Images in a sample dictionary to PyTorch Tensors.

    This class performs precise type casting based on the key:
    - 'image' is converted to a FloatTensor and scaled to [0, 1].
    - Categorical labels (e.g., 'label', 'stuff_gt') are converted to LongTensor.
    - Probabilistic/binary masks (e.g., 'objectness_gt') are converted to
      FloatTensor and unsqueezed to have a channel dimension.
    """
    def __init__(self):
        # Keys for masks that represent distinct classes.
        self.categorical_keys = ['label', 'stuff_gt', 'object_gt']
        # Keys for masks that represent probabilities or binary values.
        self.probabilistic_keys = ['objectness_gt', 'boundary_gt', 'small_object_gt']

    def __call__(self, sample: Sample) -> Sample:
        """
        Args:
            sample (Dict): A sample dictionary where values can be PIL Images.
        Returns:
            Dict: The sample dictionary with PIL Images converted to Tensors.
        """
        for key, value in sample.items():
            if not isinstance(value, Image.Image):
                continue

            if key == 'image':
                # Convert image to FloatTensor (C, H, W) in range [0.0, 1.0]
                sample[key] = TF.to_tensor(value)
            elif key in self.categorical_keys:
                # Convert categorical label to LongTensor (H, W)
                mask_np = np.array(value, dtype=np.int64)
                sample[key] = torch.from_numpy(mask_np)
            elif key in self.probabilistic_keys:
                # Convert probabilistic label to FloatTensor (1, H, W)
                mask_np = np.array(value, dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_np)
                sample[key] = mask_tensor.unsqueeze(0)
        return sample


class RandomHorizontalFlip(object):
    """
    Applies a horizontal flip randomly to the image and all associated masks.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            if 'image' in sample:
                sample['image'] = TF.hflip(sample['image'])
            for key in ALL_MASK_KEYS:
                if key in sample:
                    sample[key] = TF.hflip(sample[key])
        return sample


class RandomRotate(object):
    """
    Applies a random rotation to the image and all associated masks.
    """
    def __init__(self, degrees: float, fill_values_map: Optional[Dict[str, int]] = None):
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.fill_values_map = fill_values_map if fill_values_map is not None else _DEFAULT_FILL_VALUES

    def __call__(self, sample: Sample) -> Sample:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # Rotate image with bilinear interpolation
        image_fill = self.fill_values_map.get('image', 0)
        if 'image' in sample and isinstance(sample['image'], Image.Image):
            sample['image'] = TF.rotate(sample['image'], angle,
                                        interpolation=TF.InterpolationMode.BILINEAR,
                                        fill=image_fill)

        # Rotate all masks with nearest neighbor interpolation
        for key in ALL_MASK_KEYS:
            if key in sample and isinstance(sample[key], Image.Image):
                fill_val = self.fill_values_map.get(key, 0)
                sample[key] = TF.rotate(sample[key], angle,
                                        interpolation=TF.InterpolationMode.NEAREST,
                                        fill=fill_val)
        return sample


class RandomGaussianBlur(object):
    """
    Applies Gaussian blur to the image with a certain probability.
    """
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, sample: Sample) -> Sample:
        if 'image' in sample and isinstance(sample['image'], Image.Image) and random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            sample['image'] = sample['image'].filter(ImageFilter.GaussianBlur(radius=radius))
        return sample


class RandomScaleCrop(object):
    """
    Randomly scales and then crops the image and all associated masks to a fixed size.
    """
    def __init__(self, base_size: int, crop_size: int, scale_range: Tuple[float, float] = (0.5, 2.0),
                 fill_values_map: Optional[Dict[str, int]] = None):
        self.base_size = base_size
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.scale_range = scale_range
        self.fill_values_map = fill_values_map if fill_values_map is not None else _DEFAULT_FILL_VALUES

    def __call__(self, sample: Sample) -> Sample:
        if 'image' not in sample or not isinstance(sample['image'], Image.Image):
            raise TypeError("RandomScaleCrop requires 'image' to be a PIL Image.")

        img = sample['image']
        # Collect all masks to be transformed
        masks = {key: sample[key] for key in ALL_MASK_KEYS if key in sample and isinstance(sample[key], Image.Image)}
        
        # --- Scaling ---
        w_orig, h_orig = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        short_size = int(self.base_size * scale)
        
        if h_orig < w_orig:
            new_h, new_w = short_size, int(short_size * w_orig / h_orig)
        else:
            new_h, new_w = int(short_size * h_orig / w_orig), short_size
            
        img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
        for key in masks:
            masks[key] = TF.resize(masks[key], (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

        # --- Padding ---
        th_crop, tw_crop = self.crop_size
        pad_h, pad_w = max(th_crop - new_h, 0), max(tw_crop - new_w, 0)
        padding = [pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2)]
        
        if any(p > 0 for p in padding):
            img = TF.pad(img, padding, fill=self.fill_values_map.get('image', 0), padding_mode='constant')
            for key in masks:
                masks[key] = TF.pad(masks[key], padding, fill=self.fill_values_map.get(key, 0), padding_mode='constant')
                
        # --- Cropping ---
        i, j, h, w = RandomCrop.get_params(img, self.crop_size)
        sample['image'] = TF.crop(img, i, j, h, w)
        for key in masks:
            sample[key] = TF.crop(masks[key], i, j, h, w)

        return sample

class RandomCrop:
    """Helper class for RandomScaleCrop to get random crop parameters."""
    @staticmethod
    def get_params(img: Image.Image, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = TF.get_image_size(img)
        th, tw = output_size
        if h < th or w < tw:
            raise ValueError(f"Required crop size {output_size} is larger than input image size {(h, w)}.")
        if w == tw and h == th:
            return 0, 0, h, w
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, w


class FixScaleCrop(object):
    """
    Resizes the image to maintain aspect ratio and fit the crop size, then performs a center crop.
    """
    def __init__(self, crop_size: int):
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    def __call__(self, sample: Sample) -> Sample:
        if 'image' not in sample or not isinstance(sample['image'], Image.Image):
            raise TypeError("FixScaleCrop requires 'image' to be a PIL Image.")

        img = sample['image']
        masks = {key: sample[key] for key in ALL_MASK_KEYS if key in sample and isinstance(sample[key], Image.Image)}

        w, h = img.size
        if w > h:
            oh = self.crop_size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[1]
            oh = int(1.0 * h * ow / w)
            
        img = TF.resize(img, (oh, ow), interpolation=TF.InterpolationMode.BILINEAR)
        for key in masks:
            masks[key] = TF.resize(masks[key], (oh, ow), interpolation=TF.InterpolationMode.NEAREST)

        # Center crop
        w, h = img.size
        i = (h - self.crop_size[0]) // 2
        j = (w - self.crop_size[1]) // 2
        
        sample['image'] = TF.crop(img, i, j, self.crop_size[0], self.crop_size[1])
        for key in masks:
            sample[key] = TF.crop(masks[key], i, j, self.crop_size[0], self.crop_size[1])

        return sample


class FixedResize(object):
    """Resizes the image and all associated masks to a fixed size."""
    def __init__(self, size: int):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, sample: Sample) -> Sample:
        if 'image' in sample:
            sample['image'] = TF.resize(sample['image'], self.size, interpolation=TF.InterpolationMode.BILINEAR)
        
        for key in ALL_MASK_KEYS:
            if key in sample:
                sample[key] = TF.resize(sample[key], self.size, interpolation=TF.InterpolationMode.NEAREST)
        return sample