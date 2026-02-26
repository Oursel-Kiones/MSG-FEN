# /workspace/deep参考1/dataloaders/utils.py (最小化修改版)
"""
This module provides utility functions for data loading and visualization,
including a custom collate function for handling variable-sized inputs and
functions for decoding segmentation maps into color images.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# =========================================================================
# <<< 我们新添加的自定义 collate_fn >>>
# =========================================================================

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A custom collate_fn for handling batches where samples may have different sizes.

    Instead of stacking tensors (which would fail if sizes don't match), this
    function organizes the data into a dictionary where each key corresponds
    to a list of that key's values from all samples in the batch.

    Args:
        batch (List[Dict[str, Any]]): A list of sample dictionaries, where each
                                      dictionary is an output of the Dataset's
                                      __getitem__ method.

    Returns:
        Dict[str, Any]: A dictionary with the same keys as the input samples,
                        but where each value is a list of all corresponding
                        values from the batch.
    """
    keys = batch[0].keys()
    collated_batch = {key: [d[key] for d in batch] for key in keys}

    return collated_batch

# =========================================================================
# <<< 您提供的现有函数 (已添加类型提示和文档字符串) >>>
# =========================================================================

def decode_seg_map_sequence(label_masks: torch.Tensor, dataset: str = 'pascal') -> torch.Tensor:
    """
    Decodes a sequence of segmentation label masks into a batch of color images.

    Args:
        label_masks (torch.Tensor): A batch of label masks, typically with
                                    shape (B, H, W).
        dataset (str): The dataset name ('pascal', 'coco', or 'cityscapes') to
                       determine the color palette.

    Returns:
        torch.Tensor: A batch of decoded color images with shape (B, 3, H, W).
    """
    rgb_masks = []
    for label_mask in label_masks:
        np_mask = label_mask.numpy() if isinstance(label_mask, torch.Tensor) else label_mask
        rgb_mask = decode_segmap(np_mask, dataset, plot=False)
        rgb_masks.append(rgb_mask)
    
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


# === 关键修复点 1：为 decode_segmap 添加 is_thing 参数 ===
def decode_segmap(label_mask: np.ndarray, dataset: str = 'pascal', plot: bool = False, is_thing: bool = False) -> np.ndarray:
    """
    Decodes a segmentation class label array into a color image.

    Args:
        label_mask (np.ndarray): An (H, W) array of integer class labels.
        dataset (str): The dataset name ('pascal', 'coco', or 'cityscapes')
                       to determine the color palette.
        plot (bool, optional): If True, displays the resulting color image.
        is_thing (bool, optional): If True and dataset is 'cityscapes', uses the
                                   specific 12-class 'thing' colormap.

    Returns:
        np.ndarray: The resulting decoded color image of shape (H, W, 3)
                    with pixel values in [0, 1].
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        # --- 内部逻辑修改 ---
        if is_thing:
            n_classes = 12
            label_colours = get_cityscapes_thing_labels()
        else:
            n_classes = 19
            label_colours = get_cityscapes_labels()
        # --- 结束内部逻辑修改 ---
    else:
        raise NotImplementedError(f"No color palette implemented for dataset '{dataset}'")

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    
    # 使用 np.unique 提高效率，只遍历图片中存在的类别
    present_classes = np.unique(label_mask)
    for class_index in present_classes:
        if class_index < n_classes: # 确保索引在调色板范围内
            idx = label_mask == class_index
            rgb[idx] = label_colours[class_index]
        
    rgb_float = rgb.astype(np.float32) / 255.0

    if plot:
        plt.imshow(rgb_float)
        plt.axis('off')
        plt.show()
    
    return rgb_float


def encode_segmap(mask: np.ndarray) -> np.ndarray:
    """
    Encodes a color-coded segmentation label image back into a class map.
    (Primarily for PASCAL VOC format)

    Args:
        mask (np.ndarray): A raw, color-coded segmentation label image of
                           shape (H, W, 3).

    Returns:
        np.ndarray: A class map of shape (H, W) with integer class indices.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for i, label_color in enumerate(get_pascal_labels()):
        match_pixels = np.where(np.all(mask == label_color, axis=-1))
        label_mask[match_pixels] = i
        
    return label_mask.astype(int)


def get_cityscapes_labels() -> np.ndarray:
    """
    Returns the color palette for the Cityscapes dataset as a numpy array.
    Shape: (19, 3)
    """
    return np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ], dtype=np.uint8)

# === 关键修复点 2：新增 get_cityscapes_thing_labels 函数 ===
def get_cityscapes_thing_labels() -> np.ndarray:
    """
    Returns the specific color palette for the 12 'thing' classes of the
    Cityscapes dataset. The order corresponds to object_gt labels (0-11).
    """
    return np.array([
        # Colors are sourced from the original get_cityscapes_labels()
        [190, 153, 153],  # 0: fence (original index 4)
        [153, 153, 153],  # 1: pole (original index 5)
        [250, 170, 30],   # 2: traffic light (original index 6)
        [220, 220, 0],    # 3: traffic sign (original index 7)
        [220, 20, 60],    # 4: person (original index 11)
        [255, 0, 0],      # 5: rider (original index 12)
        [0, 0, 142],      # 6: car (original index 13)
        [0, 0, 70],       # 7: truck (original index 14)
        [0, 60, 100],     # 8: bus (original index 15)
        [0, 80, 100],     # 9: train (original index 16)
        [0, 0, 230],      # 10: motorcycle (original index 17)
        [119, 11, 32]     # 11: bicycle (original index 18)
    ], dtype=np.uint8)


def get_pascal_labels() -> np.ndarray:
    """
    Returns the color palette for the PASCAL VOC dataset as a numpy array.
    Shape: (21, 3)
    """
    return np.asarray([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)