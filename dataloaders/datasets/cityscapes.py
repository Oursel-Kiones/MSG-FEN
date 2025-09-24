# -*- coding: utf-8 -*-
"""
cityscapes.py

PyTorch Dataset for the Cityscapes dataset, specifically designed for multi-task
learning models like Panoptic-DeepLab. It generates multiple ground truth masks
for semantic, stuff, object, and boundary prediction from a single label file.

The testing and visualization code is safely encapsulated within the
`if __name__ == '__main__':` block to prevent argument parsing conflicts when
this module is imported by other scripts.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

# --- Robust Imports with Fallbacks ---
try:
    from mypath import Path
except ImportError:
    print("Warning: Could not import 'mypath.Path'. Using a temporary Path class.")
    print("         Please ensure 'mypath.py' is in your PYTHONPATH or project root.")
    class Path:
        @staticmethod
        def db_root_dir(dataset_name):
            if dataset_name == 'cityscapes':
                # !!! IMPORTANT !!!: Modify this path to your Cityscapes dataset root directory.
                # Example: '/data/cityscapes' or 'C:/datasets/cityscapes'
                return '/path/to/your/cityscapes'
            raise NotImplementedError(f"Unknown DB root for dataset: {dataset_name}")

try:
    from dataloaders import custom_transforms as tr
except ImportError:
    print("Error: Failed to import 'custom_transforms' from 'dataloaders'.")
    print("       Custom data augmentations will not be available.")
    # Define placeholder classes to prevent crashes if custom_transforms is missing
    class PlaceholderTransform:
        def __call__(self, sample): return sample
    class tr:
        RandomHorizontalFlip = staticmethod(lambda **kwargs: PlaceholderTransform())
        RandomScaleCrop = staticmethod(lambda **kwargs: PlaceholderTransform())
        RandomGaussianBlur = staticmethod(lambda **kwargs: PlaceholderTransform())
        ToTensor = staticmethod(lambda **kwargs: PlaceholderTransform())
        Normalize = staticmethod(lambda **kwargs: PlaceholderTransform())
        FixScaleCrop = staticmethod(lambda **kwargs: PlaceholderTransform())
        FixedResize = staticmethod(lambda **kwargs: PlaceholderTransform())


class CityscapesSegmentation(data.Dataset):
    """
    Cityscapes Dataset.
    This class can be used for both semantic segmentation and panoptic-style multi-task learning.
    """
    NUM_CLASSES = 19
    
    # ==================== CORE CATEGORY DEFINITIONS ====================
    # These definitions are based on the standard Cityscapes `trainId` mapping.
    
    # Stuff categories: Classes that are amorphous and non-countable (e.g., road, sky).
    STUFF_TRAIN_IDS = [0, 1, 2, 3, 8, 9, 10]  # road, sidewalk, building, wall, vegetation, terrain, sky
    NUM_STUFF_CLASSES = len(STUFF_TRAIN_IDS)  # 7

    # Object categories: Classes that represent distinct, countable instances (e.g., car, person).
    OBJECT_TRAIN_IDS = [4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18] # fence, pole, traffic light, ...
    NUM_OBJECT_CLASSES = len(OBJECT_TRAIN_IDS) # 12

    # A subset of small objects for special attention/loss weighting.
    SMALL_OBJECT_TRAIN_IDS = [4, 5, 12, 17] # fence, pole, rider, motorcycle
    # ===================================================================

    # --- Fill values for data augmentation transforms ---
    # Defines how to pad each ground truth map during transformations.
    FILL_VALUES = {
        'image': 0,
        'label': 255,           # Standard semantic segmentation GT
        'stuff_gt': 255,        # Stuff segmentation GT
        'object_gt': 255,       # Object segmentation GT
        'objectness_gt': 255,   # Binary objectness mask
        'boundary_gt': 0,       # Boundary mask (0 is non-boundary)
        'small_object_gt': 0,   # Small object mask (0 is non-small-object)
    }

    def __init__(self, args, root=None, split="train", transform=None):
        self.root = root if root is not None else Path.db_root_dir('cityscapes')
        self.split = split
        self.args = args
        self.files = {}
        self.transform_applied_by_user = transform

        # Expose class properties as instance properties for easy external access
        self.num_classes = self.NUM_CLASSES
        self.num_stuff_classes = self.NUM_STUFF_CLASSES
        self.num_object_classes = self.NUM_OBJECT_CLASSES
        self.ignore_index = 255

        # Define paths
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='_leftImg8bit.png')

        # Cityscapes official label mapping constants
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        
        # --- Create efficient Look-Up Tables (LUTs) for GT generation ---
        self.stuff_gt_map = np.full((256,), self.ignore_index, dtype=np.uint8)
        for i, train_id in enumerate(self.STUFF_TRAIN_IDS):
            self.stuff_gt_map[train_id] = i

        self.object_gt_map = np.full((256,), self.ignore_index, dtype=np.uint8)
        for i, train_id in enumerate(self.OBJECT_TRAIN_IDS):
            self.object_gt_map[train_id] = i
        # --------------------------------------------------------------------

        if not self.files[split]:
            raise FileNotFoundError(f"No files found for split='{split}' in {self.images_base}. Please check the path.")
        
        print(f"Found {len(self.files[split])} images in '{split}' split.")
        print(f"  - Stuff classes: {self.NUM_STUFF_CLASSES}, Object classes: {self.NUM_OBJECT_CLASSES}")

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # --- 1. Load image and original label file ---
        img_path = self.files[self.split][index].rstrip()
        city_folder = os.path.basename(os.path.dirname(img_path))
        base_filename = os.path.basename(img_path).replace('_leftImg8bit.png', '')
        lbl_filename = base_filename + '_gtFine_labelIds.png'
        lbl_path = os.path.join(self.annotations_base, city_folder, lbl_filename)

        _img = Image.open(img_path).convert('RGB')
        try:
            _tmp_original_labels_pil = Image.open(lbl_path)
        except FileNotFoundError:
            # Handle cases where a label might be missing by creating a blank one
            _tmp_original_labels_pil = Image.new('L', _img.size, color=0)
        
        _tmp_original_labels_np = np.array(_tmp_original_labels_pil, dtype=np.uint8)

        # --- 2. Generate all required ground truth maps ---
        
        # a. `label`: Standard 19-class semantic segmentation GT (trainId format)
        _encoded_labels_np = self.encode_segmap(_tmp_original_labels_np.copy())
        
        # b. `stuff_gt`: 7-class Stuff segmentation GT (using efficient LUT)
        _stuff_gt_np = self.stuff_gt_map[_encoded_labels_np]

        # c. `object_gt`: 12-class Object segmentation GT (using efficient LUT)
        _object_gt_np = self.object_gt_map[_encoded_labels_np]
        
        # d. `objectness_gt`: Binary objectness mask (1 for objects, 0 for stuff)
        _objectness_gt_np = np.full_like(_encoded_labels_np, self.ignore_index, dtype=np.uint8)
        _objectness_gt_np[np.isin(_encoded_labels_np, self.STUFF_TRAIN_IDS)] = 0
        _objectness_gt_np[np.isin(_encoded_labels_np, self.OBJECT_TRAIN_IDS)] = 1

        # e. `boundary_gt`: Binary boundary mask between different classes
        _boundary_gt_np = self.generate_boundary_map(_encoded_labels_np.copy())
        
        # f. `small_object_gt`: Binary mask for a specific subset of small objects
        _small_object_gt_np = np.zeros_like(_encoded_labels_np, dtype=np.uint8)
        _small_object_gt_np[np.isin(_encoded_labels_np, self.SMALL_OBJECT_TRAIN_IDS)] = 1
        
        # --- 3. Convert all numpy arrays back to PIL Images for transformation ---
        pil_sample = {
            'image': _img,
            'label': Image.fromarray(_encoded_labels_np),
            'stuff_gt': Image.fromarray(_stuff_gt_np),
            'object_gt': Image.fromarray(_object_gt_np),
            'objectness_gt': Image.fromarray(_objectness_gt_np),
            'boundary_gt': Image.fromarray(_boundary_gt_np),
            'small_object_gt': Image.fromarray(_small_object_gt_np)
        }

        # --- 4. Apply transformations ---
        if self.transform_applied_by_user:
            return self.transform_applied_by_user(pil_sample)
        
        # If no transform is passed, return the raw PIL sample
        return pil_sample

    def encode_segmap(self, mask_np):
        """Converts raw Cityscapes labelIds to trainIds (0-18 and 255)."""
        encoded_mask = np.ones_like(mask_np, dtype=np.uint8) * self.ignore_index
        for void_class_id in self.void_classes:
            encoded_mask[mask_np == void_class_id] = self.ignore_index
        for valid_class_id in self.valid_classes:
            train_id = self.class_map.get(valid_class_id, self.ignore_index)
            if train_id != self.ignore_index:
                encoded_mask[mask_np == valid_class_id] = train_id
        return encoded_mask

    def generate_boundary_map(self, label_map_train_id_np):
        """Generates a boundary map from a trainId label map."""
        h, w = label_map_train_id_np.shape
        boundary_map = np.zeros((h, w), dtype=np.uint8)
        valid_pixels_mask = (label_map_train_id_np != self.ignore_index)
        
        # Horizontal boundaries
        horizontal_diff = (label_map_train_id_np[:, :-1] != label_map_train_id_np[:, 1:]) & \
                          (valid_pixels_mask[:, :-1] & valid_pixels_mask[:, 1:])
        boundary_map[:, :-1][horizontal_diff] = 1
        boundary_map[:, 1:][horizontal_diff] = 1
        
        # Vertical boundaries
        vertical_diff = (label_map_train_id_np[:-1, :] != label_map_train_id_np[1:, :]) & \
                        (valid_pixels_mask[:-1, :] & valid_pixels_mask[1:, :])
        boundary_map[:-1, :][vertical_diff] = 1
        boundary_map[1:, :][vertical_diff] = 1
        return boundary_map

    def recursive_glob(self, rootdir='.', suffix=''):
        """Finds all files with a given suffix in a directory and its subdirectories."""
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    # --- Methods to get default transforms ---
    def get_transform(self, split=None):
        """Returns the appropriate transform composer based on the split."""
        current_split = split if split is not None else self.split
        if current_split == 'train':
            return self._get_transform_tr()
        elif current_split == 'val':
            return self._get_transform_val()
        elif current_split == 'test':
            return self._get_transform_ts()
        else:
            raise ValueError(f"Unknown split: {current_split}")

    def _get_transform_tr(self):
        base_size = getattr(self.args, 'base_size', 513)
        crop_size = getattr(self.args, 'crop_size', 513)
        return transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=base_size, crop_size=crop_size,
                               fill_values_map=self.FILL_VALUES),
            tr.RandomGaussianBlur(p=getattr(self.args, 'gaussian_blur_p', 0.0)),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _get_transform_val(self):
        crop_size = getattr(self.args, 'crop_size', 513)
        return transforms.Compose([
            tr.FixScaleCrop(crop_size=crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _get_transform_ts(self):
        target_size = getattr(self.args, 'crop_size', 513)
        return transforms.Compose([
            tr.FixedResize(size=target_size),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

# ==============================================================================
#      SELF-TESTING AND VISUALIZATION SCRIPT
# This block is only executed when the script is run directly.
# (e.g., `python dataloaders/datasets/cityscapes.py`)
# It will NOT run when this file is imported by another script, thus
# preventing argument parsing conflicts.
# ==============================================================================
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    # 1. Define a simple parser specifically for this test script.
    parser = argparse.ArgumentParser(description="Cityscapes Dataset Self-Testing Script")
    parser.add_argument('--base_size', type=int, default=513, help="Base size for data augmentation.")
    parser.add_argument('--crop_size', type=int, default=513, help="Crop size for data augmentation.")
    parser.add_argument('--gaussian_blur_p', type=float, default=0.5, help="Probability of applying Gaussian blur.")
    parser.add_argument('--data_root', type=str, default=None, help="Override the dataset root directory path.")
    args = parser.parse_args()

    print("="*60)
    print("      CityscapesSegmentation Dataset Loader Validation Script      ")
    print("="*60)
    
    # 2. Initialize the dataset with its internal training transforms.
    try:
        dataset_root = args.data_root if args.data_root else None
        dataset_builder = CityscapesSegmentation(args=args, root=dataset_root, split='train')
        train_transforms = dataset_builder.get_transform('train')
        
        cityscapes_train = CityscapesSegmentation(
            args=args, root=dataset_root, split='train', transform=train_transforms
        )
        print("Successfully initialized training dataset with transformations.")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize dataset: {e}")
        print("Please check your dataset path and file structure.")
        exit()

    if len(cityscapes_train) == 0:
        print("\n[ERROR] Cityscapes training dataset is empty. Aborting.")
        exit()

    # 3. Create a DataLoader to fetch a batch of samples.
    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=0)
    print(f"Successfully initialized DataLoader with {len(dataloader)} batches.")

    if len(dataloader) > 0:
        print("\nFetching and analyzing one batch from the DataLoader...")
        try:
            sample_batched = next(iter(dataloader))
        except Exception as e:
            print(f"\n[ERROR] Failed to fetch data from DataLoader: {e}")
            exit()
            
        # Get the first sample from the batch for analysis.
        sample = {key: val[0] for key, val in sample_batched.items()}
        
        print("\n--- Single Sample Tensor Analysis ---")
        for key, tensor in sample.items():
            unique_vals = torch.unique(tensor).tolist()
            # Format unique values string for better readability
            if len(unique_vals) > 15:
                unique_vals_str = f"[{', '.join(map(str, unique_vals[:7]))}, ..., {', '.join(map(str, unique_vals[-7:]))}]"
            else:
                unique_vals_str = str(unique_vals)
            
            print(f"  - {key:<17}: shape={str(list(tensor.shape)):<15}, dtype={str(tensor.dtype):<18}, unique_vals={unique_vals_str}")
            
            # Assert that ground truth masks are of type LongTensor
            if 'gt' in key and key not in ['boundary_gt', 'small_object_gt']:
                 assert tensor.dtype == torch.long, f"Type Error! {key} should be torch.LongTensor but is {tensor.dtype}"
            
        print("\n[SUCCESS] Data types for all classification GTs are correct (torch.long).")

        # 4. Visualize all generated ground truths for one sample.
        print("\nGenerating visualization grid...")
        def tensor_to_pil_display(tensor_img, is_mask=False, normalize_params=None):
            """Converts a Tensor to a displayable PIL image or numpy array."""
            if tensor_img.ndim == 3 and not is_mask:
                img_to_show = tensor_img.cpu().clone()
                if normalize_params: # Un-normalize if params are provided
                    mean = torch.tensor(normalize_params['mean']).view(3, 1, 1)
                    std = torch.tensor(normalize_params['std']).view(3, 1, 1)
                    img_to_show = img_to_show.mul(std).add(mean)
                img_to_show = img_to_show.clamp(0, 1)
                return transforms.ToPILImage()(img_to_show)
            elif tensor_img.ndim == 2 or (tensor_img.ndim == 3 and tensor_img.shape[0] == 1):
                if tensor_img.ndim == 3: tensor_img = tensor_img.squeeze(0)
                return tensor_img.cpu().numpy()
            return np.zeros((100, 100), dtype=np.uint8)

        norm_params = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
        
        fig, axs = plt.subplots(2, 4, figsize=(24, 12))
        axs = axs.ravel()

        display_items = {
            'image': 'Image (Augmented)', 'label': 'Label GT (0-18)',
            'stuff_gt': 'Stuff GT (0-6)', 'object_gt': 'Object GT (0-11)',
            'objectness_gt': 'Objectness GT (0,1,255)', 'boundary_gt': 'Boundary GT (0,1)',
            'small_object_gt': 'Small Object GT (0,1)'
        }
        
        cmap = 'viridis'

        axs[0].imshow(tensor_to_pil_display(sample['image'], normalize_params=norm_params))
        axs[0].set_title(display_items['image']); axs[0].axis('off')
        
        plot_idx = 1
        for key, title in display_items.items():
            if key == 'image' or key not in sample: continue
            ax = axs[plot_idx]
            tensor_np = tensor_to_pil_display(sample[key], is_mask=True)
            im = ax.imshow(tensor_np, cmap=cmap)
            ax.set_title(f"{title}\nUnique: {np.unique(tensor_np).tolist()}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
            plot_idx += 1
        
        # Hide any unused subplots
        for i in range(plot_idx, len(axs)):
            axs[i].set_visible(False)

        plt.suptitle('Validation of All Generated Ground Truths (After Augmentation)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        print("Displaying visualization plot. Close the plot window to exit.")
        plt.show()
        
    else:
        print("\nDataLoader is empty, cannot perform validation.")
        
    print("\n="*60)
    print("      Validation script finished successfully.      ")
    print("="*60)