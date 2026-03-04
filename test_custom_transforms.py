# test_custom_transforms.py (最终修正版 - 健壮的绘图逻辑)

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') # 强制使用无GUI的Agg后端
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os
from PIL import Image

try:
    from dataloaders.datasets.cityscapes import CityscapesSegmentation
except ImportError:
    print("错误: 无法从 'dataloaders.datasets.cityscapes' 导入 CityscapesSegmentation。")
    exit()

def get_argparser():
    parser = argparse.ArgumentParser(description="全面验证 Cityscapes 数据管道")
    parser.add_argument('--data_root', type=str, required=True, help="Cityscapes 数据集的根目录路径。")
    parser.add_argument('--num_samples', type=int, default=5, help="要随机抽样验证的样本数量。")
    parser.add_argument('--output_dir', type=str, default='validation_outputs', help="保存验证结果图像的目录。")
    parser.add_argument('--base_size', type=int, default=513)
    parser.add_argument('--crop_size', type=int, default=513)
    parser.add_argument('--gaussian_blur_p', type=float, default=0.5)
    return parser

def convert_for_display(data):
    if isinstance(data, torch.Tensor):
        if data.ndim == 3 and data.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            data = data.clone().mul_(std).add_(mean).clamp(0, 1)
            return data.numpy().transpose(1, 2, 0)
        else:
            if data.ndim == 3:
                data = data.squeeze(0)
            return data.cpu().numpy()
    elif isinstance(data, Image.Image):
        return np.array(data)
    return np.zeros((100, 100), dtype=np.uint8)

def validate_and_visualize_sample(sample, title_prefix="", output_path=None):
    print("-" * 30)
    print(f"{title_prefix} - 数值验证")
    print("-" * 30)

    is_tensor_sample = isinstance(sample['image'], torch.Tensor)
    if is_tensor_sample:
        for key, tensor in sample.items():
            unique_vals = torch.unique(tensor).tolist()
            if len(unique_vals) > 15:
                unique_vals_str = f"[{', '.join(map(str, unique_vals[:7]))}, ..., {', '.join(map(str, unique_vals[-7:]))}]"
            else:
                unique_vals_str = str(unique_vals)
            print(f"  - {key:<17}: shape={str(list(tensor.shape)):<15}, dtype={str(tensor.dtype):<18}, unique_vals={unique_vals_str}")
        
        assert sample['label'].dtype == torch.long and sample['label'].ndim == 2
        assert sample['stuff_gt'].dtype == torch.long and sample['stuff_gt'].ndim == 2
        assert sample['object_gt'].dtype == torch.long and sample['object_gt'].ndim == 2
        print("[断言通过] label, stuff_gt, object_gt 是 2D LongTensor。")
        assert sample['objectness_gt'].dtype == torch.float32 and sample['objectness_gt'].ndim == 3
        assert sample['boundary_gt'].dtype == torch.float32 and sample['boundary_gt'].ndim == 3
        assert sample['small_object_gt'].dtype == torch.float32 and sample['small_object_gt'].ndim == 3
        print("[断言通过] objectness_gt, boundary_gt, small_object_gt 是 3D (1,H,W) FloatTensor。")
        print("\n")

    fig, axs = plt.subplots(2, 4, figsize=(28, 12))
    axs = axs.ravel()
    fig.suptitle(title_prefix, fontsize=20, y=0.98) 

    display_items = {
        'image': 'Image', 'label': 'Label GT (0-18)', 'stuff_gt': 'Stuff GT (0-6)', 
        'object_gt': 'Object GT (0-11)', 'objectness_gt': 'Objectness GT (0,1,255)', 
        'boundary_gt': 'Boundary GT (0,1)', 'small_object_gt': 'Small Object GT (0,1)'
    }
    
    cmap = 'viridis'

    for i, (key, title) in enumerate(display_items.items()):
        if key not in sample:
            axs[i].set_visible(False); continue
        
        display_data = convert_for_display(sample[key])
        
        # ==================== 核心修改：健壮的绘图逻辑 ====================
        if key == 'image':
            axs[i].imshow(display_data)
            if isinstance(sample[key], torch.Tensor):
                img_tensor = sample[key]
                title_str = f"{title} | min: {img_tensor.min():.2f}, mean: {img_tensor.mean():.2f}, max: {img_tensor.max():.2f}"
            else:
                unique_vals_list = np.unique(display_data).tolist()
                title_str = f"{title} | Unique: {len(unique_vals_list)} values"
            axs[i].set_title(title_str, fontsize=11)

        else: # 处理所有掩码 (GTs)
            im = axs[i].imshow(display_data, cmap=cmap) # 'im' 在此定义
            unique_vals_list = np.unique(display_data).tolist()
            if len(unique_vals_list) > 10:
                 unique_vals_str = f"Unique: [{', '.join(map(str, unique_vals_list[:5]))},...,{unique_vals_list[-1]}]"
            else:
                 unique_vals_str = f"Unique: {unique_vals_list}"
            axs[i].set_title(f"{title} | {unique_vals_str}", fontsize=11)
            fig.colorbar(im, ax=axs[i], orientation='horizontal', fraction=0.046, pad=0.08) # 'im' 在此使用
        # =====================================================================

        axs[i].axis('off')

    for i in range(len(display_items), len(axs)):
        axs[i].set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.4, wspace=0.1)
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"✅ 可视化结果已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


if __name__ == '__main__':
    args = get_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print(f"      开始全面验证 Cityscapes 数据管道 (结果将保存到 ./{args.output_dir})")
    print("="*60)

    dataset_no_aug = CityscapesSegmentation(args=args, root=args.data_root, split='train', transform=None)
    transform_builder = CityscapesSegmentation(args=args, root=args.data_root)
    train_transforms = transform_builder._get_transform_tr()

    if len(dataset_no_aug) == 0:
        print(f"\n[错误] 数据集在路径 '{args.data_root}' 中未找到或为空。")
        exit()

    sample_indices = random.sample(range(len(dataset_no_aug)), k=args.num_samples)

    for i, idx in enumerate(sample_indices):
        print("\n" + "="*60)
        print(f"正在处理随机样本 {i+1}/{args.num_samples} (原始索引: {idx})")
        print("="*60)

        sample_before_aug = dataset_no_aug[idx]
        output_path_before = os.path.join(args.output_dir, f"sample_{idx}_BEFORE_aug.png")
        validate_and_visualize_sample(sample_before_aug, f"Sample {idx} - BEFORE Augmentation", output_path=output_path_before)

        sample_after_aug = train_transforms(sample_before_aug)
        output_path_after = os.path.join(args.output_dir, f"sample_{idx}_AFTER_aug.png")
        validate_and_visualize_sample(sample_after_aug, f"Sample {idx} - AFTER Augmentation", output_path=output_path_after)

    print("\n" + "="*60)
    print(f"      所有随机样本验证完毕。结果图已全部保存在 '{args.output_dir}' 文件夹中。")
    print("="*60)