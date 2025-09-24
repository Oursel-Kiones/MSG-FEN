import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- 用户需要配置的部分 -----------------

# ======================= (核心修改点) =======================
# 1. 导入你的数据集类
#    问题：原来这里写的是 'from ... import Cityscapes'，但实际的类名是 'CityscapesSegmentation'。
#    修改：将 Cityscapes 替换为 CityscapesSegmentation。
from dataloaders.datasets.cityscapes import CityscapesSegmentation as YourCityscapesClass
# ==========================================================

# 2. 指定输出路径
output_path = "/workspace/verify_stuff_gt.png"

# 3. 模拟一个 args 对象 (如果你的数据集初始化需要)
class MockArgs:
    # 在这里添加你的数据集初始化可能用到的任何 args 属性
    # 例如：
    # crop_size = 768
    pass
args = MockArgs()
# ----------------------------------------------------

# Cityscapes 调色板 (19个训练类别)
cityscapes_palette = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

# 为 Stuff 类别创建一个新的、简单的调色板 (7个类别)
stuff_palette = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], 
    [0, 255, 255], [255, 0, 255], [128, 128, 128]
]

def decode_target(mask, palette):
    """根据调色板将标签掩码转换为彩色图像"""
    # 注意：这里的输入 mask 应该是 NumPy 数组或可以转换为 NumPy 的张量
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 如果掩码是 PIL Image, 先转成 NumPy
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
        
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

print("--- 开始验证 Stuff 类别生成 ---")

try:
    # 1. 初始化你的数据集和数据加载器
    #    现在 `YourCityscapesClass` 已经指向了正确的类 `CityscapesSegmentation`
    dataset = YourCityscapesClass(args=args, split='val', transform=None) # 使用 transform=None 获取原始PIL数据
    
    # 注意：直接从 dataset 获取样本，以避免 DataLoader 可能引入的复杂转换
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查路径或 split 设置。")

    # 2. 从数据集中获取一个样本
    #    你的 __getitem__ 返回一个字典
    sample = dataset[0] 
    
    # 检查 'stuff_gt' 是否存在
    if 'stuff_gt' not in sample:
        raise KeyError("在数据样本中未找到 'stuff_gt' 键。请确认 cityscapes.py 的 __getitem__ 是否已正确修改。")

    image = sample['image']
    original_target = sample['label'] # 这是PIL Image
    stuff_target = sample['stuff_gt']   # 这也是PIL Image

    # 将PIL掩码转为Tensor进行分析
    original_target_tensor = torch.from_numpy(np.array(original_target))
    stuff_target_tensor = torch.from_numpy(np.array(stuff_target))

    print(f"原始GT (训练ID) 中的唯一值: {torch.unique(original_target_tensor).tolist()}")
    print(f"新生成Stuff GT中的唯一值: {torch.unique(stuff_target_tensor).tolist()}")
    print(f"检测到的 num_stuff_classes 的值是: {dataset.num_stuff_classes}")

    # 断言检查
    unique_stuff_ids = torch.unique(stuff_target_tensor)
    for uid in unique_stuff_ids:
        if uid != dataset.ignore_index:
            assert uid < dataset.num_stuff_classes, f"Stuff ID {uid} 超出范围 [0, {dataset.num_stuff_classes-1}]"
    print("断言检查通过：Stuff GT 中的所有有效ID都在正确范围内。")

    # 4. 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 显示原始图像 (PIL Image)
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 显示原始GT掩码 (PIL Image)
    axes[1].imshow(decode_target(original_target, cityscapes_palette))
    axes[1].set_title("Original Ground Truth (Train IDs)")
    axes[1].axis('off')

    # 显示新生成的Stuff GT掩码 (PIL Image)
    axes[2].imshow(decode_target(stuff_target, stuff_palette))
    axes[2].set_title(f"Generated Stuff Ground Truth ({dataset.num_stuff_classes} classes)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"验证图片已保存到: {output_path}")

except Exception as e:
    import traceback
    traceback.print_exc() # 打印完整的错误堆栈
    print(f"\n脚本执行出错: {e}")
    print("请检查以下几点：")
    print("1. 你是否已经将 `cityscapes.py` 的 `__init__` 和 `__getitem__` 方法按要求修改？")
    print("2. 你的数据集路径是否正确，'val' split 是否有数据？")
    print("3. 如果错误是 KeyError，说明 `stuff_gt` 没有被正确生成和返回。")