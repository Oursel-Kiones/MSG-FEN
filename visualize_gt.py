# ===================================================================================
#           visualize_gt.py - 多维真值(GT)在线生成与可视化脚本 (最终版)
#           - 定制适配于您的 CityscapesSegmentation 类 -
# ===================================================================================
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# --- 用户路径已配置 (无需修改) ---
CITYSCAPES_ROOT = "/workspace/data解压/cityscapes"
PROJECT_PATH = "/workspace/deep参考1"
# -----------------------------------

# 将项目路径添加到系统路径中，以便导入自定义模块
sys.path.insert(0, PROJECT_PATH)
try:
    # 动态导入项目中的 cityscapes.py
    from dataloaders.datasets import cityscapes
    print(f"成功从 '{PROJECT_PATH}' 导入 cityscapes dataloader.")
except ImportError as e:
    print(f"\n[错误!] 无法导入 'cityscapes' 模块。请确认路径是否正确。错误: {e}\n")
    exit()

def get_cityscapes_colors():
    """获取Cityscapes 19个类别的标准颜色 (用于将Train ID映射回彩色图)"""
    return np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ])

def decode_train_id_to_color(label_pil):
    """将包含Train ID (0-18, 255) 的PIL灰度图解码为彩色RGB图像"""
    label_np = np.array(label_pil)
    colors = get_cityscapes_colors()
    
    rgb_image = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)
    
    for train_id in range(19):
        mask = label_np == train_id
        rgb_image[mask] = colors[train_id]
        
    return Image.fromarray(rgb_image)


def visualize_and_save(dataset, indices, save_dir="."):
    """从数据集中加载指定索引的样本，并生成5张对比图。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建输出目录: {save_dir}")

    for i, index in enumerate(indices):
        try:
            # __getitem__ 返回一个包含PIL图像的字典
            sample = dataset[index]
            # 从文件名中提取基础名，用于保存
            base_filename = os.path.basename(dataset.files[dataset.split][index]).replace('_leftImg8bit.png', '')
        except IndexError:
            print(f"索引 {index} 超出范围，跳过。")
            continue

        print(f"正在生成第 {i+1}/{len(indices)} 张对比图 (样本: {base_filename})...")

        # 将PIL格式的语义标签解码为彩色图像
        color_semantic_gt = decode_train_id_to_color(sample['label'])

        fig, axs = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(f"Multi-Dimension GT Generation (Sample: {base_filename})", fontsize=16)

        # (a) 原始输入图像
        axs[0].imshow(sample['image'])
        axs[0].set_title('(a) Original Input Image', fontsize=12)
        axs[0].axis('off')

        # (b) 原始语义分割GT (彩色)
        axs[1].imshow(color_semantic_gt)
        axs[1].set_title('(b) Original Semantic GT', fontsize=12)
        axs[1].axis('off')

        # (c) 生成的 Objectness GT
        axs[2].imshow(sample['objectness_gt'], cmap='gray')
        axs[2].set_title('(c) Generated Objectness GT', fontsize=12)
        axs[2].axis('off')

        # (d) 生成的 Boundary GT
        axs[3].imshow(sample['boundary_gt'], cmap='gray')
        axs[3].set_title('(d) Generated Boundary GT', fontsize=12)
        axs[3].axis('off')

        # (e) 生成的 Small Object GT
        axs[4].imshow(sample['small_object_gt'], cmap='gray')
        axs[4].set_title('(e) Generated Small Object GT', fontsize=12)
        axs[4].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_filename = os.path.join(save_dir, f"gt_visualization_{base_filename}.png")
        plt.savefig(output_filename, dpi=200, bbox_inches='tight')
        print(f"-> 成功保存图像至: {output_filename}")
        plt.close(fig)

def main():
    """主执行函数"""
    print("--- 开始执行多维真值(GT)可视化脚本 (定制版) ---")

    # 1. 准备一个空的 'args' 对象，因为 CityscapesSegmentation 的 __init__ 需要它
    mock_args = argparse.Namespace()

    # 2. 实例化您的 CityscapesSegmentation 类
    #    【关键】: 使用正确的类名 CityscapesSegmentation
    #    【关键】: 设置 transform=None，以获取原始的PIL图像和掩码
    try:
        dataset = cityscapes.CityscapesSegmentation(
            args=mock_args,
            root=CITYSCAPES_ROOT, 
            split='val',
            transform=None  # 这将使 __getitem__ 返回一个包含PIL图像的字典
        )
    except Exception as e:
        print(f"\n[错误!] 初始化 CityscapesSegmentation 数据集失败。请确认：")
        print(f"1. Cityscapes根目录 '{CITYSCAPES_ROOT}' 是否正确？")
        print(f"2. 您的 cityscapes.py 代码没有其他依赖问题。")
        print(f"原始错误: {e}\n")
        return

    # 3. 选择几个有代表性的样本索引进行可视化
    sample_indices = [10, 50, 120, 210, 350] 

    # 4. 执行可视化并保存结果到新目录
    visualize_and_save(dataset, sample_indices, save_dir="./gt_visualizations_output")
    
    print("\n--- 脚本执行完毕 ---")
    print(f"所有可视化结果已保存在 './gt_visualizations_output' 目录中。")

if __name__ == '__main__':
    main()