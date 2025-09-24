import os
import numpy as np
from PIL import Image
from skimage import measure  # 用于寻找连通组件和属性
from collections import defaultdict
from tqdm import tqdm  # 进度条
import argparse
import matplotlib.pyplot as plt  # 用于可选的直方图可视化

# --- 为了独立运行，从 Cityscapes 数据加载器中复制关键定义 ---
# (假设 mypath.Path.db_root_dir 能工作，或者你手动替换路径)
try:
    from mypath import Path
    DEFAULT_CITYSCAPES_PATH = Path.db_root_dir('cityscapes')
except ImportError:
    # === 重要: 如果 mypath 不工作，请在这里设置你的 CITYSCAPES 路径 ===
    DEFAULT_CITYSCAPES_PATH = "/workspace/data解压/cityscapes" # <--- 如果需要，请修改此路径
    print(f"警告: mypath 模块未找到。使用默认路径: {DEFAULT_CITYSCAPES_PATH}")
    print("请确保此路径正确或修改脚本。")

NUM_CLASSES = 19
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
# 从原始ID到训练ID (0-18)的映射
class_map = dict(zip(valid_classes, range(NUM_CLASSES)))
ignore_index = 255

# 训练ID 0-18 对应的类别名称
class_names_map = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}
# --- 定义复制结束 ---

def encode_segmap(mask):
    """将原始的Cityscapes标签ID转换为训练ID (0-18)"""
    # 将所有无效类别设置为忽略索引
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    # 将有效类别映射到0-18的训练ID
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def recursive_glob(rootdir='.', suffix=''):
    """使用给定的后缀和根目录执行递归的glob搜索"""
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

def analyze_areas(dataset_root, split='val', max_images=None, connectivity=2):
    """
    分析Cityscapes真值标签中每个类别连通组件的面积分布。

    Args:
        dataset_root (str): Cityscapes数据集的根目录路径。
        split (str): 要分析的数据集分割 ('train', 'val', 'test')。默认为 'val'。
        max_images (int, optional): 要处理的最大图像数量。默认为 None (处理所有)。
        connectivity (int): 标记组件的连通性 (1表示4连通, 2表示8连通)。默认为 2。

    Returns:
        dict: 一个字典，键是类别ID (0-18)，值是该类别所有连通组件的面积列表。
    """
    print(f"开始对 Cityscapes '{split}' 分割进行面积分析...")
    print(f"数据集根目录: {dataset_root}")

    images_base = os.path.join(dataset_root, 'leftImg8bit', split)
    annotations_base = os.path.join(dataset_root, 'gtFine', split)

    image_files = recursive_glob(rootdir=images_base, suffix='_leftImg8bit.png')

    if not image_files:
        raise FileNotFoundError(f"在 {images_base} 中没有找到图像文件。请检查数据集路径和分割名称。")

    if max_images is not None and max_images < len(image_files):
        print(f"将处理 {max_images} 张图像的子集 (总共 {len(image_files)} 张)。")
        image_files = image_files[:max_images]
    else:
        print(f"将处理所有找到的 {len(image_files)} 张图像。")

    class_areas = defaultdict(list)

    for img_path in tqdm(image_files, desc=f"分析 '{split}' 图像"):
        city_name = os.path.basename(os.path.dirname(img_path))
        base_filename = os.path.basename(img_path).replace('_leftImg8bit.png', '')
        lbl_filename = base_filename + '_gtFine_labelIds.png'
        lbl_path = os.path.join(annotations_base, city_name, lbl_filename)

        if not os.path.exists(lbl_path):
            continue

        try:
            _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
            label_map = encode_segmap(_tmp)

            unique_classes = np.unique(label_map)
            for class_id in unique_classes:
                if class_id >= NUM_CLASSES or class_id < 0:
                    continue

                binary_mask = (label_map == class_id)
                labeled_mask, num_labels = measure.label(binary_mask, background=0,
                                                         connectivity=connectivity, return_num=True)

                if num_labels > 0:
                    regions = measure.regionprops(labeled_mask)
                    for region in regions:
                        class_areas[class_id].append(region.area)

        except Exception as e:
            print(f"\n处理文件 {lbl_path} 时出错: {e}")

    print("\n面积分析完成。")
    return class_areas

def suggest_thresholds(class_areas, percentile=5, min_threshold=10):
    """
    基于百分位数统计建议面积阈值。

    Args:
        class_areas (dict): 每个类别ID的面积列表字典。
        percentile (int): 用于阈值计算的百分位数 (例如, 5 表示第5百分位数)。
        min_threshold (int): 允许的最小阈值。

    Returns:
        dict: 将类别ID (0-18) 映射到建议阈值的字典。
    """
    print(f"\n根据第 {percentile} 百分位数计算建议阈值 (最小阈值 = {min_threshold})...")
    thresholds = {}
    print("-" * 80)
    print(f"{'类别ID':<10} {'类别名称':<15} {'组件数量':<10} {'最小面积':<10} {'中位面积':<12} {f'第{percentile}百分位':<12} {'建议阈值':<15}")
    print("-" * 80)

    for class_id in range(NUM_CLASSES):
        class_name = class_names_map.get(class_id, '未知')
        if class_id in class_areas and class_areas[class_id]:
            areas = np.array(class_areas[class_id])
            count = len(areas)
            min_area = np.min(areas)
            median_area = np.median(areas)
            perc_value = np.percentile(areas, percentile)
            suggested = max(min_threshold, int(np.ceil(perc_value)))

            thresholds[class_id] = suggested
            print(f"{class_id:<10} {class_name:<15} {count:<10} {min_area:<10} {median_area:<12.1f} {perc_value:<12.1f} {suggested:<15}")
        else:
            thresholds[class_id] = 0
            print(f"{class_id:<10} {class_name:<15} {'0':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'0':<15}")

    print("-" * 80)
    print("\n建议的 min_area_thresholds 字典:")
    threshold_str = "min_area_thresholds = {\n"
    items_per_line = 4
    count = 0
    for k, v in thresholds.items():
        threshold_str += f"    {k}: {v},"
        count += 1
        if count % items_per_line == 0:
            threshold_str += "\n"
        else:
            threshold_str += " "
    threshold_str = threshold_str.strip().rstrip(',') + "\n}"
    print(threshold_str)

    return thresholds

def plot_histograms(class_areas, bins=50):
    """(可选) 绘制每个类别面积分布的直方图"""
    print("\n正在生成面积直方图...")
    num_classes_with_data = len(class_areas)
    cols = 4
    rows = int(np.ceil(num_classes_with_data / cols))
    if rows == 0: return # No data to plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for class_id in sorted(class_areas.keys()):
        if class_areas[class_id]:
            ax = axes[plot_idx]
            areas = np.array(class_areas[class_id])
            
            # 使用对数刻度以更好地可视化差异巨大的面积
            ax.hist(areas, bins=bins, log=True)
            ax.set_yscale('log')
            ax.set_ylabel('数量 (对数刻度)')

            class_name = class_names_map.get(class_id, f'ID {class_id}')
            ax.set_title(f"{class_name} (ID {class_id})")
            ax.set_xlabel('面积 (像素)')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle("真值组件的面积分布", fontsize=16)
    try:
        save_path = "cityscapes_area_histograms.png"
        plt.savefig(save_path)
        print(f"直方图已保存至 {save_path}")
    except Exception as e:
        print(f"保存直方图时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析Cityscapes真值组件的面积")
    parser.add_argument('--dataset-path', type=str, default=DEFAULT_CITYSCAPES_PATH,
                        help='Cityscapes数据集的根目录路径。')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help="要分析的数据集分割。默认为 'val'。")
    parser.add_argument('--max-images', type=int, default=None,
                        help='要处理的最大图像数量 (用于快速分析)。')
    parser.add_argument('--percentile', type=int, default=5,
                        help='用于建议阈值的面积分布百分位数。默认为 5。')
    parser.add_argument('--min-threshold', type=int, default=10,
                        help='最小建议面积阈值。默认为 10。')
    parser.add_argument('--connectivity', type=int, default=2, choices=[1, 2],
                        help='标记组件的连通性 (1: 4-连通, 2: 8-连通)。默认为 2。')
    parser.add_argument('--plot', action='store_true',
                        help='生成并保存面积分布的直方图。')

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"错误: 数据集路径未找到或不是一个目录: {args.dataset_path}")
        exit(1)
    gt_fine_path = os.path.join(args.dataset_path, 'gtFine', args.split)
    if not os.path.isdir(gt_fine_path):
         print(f"错误: 真值目录未找到: {gt_fine_path}")
         print("请确保Cityscapes数据集结构正确。")
         exit(1)

    collected_areas = analyze_areas(args.dataset_path, args.split, args.max_images, args.connectivity)

    if collected_areas:
        suggested_thresholds = suggest_thresholds(collected_areas, args.percentile, args.min_threshold)
        if args.plot:
            plot_histograms(collected_areas)
    else:
        print("没有收集到任何面积数据。跳过阈值建议。")

    print("\n分析结束。")