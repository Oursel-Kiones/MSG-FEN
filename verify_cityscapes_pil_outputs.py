import os
import sys # 需要 sys 来修改路径 (如果 mypath.py 不在标准路径中)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# --- 关键修改：导入自定义的 Path 类 ---
# 假设 mypath.py 在 /workspace/deep参考1/ 目录下，
# 并且 verify_cityscapes_pil_outputs.py 也在类似 /workspace/deep参考1/ 的地方运行
# 或者 mypath.py 所在的目录在 PYTHONPATH 中。

# 选项1: 如果 mypath.py 在的目录是 verify_cityscapes_pil_outputs.py 的父目录或已知相对路径
# 例如，如果 verify_cityscapes_pil_outputs.py 在 /workspace/deep参考1/some_folder/
# mypath_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 父目录
# sys.path.insert(0, mypath_dir)

# 选项2: 如果 mypath.py 和 cityscapes.py 都在 Python 的搜索路径中 (更推荐的配置)
try:
    from mypath import Path  # 直接从 mypath.py 导入 Path 类
    print("成功从 'mypath.py' 导入 Path。")
    # 获取 Cityscapes 数据集的基本路径
    CITYSCAPES_ROOT = Path.db_root_dir('cityscapes')
    print(f"Cityscapes root directory from mypath.Path: {CITYSCAPES_ROOT}")
except ImportError as e_path:
    print(f"错误：无法从 'mypath.py' 导入 Path: {e_path}")
    print("请确保 'mypath.py' 位于 Python 搜索路径中，或者调整 sys.path。")
    print(f"当前 sys.path: {sys.path}")
    CITYSCAPES_ROOT = '/workspace/data解压/cityscapes' # 使用一个后备的硬编码路径
    print(f"警告: 将使用后备 Cityscapes 根目录: {CITYSCAPES_ROOT}")
    # 定义一个临时的 Path 类，如果导入失败，以便脚本的其余部分可以尝试运行
    if 'Path' not in locals():
        class Path:
            @staticmethod
            def db_root_dir(dataset_name):
                if dataset_name == 'cityscapes':
                    return '/workspace/data解压/cityscapes' # 后备路径
                raise NotImplementedError(f"临时 Path 类未实现: {dataset_name}")
        print("使用临时的 Path 类定义。")


# --- 导入 CityscapesSegmentation ---
# 假设 cityscapes.py 在 dataloaders.datasets 目录下
# 并且 dataloaders 目录与 mypath.py 在同一级别，或者可以通过 PYTHONPATH 访问
# 如果 dataloaders 在 /workspace/deep参考1/dataloaders/
# 你可能需要确保 /workspace/deep参考1/ 在 sys.path 中
# 通常，如果你的项目根目录 (如 /workspace/deep参考1/) 在 PYTHONPATH 中，
# 或者你从项目根目录运行脚本，那么下面的导入应该有效。

# 将项目根目录添加到 sys.path (如果需要)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设脚本在类似 'utils' 的子目录
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)

try:
    from dataloaders.datasets.cityscapes import CityscapesSegmentation
    print("成功从 'dataloaders.datasets.cityscapes' 导入 CityscapesSegmentation。")
except ImportError as e_cs:
    print(f"错误：无法导入 CityscapesSegmentation: {e_cs}")
    print("请确保 cityscapes.py 文件存在于 'dataloaders/datasets/' 目录下，")
    print("并且其父目录 ('dataloaders' 的父目录) 在 Python 的搜索路径中 (sys.path)。")
    print(f"当前 sys.path: {sys.path}")
    exit()


def verify_pil_mask(mask_pil, mask_name, expected_unique_values_sets, ignore_index_val=None):
    """
    验证单个 PIL 掩码。

    Args:
        mask_pil (PIL.Image): 要验证的 PIL 图像掩码。
        mask_name (str): 掩码的名称 (e.g., 'label').
        expected_unique_values_sets (list of set/list):
            一个集合列表，其中每个集合代表一组允许的唯一值。
            例如: [{'0-18', 255}] 或 [[0, 1], [0, 1, 255]]
            如果 mask_np 中的唯一值是其中任何一个集合的子集，则验证通过。
            特殊字符串 '0-N' 会被扩展为 range(N+1)。
        ignore_index_val (int, optional): 掩码的忽略索引值。
    """
    if not isinstance(mask_pil, Image.Image):
        print(f"  [ERROR] '{mask_name}' 不是一个 PIL Image. 类型为: {type(mask_pil)}")
        return False
    if mask_pil.mode != 'L':
        print(f"  [ERROR] '{mask_name}' PIL Image mode 不是 'L'. Mode 为: {mask_pil.mode}")
        return False

    mask_np = np.array(mask_pil)
    unique_values_in_mask = set(np.unique(mask_np).tolist())
    print(f"  '{mask_name}' (PIL 'L', np.uint8) unique values: {sorted(list(unique_values_in_mask))}")

    is_valid = False
    for expected_set_raw in expected_unique_values_sets:
        current_expected_set = set()
        for item in expected_set_raw:
            if isinstance(item, str) and '-' in item:
                try:
                    start_str, end_str = item.split('-')
                    start, end = int(start_str), int(end_str)
                    current_expected_set.update(range(start, end + 1))
                except ValueError:
                    print(f"    [警告] 无法解析范围字符串 '{item}' for '{mask_name}'.")
                    current_expected_set.add(item) # 按原样添加以防万一
            else:
                current_expected_set.add(item)

        if unique_values_in_mask.issubset(current_expected_set):
            is_valid = True
            break # 找到一个匹配的期望值集

    if not is_valid:
        print(f"  [ERROR] '{mask_name}' 包含未预期的值。期望子集于: {expected_unique_values_sets}, 实际得到: {sorted(list(unique_values_in_mask))}")
        return False
    else:
        print(f"  '{mask_name}' values OK.")
        return True

def main(args_cli):
    print("--- 开始验证 CityscapesSegmentation PIL 输出 ---")

    class DummyArgs:
        base_size = 513
        crop_size = 513
        gaussian_blur_p = 0.0
        # 添加 CityscapesSegmentation 可能需要的其他 args 属性

    dataset_args = DummyArgs()

    print(f"\n正在初始化 CityscapesSegmentation for split '{args_cli.split}' with transform=None...")
    try:
        # 使用从 mypath.Path 获取的 CITYSCAPES_ROOT (或后备路径)
        # CityscapesSegmentation 的 __init__ 方法中 root 参数优先于内部 Path.db_root_dir 调用
        dataset = CityscapesSegmentation(args=dataset_args, 
                                         root=CITYSCAPES_ROOT, # 显式传递 root
                                         split=args_cli.split, 
                                         transform=None)
    except Exception as e:
        print(f"初始化 CityscapesSegmentation 时发生错误: {e}")
        print(f"请确保数据集路径 '{CITYSCAPES_ROOT}' 正确且数据集存在。")
        return

    if len(dataset) == 0:
        print(f"错误：在 split '{args_cli.split}' 中未找到数据。请检查数据集路径和文件。")
        return

    print(f"数据集 '{args_cli.split}' 初始化成功，包含 {len(dataset)} 个样本。")

    num_classes = dataset.NUM_CLASSES
    ignore_index = dataset.ignore_index

    expected_values_specs = {
        'label': {
            "sets": [[f'0-{num_classes - 1}', ignore_index]],
            "description": f"0-{num_classes-1} (trainId) or {ignore_index} (ignore)"
        },
        'objectness_gt': {
            "sets": [[0, 1, ignore_index]],
            "description": f"0 (stuff), 1 (object), or {ignore_index} (ignore)"
        },
        'boundary_gt': {
            "sets": [[0, 1]],
            "description": "0 (non-boundary) or 1 (boundary)"
        },
        'small_object_gt': {
            "sets": [[0, 1]],
            "description": "0 (not small object/ignored in label) or 1 (small object)"
        }
    }

    num_samples_to_verify = min(args_cli.num_samples, len(dataset))
    print(f"\n将验证 {num_samples_to_verify} 个样本的 PIL 掩码...")

    all_samples_valid = True
    for i in range(num_samples_to_verify):
        sample_index = np.random.randint(0, len(dataset)) if args_cli.random_samples else i
        print(f"\n--- 验证样本索引: {sample_index} ---")
        try:
            pil_sample = dataset[sample_index]
        except Exception as e:
            print(f"  [ERROR] 获取样本 {sample_index} 时出错: {e}")
            all_samples_valid = False
            continue

        if not isinstance(pil_sample, dict):
            print(f"  [ERROR] dataset[{sample_index}] 返回的不是字典类型，而是 {type(pil_sample)}")
            all_samples_valid = False
            continue

        if 'image' not in pil_sample or not isinstance(pil_sample['image'], Image.Image):
            print(f"  [ERROR] 'image' 键不存在或其值不是 PIL Image.")
            all_samples_valid = False
        elif pil_sample['image'].mode != 'RGB':
            print(f"  [ERROR] 'image' PIL Image mode 不是 'RGB'. Mode 为: {pil_sample['image'].mode}")
            all_samples_valid = False
        else:
            print(f"  'image' PIL Image (mode RGB) OK.")

        current_sample_all_masks_valid = True
        for mask_key, spec in expected_values_specs.items():
            if mask_key not in pil_sample:
                print(f"  [ERROR] 样本中未找到掩码键: '{mask_key}'")
                all_samples_valid = False
                current_sample_all_masks_valid = False
                continue
            
            print(f"  正在验证 '{mask_key}' (期望: {spec['description']})...")
            if not verify_pil_mask(pil_sample[mask_key], mask_key, spec["sets"], ignore_index_val=ignore_index):
                all_samples_valid = False
                current_sample_all_masks_valid = False
        
        if not current_sample_all_masks_valid:
            print(f"  样本 {sample_index} 中的一个或多个掩码验证失败。")

        if args_cli.visualize and current_sample_all_masks_valid:
            print(f"  正在可视化样本 {sample_index} 的 PIL 输出...")
            num_items_to_show = len(pil_sample)
            fig, axs = plt.subplots(1, num_items_to_show, figsize=(5 * num_items_to_show, 6))
            
            fig_title = f"Raw PIL Outputs - Sample Index: {sample_index}"
            if hasattr(dataset, 'files') and hasattr(dataset, 'split') and dataset.split in dataset.files and \
               isinstance(dataset.files[dataset.split], list) and sample_index < len(dataset.files[dataset.split]):
                img_filename = os.path.basename(dataset.files[dataset.split][sample_index])
                fig_title += f"\n{img_filename}"

            fig.suptitle(fig_title, fontsize=14)

            item_idx = 0
            for key, value in pil_sample.items():
                if item_idx >= num_items_to_show: break
                ax = axs[item_idx] if num_items_to_show > 1 else axs

                if isinstance(value, Image.Image):
                    ax.imshow(np.array(value))
                    title_str = f"{key}\nMode: {value.mode}, Size: {value.size}"
                    if value.mode == 'L':
                        try:
                            unique_vals_viz = np.unique(np.array(value))
                            title_str += f"\nUnique: {unique_vals_viz.tolist()}"
                        except Exception:
                            pass
                    ax.set_title(title_str, fontsize=10)
                else:
                    ax.text(0.5, 0.5, f"Cannot display\n{key}\nType: {type(value)}",
                              ha='center', va='center', fontsize=10)
                ax.axis('off')
                item_idx += 1
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
        elif args_cli.visualize and not current_sample_all_masks_valid:
             print(f"  由于样本 {sample_index} 验证失败，跳过可视化。")

    print("\n--- 验证结束 ---")
    if all_samples_valid:
        print("所有被检查样本的 PIL 掩码均符合目标值域！")
    else:
        print("一个或多个样本的 PIL 掩码不符合目标值域。请检查上面的 [ERROR] 信息。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="验证 CityscapesSegmentation 返回的原始 PIL 掩码。")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help="要验证的数据集划分 (默认: train)")
    parser.add_argument('--num_samples', type=int, default=3,
                        help="要验证的样本数量 (默认: 3)")
    parser.add_argument('--random_samples', action='store_true',
                        help="是否随机选择样本进行验证，否则从头开始选择。")
    parser.add_argument('--visualize', action='store_true',
                        help="是否可视化每个已验证样本的 PIL 图像和掩码。")
    
    cli_args = parser.parse_args()
    main(cli_args)