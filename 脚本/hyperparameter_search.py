import subprocess
import os
import re
import argparse
from itertools import product
from datetime import datetime

# ==============================================================================
# 1. 配置区域: 在这里定义你的实验
# ==============================================================================

# --- 定义你的Python解释器和训练脚本的路径 ---
PYTHON_EXECUTABLE = "/root/miniforge3/envs/tf_deep/bin/python"
TRAIN_SCRIPT_PATH = "/workspace/deep参考1/train1.py"

# --- 定义超参数搜索网格 ---
# 你可以轻松地在这里添加、删除或修改要测试的值
SEARCH_GRID = {
    '--lr': [0.001, 0.005, 0.01],
    '--pos-weight': [10.0, 15.0, 20.0, 25.0]
}

# --- 定义固定的基础命令参数 ---
BASE_COMMAND_ARGS = [
    "--dataset", "cityscapes",
    "--training-stage", "1",
    "--batch-size", "4",  # 在搜索时可以使用稍大的batch-size以加快速度
    "--workers", "8"
]

# ==============================================================================

def run_experiment(lr, pos_weight, search_epochs):
    """
    运行单次训练实验并返回验证损失。
    """
    # 动态生成一个唯一的实验名称，避免日志和模型覆盖
    timestamp = datetime.now().strftime("%H%M%S")
    checkname = f"search_lr{lr}_pw{pos_weight}_{timestamp}"
    
    command = [
        PYTHON_EXECUTABLE,
        TRAIN_SCRIPT_PATH,
        *BASE_COMMAND_ARGS,
        "--epochs", str(search_epochs),
        "--lr", str(lr),
        "--pos-weight", str(pos_weight),
        "--checkname", checkname
    ]
    
    print("-" * 80)
    print(f"🚀 Executing: {' '.join(command)}")
    print("-" * 80)

    try:
        # 运行子进程并捕捉输出
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # 如果脚本返回非零退出码（即出错），则会抛出异常
        )
        
        # 使用正则表达式从输出中解析验证损失
        output = result.stdout
        match = re.search(r"Validation: Val Loss: (\d+\.\d+)", output)
        
        if match:
            val_loss = float(match.group(1))
            print(f"✅ Success! Validation Loss: {val_loss:.4f}")
            return val_loss
        else:
            print("⚠️ Warning: Could not parse validation loss from output.")
            return float('inf') # 返回无穷大表示失败

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during experiment execution for lr={lr}, pos_weight={pos_weight}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return float('nan') # 返回nan表示训练崩溃


def main(search_epochs):
    """
    主函数，协调整个超参数搜索过程。
    """
    print("🔥 Starting Hyperparameter Search...")
    
    # 从搜索网格生成所有参数组合
    param_names = list(SEARCH_GRID.keys())
    param_values = list(SEARCH_GRID.values())
    combinations = list(product(*param_values))
    
    results = []
    
    print(f"Total combinations to test: {len(combinations)}")
    
    for i, combo in enumerate(combinations):
        lr_val, pos_weight_val = combo
        
        print(f"\n--- Running Combination {i+1}/{len(combinations)} ---")
        
        val_loss = run_experiment(lr_val, pos_weight_val, search_epochs)
        
        results.append({
            'lr': lr_val,
            'pos_weight': pos_weight_val,
            'val_loss': val_loss
        })

    # --- 报告最终结果 ---
    print("\n\n" + "="*30 + " SEARCH COMPLETE " + "="*30)
    
    # 过滤掉失败的(nan)并按损失排序
    successful_results = [r for r in results if r['val_loss'] is not float('nan')]
    successful_results.sort(key=lambda x: x['val_loss'])
    
    if not successful_results:
        print("No successful runs completed. Please check the errors above.")
        return

    print("📊 Results sorted by Validation Loss (lower is better):")
    print("-" * 50)
    print(f"{'Learning Rate':<15} | {'Pos Weight':<15} | {'Validation Loss':<20}")
    print("-" * 50)
    
    for res in successful_results:
        print(f"{res['lr']:<15.5f} | {res['pos_weight']:<15.1f} | {res['val_loss']:.4f}")
        
    print("-" * 50)
    
    best_run = successful_results[0]
    print("\n🏆 Best Hyperparameters Found 🏆")
    print(f"   Learning Rate: {best_run['lr']}")
    print(f"   Pos Weight:    {best_run['pos_weight']}")
    print(f"   Best Val Loss: {best_run['val_loss']:.4f}")
    print("\nRecommendation: Use these parameters for your full training run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Search for train1.py")
    parser.add_argument(
        '--search-epochs',
        type=int,
        default=1,
        help="Number of epochs to train for each hyperparameter combination during the search."
    )
    args = parser.parse_args()
    
    main(args.search_epochs)