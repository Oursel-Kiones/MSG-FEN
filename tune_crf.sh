#!/bin/bash

# --- 配置区 ---
PYTHON_CMD="python /workspace/deep参考1/train.py" # 你的 train.py 脚本路径
MODEL_PATH="/workspace/model_best.pth.tar"        # 你的最佳模型路径
LOG_FILE="crf_tuning_log_round2.txt"              # 新的日志文件名，避免覆盖旧的
GPU_ID="0"                                        # 使用的 GPU ID
TEST_BATCH_SIZE=4                                 # 验证时的批大小
# !!! 重要：如果你的 crop/base size 不是 513，请务必在这里设置正确的值 !!!
# CROP_SIZE=513                                   # 确认或取消注释并设置
# BASE_SIZE=513                                   # 确认或取消注释并设置

# --- CRF 参数测试范围 (第二轮 - 更聚焦) ---
# 目标：降低平滑权重，提高颜色敏感度
ITERS_LIST=(5)              # 固定迭代次数，上次结果显示影响不大
SXY_G_LIST=(1 3)            # 测试更小的空间范围 + 默认
COMPAT_G_LIST=(1 2 3)       # 测试更低的平滑权重
SXY_B_LIST=(80)             # 固定上次稍优的空间范围
SRGB_B_LIST=(5 7 10)        # 测试更敏感的颜色阈值 + 上次最优
COMPAT_B_LIST=(1 2 3)       # 测试极低的平滑权重 (关键！)

# --- train.py 的基础参数 (不包括 CRF 特定参数) ---
BASE_ARGS="--dataset cityscapes"
BASE_ARGS+=" --backbone resnet"
BASE_ARGS+=" --gpu-ids $GPU_ID"
BASE_ARGS+=" --resume $MODEL_PATH"
BASE_ARGS+=" --use_crf"           # 启用 CRF
BASE_ARGS+=" --start_epoch 1"     # 触发初始验证（利用不修改代码的技巧）
BASE_ARGS+=" --epochs 1"          # 只运行一个 epoch（我们只关心初始验证结果）
BASE_ARGS+=" --eval-interval 1"
BASE_ARGS+=" --test-batch-size $TEST_BATCH_SIZE"
# 如果需要，取消下面行的注释并设置
# BASE_ARGS+=" --crop-size $CROP_SIZE"
# BASE_ARGS+=" --base-size $BASE_SIZE"
# 除非你想仔细对比某个组合的可视化结果，否则建议注释掉，避免产生过多文件
# BASE_ARGS+=" --save_val_results"

# --- 脚本开始 ---
echo "开始 CRF 参数调优 (第二轮)..."
echo "结果将记录在: $LOG_FILE"
echo ""

# 初始化日志文件
echo "CRF 参数调优日志 (Round 2)" > "$LOG_FILE"
echo "时间戳: $(date)" >> "$LOG_FILE"
echo "模型: $MODEL_PATH" >> "$LOG_FILE"
echo "参数 (iters, sxy_g, compat_g, sxy_b, srgb_b, compat_b) | 提取到的mIoU" >> "$LOG_FILE"
echo "---------------------------------------------------------------------------------" >> "$LOG_FILE"

# 运行计数器
run_count=0
total_runs=$((${#ITERS_LIST[@]} * ${#SXY_G_LIST[@]} * ${#COMPAT_G_LIST[@]} * ${#SXY_B_LIST[@]} * ${#SRGB_B_LIST[@]} * ${#COMPAT_B_LIST[@]}))
echo "总计需要运行 $total_runs 次组合。"

# 嵌套循环遍历参数组合
for iters in "${ITERS_LIST[@]}"; do
  for sxy_g in "${SXY_G_LIST[@]}"; do
    for compat_g in "${COMPAT_G_LIST[@]}"; do
      for sxy_b in "${SXY_B_LIST[@]}"; do
        for srgb_b in "${SRGB_B_LIST[@]}"; do
          for compat_b in "${COMPAT_B_LIST[@]}"; do

            run_count=$((run_count + 1))
            PARAMS_STR="$iters, $sxy_g, $compat_g, $sxy_b, $srgb_b, $compat_b"
            echo "--- [运行 $run_count / $total_runs] 测试参数: $PARAMS_STR ---"

            # 构建当前运行的 CRF 参数
            CRF_ARGS="--crf_iters $iters"
            CRF_ARGS+=" --crf_sxy_g $sxy_g"
            CRF_ARGS+=" --crf_compat_g $compat_g"
            CRF_ARGS+=" --crf_sxy_b $sxy_b"
            CRF_ARGS+=" --crf_srgb_b $srgb_b"
            CRF_ARGS+=" --crf_compat_b $compat_b"

            # 构建完整的命令
            FULL_CMD="$PYTHON_CMD $BASE_ARGS $CRF_ARGS"
            echo "执行命令: $FULL_CMD"

            # 执行命令并捕获输出 (标准输出和标准错误)
            run_output=$($FULL_CMD 2>&1)

            # --- 尝试从 *第一个* 验证块提取 mIoU ---
            mIoU_value=$(echo "$run_output" | sed -n '/Running initial validation/,/Starting Training/p' | grep 'mIoU:' | head -n 1 | sed -n 's/.*mIoU:\([0-9.]*\).*/\1/p')

            # 检查是否成功提取
            if [[ -z "$mIoU_value" ]]; then
                mIoU_value="提取错误"
                echo "错误: 无法为参数 $PARAMS_STR 提取 mIoU"
            fi

            echo "提取到的 mIoU: $mIoU_value"

            # 记录结果
            echo "$PARAMS_STR | $mIoU_value" >> "$LOG_FILE"
            echo "--- 运行结束。结果已记录。 ---"
            echo "" # 添加空行增加可读性

          done # compat_b 循环结束
        done # srgb_b 循环结束
      done # sxy_b 循环结束
    done # compat_g 循环结束
  done # sxy_g 循环结束
done # iters 循环结束

echo "--- CRF 参数调优完成 (第二轮) ---"
echo "总运行次数: $run_count"
echo "结果记录在: $LOG_FILE"
echo "请查看日志文件以找到最佳 mIoU 及对应的参数。"
echo "你可以使用类似下面的命令对日志文件进行排序 (忽略表头和错误行):"
echo "grep -v '提取错误' $LOG_FILE | grep '|' | sort -t '|' -k2 -nr"