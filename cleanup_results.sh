#!/bin/bash

# ================== 配置 ==================
# !!! 极度危险：请务必确认这是你想要清空的正确路径 !!!
TARGET_DIR="/workspace/deep参考1/result/cityscapes/deeplab-resnet"
# 清理间隔（秒），1小时 = 3600秒
CLEANUP_INTERVAL=3600
# ==========================================

echo "自动清理脚本启动..."
echo "目标目录: ${TARGET_DIR}"
echo "清理间隔: ${CLEANUP_INTERVAL} 秒"

# --- 函数：执行清理操作 ---
perform_cleanup() {
    # 检查目录是否存在
    if [ -d "$TARGET_DIR" ]; then
        echo "$(date): 正在清理目录 ${TARGET_DIR} ..."
        # 进入目标目录
        cd "$TARGET_DIR" || { echo "错误：无法进入目录 ${TARGET_DIR}"; exit 1; }
        # 删除目录下的所有文件和子目录（不删除目录本身）
        # 使用 find 更安全一点，避免意外的 shell 扩展问题
        find . -mindepth 1 -delete
        # 或者使用 rm (确保你在正确的目录！)
        # rm -rf *
        echo "$(date): 清理完成。"
        # 返回上一级目录（可选，保持脚本干净）
        cd - > /dev/null
    else
        echo "$(date): 警告：目录 ${TARGET_DIR} 不存在，跳过清理。"
    fi
}

# --- 初始清理 ---
echo "$(date): 执行初始清理..."
perform_cleanup

# --- 启动定时清理循环 ---
echo "$(date): 启动每隔 ${CLEANUP_INTERVAL} 秒的定时清理..."
while true; do
    # 等待指定间隔
    sleep "$CLEANUP_INTERVAL"
    # 执行清理
    perform_cleanup
done

echo "脚本意外退出。" # 正常情况下不会执行到这里
exit 0