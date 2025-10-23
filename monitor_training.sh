#!/bin/bash

# 训练监控脚本
echo "=== 双卡分布式训练监控 ==="
echo "时间: $(date)"
echo ""

# 检查训练进程
echo "=== 训练进程状态 ==="
ps aux | grep -E "(torchrun|python.*base_train)" | grep -v grep
echo ""

# GPU状态
echo "=== GPU状态 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
echo ""

# 训练日志最后几行
echo "=== 最新训练日志 ==="
if [ -f "long_training.log" ]; then
    tail -5 long_training.log
else
    echo "日志文件不存在"
fi
echo ""

# 训练进度估算
if [ -f "long_training.log" ]; then
    echo "=== 训练进度估算 ==="
    CURRENT_STEP=$(tail -1 long_training.log | grep -o "step [0-9]*" | grep -o "[0-9]*" | head -1)
    if [ ! -z "$CURRENT_STEP" ]; then
        TOTAL_STEPS=151040
        PROGRESS=$(echo "scale=2; $CURRENT_STEP * 100 / $TOTAL_STEPS" | bc -l)
        echo "当前步数: $CURRENT_STEP / $TOTAL_STEPS"
        echo "完成进度: ${PROGRESS}%"
        
        # 估算剩余时间（基于当前速度）
        REMAINING_STEPS=$((TOTAL_STEPS - CURRENT_STEP))
        echo "剩余步数: $REMAINING_STEPS"
        echo "预计剩余时间: 请查看日志中的时间信息"
    fi
fi