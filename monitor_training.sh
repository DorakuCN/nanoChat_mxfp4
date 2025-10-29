#!/bin/bash

# nanochat训练监控脚本
# 用法: bash monitor_training.sh

echo "=== nanochat训练监控 ==="
echo "时间: $(date)"
echo

# 检查tmux会话状态
echo "=== TMUX会话状态 ==="
tmux list-sessions 2>/dev/null || echo "没有活动的tmux会话"
echo

# 检查GPU状态
echo "=== GPU状态 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
echo

# 检查训练进程
echo "=== 训练进程 ==="
ps aux | grep -E "(torchrun|python.*base_train)" | grep -v grep
echo

# 显示最新训练日志
echo "=== 最新训练进度 ==="
if tmux has-session -t nanochat_training 2>/dev/null; then
    tmux capture-pane -t nanochat_training -p | tail -5
else
    echo "tmux会话不存在"
fi
echo

# 显示训练日志文件
echo "=== 最新日志文件 ==="
ls -la logs/train_dual_gpu_$(date +%Y%m%d)_*.log 2>/dev/null | tail -1
echo

echo "=== 监控命令 ==="
echo "查看tmux会话: tmux attach -t nanochat_training"
echo "查看实时日志: tmux capture-pane -t nanochat_training -p"
echo "停止训练: tmux kill-session -t nanochat_training"