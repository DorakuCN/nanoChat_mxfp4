#!/bin/bash

# 统一的双卡分布式训练入口脚本
# 复用现有的环境设置，避免重复配置

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# 检查参数
if [ $# -eq 0 ]; then
    log_error "请提供配置文件或使用 --help 查看帮助"
    echo "用法: $0 <config_file> [额外参数...]"
    echo "示例: $0 configs/dual_gpu/conservative.py"
    echo "示例: $0 configs/dual_gpu/balanced.py"
    exit 1
fi

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    cat << EOF
双卡分布式训练脚本

用法:
    $0 <config_file> [额外参数...]

配置文件:
    configs/dual_gpu/conservative.py  - 保守配置 (稳定运行)
    configs/dual_gpu/balanced.py      - 平衡配置 (性能与稳定性平衡)
    configs/dual_gpu/aggressive.py    - 激进配置 (最大化显存利用率)

示例:
    $0 configs/dual_gpu/conservative.py
    $0 configs/dual_gpu/balanced.py
    $0 configs/dual_gpu/aggressive.py

环境要求:
    - 至少2个GPU
    - 本地编译的PyTorch (支持libuv)
    - CUDA 13.0
EOF
    exit 0
fi

CONFIG_FILE="$1"
shift  # 移除第一个参数，保留其他参数

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

log_info "使用配置文件: $CONFIG_FILE"

# 检查GPU状态
log_info "检查GPU状态..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi 未找到"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    log_error "需要至少2个GPU，当前只有 $GPU_COUNT 个"
    exit 1
fi

# 显示GPU信息
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader

# 设置环境变量
log_info "设置环境变量..."

# PyTorch本地路径
export TORCH_LOCAL_PATH="/home/llama/Tools/pytorch"
export PYTHONPATH="$TORCH_LOCAL_PATH:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TORCH_LOCAL_PATH/torch/lib:${LD_LIBRARY_PATH:-}"

# 修复libuv问题
export USE_LIBUV=0
export TORCHELASTIC_USE_AGENT_STORE=0

# CUDA配置
export CUDA_HOME="/usr/local/cuda-13.0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# 编译模式控制（默认禁用以节省显存）
current_compile="${TORCH_COMPILE:-0}"
log_info "TORCH_COMPILE设置为: $current_compile"

# 分布式训练配置
export MASTER_ADDR="localhost"
export MASTER_PORT="${MASTER_PORT:-12355}"

# 检查端口是否被占用，如果被占用则寻找下一个可用端口
check_port_availability() {
    local port=$1
    while lsof -i :"$port" &> /dev/null; do
        log_warn "端口 $port 已被占用，尝试下一个端口..."
        port=$((port + 1))
    done
    export MASTER_PORT=$port
    log_success "使用端口: $MASTER_PORT"
}

check_port_availability "$MASTER_PORT"

export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES="0,1"

# 日志配置
export WANDB_RUN="dummy"
export WANDB_MODE="disabled"

# 性能优化
export OMP_NUM_THREADS=1
# TORCH_COMPILE 由外部环境变量控制
export TORCH_COMPILE="$current_compile"

log_info "环境配置完成"

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="logs/train_dual_gpu_${TIMESTAMP}.log"

log_info "日志文件: $LOG_FILE"

# 启动训练
log_info "启动分布式训练..."

cd /home/llama/Tools/nanochat

# 使用exec确保信号正确传递
exec torchrun \
    --nproc-per-node=2 \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --nnodes=1 \
    --node-rank=0 \
    scripts/base_train.py \
    "$CONFIG_FILE" \
    "$@" 2>&1 | tee "$LOG_FILE"
