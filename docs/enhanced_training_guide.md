# 增强版双卡分布式训练指南

## 概述

本指南介绍如何使用重构后的训练脚本进行双卡分布式训练，解决了libuv、日志、资源检查、异常处理等问题。

## 脚本架构

### 1. 统一入口脚本 (`train_dual_gpu.sh`)
- **功能**: 简化的统一入口，复用环境设置
- **特点**: 参数化配置，自动环境检测
- **用法**: `./train_dual_gpu.sh <config_file>`

### 2. 增强版脚本 (`train_dual_gpu_enhanced.sh`)
- **功能**: 完整的参数化训练脚本
- **特点**: 支持CLI参数，自动资源检查，智能错误处理
- **用法**: `./train_dual_gpu_enhanced.sh --device_batch_size=16 --total_batch_size=49152`

### 3. 监控脚本 (`monitor_training.sh`)
- **功能**: 实时监控训练状态
- **特点**: GPU状态、训练进度、系统资源监控
- **用法**: `./monitor_training.sh --log=logs/train_dual_gpu_20241021_160000.log`

## 配置文件管理

### 配置目录结构
```
configs/dual_gpu/
├── __init__.py
├── conservative.py    # 保守配置 (稳定运行)
├── balanced.py        # 平衡配置 (性能与稳定性平衡)
└── aggressive.py      # 激进配置 (最大化显存利用率)
```

### 配置说明

#### 保守配置 (`conservative.py`)
```python
depth = 12
device_batch_size = 8
total_batch_size = 24576
max_seq_len = 1536
# 预期显存使用量: 15-16GB (48%)
```

#### 平衡配置 (`balanced.py`)
```python
depth = 16
device_batch_size = 12
total_batch_size = 49152
max_seq_len = 2048
# 预期显存使用量: 20-22GB (65%)
```

#### 激进配置 (`aggressive.py`)
```python
depth = 20
device_batch_size = 14
total_batch_size = 86016
max_seq_len = 3072
# 预期显存使用量: 28-30GB (90%)
```

## 使用方法

### 1. 基础使用

```bash
# 使用保守配置
./train_dual_gpu.sh configs/dual_gpu/conservative.py

# 使用平衡配置
./train_dual_gpu.sh configs/dual_gpu/balanced.py

# 使用激进配置
./train_dual_gpu.sh configs/dual_gpu/aggressive.py
```

### 2. 参数化使用

```bash
# 自定义批次大小
./train_dual_gpu_enhanced.sh \
    --device_batch_size=16 \
    --total_batch_size=49152 \
    --config=configs/dual_gpu/balanced.py

# 自定义迭代次数和评估频率
./train_dual_gpu_enhanced.sh \
    --num_iterations=50 \
    --eval_every=25 \
    --config=configs/dual_gpu/conservative.py
```

### 3. 监控训练

```bash
# 在另一个终端启动监控
./monitor_training.sh --interval=3 --log=logs/train_dual_gpu_20241021_160000.log
```

## 环境配置

### 自动设置的环境变量

```bash
# PyTorch本地路径
export TORCH_LOCAL_PATH="/home/llama/Tools/pytorch"
export PYTHONPATH="$TORCH_LOCAL_PATH:$PYTHONPATH"
export LD_LIBRARY_PATH="$TORCH_LOCAL_PATH/torch/lib:$LD_LIBRARY_PATH"

# 修复libuv问题
export USE_LIBUV=0
export TORCHELASTIC_USE_AGENT_STORE=0

# CUDA配置
export CUDA_HOME="/usr/local/cuda-13.0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 分布式训练配置
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES="0,1"

# 日志配置
export WANDB_RUN="dummy"
export WANDB_MODE="disabled"
```

## 错误处理

### 1. OOM错误处理
```bash
# 检测到OOM时的建议
1. 减小 device_batch_size (当前: 16)
2. 减小 total_batch_size (当前: 49152)
3. 设置 TORCH_COMPILE=0 禁用编译优化
```

### 2. wandb登录问题
```bash
# 检测到wandb问题时的建议
1. 确保 use_dummy_wandb = True
2. 设置 WANDB_MODE=disabled
```

### 3. libuv问题
```bash
# 检测到libuv问题时的建议
1. 确保 USE_LIBUV=0
2. 确保 TORCHELASTIC_USE_AGENT_STORE=0
```

## 资源检查

### 1. GPU状态检查
- 自动检测GPU数量（需要至少2个）
- 检查显存可用性（建议至少20GB）
- 显示GPU详细信息

### 2. 端口占用检查
- 自动检测端口冲突
- 自动寻找可用端口
- 最多尝试10个端口

### 3. 配置验证
- 检查配置文件存在性
- 验证批次大小整除性
- 确保参数合理性

## 日志系统

### 1. 日志文件
- 自动创建带时间戳的日志文件
- 同时输出到控制台和文件
- 日志目录: `logs/`

### 2. 日志格式
```
[INFO] 2024-10-21 16:00:00 启动增强版双卡分布式训练...
[INFO] 2024-10-21 16:00:01 检查GPU状态...
[SUCCESS] 2024-10-21 16:00:02 配置验证通过
```

### 3. 错误分析
- 自动分析日志中的错误类型
- 提供针对性的解决建议
- 记录详细的错误信息

## 性能优化

### 1. 显存优化
- 使用 `expandable_segments:True` 减少内存碎片
- 设置 `max_split_size_mb=512` 限制内存块大小
- 自动调整批次大小避免OOM

### 2. 计算优化
- 启用 `TORCH_COMPILE=1` 编译优化
- 设置 `OMP_NUM_THREADS=1` 避免CPU竞争
- 使用本地编译的PyTorch获得最佳性能

### 3. 分布式优化
- 使用NCCL进行GPU间通信
- 优化数据加载和预处理
- 平衡计算和通信开销

## 最佳实践

### 1. 配置选择
- **开发阶段**: 使用保守配置确保稳定性
- **测试阶段**: 使用平衡配置验证性能
- **生产阶段**: 使用激进配置最大化效率

### 2. 监控建议
- 始终使用监控脚本观察训练状态
- 关注显存使用率和GPU利用率
- 及时处理OOM和性能问题

### 3. 故障排除
- 遇到OOM时逐步减小批次大小
- 遇到libuv问题时检查环境变量
- 遇到wandb问题时确保使用dummy模式

## 示例工作流

### 1. 完整训练流程
```bash
# 1. 启动训练（保守配置）
./train_dual_gpu.sh configs/dual_gpu/conservative.py

# 2. 在另一个终端启动监控
./monitor_training.sh --log=logs/train_dual_gpu_20241021_160000.log

# 3. 观察训练状态，确认稳定后可以尝试更激进的配置
./train_dual_gpu.sh configs/dual_gpu/balanced.py
```

### 2. 参数调优流程
```bash
# 1. 从保守配置开始
./train_dual_gpu_enhanced.sh --config=configs/dual_gpu/conservative.py

# 2. 逐步增加批次大小
./train_dual_gpu_enhanced.sh \
    --device_batch_size=12 \
    --total_batch_size=36864 \
    --config=configs/dual_gpu/conservative.py

# 3. 继续增加直到找到最优配置
./train_dual_gpu_enhanced.sh \
    --device_batch_size=16 \
    --total_batch_size=49152 \
    --config=configs/dual_gpu/conservative.py
```

## 总结

重构后的训练脚本解决了以下关键问题：

1. **统一入口**: 避免脚本差异，集中维护环境配置
2. **参数化设计**: 支持CLI参数，快速实验不同配置
3. **libuv防护**: 显式设置环境变量，避免分布式训练问题
4. **资源检查**: 自动检测GPU状态和端口占用
5. **日志系统**: 完整的日志记录和错误分析
6. **异常处理**: 智能错误检测和解决建议
7. **配置管理**: 集中管理配置文件，易于维护

这些改进大大提升了训练脚本的可复现性、稳定性和易用性。
