# NanoChat 分布式训练指南

## 概述

NanoChat 支持多GPU分布式训练，使用 PyTorch 的 `torchrun` 和分布式数据并行 (DDP) 技术。本指南详细说明如何配置和运行双卡分布式训练。

## 架构支持

### 当前支持的分布式组件

1. **DistMuon**: 分布式版本的Muon优化器
   - 支持梯度分片和同步
   - 自动处理参数分片
   - 适用于2D参数（线性层权重）

2. **DistAdamW**: 分布式版本的AdamW优化器
   - 支持梯度平均
   - 适用于嵌入层和输出层
   - 自动处理优化器状态同步

3. **分布式数据加载器**: 
   - 自动分片数据到不同GPU
   - 支持流式数据加载
   - 自动处理数据同步

## 环境要求

### 硬件要求
- 至少2个NVIDIA GPU
- 每个GPU至少8GB显存（推荐16GB+）
- GPU间支持NCCL通信

### 软件要求
- CUDA 11.8+ 或 12.0+
- PyTorch 2.0+ (支持分布式训练)
- NCCL (NVIDIA Collective Communications Library)

## 配置步骤

### 1. 环境变量设置

```bash
# 基本设置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="0,1"  # 指定使用的GPU

# 分布式训练设置
export MASTER_ADDR="localhost"      # 主节点地址
export MASTER_PORT="12355"          # 主节点端口
export WORLD_SIZE=2                 # 总进程数
export NCCL_DEBUG=INFO              # NCCL调试信息
```

### 2. 启动分布式训练

#### 方法1: 使用torchrun (推荐)

```bash
torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=12355 \
    --nnodes=1 \
    --node_rank=0 \
    scripts/base_train.py \
    --depth=20 \
    --device_batch_size=32 \
    --total_batch_size=1048576
```

#### 方法2: 使用提供的脚本

```bash
bash speedrun_distributed.sh
```

### 3. 参数配置

#### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--nproc_per_node` | 每个节点的进程数 | 2 (双卡) |
| `--device_batch_size` | 每GPU批次大小 | 32 |
| `--total_batch_size` | 总批次大小 | 1048576 |
| `--depth` | 模型深度 | 20 |
| `--max_seq_len` | 最大序列长度 | 2048 |

#### 批次大小计算

```
总批次大小 = device_batch_size × max_seq_len × world_size × grad_accum_steps
```

例如：
- device_batch_size = 32
- max_seq_len = 2048  
- world_size = 2 (双卡)
- 需要的grad_accum_steps = 1048576 / (32 × 2048 × 2) = 8

## 代码架构分析

### 1. 分布式初始化 (`nanochat/common.py`)

```python
def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
```

### 2. 分布式优化器 (`nanochat/muon.py`, `nanochat/adamw.py`)

```python
# 自动选择分布式或单GPU优化器
AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
MuonFactory = DistMuon if ddp else Muon
```

### 3. 分布式数据加载 (`nanochat/dataloader.py`)

```python
def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    # 数据分片: start=ddp_rank, step=ddp_world_size
    for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
        # 处理分片数据
```

## 性能优化建议

### 1. 批次大小优化

- **小模型**: device_batch_size = 16-32
- **中等模型**: device_batch_size = 32-64  
- **大模型**: device_batch_size = 64-128

### 2. 内存优化

- 使用 `expandable_segments:True` 减少内存碎片
- 调整 `grad_accum_steps` 平衡内存使用和训练效率
- 监控GPU内存使用情况

### 3. 通信优化

- 使用高速互连（如NVLink）提高GPU间通信速度
- 调整NCCL参数优化通信性能
- 考虑使用混合精度训练

## 故障排除

### 常见问题

1. **NCCL错误**
   ```
   解决方案: 设置 export NCCL_DEBUG=INFO 查看详细错误信息
   ```

2. **内存不足**
   ```
   解决方案: 减少 device_batch_size 或增加 grad_accum_steps
   ```

3. **进程同步问题**
   ```
   解决方案: 确保所有进程使用相同的随机种子
   ```

4. **数据加载不均衡**
   ```
   解决方案: 检查数据分片逻辑，确保数据均匀分布
   ```

### 调试技巧

1. **启用详细日志**:
   ```bash
   export NCCL_DEBUG=INFO
   export TORCH_DISTRIBUTED_DEBUG=DETAIL
   ```

2. **单GPU测试**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/base_train.py
   ```

3. **检查GPU状态**:
   ```bash
   nvidia-smi -l 1
   ```

## 监控和日志

### 训练监控

- 使用 `nvidia-smi` 监控GPU使用率
- 使用 `htop` 监控CPU和内存使用
- 查看训练日志中的损失和指标

### 日志分析

- 每个进程会输出独立的日志
- 主进程 (rank 0) 负责保存检查点
- 使用wandb记录训练指标

## 最佳实践

1. **渐进式扩展**: 从单GPU开始，逐步扩展到多GPU
2. **性能测试**: 在不同配置下测试性能，找到最优参数
3. **资源监控**: 持续监控GPU和内存使用情况
4. **错误处理**: 设置适当的错误处理和恢复机制
5. **检查点管理**: 定期保存检查点，避免训练中断

## 扩展配置

### 4卡训练配置

```bash
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=12355 \
    scripts/base_train.py \
    --device_batch_size=16 \
    --total_batch_size=2097152
```

### 8卡训练配置

```bash
torchrun \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=12355 \
    scripts/base_train.py \
    --device_batch_size=8 \
    --total_batch_size=4194304
```

## 总结

NanoChat的分布式训练架构设计良好，支持自动的分布式优化器和数据加载。通过合理配置参数和监控系统资源，可以实现高效的分布式训练。建议从双卡配置开始，逐步扩展到更多GPU。
