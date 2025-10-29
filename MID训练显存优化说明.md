# MID训练显存优化方案

## 问题分析

根据你的分析，MID训练显存使用量从~10GB翻倍到~16GB的主要原因是：

1. **强制torch.compile**: `scripts/mid_train.py:76` 直接执行 `model = torch.compile(model, dynamic=False)`
2. **编译缓存常驻**: TorchDynamo生成的Triton kernel和workspace常驻GPU显存
3. **双重模型**: 编译版模型和原始`orig_model`同时存在
4. **内存分配策略**: `PYTORCH_CUDA_ALLOC_CONF`的扩展段在编译模式下利用率更高

## 解决方案

### 1. 修改了 `scripts/mid_train.py`

```python
# 原来的强制编译
model = torch.compile(model, dynamic=False)

# 修改为可选编译
compile_model = os.environ.get("TORCH_COMPILE", "0") == "1"
if compile_model:
    print0("🔥 Using torch.compile for faster training (higher memory usage)")
    model = torch.compile(model, dynamic=False)
else:
    print0("⚡ Using eager mode for lower memory usage")
```

### 2. 创建了三个启动脚本

#### A. `启动MID训练_低显存版.sh`
- **显存使用**: ~10GB (vs 编译版~16GB)
- **训练时间**: ~10-15小时 (稍慢但更稳定)
- **特点**: 禁用torch.compile，使用eager模式
- **适用**: 显存紧张或需要稳定训练的场景

#### B. `启动MID训练_可配置版.sh`
- **交互式选择**: 用户可选择eager或compile模式
- **灵活配置**: 根据需求动态调整
- **端口隔离**: 使用不同端口避免冲突

#### C. `启动MID训练_修复版.sh` (原有)
- **保持现状**: 继续使用编译模式
- **高性能**: ~16GB显存，更快训练速度

## 使用方法

### 启动低显存版本
```bash
bash 启动MID训练_低显存版.sh
```

### 启动可配置版本
```bash
bash 启动MID训练_可配置版.sh
# 然后选择模式：
# 1) Eager模式 (低显存 ~10GB)
# 2) Compile模式 (高显存 ~16GB)
# 3) 保持当前运行
```

### 手动控制编译
```bash
# 禁用编译 (低显存)
export TORCH_COMPILE=0
bash 启动MID训练_修复版.sh

# 启用编译 (高性能)
export TORCH_COMPILE=1
bash 启动MID训练_修复版.sh
```

## 技术细节

### 环境变量控制
- `TORCH_COMPILE=0`: 使用eager模式，低显存
- `TORCH_COMPILE=1`: 使用compile模式，高性能

### 内存分配优化
- **低显存版**: `max_split_size_mb=128`
- **编译版**: `max_split_size_mb=256`

### 端口隔离
- **修复版**: `MASTER_PORT=12355`
- **低显存版**: `MASTER_PORT=12356`
- **可配置版**: `MASTER_PORT=12357`

## 预期效果

| 模式 | 显存使用 | 训练时间 | 吞吐量 | 稳定性 |
|------|----------|----------|--------|--------|
| Eager | ~10GB | ~10-15h | 较低 | 高 |
| Compile | ~16GB | ~8-12h | 较高 | 中 |

## 建议

1. **首次尝试**: 使用低显存版本验证训练流程
2. **生产环境**: 根据显存情况选择合适的模式
3. **调试阶段**: 使用eager模式便于问题排查
4. **性能优化**: 确认稳定后使用compile模式提升速度

## 监控命令

```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/mid_train_*.log

# 查看训练进程
ps aux | grep torchrun
```
