# RL训练显存利用优化方案

## 当前状态分析

### 当前配置
- `device_batch_size = 4` (每个GPU的微批次)
- `examples_per_step = 32` (每个训练step处理的样本数)
- `num_samples = 16` (每个问题生成的序列数)
- **显存使用：~2.6GB / 32GB (利用率仅8%)**

### 显存占用分析
```
模型参数:          ~1-2GB
优化器状态:        ~1GB
KV cache (优化后): ~256MB/batch
训练激活值:        <1GB
------------------------
总计:              ~3GB (利用率极低)
```

## 优化建议

### 推荐配置（中等提升）
```bash
DEVICE_BATCH_SIZE=8      # 从4增加到8 (2x)
EXAMPLES_PER_STEP=64     # 从32增加到64 (2x)
NUM_SAMPLES=32           # 从16增加到32 (2x)
```
**预期显存使用：~10-15GB (利用率50-70%)**

### 激进配置（最大化利用）
```bash
DEVICE_BATCH_SIZE=12     # 从4增加到12 (3x)
EXAMPLES_PER_STEP=96    # 从32增加到96 (3x)
NUM_SAMPLES=24          # 从16增加到24 (1.5x)
```
**预期显存使用：~18-25GB (利用率75-85%)**

## 显存占用计算

### KV Cache显存计算
```
KV Cache大小 = num_layers × 2 (K/V) × batch_size × num_heads × seq_len × head_dim × dtype_size

优化后（按实际需要）:
- 单次batch: 16层 × 2 × 8batch × 8heads × 512seq × 128dim × 2bytes = ~512MB
- 训练循环中: 最多2-3个batch同时存在 = ~1.5GB
```

### 训练激活值估算
```
activation_memory ≈ device_batch_size × seq_len × model_dim × num_layers × 2bytes
= 8 × 512 × 1024 × 16 × 2 ≈ 134MB per pass

总激活值（考虑梯度累积）:
≈ 134MB × 4 passes ≈ 536MB
```

### 总显存估算（推荐配置）
```
模型参数:           ~2GB
优化器状态:         ~2GB
KV cache:           ~1.5GB
训练激活值:         ~2GB
中间张量/缓存:      ~2-4GB
------------------------
总计:               ~10-15GB per GPU
```

## 性能影响

### 训练速度
- **增加batch size**: 训练速度可能提升（更好的GPU利用率）
- **增加samples**: 采样时间线性增加，但每个step的样本质量更高
- **总体**: 每个step时间可能增加，但训练效率更高

### 梯度质量
- **更多samples**: 更准确的reward估计，梯度更稳定
- **更多examples**: 每个step的梯度更稳定，训练更平滑

## 使用建议

### 1. 渐进式增加
```bash
# 第一步：先增加到中等配置
bash 启动RL训练_高显存利用版.sh

# 监控显存使用
watch -n 2 nvidia-smi

# 如果显存充足，可以进一步增加
```

### 2. 监控指标
- **显存使用**: 应稳定在10-20GB，不应超过28GB
- **GPU利用率**: 应保持在80%以上
- **训练速度**: 每个step的时间可能增加，但总体效率提升

### 3. 如果OOM（显存不足）
可以逐步降低：
- `device_batch_size` 从8降到6
- `num_samples` 从32降到24
- `max_new_tokens` 从256降到192（如果序列较短）

## 参数对比表

| 参数 | 当前值 | 推荐值 | 激进值 | 说明 |
|------|--------|--------|--------|------|
| device_batch_size | 4 | 8 | 12 | 每个GPU的微批次 |
| examples_per_step | 32 | 64 | 96 | 每个step的样本数 |
| num_samples | 16 | 32 | 24 | 每个问题的序列数 |
| 显存使用/GPU | ~3GB | ~12GB | ~20GB | 预期显存占用 |
| 显存利用率 | 8% | 40% | 65% | 相对于32GB |
| 训练速度 | 基准 | +20-30% | +10-20% | 相对提升 |

## 注意事项

1. **确保可整除**: 
   - `NUM_SAMPLES % DEVICE_BATCH_SIZE == 0`
   - `EXAMPLES_PER_STEP % NUM_GPUS == 0`

2. **显存监控**: 
   - 首次运行建议监控显存，确保不超过28GB
   - 如果接近上限，适当降低参数

3. **训练稳定性**: 
   - 增加batch size通常能提升训练稳定性
   - 但需要确保学习率合适

4. **采样质量**: 
   - 增加`num_samples`可以提升reward估计的准确性
   - 但会增加采样时间

## 快速启动

```bash
# 使用高显存利用版（推荐配置）
bash 启动RL训练_高显存利用版.sh

# 监控训练
tail -f logs/chat_rl_train_highmem_*.log

# 监控显存
watch -n 2 nvidia-smi
```

