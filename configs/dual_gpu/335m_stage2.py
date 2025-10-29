# 335M模型第二阶段训练配置
# 基于185M模型成功经验，优化显存使用
# 目标：训练335M参数模型 (d24, 24层)

run = "dual_gpu_335m_stage2"
depth = 24  # 模型深度：从16层增加到24层
device_batch_size = 8   # ✅ 保守配置：从10减少到8
total_batch_size = 24576  # 8 * 1536 * 2
max_seq_len = 1536  # ✅ 保持成功配置：避免OOM
eval_every = 100
sample_every = 100
use_dummy_wandb = True
activation_checkpoint = True  # 通过激活检查点节省显存

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数
eval_tokens = 1 * 24576  # 匹配batch size
core_metric_every = 1000
core_metric_max_per_task = 500

# 显存优化策略:
# - batch_size=8 vs 10: 减少logits显存从3.9GB至3.1GB
# - seq_len=1536: 保持成功配置
# - depth=24: 增加模型容量至335M参数
# - 预期显存需求: ~28GB (安全裕度: 4GB)

# 关键改进:
# 1. 基于185M成功经验
# 2. 保守的batch size配置
# 3. 保持稳定的序列长度
# 4. 增加模型深度提升容量
