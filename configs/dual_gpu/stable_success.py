# 稳定成功的双卡训练配置
# 基于 train_dual_gpu_20251023_154520.log 的成功配置
# 已验证可稳定运行至39,000步

run = "dual_gpu_stable_success"
depth = 16  # 模型深度
device_batch_size = 10  # ✅ 成功配置：避免logits OOM
total_batch_size = 30720  # 10 * 1536 * 2
max_seq_len = 1536  # ✅ 成功配置：平衡显存和性能
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
eval_tokens = 1 * 30720  # 匹配batch size
core_metric_every = 1000
core_metric_max_per_task = 500

# 关键差异说明:
# - batch_size=10 vs 20: 减少logits显存从15.7GB至7.8GB
# - seq_len=1536 vs 3072: 减少logits显存从7.8GB至3.9GB
# - 总显存需求: ~25GB (安全裕度: 7GB)

