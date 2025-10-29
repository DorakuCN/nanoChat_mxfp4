# 335M模型激进配置
# 在保守配置成功后，尝试更高性能配置
# 目标：最大化335M模型性能

run = "dual_gpu_335m_aggressive"
depth = 24  # 模型深度：24层
device_batch_size = 10   # ✅ 尝试恢复到185M成功配置
total_batch_size = 30720  # 10 * 1536 * 2
max_seq_len = 2048  # ✅ 适度增加序列长度
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

# 激进配置说明:
# - batch_size=10: 恢复到185M成功配置
# - seq_len=2048: 适度增加序列长度 (vs 1536)
# - depth=24: 335M参数模型
# - 预期logits显存: 10 * 2048 * 65536 * 4 = 5.2GB
# - 预期总显存: ~30GB (接近极限)

# 使用条件:
# 1. 保守配置(335m_stage2)成功运行
# 2. GPU显存使用稳定
# 3. 无OOM错误
