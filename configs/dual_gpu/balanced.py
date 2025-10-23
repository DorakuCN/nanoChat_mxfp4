# 平衡的双卡分布式训练配置
# 目标：平衡性能和稳定性

run = "dual_gpu_balanced"
depth = 16  # 平衡的模型深度
device_batch_size = 12  # 平衡的批次大小
total_batch_size = 49152  # 平衡的总批次大小 (48K tokens)
max_seq_len = 2048  # 平衡的序列长度
eval_every = 100
sample_every = 100
use_dummy_wandb = True

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数
eval_tokens = 5 * 49152  # 减少评估tokens数量
core_metric_every = 1000
core_metric_max_per_task = 500
