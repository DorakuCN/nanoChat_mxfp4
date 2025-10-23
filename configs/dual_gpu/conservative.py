# 保守的双卡分布式训练配置
# 目标：稳定运行，避免OOM

run = "dual_gpu_conservative"
depth = 12  # 保守的模型深度
device_batch_size = 8  # 保守的批次大小
total_batch_size = 24576  # 保守的总批次大小 (24K tokens)
max_seq_len = 1536  # 保守的序列长度
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
eval_tokens = 10 * 24576  # 减少评估tokens数量
core_metric_every = 1000
core_metric_max_per_task = 500
