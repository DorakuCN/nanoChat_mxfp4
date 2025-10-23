# 激进的双卡分布式训练配置
# 目标：最大化显存利用率，接近极限

run = "dual_gpu_aggressive"
depth = 20  # 激进的模型深度
device_batch_size = 14  # 激进的批次大小
total_batch_size = 86016  # 激进的总批次大小 (84K tokens)
max_seq_len = 3072  # 激进的序列长度
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
eval_tokens = 2 * 86016  # 减少评估tokens数量
core_metric_every = 1000
core_metric_max_per_task = 500
