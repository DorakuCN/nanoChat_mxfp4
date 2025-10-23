# 平衡显存使用配置 - 26GB显存使用
# 目标：在稳定性和显存利用率之间取得平衡

run = "dual_gpu_balanced_26gb"  # 平衡配置的run名称
depth = 13  # 适中的模型深度
device_batch_size = 10  # 适中的批次大小
total_batch_size = 30720  # 适中的总批次大小 (31K tokens)
max_seq_len = 1664  # 适中的序列长度
eval_every = 120
sample_every = 120
use_dummy_wandb = False  # 启用真实的WandB日志记录

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数
eval_tokens = 12 * 30720  # 适中的评估tokens数量
core_metric_every = 1200
core_metric_max_per_task = 600
