# 高显存使用配置 - 31GB显存使用
# 目标：最大化显存利用率，提高训练效率

run = "dual_gpu_high_memory_31gb"  # 高显存配置的run名称
depth = 16  # 增加模型深度
device_batch_size = 16  # 增加批次大小
total_batch_size = 49152  # 增加总批次大小 (49K tokens)
max_seq_len = 2048  # 增加序列长度
eval_every = 200
sample_every = 200
use_dummy_wandb = False  # 启用真实的WandB日志记录

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数
eval_tokens = 20 * 49152  # 增加评估tokens数量
core_metric_every = 2000
core_metric_max_per_task = 1000
