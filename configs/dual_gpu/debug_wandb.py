# 调试WandB配置
run = "debug_wandb_test"
depth = 4
device_batch_size = 2
total_batch_size = 1024
max_seq_len = 512
eval_every = 10
sample_every = 10
use_dummy_wandb = False  # 强制启用真实WandB

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数
eval_tokens = 1000
core_metric_every = 100
core_metric_max_per_task = 50
