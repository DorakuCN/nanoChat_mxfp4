# 修正的26GB显存使用配置
# 目标：修复批次大小配置错误，实现26GB显存使用

run = "dual_gpu_corrected_26gb"  # 修正配置的run名称
depth = 13  # 适中的模型深度
device_batch_size = 10  # 适中的批次大小
total_batch_size = 33280  # 修正：必须能被 world_tokens_per_fwdbwd 整除
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
eval_tokens = 12 * 33280  # 修正的评估tokens数量
core_metric_every = 1200
core_metric_max_per_task = 600
