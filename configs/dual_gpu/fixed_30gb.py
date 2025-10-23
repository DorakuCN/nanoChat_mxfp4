# 修复的30GB显存使用配置
# 目标：修复模型维度问题，实现30GB显存使用

run = "dual_gpu_fixed_30gb"  # 修复配置的run名称
depth = 14  # 修复：确保model_dim能被num_heads整除
device_batch_size = 12  # 增加批次大小
total_batch_size = 46080  # 修复：确保能被world_tokens_per_fwdbwd整除
max_seq_len = 1920  # 增加序列长度
eval_every = 150
sample_every = 150
use_dummy_wandb = False  # 启用真实的WandB日志记录

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数
eval_tokens = 15 * 46080  # 修复的评估tokens数量
core_metric_every = 1500
core_metric_max_per_task = 750
