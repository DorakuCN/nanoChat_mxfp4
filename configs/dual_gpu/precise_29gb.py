# 精确29GB显存利用率的双卡分布式训练配置
# 目标：充分利用32GB显存，显存使用量达到29GB左右，避免OOM

run = "dual_gpu_precise_29gb"
depth = 20  # 适中的模型深度
device_batch_size = 16  # 适中的批次大小
total_batch_size = 98304  # 确保可整除 (16 * 3072 * 2 = 98304)
max_seq_len = 3072  # 保持较长的序列长度
eval_every = 100
sample_every = 100
use_dummy_wandb = True

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数 - 减少评估时的显存使用
eval_tokens = 1 * 98304  # 减少评估tokens数量
core_metric_every = 1000
core_metric_max_per_task = 500
