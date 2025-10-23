# 目标29GB显存利用率的双卡分布式训练配置
# 目标：充分利用32GB显存，显存使用量达到29GB左右

run = "dual_gpu_target_29gb"
depth = 22  # 增加模型深度以充分利用显存
device_batch_size = 15  # 适中的批次大小
total_batch_size = 92160  # 确保可整除 (15 * 3072 * 2 = 92160)
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
eval_tokens = 1 * 92160  # 减少评估tokens数量
core_metric_every = 1000
core_metric_max_per_task = 500
