# 高显存利用率的双卡分布式训练配置
# 目标：充分利用32GB显存，显存使用量达到28GB左右

run = "dual_gpu_high_memory"
depth = 20  # 增加模型深度以充分利用显存
device_batch_size = 16  # 增加批次大小
total_batch_size = 98304  # 增加总批次大小 (96K tokens)
max_seq_len = 3072  # 增加序列长度以更好地利用显存
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
eval_tokens = 2 * 98304  # 减少评估tokens数量
core_metric_every = 1000
core_metric_max_per_task = 500
