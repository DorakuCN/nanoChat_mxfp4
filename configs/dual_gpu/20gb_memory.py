# 20GB显存利用率的双卡分布式训练配置
# 目标：显存使用量达到20GB左右，进行长时间稳定训练
# 使用eager模式避免torch.compile的显存开销

run = "dual_gpu_20gb_memory"
depth = 10  # 适中的模型深度
device_batch_size = 6  # 保守的批次大小，确保稳定运行
total_batch_size = 36864  # 确保可整除 (6 * 3072 * 2 = 36864)
max_seq_len = 3072  # 保持较长的序列长度
eval_every = 0  # 关闭验证以节省显存
sample_every = 0  # 关闭采样以节省显存
use_dummy_wandb = True

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数 - 关闭所有评估以节省显存
eval_tokens = 0  # 关闭评估
core_metric_every = 0  # 关闭核心指标计算
core_metric_max_per_task = 0
