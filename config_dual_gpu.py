# 双卡分布式训练配置
run = "dual_gpu_distributed"
depth = 16
device_batch_size = 16
total_batch_size = 524288
max_seq_len = 2048
eval_every = 50
sample_every = 50
use_dummy_wandb = True  # 禁用wandb以避免需要API key
