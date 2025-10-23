# NanoChat 快速开始指南

## 环境准备

### 1. 安装依赖
```bash
# 安装Python依赖
pip install -e .

# 或使用uv
uv sync
```

### 2. 检查GPU环境
```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 检查GPU数量
python -c "import torch; print(torch.cuda.device_count())"
```

## 训练流程

### 阶段1: Tokenizer训练
```bash
# 训练自定义tokenizer
python scripts/tok_train.py --vocab_size=65536 --data_dir=<data_path>

# 验证tokenizer
python scripts/tok_eval.py
```

### 阶段2: 数据准备
```bash
# 下载FineWeb-Edu数据
python nanochat/dataset.py -n 100  # 下载100个文件

# 检查数据
python -c "from nanochat.dataset import parquets_iter_batched; print('Data ready')"
```

### 阶段3: 基础预训练
```bash
# 单GPU训练
python scripts/base_train.py configs/dual_gpu/stable_26gb.py

# 多GPU训练
torchrun --nproc_per_node=2 scripts/base_train.py configs/dual_gpu/stable_26gb.py
```

### 阶段4: 中期指令训练
```bash
# 加载base checkpoint进行mid训练
python scripts/mid_train.py --base_checkpoint=base_checkpoints/latest.pt
```

### 阶段5: 监督微调
```bash
# 使用mid checkpoint进行SFT
python scripts/chat_sft.py --mid_checkpoint=mid_checkpoints/latest.pt
```

### 阶段6: 强化学习
```bash
# 使用SFT checkpoint进行RL
python scripts/chat_rl.py --sft_checkpoint=chatsft_checkpoints/latest.pt
```

## 配置选择

### 根据GPU内存选择配置
- **24GB GPU**: `configs/dual_gpu/stable_24gb.py`
- **26GB GPU**: `configs/dual_gpu/stable_26gb.py`
- **多GPU**: `configs/dual_gpu/balanced.py`

### 自定义配置
```python
# 创建自定义配置文件
run = "my_experiment"
depth = 16
device_batch_size = 8
total_batch_size = 262144
max_seq_len = 2048
eval_every = 100
```

## 监控训练

### 1. 日志监控
```bash
# 实时查看训练日志
tail -f logs/train_dual_gpu_*.log

# 监控GPU使用
watch -n 1 nvidia-smi
```

### 2. Wandb集成
```bash
# 启用wandb日志
python scripts/base_train.py configs/dual_gpu/stable_26gb.py --run=my_experiment
```

### 3. 检查点管理
```bash
# 查看最新checkpoint
ls -la base_checkpoints/

# 恢复训练
python scripts/base_train.py --resume_from=base_checkpoints/step_1000.pt
```

## 评估模型

### 1. 基础评估
```bash
# CORE benchmark评估
python -m scripts.base_eval --checkpoint=base_checkpoints/latest.pt

# Bits-per-byte评估
python -m scripts.base_loss --checkpoint=base_checkpoints/latest.pt
```

### 2. 对话评估
```bash
# 任务准确率评估
python -m scripts.chat_eval --checkpoint=chatsft_checkpoints/latest.pt

# 交互式对话测试
python scripts/chat_cli.py --checkpoint=chatsft_checkpoints/latest.pt
```

## 常见问题

### 1. 内存不足
- 减少`device_batch_size`
- 减少`max_seq_len`
- 使用梯度累积

### 2. 训练速度慢
- 检查数据加载速度
- 使用CUDAPrefetcher
- 调整`eval_every`频率

### 3. 收敛问题
- 调整学习率
- 检查数据质量
- 增加训练步数

## 高级用法

### 1. 自定义任务
```python
# 在tasks/目录添加新任务
class MyTask:
    def __getitem__(self, idx):
        return {"input": "...", "output": "..."}
    
    def evaluate(self, predictions):
        return {"accuracy": 0.95}
```

### 2. 分布式训练
```bash
# 多节点训练
torchrun --nnodes=2 --nproc_per_node=4 scripts/base_train.py
```

### 3. 模型导出
```python
# 导出为HuggingFace格式
from nanochat.checkpoint_manager import load_model
model = load_model("chatsft_checkpoints/latest.pt")
model.save_pretrained("my_model")
```

## 性能优化建议

1. **数据加载**: 使用SSD存储，增加worker数量
2. **内存管理**: 定期清理checkpoint，使用混合精度
3. **计算优化**: 启用torch.compile，使用Flash Attention
4. **网络优化**: 使用InfiniBand，优化DDP通信

## 故障排除

### 检查清单
- [ ] CUDA环境正确安装
- [ ] 数据路径可访问
- [ ] 配置文件语法正确
- [ ] 磁盘空间充足
- [ ] 网络连接正常（wandb）

### 日志分析
```bash
# 查找错误信息
grep -i error logs/train_dual_gpu_*.log

# 分析内存使用
grep -i "cuda out of memory" logs/train_dual_gpu_*.log
```
