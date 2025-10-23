# NanoChat 脚本参考指南

## 训练脚本

### 核心训练流程

#### 1. `scripts/base_train.py` - 基础预训练
**用途**: 在大规模文本数据上进行GPT模型预训练

**主要参数**:
- `--config`: 配置文件路径
- `--depth`: 模型深度
- `--device_batch_size`: 单卡批大小
- `--total_batch_size`: 总批大小
- `--max_seq_len`: 最大序列长度
- `--eval_every`: 评估频率

**使用示例**:
```bash
python scripts/base_train.py configs/dual_gpu/stable_26gb.py
torchrun --nproc_per_node=2 scripts/base_train.py configs/dual_gpu/stable_26gb.py
```

#### 2. `scripts/mid_train.py` - 中期指令训练
**用途**: 在指令数据上进行混合训练，连接预训练和微调

**主要参数**:
- `--base_checkpoint`: 基础模型checkpoint路径
- `--init_lr_frac`: 初始学习率比例
- `--target_examples_per_step`: 每步目标样本数

**使用示例**:
```bash
python scripts/mid_train.py --base_checkpoint=base_checkpoints/latest.pt
```

#### 3. `scripts/chat_sft.py` - 监督微调
**用途**: 在对话数据上进行监督微调，对齐模型行为

**主要参数**:
- `--mid_checkpoint`: 中期模型checkpoint路径
- `--device_batch_size`: 批大小（默认4）
- `--target_examples_per_step`: 每步目标样本数

**使用示例**:
```bash
python scripts/chat_sft.py --mid_checkpoint=mid_checkpoints/latest.pt
```

#### 4. `scripts/chat_rl.py` - 强化学习
**用途**: 通过奖励信号优化模型，实现RLHF

**主要参数**:
- `--sft_checkpoint`: SFT模型checkpoint路径
- `--num_samples`: 每个问题的样本数
- `--reward_weight`: 奖励权重

**使用示例**:
```bash
python scripts/chat_rl.py --sft_checkpoint=chatsft_checkpoints/latest.pt
```

## Tokenizer脚本

### 5. `scripts/tok_train.py` - Tokenizer训练
**用途**: 训练自定义RustBPE tokenizer

**主要参数**:
- `--vocab_size`: 词汇表大小
- `--data_dir`: 训练数据目录
- `--output_dir`: 输出目录

**使用示例**:
```bash
python scripts/tok_train.py --vocab_size=65536 --data_dir=./data
```

### 6. `scripts/tok_eval.py` - Tokenizer评估
**用途**: 评估tokenizer性能和token_bytes.pt

**使用示例**:
```bash
python scripts/tok_eval.py
```

## 评估脚本

### 7. `scripts/base_eval.py` - 基础评估
**用途**: 运行CORE benchmark评估

**主要参数**:
- `--checkpoint`: 模型checkpoint路径
- `--tasks`: 评估任务列表

**使用示例**:
```bash
python -m scripts.base_eval --checkpoint=base_checkpoints/latest.pt
```

### 8. `scripts/base_loss.py` - 损失评估
**用途**: 计算bits-per-byte等损失指标

**使用示例**:
```bash
python -m scripts.base_loss --checkpoint=base_checkpoints/latest.pt
```

### 9. `scripts/chat_eval.py` - 对话评估
**用途**: 评估对话任务的准确率

**使用示例**:
```bash
python -m scripts.chat_eval --checkpoint=chatsft_checkpoints/latest.pt
```

## 交互脚本

### 10. `scripts/chat_cli.py` - 命令行对话
**用途**: 与训练好的模型进行交互式对话

**主要参数**:
- `--checkpoint`: 模型checkpoint路径
- `--max_tokens`: 最大生成token数

**使用示例**:
```bash
python scripts/chat_cli.py --checkpoint=chatsft_checkpoints/latest.pt
```

### 11. `scripts/chat_web.py` - Web界面
**用途**: 提供Web界面进行模型对话

**使用示例**:
```bash
python scripts/chat_web.py --checkpoint=chatsft_checkpoints/latest.pt
```

## 辅助脚本

### 12. `scripts/mid_train.py` - 中期训练
**用途**: 在指令数据上进行混合训练

### 13. `scripts/chat_rl.py` - 强化学习
**用途**: 实现GRPO风格的RLHF训练

## 配置文件

### 配置文件位置
- `configs/dual_gpu/`: 双GPU配置
- `config_dual_gpu.py`: 基础双GPU配置

### 配置参数说明
```python
# 基本配置
run = "experiment_name"           # 实验名称
depth = 16                       # 模型深度
device_batch_size = 8            # 单卡批大小
total_batch_size = 262144        # 总批大小
max_seq_len = 2048              # 最大序列长度

# 训练参数
eval_every = 100                # 评估频率
sample_every = 100              # 采样频率
use_dummy_wandb = True          # 使用虚拟wandb

# 优化参数
grad_clip = 1.0                 # 梯度裁剪
embedding_lr = 0.2              # embedding学习率
unembedding_lr = 0.004          # unembedding学习率
matrix_lr = 0.02                # 矩阵学习率
weight_decay = 0.0              # 权重衰减
```

## 常用命令组合

### 完整训练流程
```bash
# 1. 训练tokenizer
python scripts/tok_train.py --vocab_size=65536

# 2. 基础预训练
python scripts/base_train.py configs/dual_gpu/stable_26gb.py

# 3. 中期训练
python scripts/mid_train.py --base_checkpoint=base_checkpoints/latest.pt

# 4. 监督微调
python scripts/chat_sft.py --mid_checkpoint=mid_checkpoints/latest.pt

# 5. 强化学习
python scripts/chat_rl.py --sft_checkpoint=chatsft_checkpoints/latest.pt
```

### 评估流程
```bash
# 基础评估
python -m scripts.base_eval --checkpoint=base_checkpoints/latest.pt
python -m scripts.base_loss --checkpoint=base_checkpoints/latest.pt

# 对话评估
python -m scripts.chat_eval --checkpoint=chatsft_checkpoints/latest.pt

# 交互测试
python scripts/chat_cli.py --checkpoint=chatsft_checkpoints/latest.pt
```

### 调试和监控
```bash
# 检查tokenizer
python scripts/tok_eval.py

# 监控训练
tail -f logs/train_dual_gpu_*.log

# 检查GPU使用
watch -n 1 nvidia-smi
```

## 脚本开发指南

### 添加新脚本
1. 在`scripts/`目录创建新脚本
2. 导入必要的nanochat模块
3. 使用`nanochat.configurator`处理配置
4. 实现标准的训练/评估循环

### 脚本模板
```python
#!/usr/bin/env python3
import argparse
from nanochat.configurator import Configurator
from nanochat.common import setup_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    
    # 加载配置
    config = Configurator(args.config)
    
    # 设置训练环境
    setup_training()
    
    # 实现具体逻辑
    pass

if __name__ == "__main__":
    main()
```
