# NanoChat 训练Pipeline概览

## 项目架构

NanoChat项目采用四阶段训练流程，从基础预训练到强化学习，构建完整的语言模型训练pipeline。

## 四阶段训练流程

### 1. 基础预训练 (Base Pretraining)
- **脚本**: `scripts/base_train.py`
- **目标**: 在大规模文本数据上进行预训练
- **数据**: FineWeb-Edu parquet分片
- **模型**: GPT架构，支持多GPU分布式训练

### 2. 中期指令混合训练 (Mid Training)
- **脚本**: `scripts/mid_train.py`
- **目标**: 在指令数据上进行混合训练
- **数据**: SmolTalk/MMLU/GSM8K任务混合
- **特点**: 加载base checkpoint，使用缩放学习率

### 3. 监督微调 (Supervised Fine-Tuning)
- **脚本**: `scripts/chat_sft.py`
- **目标**: 对齐模型行为到人类偏好
- **数据**: 对话格式的监督数据
- **评估**: MMLU/ARC等多选题准确率

### 4. 强化学习 (Reinforcement Learning)
- **脚本**: `scripts/chat_rl.py`
- **目标**: 通过奖励信号优化模型
- **方法**: GRPO风格的RLHF
- **任务**: GSM8K数学问题生成

## 核心组件

### 共享组件
- **数据流**: `nanochat/dataset.py:22`, `nanochat/dataloader.py:10`
- **模型**: `nanochat/gpt.py:27`, `nanochat/gpt.py:231`
- **配置**: `nanochat/configurator.py:26`
- **计算**: `nanochat/common.py:92`
- **报告**: `nanochat/report.py:80`

### 分布式训练
- 支持多GPU/单GPU运行
- 自动检测RANK/LOCAL_RANK
- 使用torchrun DDP
- CUDA设备自动设置

### 日志系统
- 默认使用DummyWandb
- 支持真实wandb项目
- 可通过`--run`参数或配置文件覆盖

## 数据Pipeline

### 数据源
- **预训练**: FineWeb-Edu parquet分片
- **指令训练**: TaskMixture组合多源任务
- **SFT**: ARC/GSM8K/SmolTalk
- **RL**: GSM8K reward/eval API

### 数据处理
- `parquets_iter_batched`处理分片和DDP rank
- `tokenizing_distributed_data_loader`并行编码
- 双端队列累积token生成输入/标签矩阵
- CUDAPrefetcher优化H2D传输

## Tokenizer系统

### 训练
- 脚本: `scripts/tok_train.py`
- 输出: RustBPE + token_bytes.pt
- 位置: `~/.cache/nanochat/tokenizer`

### 使用
- `get_tokenizer()`自动加载
- `get_token_bytes()`返回UTF-8字节长度
- 支持对话渲染和特殊token

## 训练配置

### 超参数
- 模型维度: `model_dim = depth * 64`
- 头数: `ceil(model_dim/128)`
- 序列长度: 默认2048
- 批大小: 通过total_batch_size计算

### 优化器
- **Muon**: 线性层优化
- **AdamW**: embedding/lm_head优化
- 支持梯度裁剪和学习率调度

## 评估系统

### 指标
- **Val bits-per-byte**: 预训练质量
- **CORE benchmark**: 综合评估
- **任务准确率**: MMLU/ARC等
- **Pass@k**: 生成质量评估

### 工具
- `scripts/base_eval.py`: CORE评估
- `scripts/base_loss.py`: bpb评估
- `scripts/chat_eval.py`: 任务准确率

## Checkpoint管理

### 保存
- 仅rank0写入模型/优化器/元数据
- 自动处理torch.compile前缀
- 支持最新step自动加载

### 位置
- Base: `base_checkpoints/`
- Mid: `mid_checkpoints/`
- SFT: `chatsft_checkpoints/`
- RL: `chatrl_checkpoints/`

## 使用流程

### 1. 准备阶段
```bash
# 训练tokenizer
python scripts/tok_train.py --vocab_size=65536

# 下载数据
python nanochat/dataset.py -n <num_files>
```

### 2. 训练阶段
```bash
# 基础预训练
python scripts/base_train.py configs/dual_gpu/stable_26gb.py

# 中期训练
python scripts/mid_train.py --base_checkpoint=<path>

# 监督微调
python scripts/chat_sft.py --mid_checkpoint=<path>

# 强化学习
python scripts/chat_rl.py --sft_checkpoint=<path>
```

### 3. 评估阶段
```bash
# 独立评估
python -m scripts.base_eval
python -m scripts.chat_eval
```

## 扩展指南

### 添加新任务
1. 在`tasks/`模块实现新任务类
2. 实现`__getitem__`、`reward`、`evaluate`方法
3. 纳入相应阶段的TaskMixture

### 自定义配置
1. 复制`configs/dual_gpu/`下的配置文件
2. 修改超参数和路径
3. 通过`--config`参数使用

## 监控和调试

### 日志检查
- 定期检查wandb/日志
- 监控`~/.cache/nanochat`下的checkpoint
- 确认评估频率符合资源预算

### 性能优化
- 调整`eval_every`、`core_metric_every`频率
- 根据GPU内存调整批大小
- 使用CUDAPrefetcher优化数据传输
