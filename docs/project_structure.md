# NanoChat 项目结构说明

## 整理后的项目结构

### 📁 核心目录

```
nanochat/
├── nanochat/              # 核心Python包
│   ├── __init__.py
│   ├── gpt.py            # GPT模型实现
│   ├── tokenizer.py      # Tokenizer系统
│   ├── dataset.py        # 数据集处理
│   ├── dataloader.py     # 数据加载器
│   ├── configurator.py   # 配置管理
│   ├── common.py         # 通用工具
│   ├── checkpoint_manager.py  # Checkpoint管理
│   ├── report.py         # 报告生成
│   └── ...              # 其他核心模块
├── scripts/              # 训练和评估脚本
│   ├── base_train.py     # 基础预训练
│   ├── mid_train.py      # 中期指令训练
│   ├── chat_sft.py       # 监督微调
│   ├── chat_rl.py        # 强化学习
│   ├── tok_train.py      # Tokenizer训练
│   ├── base_eval.py      # 基础评估
│   ├── chat_eval.py      # 对话评估
│   └── ...              # 其他脚本
├── tasks/                # 任务定义
│   ├── arc.py           # ARC任务
│   ├── gsm8k.py         # GSM8K任务
│   ├── mmlu.py          # MMLU任务
│   └── ...              # 其他任务
├── experiments/          # 实验代码
│   └── mxfp4/           # MXFP4实验
├── configs/              # 配置文件
│   └── dual_gpu/        # 双GPU配置
├── docs/                 # 文档
│   ├── pipeline_overview.md      # Pipeline概览
│   ├── quick_start_guide.md      # 快速开始指南
│   ├── scripts_reference.md      # 脚本参考
│   ├── project_structure.md      # 项目结构说明
│   └── ...              # 其他文档
├── tests/                # 测试代码
├── dev/                  # 开发工具
└── archive/              # 归档目录
    ├── logs/            # 历史日志
    ├── configs/         # 重复配置
    ├── scripts/         # 重复脚本
    └── test_logs/       # 测试日志
```

### 📄 核心文件

#### 训练脚本
- `train_dual_gpu.sh` - 主要双GPU训练脚本
- `train_sm120.sh` - SM120训练脚本
- `speedrun.sh` - 快速运行脚本
- `run1000.sh` - 标准运行脚本

#### 配置文件
- `config_dual_gpu.py` - 基础双GPU配置
- `configs/dual_gpu/` - 各种内存配置

#### 项目文件
- `pyproject.toml` - Python项目配置
- `README.md` - 项目说明
- `.gitignore` - Git忽略规则

## 四阶段训练流程

### 1. 基础预训练 (Base Pretraining)
- **入口**: `scripts/base_train.py`
- **配置**: `configs/dual_gpu/*.py`
- **数据**: FineWeb-Edu parquet分片
- **输出**: `base_checkpoints/`

### 2. 中期指令训练 (Mid Training)
- **入口**: `scripts/mid_train.py`
- **输入**: base checkpoint
- **数据**: 任务混合数据
- **输出**: `mid_checkpoints/`

### 3. 监督微调 (SFT)
- **入口**: `scripts/chat_sft.py`
- **输入**: mid checkpoint
- **数据**: 对话数据
- **输出**: `chatsft_checkpoints/`

### 4. 强化学习 (RL)
- **入口**: `scripts/chat_rl.py`
- **输入**: SFT checkpoint
- **数据**: 奖励数据
- **输出**: `chatrl_checkpoints/`

## 核心组件说明

### 数据流
- `nanochat/dataset.py` - 数据源管理
- `nanochat/dataloader.py` - 数据加载和tokenization
- `tasks/` - 各种评估任务定义

### 模型
- `nanochat/gpt.py` - GPT模型实现
- `nanochat/tokenizer.py` - Tokenizer系统
- `nanochat/muon.py` - Muon优化器

### 训练
- `nanochat/engine.py` - 训练引擎
- `nanochat/execution.py` - 执行管理
- `nanochat/common.py` - 通用训练工具

### 评估
- `nanochat/core_eval.py` - CORE评估
- `nanochat/loss_eval.py` - 损失评估
- `scripts/*_eval.py` - 各种评估脚本

### 管理
- `nanochat/checkpoint_manager.py` - Checkpoint管理
- `nanochat/configurator.py` - 配置管理
- `nanochat/report.py` - 报告生成

## 使用指南

### 快速开始
1. 查看 [Quick Start Guide](quick_start_guide.md)
2. 阅读 [Pipeline Overview](pipeline_overview.md)
3. 参考 [Scripts Reference](scripts_reference.md)

### 配置选择
- 根据GPU内存选择 `configs/dual_gpu/` 下的配置
- 修改 `config_dual_gpu.py` 进行自定义配置

### 训练流程
```bash
# 1. 基础预训练
python scripts/base_train.py configs/dual_gpu/stable_26gb.py

# 2. 中期训练
python scripts/mid_train.py --base_checkpoint=base_checkpoints/latest.pt

# 3. 监督微调
python scripts/chat_sft.py --mid_checkpoint=mid_checkpoints/latest.pt

# 4. 强化学习
python scripts/chat_rl.py --sft_checkpoint=chatsft_checkpoints/latest.pt
```

## 归档说明

### archive/ 目录
包含所有历史文件，按类型分类：
- `logs/` - 训练日志
- `configs/` - 重复配置文件
- `scripts/` - 重复脚本文件
- `test_logs/` - 测试日志

### 清理内容
- 删除了所有 `__pycache__` 目录
- 删除了所有 `.pyc` 文件
- 移动了重复和过时的文件到归档目录

## 扩展指南

### 添加新任务
1. 在 `tasks/` 目录创建新任务类
2. 实现必要的方法
3. 在相应阶段集成

### 自定义配置
1. 复制现有配置文件
2. 修改参数
3. 通过 `--config` 参数使用

### 添加新脚本
1. 在 `scripts/` 目录创建
2. 使用标准模板
3. 集成到训练流程中

## 维护建议

1. **定期清理**: 将过时文件移动到 `archive/`
2. **文档更新**: 保持文档与代码同步
3. **配置管理**: 统一管理配置文件
4. **日志管理**: 定期归档训练日志
5. **测试覆盖**: 保持测试代码更新
