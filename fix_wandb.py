#!/usr/bin/env python3
"""
WandB修复脚本 - 确保WandB正确初始化和显示训练曲线
"""

import os
import sys

def check_wandb_installation():
    """检查WandB安装状态"""
    try:
        import wandb
        print(f"✅ WandB已安装，版本: {wandb.__version__}")
        return True
    except ImportError:
        print("❌ WandB未安装")
        return False

def check_wandb_login():
    """检查WandB登录状态"""
    try:
        import wandb
        # 尝试获取当前用户
        user = wandb.api.default_entity
        if user:
            print(f"✅ WandB已登录，用户: {user}")
            return True
        else:
            print("❌ WandB未登录")
            return False
    except Exception as e:
        print(f"❌ WandB登录检查失败: {e}")
        return False

def fix_wandb_config():
    """修复WandB配置"""
    config_path = "/home/llama/Tools/nanochat/configs/dual_gpu/conservative_with_wandb.py"
    
    # 读取配置文件
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 确保use_dummy_wandb = False
    if "use_dummy_wandb = False" not in content:
        print("🔧 修复配置文件中的WandB设置...")
        content = content.replace("use_dummy_wandb = True", "use_dummy_wandb = False")
        with open(config_path, 'w') as f:
            f.write(content)
        print("✅ 配置文件已修复")
    else:
        print("✅ 配置文件WandB设置正确")

def create_debug_config():
    """创建调试配置文件"""
    debug_config = """# 调试WandB配置
run = "debug_wandb_test"
depth = 4
device_batch_size = 2
total_batch_size = 1024
max_seq_len = 512
eval_every = 10
sample_every = 10
use_dummy_wandb = False  # 强制启用真实WandB

# 优化参数
grad_clip = 1.0
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0

# 评估参数
eval_tokens = 1000
core_metric_every = 100
core_metric_max_per_task = 50
"""
    
    with open("/home/llama/Tools/nanochat/configs/dual_gpu/debug_wandb.py", 'w') as f:
        f.write(debug_config)
    print("✅ 创建调试配置文件: configs/dual_gpu/debug_wandb.py")

def main():
    print("🔧 WandB修复脚本启动...")
    
    # 检查WandB安装
    if not check_wandb_installation():
        print("请先安装WandB: pip install wandb")
        return
    
    # 检查WandB登录
    if not check_wandb_login():
        print("请先登录WandB: wandb login")
        return
    
    # 修复配置文件
    fix_wandb_config()
    
    # 创建调试配置
    create_debug_config()
    
    print("\n🎯 修复完成！现在可以运行:")
    print("1. 测试WandB: ./train_dual_gpu.sh configs/dual_gpu/debug_wandb.py")
    print("2. 正式训练: ./train_dual_gpu.sh configs/dual_gpu/conservative_with_wandb.py")

if __name__ == "__main__":
    main()
