# PyTorch libuv支持和双卡分布式训练总结

## 📋 问题描述

在尝试运行双卡分布式训练时，遇到了以下错误：
```
torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built without libuv support, run with USE_LIBUV=0 to disable it.
```

## 🔍 问题分析

### 根本原因

1. **libuv版本不匹配**:
   - 系统通过apt安装的libuv版本: 1.48.0
   - PyTorch构建系统使用的libuv版本: 1.41.0 (来自tensorpipe/third_party/libuv)

2. **构建配置问题**:
   - PyTorch虽然配置了`USE_LIBUV=ON`，但实际编译时没有正确启用libuv支持
   - 构建后的库文件没有复制到PyTorch安装目录，导致运行时加载的是旧版本

3. **运行时检测机制**:
   - PyTorch通过`::c10d::detail::is_libuv_tcpstore_backend_available()`函数检测libuv支持
   - 该函数检查编译时是否定义了`TORCH_USE_LIBUV`宏

## 🛠️ 解决方案

### 步骤 1: 重新配置PyTorch构建

```bash
cd /home/llama/Tools/pytorch/build_sm120
cmake .. -DUSE_LIBUV=ON
```

### 步骤 2: 重新构建关键组件

```bash
ninja torch_cpu torch_python
```

### 步骤 3: 复制新构建的库到PyTorch安装目录

```bash
cp build_sm120/lib/libtorch_cpu.so torch/lib/
cp build_sm120/lib/libtorch_python.so torch/lib/
```

### 步骤 4: 验证libuv支持

```python
import sys
sys.path.insert(0, '/home/llama/Tools/pytorch')
import torch
import torch.distributed as dist
import os

# 设置环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

# 测试分布式初始化
dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
print('✅ 分布式训练初始化成功 - libuv支持正常!')
dist.destroy_process_group()
```

### 步骤 5: 修复训练脚本配置

1. **更新`train_dual_gpu.sh`**:
   ```bash
   # 设置PyTorch路径
   export TORCH_LOCAL_PATH="/home/llama/Tools/pytorch"
   export PYTHONPATH="${TORCH_LOCAL_PATH}:${PYTHONPATH}"
   export LD_LIBRARY_PATH="${TORCH_LOCAL_PATH}/torch/lib:${LD_LIBRARY_PATH}"
   ```

2. **修复配置文件参数传递**:
   - 原先: `scripts/base_train.py --config=config_dual_gpu.py` ❌
   - 修正: `scripts/base_train.py config_dual_gpu.py` ✅

3. **修复wandb配置覆盖问题**:
   - 在`scripts/base_train.py`中添加检查，避免覆盖配置文件中的`use_dummy_wandb`设置
   ```python
   if 'use_dummy_wandb' not in globals():
       use_dummy_wandb = run == "dummy" or not master_process or not wandb_available
   ```

## ✅ 验证结果

### 系统状态

```bash
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090 D      Off |   00000000:21:00.0 Off |                  N/A |
| 33%   47C    P5             61W /  600W |    6407MiB /  32607MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 5090 D      Off |   00000000:49:00.0 Off |                  N/A |
| 38%   49C    P1             65W /  600W |    6381MiB /  32607MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### 训练进程

- ✅ 1个torchrun主进程
- ✅ 2个训练进程 (每个GPU一个)
- ✅ 双GPU内存使用: ~6.3GB each
- ✅ CPU使用率: 97-98% (说明正在进行计算)

### 训练输出

```
Distributed world size: 2
Total batch size 524,288 => gradient accumulation steps: 8
Total number of training tokens: 6,710,886,400
Step 00000 | Validation bpb: 3.3109
```

## 📝 关键要点

1. **libuv是PyTorch分布式训练的可选依赖**:
   - 默认使用TCPStore进行进程间通信
   - libuv提供更好的异步I/O性能
   - 需要在编译时明确启用

2. **构建系统的正确配置**:
   - 使用`cmake .. -DUSE_LIBUV=ON`配置
   - 确保编译后的库文件复制到正确的位置
   - 验证运行时加载的是新构建的版本

3. **分布式训练的环境变量**:
   - `TORCH_LOCAL_PATH`: PyTorch本地构建路径
   - `PYTHONPATH`: Python模块搜索路径
   - `LD_LIBRARY_PATH`: 动态链接库搜索路径

4. **配置文件的正确处理**:
   - 配置文件中的设置应该优先于脚本中的默认值
   - 使用`globals()`检查变量是否已定义

## 🚀 后续优化建议

1. **性能调优**:
   - 调整`device_batch_size`以充分利用GPU内存
   - 监控GPU利用率，确保达到80%以上
   - 使用`torch.profiler`分析性能瓶颈

2. **代码改进**:
   - 将PyTorch路径配置添加到环境变量或配置文件
   - 创建自动化脚本验证libuv支持
   - 添加分布式训练的健康检查

3. **文档更新**:
   - 记录libuv支持的必要性和优势
   - 提供详细的构建和测试步骤
   - 添加常见问题和解决方案

## 📚 参考资料

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch C10D Documentation](https://pytorch.org/docs/stable/distributed.html)
- [libuv Official Documentation](https://libuv.org/)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)

---

**构建时间**: 2025-10-21 14:50  
**测试时间**: 2025-10-21 14:58  
**状态**: ✅ 成功

