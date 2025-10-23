# Wandb Configuration for Nanochat

This document explains how wandb (Weights & Biases) is configured and used in the nanochat project.

## Setup

Wandb has been configured with the following API key: `fa8c8e83fe896ace8d63a99e295a7cb80dd6f288`

## Configuration

### API Key Integration
The wandb API key `fa8c8e83fe896ace8d63a99e295a7cb80dd6f288` is now directly integrated into all training scripts:
- `scripts/base_train.py`
- `scripts/base_train_mxfp4.py`
- `scripts/chat_sft.py`
- `scripts/chat_rl.py`

The API key is automatically set via `os.environ.setdefault("WANDB_API_KEY", "fa8c8e83fe896ace8d63a99e295a7cb80dd6f288")` when wandb is available and not in dummy mode.

### Project Settings
- **Base Training**: `nanochat`
- **SFT Training**: `nanochat-sft`
- **RL Training**: `nanochat-rl`
- **Entity**: Default (personal account)

## Usage

### Training Scripts
All training scripts (`base_train.py`, `base_train_mxfp4.py`, `chat_sft.py`, `chat_rl.py`) now include the wandb API key directly in the code, eliminating the need for external configuration files.

### Environment Variables
The training script `train_dual_gpu.sh` has been updated to enable wandb by default:
```bash
export WANDB_PROJECT="nanochat"
export WANDB_ENTITY=""  # Use default entity
```

### Disabling Wandb
To disable wandb for a run, set:
```bash
export WANDB_MODE="disabled"
# or
export WANDB_RUN="dummy"
```

## Viewing Results

After running training, you can view your results at:
- **Project Dashboard**: https://wandb.ai/doraku-suzaku-personal/nanochat
- Individual runs will have their own URLs displayed in the console output

## Features

- **Automatic Code Tracking**: All code changes are automatically tracked
- **Configuration Logging**: All training hyperparameters are logged
- **Metrics Logging**: Training loss, validation metrics, etc.
- **Model Artifacts**: Model checkpoints can be logged as artifacts
- **Distributed Training Support**: Only the master process logs to wandb

## Integration Status

✅ Wandb API key configured  
✅ Project configuration set up  
✅ Training scripts updated  
✅ Test run completed successfully  
✅ Documentation created  

The wandb integration is now ready for use with your nanochat training runs.
