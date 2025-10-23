#!/usr/bin/env python3
"""
Optimized training script with memory-efficient validation.
This script addresses the OOM issue during validation by using smaller batch sizes for evaluation.
"""

import os
import sys
import time
import math
import wandb
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# Configuration (can be overridden by command line or config file)
# -----------------------------------------------------------------------------

# Model architecture
depth = 16
model_dim = 1536
num_heads = 12
num_kv_heads = 4
vocab_size = 50304
max_seq_len = 2048

# Training hyperparameters
device_batch_size = 16
total_batch_size = 524288
max_steps = 100000
learning_rate = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_steps = 1000
lr_decay_steps = 100000
min_lr = 1e-5

# Data
data_dir = "data"
dataset = "openwebtext"
tokenizer = "gpt2"

# Evaluation
eval_every = 50
sample_every = 50
eval_tokens = 20 * 1048576  # Reduced for memory efficiency
core_metric_every = 1000
core_metric_max_per_task = 500

# Logging
run = "nanochat"
use_dummy_wandb = False

# Checkpointing
save_every = 1000
load_checkpoint = None

# Optimization
use_flash_attention = True
use_sdpa = True
compile_model = True

# Memory optimization for validation
eval_batch_size = 8  # Smaller batch size for validation
eval_max_seq_len = 1024  # Shorter sequence length for validation

# -----------------------------------------------------------------------------
# Load configuration overrides
# -----------------------------------------------------------------------------

exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Initialize distributed training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    compute_init()
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = 'cuda'
    master_process = True
    seed_offset = 0

# Set random seed
torch.manual_seed(1337 + seed_offset)

# Create model
config = GPTConfig(
    n_layer=depth,
    n_embd=model_dim,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    vocab_size=vocab_size,
    sequence_len=max_seq_len,
)
model = GPT(config)
model = model.to(device)

# Initialize distributed training
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])

# Compile model for faster training
if compile_model:
    model = torch.compile(model)

# Create optimizer
optimizer = model.parameters()
if hasattr(model, 'module'):
    optimizer = model.module.parameters()
optimizer = torch.optim.AdamW(optimizer, lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

# Create scaler for mixed precision
scaler = GradScaler()

# Create learning rate scheduler
def get_lr(step):
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    elif step < lr_decay_steps:
        return learning_rate
    else:
        return min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * (step - lr_decay_steps) / (max_steps - lr_decay_steps)))

# Create data loaders
train_loader = tokenizing_distributed_data_loader(
    data_dir=data_dir,
    dataset=dataset,
    tokenizer=tokenizer,
    batch_size=device_batch_size,
    max_seq_len=max_seq_len,
    ddp_rank=ddp_rank,
    ddp_world_size=ddp_world_size,
)

def build_val_loader():
    """Build validation data loader with smaller batch size for memory efficiency."""
    return tokenizing_distributed_data_loader(
        data_dir=data_dir,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,  # Use smaller batch size for validation
        max_seq_len=eval_max_seq_len,  # Use shorter sequence length for validation
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

# Get tokenizer and token bytes
tokenizer_obj = get_tokenizer(tokenizer)
token_bytes = get_token_bytes(tokenizer_obj)

# wandb logging init
wandb_available = wandb is not None
if 'use_dummy_wandb' not in globals():
    use_dummy_wandb = run == "dummy" or not master_process or not wandb_available
if master_process and not wandb_available and run != "dummy":
    print0("wandb is not installed, falling back to DummyWandb. Install wandb to enable logging.")
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=locals())

# Load checkpoint if specified
step = 0
if load_checkpoint is not None:
    checkpoint = torch.load(load_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']

# Create engine
engine = Engine(
    model=model,
    optimizer=optimizer,
    scaler=scaler,
    device=device,
    compile_model=compile_model,
)

# Training loop
model.train()
min_val_bpb = float('inf')
grad_accum_steps = total_batch_size // (device_batch_size * ddp_world_size)

print0(f"Starting training with {ddp_world_size} GPUs")
print0(f"Batch size: {device_batch_size} per GPU, {total_batch_size} total")
print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Memory optimization settings
torch.cuda.empty_cache()
if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
    torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve 5% for validation

autocast_ctx = autocast()

for step in range(step, max_steps):
    # Get learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Training step
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    
    with autocast_ctx:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss = loss / grad_accum_steps

    scaler.scale(loss).backward()

    if (step + 1) % grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Logging
    if step % 10 == 0:
        print0(f"Step {step:05d} | Loss: {loss.item() * grad_accum_steps:.4f} | LR: {lr:.2e}")

    # Evaluation
    last_step = step == max_steps - 1
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (eval_batch_size * eval_max_seq_len * ddp_world_size)
        
        # Clear cache before validation
        torch.cuda.empty_cache()
        
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        
        wandb_run.log({
            "step": step,
            "train_loss": loss.item() * grad_accum_steps,
            "val_bpb": val_bpb,
            "lr": lr,
        })
        
        model.train()

    # Sampling
    if last_step or step % sample_every == 0:
        model.eval()
        with torch.no_grad():
            # Generate sample
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
            sample_str = tokenizer_obj.decode(sample[0].tolist())
            print0(f"Step {step:05d} | Sample: {sample_str[:100]}...")
        model.train()

    # Checkpointing
    if last_step or (step > 0 and step % save_every == 0):
        if master_process:
            save_checkpoint(model, optimizer, step, run)

# Cleanup
if ddp:
    compute_cleanup()

print0("Training completed!")
