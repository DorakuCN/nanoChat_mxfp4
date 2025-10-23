"""
Experimental MXFP4/NVFP4 training script.

This file is a copy of scripts/base_train.py with additional hooks that allow
experimenting with forthcoming 4-bit floating-point precisions on Blackwell
class GPUs. The original training script remains untouched.
"""

import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

try:
    import wandb
except ModuleNotFoundError:  # wandb is optional when running with WANDB_RUN=dummy
    wandb = None

from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.gpt import GPTConfig
from scripts.base_eval import evaluate_model

from experiments.mxfp4.dtypes import (
    PrecisionConfig,
    resolve_precision_config,
    PrecisionNotAvailableError,
)
from experiments.mxfp4.gpt_mxfp4 import (
    PrecisionAwareGPT,
    PrecisionAwareGPTConfig,
)

print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy"
depth = 20
max_seq_len = 2048
num_iterations = -1
target_flops = -1.0
target_param_data_ratio = 20
device_batch_size = 32
total_batch_size = 524288
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
eval_every = 250
eval_tokens = 20 * 524288
core_metric_every = 2000
core_metric_max_per_task = 500
sample_every = 2000
model_tag = ""

# Precision specific knobs
precision = "bfloat16"
precision_embedding = None
precision_rotary = None
precision_logits = "float32"
precision_autocast = None
precision_allow_fallback = True

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open(os.path.join("nanochat", "configurator.py")).read())
user_config = {k: globals()[k] for k in config_keys}

# Resolve precision choices early so we can fail fast if the dtype is missing.
precision_config = PrecisionConfig(
    compute=precision,
    embedding=precision_embedding,
    rotary=precision_rotary,
    logits=precision_logits,
    autocast=precision_autocast,
    allow_fallback=precision_allow_fallback,
)
try:
    resolved_precision = resolve_precision_config(precision_config)
except PrecisionNotAvailableError as exc:
    print0(str(exc))
    raise SystemExit(1)

print0(
    "Precision settings "
    f"(compute={resolved_precision.compute}, "
    f"embedding={resolved_precision.embedding}, "
    f"rotary={resolved_precision.rotary}, "
    f"logits={resolved_precision.logits}, "
    f"autocast={resolved_precision.autocast})"
)

# -----------------------------------------------------------------------------

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=resolved_precision.autocast)

wandb_available = wandb is not None
use_dummy_wandb = run == "dummy" or not master_process or not wandb_available
if master_process and not wandb_available and run != "dummy":
    print0("wandb is not installed, falling back to DummyWandb. Install wandb to enable logging.")

wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = "
    f"{tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {total_batch_size:,} => gradient accumulation steps: "
    f"{grad_accum_steps}"
)

# -----------------------------------------------------------------------------
# Initialize the Model

model_config = GPTConfig(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
precision_aware_config = PrecisionAwareGPTConfig(gpt=model_config, precision=precision_config)

with torch.device("meta"):
    model = PrecisionAwareGPT(precision_aware_config)
model.to_empty(device="cuda")
model.init_weights()
orig_model = model
if os.environ.get("TORCH_COMPILE", "0") == "1":
    print0("使用torch.compile模式（MXFP4）")
    model = torch.compile(model, dynamic=False)
else:
    print0("使用eager模式（MXFP4）")

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(
    f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}"
)
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Optimizers / data
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            inputs_cpu, targets_cpu = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            inputs = inputs_cpu.to(device=self.device, dtype=torch.int32, non_blocking=True)
            targets = targets_cpu.to(device=self.device, dtype=torch.int64, non_blocking=True)
        self.next_batch = (inputs, targets)

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def __iter__(self):
        return self


train_loader = tokenizing_distributed_data_loader(
    device_batch_size,
    max_seq_len,
    split="train",
    device=None,
)
train_prefetcher = CUDAPrefetcher(train_loader, device)
x, y = next(train_prefetcher)


def build_val_loader():
    loader = tokenizing_distributed_data_loader(
        device_batch_size,
        max_seq_len,
        split="val",
        device=None,
    )
    return CUDAPrefetcher(loader, device)

# -----------------------------------------------------------------------------
# Schedulers

warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0


def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# -----------------------------------------------------------------------------
# Training loop

min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    if eval_every > 0 and (last_step or (step > 0 and step % eval_every == 0)):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (
            device_batch_size * max_seq_len * ddp_world_size
        )
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(
                orig_model, tokenizer, device, max_per_task=core_metric_max_per_task
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    if master_process and sample_every > 0 and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
            print0(tokenizer.decode(sample[0]))
        model.train()

    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": dict(model_config.__dict__),
                "precision_config": precision_config.__dict__,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            },
        )

    if last_step:
        break

    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        if not last_step or micro_step != grad_accum_steps - 1:
            x, y = next(train_prefetcher)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | "
        f"total time: {total_training_time/60:.2f}m"
    )
    if step % 100 == 0:
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
            }
        )

print0(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

from nanochat.report import get_report

get_report().log(
    section="Base model training (experimental mxfp4)",
    data=[
        user_config,
        {
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": warmup_ratio,
            "warmdown_ratio": warmdown_ratio,
            "final_lr_frac": final_lr_frac,
            "precision": precision_config.__dict__,
        },
        {
            "Minimum validation bpb": min_val_bpb,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results["core_metric"],
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time/60:.2f}m",
            "Peak memory usage": f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB",
        },
    ],
)

wandb_run.finish()
compute_cleanup()
