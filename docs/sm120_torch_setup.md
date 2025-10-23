# nanochat Training Overview (SM120 / CUDA 13)

This note summarises the training pipeline inside `nanochat/` and documents the
extra steps required to train with the locally built Torch that targets
Ada-Next (`sm_120`, CUDA 13, RTX 5090D).

## Code structure for training

- `scripts/tok_train.py` / `scripts/tok_eval.py`: train & validate the BPE
  tokenizer using `nanochat.tokenizer` utilities, with text shards prepared by
  `nanochat.dataset`.
- `scripts/base_train.py`: pretraining entry point. It builds a GPT model from
  `nanochat.gpt.GPT`, orchestrates distributed data loading via
  `nanochat.dataloader.tokenizing_distributed_data_loader`, and logs metrics
  through the `nanochat.engine.Engine` helper.
- `scripts/base_loss.py` / `scripts/base_eval.py`: post-train diagnostics for
  loss sweeps and CORE evaluations using the saved checkpoints via
  `nanochat.checkpoint_manager`.
- `scripts/mid_train.py`: mid-training stage that adapts the base model to chat
  style data and multiple-choice consumption.
- `scripts/chat_sft.py`: supervised fine-tuning stage that performs
  sequence-to-sequence style updates.
- `scripts/chat_eval.py`: evaluation harness for mid/SFT/RL checkpoints.
- `nanochat/engine.py`: wraps optimisation (Muon + AdamW) and mixed precision
  execution including gradient accumulation.
- `nanochat/report.py`: generates `report.md` summarising every phase.

The `speedrun.sh` script stitches these phases into an end-to-end pipeline, and
`run1000.sh` is the extended 41 h preset. Both now expect the local SM120 Torch
build by default.

## Torch dependency changes

- `pyproject.toml` now pins `torch>=2.10.0.dev0` and configures `uv` to source
  it from `../pytorch` (editable install). This resolves to the local
  2.10.0a0+git build that carries the CUDA 13 / `sm_120` support.
- `uv.lock` has been rewritten accordingly and drops the CUDA 12 wheel metadata
  so dependency resolution stays offline-friendly.

If your tree lives elsewhere, override the path by exporting
`TORCH_LOCAL_PATH=/abs/path/to/pytorch` before running `uv sync`.

## Runtime environment expectations

Both `speedrun.sh` and `run1000.sh` now:

1. Create/activate the project venv via `uv`.
2. Honour `NANOCHAT_BASE_DIR` (defaults to `~/.cache/nanochat`, so set it first if you
   want to reuse `/home/llama/Train/nanochat`).
3. Export defaults for the local Torch build:
   - `TORCH_LOCAL_PATH` – defaults to `$HOME/Tools/pytorch`.
   - `CUDA_HOME` – defaults to `/usr/local/cuda-13.0`.
   - `LD_LIBRARY_PATH` – prepends `$TORCH_LOCAL_PATH/torch/lib` and
     `$CUDA_HOME/lib64` so the compiled extensions resolve.
   - `TORCH_CUDA_ARCH_LIST` – defaults to `12.0` to target `sm_120`.
   - `PYTHONPATH` – injects `$TORCH_LOCAL_PATH` so the editable build is
     discoverable even before `pip install`.
   - `USE_LIBUV` – defaults to `0` so `torchrun` uses the built-in TCP store
     even though the local Torch build omits libuv.
4. Run a small Python probe that ensures Torch is imported from the requested
   path and prints the detected CUDA capability.
   - If the import fails (or resolves to a different location) the scripts now
     fall back to installing the latest CUDA 13 nightly wheels automatically.

### Multi-GPU launcher quick tips

- `torchrun` works out of the box with two local GPUs, e.g.
  `./train_sm120.sh torchrun --standalone --nproc_per_node=2 -m scripts.base_train`.
  The launcher transparently proxies the call through
  `python -m nanochat.torchrun_no_libuv`, which disables the libuv transport
  in both the Elastic agent and the classic `env://` rendezvous path.
- All launch helpers (`train_sm120.sh`, `speedrun.sh`, `run1000.sh`) export
  `USE_LIBUV=0` and route `torchrun` invocations through the wrapper so Torch
  never requests libuv. Set `USE_LIBUV=1` and bypass the wrapper only if your
  Torch build explicitly enables libuv.

You can adjust these exports (e.g. custom CUDA install) by setting the
respective environment variables before launching the scripts.

## Quick validation

After `uv sync` finishes, run either training script until the Torch probe
prints something like:

```
[speedrun] torch 2.10.0a0+gitba93d56 from /home/llama/Tools/pytorch/torch/__init__.py
[speedrun] CUDA capability: 12.0
```

To double-check from Python:

```
source .venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_properties(0))
PY
```

The training pipeline then proceeds exactly as before: tokenizer → base
pretrain → midtrain → SFT (→ optional RL) with all GPU kernels executing through
the SM120 build.

## Fresh uv environment quickstart

When creating a brand-new `.venv` with `uv`, install the local Torch build
before syncing the rest of the dependencies to avoid repeated downloads of the
PyTorch wheels:

```bash
uv venv
source .venv/bin/activate
pip install --no-deps /home/llama/Tools/pytorch
uv sync --frozen
```

`pip install --no-deps` keeps `pip` from reaching out to PyPI for CUDA wheels
and simply links the locally built SM120 package into the virtual environment.
After this step, the regular training scripts (`speedrun.sh`, `run1000.sh`, or
`torchrun -m scripts.base_train …`) work as described above.

## Unified launcher script

For day-to-day runs, use the helper at the repo root:

```bash
./train_sm120.sh [command]
```

It activates the conda `base` environment, exports the SM120 Torch variables,
verifies the CUDA device, and then executes the supplied command (default:
`python -m scripts.base_train`). For example:

```bash
./train_sm120.sh bash speedrun.sh
./train_sm120.sh torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```
