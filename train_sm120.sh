#!/usr/bin/env bash

# Unified launcher for nanochat training on the local SM120 / CUDA 13 stack.
# It activates the conda base environment, exports the GPU-specific variables,
# and then executes the chosen training command (defaults to base pretraining).

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Conda base environment activation

CONDA_PREFIX_ROOT=${CONDA_PREFIX_ROOT:-"$HOME/anaconda3"}
CONDA_SH="$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
    echo "[train_sm120] Unable to locate conda at $CONDA_SH" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate base

# -----------------------------------------------------------------------------
# Core environment variables for the locally built Torch

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/Train/nanochat}"
export TORCH_LOCAL_PATH="${TORCH_LOCAL_PATH:-$HOME/Tools/pytorch}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TORCH_LOCAL_PATH/torch/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export USE_LIBUV=0
export TORCHELASTIC_USE_AGENT_STORE="${TORCHELASTIC_USE_AGENT_STORE:-0}"
if [ -d "$TORCH_LOCAL_PATH/torch" ]; then
    case ":${PYTHONPATH:-}:" in
        *:"$TORCH_LOCAL_PATH":*) ;;
        *) export PYTHONPATH="${TORCH_LOCAL_PATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
fi

mkdir -p "$NANOCHAT_BASE_DIR"

echo "[train_sm120] Using conda env: $(conda info --json | python -c 'import json,sys; print(json.load(sys.stdin)["active_prefix_name"])')"
echo "[train_sm120] NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "[train_sm120] TORCH_LOCAL_PATH=$TORCH_LOCAL_PATH"
echo "[train_sm120] CUDA_HOME=$CUDA_HOME"

# Validate that torch is importable from the configured path
python - <<'PY'
import os, sys
torch_root = os.environ.get("TORCH_LOCAL_PATH", "")
try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"[train_sm120] ERROR: unable to import torch ({exc})\n")
    hint = torch_root or "<path-to-pytorch>"
    sys.stderr.write(f"[train_sm120] Hint: run 'uv sync' or install the local build with 'pip install --no-deps {hint}'\n")
    sys.exit(1)
else:
    print(f"[train_sm120] torch {torch.__version__} from {torch.__file__}")
    if torch_root and torch.__file__ and not os.path.realpath(torch.__file__).startswith(os.path.realpath(torch_root)):
        print(f"[train_sm120] WARNING: torch resolved to {torch.__file__}, expected under {torch_root}")
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"[train_sm120] CUDA device: {device} (sm_{cap[0]}{cap[1]})")
    else:
        print("[train_sm120] WARNING: torch.cuda.is_available() is False")
PY

# -----------------------------------------------------------------------------
# Command dispatch

# Wrap torchrun to ensure libuv stays disabled on local builds.
if [ "${1-}" = "torchrun" ]; then
    shift
    has_rdzv_conf=0
    for arg in "$@"; do
        case "$arg" in
            --rdzv_conf=*|--rdzv-conf=*|--rdzv_conf|--rdzv-conf)
                has_rdzv_conf=1
                break
                ;;
        esac
    done
    if [ "$has_rdzv_conf" -eq 0 ]; then
        set -- python -m nanochat.torchrun_no_libuv --rdzv_conf=use_libuv=0 "$@"
    else
        set -- python -m nanochat.torchrun_no_libuv "$@"
    fi
fi

if [ $# -eq 0 ]; then
    set -- python -m scripts.base_train
fi

echo "[train_sm120] Executing: $*"
exec "$@"
