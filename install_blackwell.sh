#!/bin/bash

# NanoChat Blackwell (sm_120) Installation Script
# This script installs PyTorch nightly build with Blackwell architecture support

set -e

echo "🚀 Installing NanoChat with Blackwell (sm_120) support..."

# Set environment variables
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the nanochat project root directory"
    exit 1
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.bashrc" 2>/dev/null || true
    source "$HOME/.zshrc" 2>/dev/null || true
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch nightly with CUDA 13.0 support for Blackwell
echo "🔥 Installing PyTorch nightly with Blackwell (sm_120) support..."
uv pip install --python .venv/bin/python --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Install other dependencies
echo "📚 Installing other dependencies..."
uv sync

# Install Rust for tokenizer compilation
if ! command -v cargo &> /dev/null; then
    echo "🦀 Installing Rust compiler..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Build the rustbpe tokenizer
echo "🔨 Building rustbpe tokenizer..."
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Verify PyTorch installation and GPU support
echo "✅ Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} (Compute Capability: {props.major}.{props.minor})')
        if props.major >= 12:  # Blackwell architecture is sm_120 (compute capability 12.0)
            print(f'  ✅ Blackwell architecture support detected!')
        else:
            print(f'  ⚠️  Non-Blackwell GPU detected')
else:
    print('❌ CUDA not available')
"

# Download evaluation bundle
echo "📊 Downloading evaluation bundle..."
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
    echo "✅ Evaluation bundle downloaded"
else
    echo "✅ Evaluation bundle already exists"
fi

# Download initial training data
echo "📥 Downloading initial training data (8 shards for tokenizer training)..."
python -m nanochat.dataset -n 8

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Train tokenizer:"
echo "   python -m scripts.tok_train --max_chars=2000000000"
echo ""
echo "2. Evaluate tokenizer:"
echo "   python -m scripts.tok_eval"
echo ""
echo "3. Download full training data (in background):"
echo "   python -m nanochat.dataset -n 240 &"
echo ""
echo "4. Start training with Blackwell support:"
echo "   torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20"
echo ""
echo "💡 For full automation, use:"
echo "   bash speedrun.sh"
echo ""
