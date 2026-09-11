#!/bin/bash
# ESM-2 Enzyme Prediction — Environment Setup
# Run this script to create a virtual environment and install dependencies

set -e

echo "============================================"
echo "ESM-2 Enzyme Prediction — Setup"
echo "============================================"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version — for GPU, use --index-url with CUDA)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ESM-2 and dependencies
echo ""
echo "Installing remaining dependencies..."
pip install fair-esm transformers pandas numpy scikit-learn matplotlib seaborn requests tqdm jupyter

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To use:"
echo "  source venv/bin/activate"
echo "  python scripts/run_pipeline.py"
echo "============================================"
