#!/usr/bin/env bash
# ============================================================
#  LPR System — macOS / Linux installer
#  Run once after cloning / unzipping the project.
#    chmod +x install.sh && ./install.sh
# ============================================================
set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch (CPU build)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# --- GPU (NVIDIA CUDA 11.8) — uncomment and comment the CPU line above ---
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing remaining dependencies..."
pip install -e .

echo ""
echo "Running install verification..."
python verify_install.py

echo ""
echo "Done! Activate the environment with:"
echo "  source venv/bin/activate"
