@echo off
REM ============================================================
REM  LPR System — Windows installer
REM  Run this once after cloning / unzipping the project.
REM ============================================================

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch (CPU build)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM --- GPU (NVIDIA CUDA 11.8) — uncomment the line below and comment the CPU line above ---
REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing remaining dependencies...
pip install -e .

echo.
echo Running install verification...
python verify_install.py

echo.
echo Done! Activate the environment with:
echo   venv\Scripts\activate
pause
