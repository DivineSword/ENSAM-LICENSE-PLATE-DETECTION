# License Plate Recognition System

Real-time Moroccan license plate recognition using YOLOv8 and EasyOCR.

## Quick start

### Windows
```bat
install.bat
```

### macOS / Linux
```bash
chmod +x install.sh && ./install.sh
```

Both scripts create a virtual environment, install PyTorch from its official index, install
the rest of the dependencies, and run `verify_install.py` to confirm everything works.

## Manual install (if you prefer)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip

# Step 1 — PyTorch (CPU)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — everything else
pip install -e .

# Step 3 — verify
python verify_install.py
```

For NVIDIA GPU support replace `cpu` with `cu118` or `cu121` in the PyTorch install line.

## Project structure

```
PROJET1/
├── src/
│   └── lpr/
│       ├── camera/        camera capture
│       ├── detection/     YOLOv8 object detection (lazy-loaded)
│       ├── ocr/           EasyOCR text reading (lazy-loaded)
│       └── utils/         shared helpers
├── models/                .pt weight files (not tracked in git)
├── tests/
├── main.py
├── pyproject.toml
├── requirements.txt
├── install.bat            Windows one-click setup
├── install.sh             macOS/Linux one-click setup
└── verify_install.py      post-install check
```

## Running

```bash
python main.py
```

Press **Q** to quit. Update the model paths at the top of `main.py` to point to your
YOLO_2 and CNN model files when they are ready.

## Requirements

Python 3.9 or higher.
