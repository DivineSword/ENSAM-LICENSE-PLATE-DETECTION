"""
verify_install.py
-----------------
Run after installing to confirm every dependency loaded correctly.

    python verify_install.py
"""

import importlib
import sys


CHECKS = [
    # (import_name, pip_name, minimum_version_attr)
    ("cv2",          "opencv-python",  "__version__"),
    ("PIL",          "pillow",         "__version__"),
    ("torch",        "torch",          "__version__"),
    ("torchvision",  "torchvision",    "__version__"),
    ("ultralytics",  "ultralytics",    "__version__"),
    ("easyocr",      "easyocr",        "__version__"),
    ("numpy",        "numpy",          "__version__"),
    ("matplotlib",   "matplotlib",     "__version__"),
    ("scipy",        "scipy",          "__version__"),
    ("tqdm",         "tqdm",           "__version__"),
    ("dotenv",       "python-dotenv",  "__version__"),
]

ok = True
col = 28

print(f"\n{'Package':<{col}} {'Status':<12} Version")
print("-" * 55)

for mod, pip_name, ver_attr in CHECKS:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, ver_attr, "n/a")
        print(f"{pip_name:<{col}} {'OK':<12} {version}")
    except ImportError as exc:
        print(f"{pip_name:<{col}} {'MISSING':<12} run: pip install {pip_name}")
        ok = False

# Extra: check that lpr package itself is importable
print()
try:
    import lpr  # noqa: F401
    print(f"{'lpr package':<{col}} {'OK'}")
except ImportError:
    print(f"{'lpr package':<{col}} {'NOT FOUND'} — did you run: pip install -e . ?")
    ok = False

# Check numpy version is below 2.0
try:
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        print(
            "\nWARNING: numpy>=2.0 detected. Some packages may break. "
            "Downgrade with: pip install 'numpy<2.0'"
        )
        ok = False
except Exception:
    pass

print()
if ok:
    print("All checks passed — you are ready to run:  python main.py")
else:
    print("Some checks failed. Fix the items above and re-run this script.")

sys.exit(0 if ok else 1)
