"""Environment check: prints the version of every required library and CUDA status.

Usage:
    python test_env.py
"""

import importlib
import platform
import sys

# (display name, import name)
LIBRARIES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("opencv-python", "cv2"),
    ("pillow", "PIL"),
    ("matplotlib", "matplotlib"),
    ("h5py", "h5py"),
    ("pandas", "pandas"),
    ("streamlit", "streamlit"),
    ("tqdm", "tqdm"),
]


def main() -> int:
    print(f"Python   : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print("-" * 44)

    missing = []
    for name, module_name in LIBRARIES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{name:<15}: {version}")
        except Exception as exc:  # broken installs can raise more than ImportError
            missing.append(name)
            print(f"{name:<15}: NOT AVAILABLE ({type(exc).__name__}: {exc})")

    print("-" * 44)
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"CUDA available : {cuda}")
        if cuda:
            print(f"CUDA version   : {torch.version.cuda}")
            print(f"GPU count      : {torch.cuda.device_count()}")
            print(f"GPU 0          : {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU (training will be slow; use Colab for a GPU).")
    except Exception:
        print("CUDA available : unknown (torch is not installed)")

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return 1

    print("\nAll libraries imported successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
