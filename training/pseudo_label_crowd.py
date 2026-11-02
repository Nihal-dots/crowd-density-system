"""Auto-label the unannotated phone photos in data/raw/crowd with a trained model.

The photos have no head annotations, so a trained CSRNet (e.g. the ShanghaiTech
Part A best_model.pth) predicts their density maps instead ("pseudo-labels").
Each photo is shrunk so its long side is --max-side pixels; the resized image
and its density map (upsampled to the image size, summing to the predicted
count) are saved to data/processed/crowd/.

Pseudo-labels only repeat what the labelling model already believes, so they
are used for training only, never to report accuracy.

Usage (from the project root):
    python training/pseudo_label_crowd.py --checkpoint <part_A best_model.pth>
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.csrnet import CSRNet  # noqa: E402
from training.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from training.density_map import PROCESSED_DIR  # noqa: E402

CROWD_RAW = PROJECT_ROOT / "data" / "raw" / "crowd"
CROWD_PROCESSED = PROCESSED_DIR / "crowd"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" not in checkpoint:  # a bare state dict
        checkpoint = {"model": checkpoint}
    model = CSRNet(pretrained=False, backend_dilation=checkpoint.get("dilation", 2))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def pseudo_label(checkpoint_path, max_side=1024, raw_dir=CROWD_RAW, out_dir=CROWD_PROCESSED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    image_dir = Path(out_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    counts = []
    for raw_path in tqdm(sorted(Path(raw_dir).glob("*.jpg")), desc="crowd pseudo-labels"):
        with Image.open(raw_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")  # phone photos store rotation in EXIF
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            img.save(image_dir / raw_path.name, quality=95)
            image = np.asarray(img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)).unsqueeze(0)
        predicted = model(tensor.to(device)).squeeze().clamp(min=0).cpu().numpy().astype(np.float32)
        count = float(predicted.sum(dtype=np.float64))

        # Upsample to the image size, then rescale so the map still sums to the count
        height, width = image.shape[:2]
        density = cv2.resize(predicted, (width, height), interpolation=cv2.INTER_LINEAR)
        total = float(density.sum(dtype=np.float64))
        if total > 0:
            density *= count / total

        with h5py.File(Path(out_dir) / f"{raw_path.stem}.h5", "w") as f:
            dset = f.create_dataset("density", data=density.astype(np.float32), compression="gzip", compression_opts=4)
            dset.attrs["count"] = count
            dset.attrs["kernel"] = "pseudo-label"
            dset.attrs["source_checkpoint"] = str(checkpoint_path)
        counts.append(count)

    if counts:
        print(f"Labelled {len(counts)} photos: predicted counts min {min(counts):.0f}, "
              f"median {np.median(counts):.0f}, max {max(counts):.0f}")
    print(f"Saved to {Path(out_dir)}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained model, e.g. part_A/best_model.pth")
    parser.add_argument("--max-side", type=int, default=1024, help="resize photos so the long side is at most this")
    args = parser.parse_args()
    return 0 if pseudo_label(args.checkpoint, args.max_side) else 1


if __name__ == "__main__":
    sys.exit(main())
