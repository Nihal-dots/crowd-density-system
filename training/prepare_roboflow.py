"""Generate density maps for the Roboflow "crowd density" dataset.

The export (TensorFlow CSV format) has one head bounding box per row in
<split>/_annotations.csv. Each box centre becomes a head point, and the map
uses the same geometry-adaptive kernels as ShanghaiTech Part A. Images that
have no rows in the CSV get an empty map (count 0).

Maps are saved to data/processed/roboflow/<split>/<image stem>.h5 (dataset
"density", attribute "count").

Usage (from the project root):
    python training/prepare_roboflow.py
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.density_map import PROCESSED_DIR, SUM_TOLERANCE, generate_density_map_adaptive  # noqa: E402

ROBOFLOW_DIR = PROJECT_ROOT / "data" / "raw" / "crowd density tensorflow"
ROBOFLOW_SPLITS = ("train", "valid", "test")
ROBOFLOW_PROCESSED = PROCESSED_DIR / "roboflow"


def load_head_points(split_dir):
    """Return {filename: (N, 2) array of [x, y] box centres} from _annotations.csv."""
    points = defaultdict(list)
    with open(split_dir / "_annotations.csv", newline="") as f:
        for row in csv.DictReader(f):
            x = (float(row["xmin"]) + float(row["xmax"])) / 2
            y = (float(row["ymin"]) + float(row["ymax"])) / 2
            points[row["filename"]].append((x, y))
    return {name: np.array(pts, dtype=np.float64) for name, pts in points.items()}


def prepare_split(split, raw_dir=ROBOFLOW_DIR, out_root=ROBOFLOW_PROCESSED):
    """Write one density map per image in the split; return the worst sum error."""
    split_dir = raw_dir / split
    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    points_by_image = load_head_points(split_dir)

    max_error = 0.0
    for image_path in tqdm(sorted(split_dir.glob("*.jpg")), desc=f"roboflow/{split}"):
        points = points_by_image.get(image_path.name, np.zeros((0, 2)))
        with Image.open(image_path) as img:
            width, height = img.size
        density = generate_density_map_adaptive((height, width), points)
        max_error = max(max_error, abs(float(density.sum(dtype=np.float64)) - len(points)))

        with h5py.File(out_dir / f"{image_path.stem}.h5", "w") as f:
            dset = f.create_dataset("density", data=density, compression="gzip", compression_opts=4)
            dset.attrs["count"] = len(points)
            dset.attrs["kernel"] = "adaptive"
    return max_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=ROBOFLOW_SPLITS, default=list(ROBOFLOW_SPLITS))
    args = parser.parse_args()

    ok = True
    for split in args.splits:
        error = prepare_split(split)
        passed = error < SUM_TOLERANCE
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {split}: max |density.sum() - count| = {error:.2e}")
    print(f"Density maps: {ROBOFLOW_PROCESSED.relative_to(PROJECT_ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
