"""Generate ground-truth density maps for the ShanghaiTech dataset.

Part A uses geometry-adaptive Gaussian kernels (sigma depends on the distance
to neighbouring heads); Part B uses a fixed sigma. Every map is renormalised so
that it sums to the number of annotated heads.

Maps are saved to data/processed/<part>/<split>/IMG_n.h5 (dataset "density").

Usage (from the project root):
    python training/density_map.py                # generate, test, save figure
    python training/density_map.py --verify-only  # only run the sum test
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # no display needed; works on Colab and headless machines
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.inspect_dataset import (  # noqa: E402
    EXPECTED_COUNTS,
    RESULTS_DIR,
    find_dataset_root,
    load_mat_annotations,
    validate_split,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FIXED_SIGMA = 15.0
SIGMA_MIN, SIGMA_MAX = 3.0, 25.0
TRUNCATE = 4.0  # kernel radius in units of sigma
SUM_TOLERANCE = 1e-3

# Parts that use a fixed sigma instead of geometry-adaptive kernels
FIXED_SIGMA_PARTS = {"part_B"}


def _render(shape, points, sigmas):
    """Sum one normalised Gaussian per head, then renormalise to the head count."""
    height, width = shape
    density = np.zeros((height, width), dtype=np.float64)
    count = len(points)
    if count == 0:
        return density.astype(np.float32)

    # Keep every head inside the image so it contributes to the count
    xs = np.clip(points[:, 0], 0, width - 1)
    ys = np.clip(points[:, 1], 0, height - 1)

    for x, y, sigma in zip(xs, ys, sigmas):
        radius = int(TRUNCATE * sigma + 0.5)
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, width)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, height)

        gx = np.exp(-((np.arange(x0, x1) - x) ** 2) / (2 * sigma**2))
        gy = np.exp(-((np.arange(y0, y1) - y) ** 2) / (2 * sigma**2))
        kernel = np.outer(gy, gx)
        # Normalise after cropping at the border so each head still adds 1
        density[y0:y1, x0:x1] += kernel / kernel.sum()

    density *= count / density.sum()
    return density.astype(np.float32)


def generate_density_map_fixed(shape, points, sigma=FIXED_SIGMA):
    """Density map with the same Gaussian sigma for every head.

    shape is (height, width); points is an (N, 2) array of [x, y].
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return _render(shape, points, np.full(len(points), float(sigma)))


def generate_density_map_adaptive(shape, points, k=3, beta=0.3):
    """Density map with geometry-adaptive Gaussian kernels.

    sigma_i = beta * mean distance from head i to its k nearest neighbours,
    clipped to [SIGMA_MIN, SIGMA_MAX]. Falls back to a fixed sigma when there
    are not enough heads (N <= k). shape is (height, width); points is an
    (N, 2) array of [x, y]. The returned float32 map sums to N.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(points)
    if count <= k:
        return generate_density_map_fixed(shape, points, FIXED_SIGMA)

    # k + 1 because the nearest neighbour of each point is the point itself
    distances, _ = KDTree(points).query(points, k=k + 1)
    sigmas = np.clip(beta * distances[:, 1:].mean(axis=1), SIGMA_MIN, SIGMA_MAX)
    return _render(shape, points, sigmas)


def iter_splits():
    """Yield (part, split, [(image_path, gt_path), ...]) for every split."""
    dataset_root = find_dataset_root()
    for (part, split), expected in EXPECTED_COUNTS.items():
        _, pairs = validate_split(dataset_root / part / split, expected)
        yield part, split, pairs


def output_path(part, split, image_path):
    return PROCESSED_DIR / part / split / f"{image_path.stem}.h5"


def precompute_all(parts=None, kernel=None, out_root=None):
    """Generate and save a density map for every image in the dataset.

    parts: limit to these parts, e.g. ["part_A"] (default: all).
    kernel: force "adaptive" or "fixed" (default: adaptive for Part A, fixed for Part B).
    out_root: output folder (default: data/processed).
    """
    out_root = Path(out_root or PROCESSED_DIR)
    for part, split, pairs in iter_splits():
        if parts is not None and part not in parts:
            continue
        out_dir = out_root / part / split
        out_dir.mkdir(parents=True, exist_ok=True)
        split_kernel = kernel or ("fixed" if part in FIXED_SIGMA_PARTS else "adaptive")
        generate = generate_density_map_fixed if split_kernel == "fixed" else generate_density_map_adaptive

        for image_path, gt_path in tqdm(pairs, desc=f"{part}/{split} ({split_kernel})"):
            points, count = load_mat_annotations(gt_path)
            with Image.open(image_path) as img:
                width, height = img.size
            density = generate((height, width), points)

            with h5py.File(out_dir / f"{image_path.stem}.h5", "w") as f:
                dset = f.create_dataset("density", data=density, compression="gzip", compression_opts=4)
                dset.attrs["count"] = count
                dset.attrs["kernel"] = split_kernel


def verify_all():
    """Test that every saved density map sums to its head count.

    Returns True when |density.sum() - count| < SUM_TOLERANCE for every image.
    """
    print("\n=== Density sum test ===")
    overall_max, failures, total = 0.0, [], 0
    for part, split, pairs in iter_splits():
        split_max, checked = 0.0, 0
        for image_path, gt_path in pairs:
            name = f"{part}/{split}/{image_path.stem}"
            h5_path = output_path(part, split, image_path)
            if not h5_path.is_file():
                failures.append(f"{name}: density map missing")
                continue
            _, count = load_mat_annotations(gt_path)
            with h5py.File(h5_path, "r") as f:
                density = f["density"][:]
            with Image.open(image_path) as img:
                width, height = img.size
            if density.shape != (height, width):
                failures.append(f"{name}: shape {density.shape} != image {(height, width)}")
            error = abs(float(density.sum(dtype=np.float64)) - count)
            split_max = max(split_max, error)
            checked += 1
            if not error < SUM_TOLERANCE:
                failures.append(f"{name}: sum error {error:.6f} (count {count})")
        total += checked
        overall_max = max(overall_max, split_max)
        print(f"{part}/{split:<11} checked {checked:>3} maps, max error {split_max:.2e}")

    print(f"Max |density.sum() - count| over {total} maps: {overall_max:.2e} (tolerance {SUM_TOLERANCE:.0e})")
    for failure in failures:
        print(f"    FAIL {failure}")
    print("PASSED" if not failures else f"FAILED ({len(failures)} problems)")
    return not failures


def save_example_figure(out_path, part="part_A", split="train_data", index=0):
    """Save a 4-panel figure: image, head points, density map, overlay."""
    pairs = next(p for pt, sp, p in iter_splits() if (pt, sp) == (part, split))
    image_path, gt_path = pairs[index]
    points, count = load_mat_annotations(gt_path)
    with Image.open(image_path) as img:
        image = np.asarray(img.convert("RGB"))
    with h5py.File(output_path(part, split, image_path), "r") as f:
        density = f["density"][:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_image, ax_points, ax_density, ax_overlay = axes.ravel()

    ax_image.imshow(image)
    ax_image.set_title(f"Image - {part}/{split}/{image_path.name}")

    ax_points.imshow(image)
    ax_points.scatter(points[:, 0], points[:, 1], s=6, c="red", linewidths=0)
    ax_points.set_title(f"Head points ({count})")

    ax_density.imshow(density, cmap="jet")
    ax_density.set_title(f"Density map (sum = {density.sum(dtype=np.float64):.2f})")

    ax_overlay.imshow(image)
    ax_overlay.imshow(density, cmap="jet", alpha=0.6)
    ax_overlay.set_title("Overlay")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-only", action="store_true", help="skip generation and only run the sum test")
    args = parser.parse_args()

    if not args.verify_only:
        precompute_all()
    passed = verify_all()

    if not args.verify_only:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        figure_path = RESULTS_DIR / "density_map_example.png"
        save_example_figure(figure_path)
        print(f"\nSaved figure : {figure_path.relative_to(PROJECT_ROOT)}")
        print(f"Density maps : {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
