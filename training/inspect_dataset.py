"""Validate and summarise the ShanghaiTech crowd counting dataset.

Checks that every image has a ground-truth file, prints per-split statistics,
and writes a sample figure and a JSON summary to research/results/.

Usage (from the project root):
    python training/inspect_dataset.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; works on Colab and headless machines
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_NAME = "shanghaitech"
RESULTS_DIR = PROJECT_ROOT / "research" / "results"

# (part, split) -> expected number of images
EXPECTED_COUNTS = {
    ("part_A", "train_data"): 300,
    ("part_A", "test_data"): 182,
    ("part_B", "train_data"): 400,
    ("part_B", "test_data"): 316,
}
EXPECTED_TOTAL_HEADS = {"part_A": 241_677, "part_B": 88_488}

IMAGE_PATTERN = re.compile(r"^IMG_(\d+)\.jpg$", re.IGNORECASE)
GT_PATTERN = re.compile(r"^GT_IMG_(\d+)\.mat$", re.IGNORECASE)


def find_dataset_root() -> Path:
    """Locate the dataset folder ignoring case (Windows vs. Linux/Colab)."""
    if RAW_DIR.is_dir():
        for entry in RAW_DIR.iterdir():
            if entry.is_dir() and entry.name.lower() == DATASET_NAME:
                return entry
    raise FileNotFoundError(f"Dataset folder '{DATASET_NAME}' not found in {RAW_DIR}")


def load_mat_annotations(path):
    """Read a ShanghaiTech ground-truth file.

    Returns (points, count): points is an (N, 2) float array of head
    positions as [x, y] in pixels, count is the number of heads N.
    """
    mat = loadmat(str(path))
    points = np.asarray(mat["image_info"][0, 0][0, 0][0], dtype=np.float32)
    points = points.reshape(-1, 2)
    return points, int(points.shape[0])


def index_files(folder: Path, pattern: re.Pattern):
    """Map image number -> file for names matching pattern; also return the rest."""
    matched, other = {}, []
    if not folder.is_dir():
        return matched, other
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if m:
            matched[int(m.group(1))] = entry
        else:
            other.append(entry.name)
    return matched, other


def validate_split(split_dir: Path, expected: int):
    """Check image / ground-truth pairing for one split."""
    images, other_images = index_files(split_dir / "images", IMAGE_PATTERN)
    gts, other_gts = index_files(split_dir / "ground-truth", GT_PATTERN)

    missing_gt = [f"GT_IMG_{n}.mat" for n in sorted(set(images) - set(gts))]
    missing_image = [f"IMG_{n}.jpg" for n in sorted(set(gts) - set(images))]
    unexpected = [f"images/{n}" for n in other_images] + [f"ground-truth/{n}" for n in other_gts]

    report = {
        "num_images": len(images),
        "num_ground_truth": len(gts),
        "expected_images": expected,
        "missing_ground_truth": missing_gt,
        "missing_images": missing_image,
        "unexpected_files": unexpected,
    }
    report["ok"] = (
        len(images) == expected
        and len(gts) == expected
        and not missing_gt
        and not missing_image
        and not unexpected
    )
    pairs = [(images[n], gts[n]) for n in sorted(set(images) & set(gts))]
    return report, pairs


def compute_stats(pairs):
    """Head count and image size statistics for a list of (image, gt) pairs."""
    counts, widths, heights = [], [], []
    out_of_bounds = 0
    for image_path, gt_path in pairs:
        points, count = load_mat_annotations(gt_path)
        with Image.open(image_path) as img:
            width, height = img.size
        counts.append(count)
        widths.append(width)
        heights.append(height)
        if count:
            inside = (
                (points[:, 0] >= 0) & (points[:, 0] < width)
                & (points[:, 1] >= 0) & (points[:, 1] < height)
            )
            out_of_bounds += int((~inside).sum())

    counts = np.array(counts)
    return {
        "num_images": len(pairs),
        "min_count": int(counts.min()),
        "mean_count": round(float(counts.mean()), 2),
        "max_count": int(counts.max()),
        "total_heads": int(counts.sum()),
        "min_width": int(min(widths)),
        "max_width": int(max(widths)),
        "min_height": int(min(heights)),
        "max_height": int(max(heights)),
        "points_outside_image": out_of_bounds,
    }


def save_samples(samples, out_path: Path):
    """Plot one example per split with red dots on the annotated heads."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, image_path, gt_path) in zip(axes.ravel(), samples):
        points, count = load_mat_annotations(gt_path)
        with Image.open(image_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.scatter(points[:, 0], points[:, 1], s=6, c="red", linewidths=0)
        ax.set_title(f"{title} - {image_path.name} ({count} heads)")
        ax.axis("off")
    for ax in axes.ravel()[len(samples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    dataset_root = find_dataset_root()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {dataset_root}\n")

    stats = {"dataset_root": str(dataset_root.relative_to(PROJECT_ROOT).as_posix()), "splits": {}, "parts": {}}
    samples = []
    all_ok = True

    print("=== Validation ===")
    split_pairs = {}
    for (part, split), expected in EXPECTED_COUNTS.items():
        report, pairs = validate_split(dataset_root / part / split, expected)
        split_pairs[(part, split)] = pairs
        stats["splits"][f"{part}/{split}"] = {"validation": report}
        all_ok &= report["ok"]

        status = "OK" if report["ok"] else "PROBLEM"
        print(
            f"{part}/{split:<11} images: {report['num_images']:>3} "
            f"(expected {expected}), ground-truth: {report['num_ground_truth']:>3}  [{status}]"
        )
        for label, key in [
            ("missing ground-truth", "missing_ground_truth"),
            ("ground-truth without image", "missing_images"),
            ("unexpected file", "unexpected_files"),
        ]:
            for name in report[key]:
                print(f"    {label}: {name}")
    print("All images have a matching ground-truth file." if all_ok else "Validation found problems (see above).")

    print("\n=== Statistics ===")
    for (part, split), pairs in split_pairs.items():
        if not pairs:
            print(f"{part}/{split}: no image / ground-truth pairs found")
            continue
        s = compute_stats(pairs)
        stats["splits"][f"{part}/{split}"].update(s)
        print(f"{part}/{split}")
        print(f"    images      : {s['num_images']}")
        print(f"    head count  : min {s['min_count']}, mean {s['mean_count']:.2f}, max {s['max_count']}")
        print(f"    total heads : {s['total_heads']:,}")
        print(
            f"    image sizes : width {s['min_width']}-{s['max_width']}, "
            f"height {s['min_height']}-{s['max_height']}"
        )
        if s["points_outside_image"]:
            print(f"    note        : {s['points_outside_image']} head points lie outside the image bounds")
        samples.append((f"{part}/{split}", *pairs[0]))

    print("\n=== Totals per part ===")
    for part, expected_heads in EXPECTED_TOTAL_HEADS.items():
        part_splits = [v for k, v in stats["splits"].items() if k.startswith(part) and "total_heads" in v]
        total_heads = sum(v["total_heads"] for v in part_splits)
        total_images = sum(v["num_images"] for v in part_splits)
        stats["parts"][part] = {
            "num_images": total_images,
            "total_heads": total_heads,
            "expected_total_heads": expected_heads,
            "difference": total_heads - expected_heads,
        }
        print(
            f"{part}: {total_images} images, {total_heads:,} heads "
            f"(expected about {expected_heads:,}, difference {total_heads - expected_heads:+,})"
        )

    samples_path = RESULTS_DIR / "dataset_samples.png"
    save_samples(samples, samples_path)
    stats_path = RESULTS_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nSaved samples : {samples_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved stats   : {stats_path.relative_to(PROJECT_ROOT)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
