"""PyTorch Datasets for ShanghaiTech / Roboflow images and precomputed density maps.

Each item is (image, target, count):
    image  : float tensor (3, H, W), ImageNet-normalised
    target : float tensor (1, H/8, W/8), density map at the CSRNet output size
    count  : number of heads in the returned region (target.sum() matches it)

Density maps must exist first: run training/density_map.py (ShanghaiTech)
or training/prepare_roboflow.py (Roboflow).

Usage (from the project root):
    python training/dataset.py            # test both parts
    python training/dataset.py --part B
    python training/dataset.py --dataset roboflow
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.density_map import PROCESSED_DIR  # noqa: E402
from training.inspect_dataset import IMAGE_PATTERN, find_dataset_root, index_files  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DOWNSAMPLE = 8  # CSRNet output is 1/8 of the input resolution
PART_A_CROP = (256, 256)  # (height, width); Part A images vary in size


def downsample_density(density, factor=DOWNSAMPLE):
    """Resize a density map to 1/factor of its size while keeping its sum.

    cv2.resize averages pixels, which shrinks the sum by about factor**2
    (64x), so the result is multiplied by the area ratio to restore it.
    """
    height, width = density.shape
    out_h, out_w = height // factor, width // factor
    small = cv2.resize(density, (out_w, out_h), interpolation=cv2.INTER_AREA)
    small = small * ((height * width) / (out_h * out_w))

    before = float(density.sum(dtype=np.float64))
    after = float(small.sum(dtype=np.float64))
    assert abs(after - before) <= 1e-2 + 1e-4 * before, (
        f"count not preserved by downsampling: {before:.4f} -> {after:.4f}"
    )
    return small.astype(np.float32)


def pair_with_density_maps(image_paths, density_dir, prepare_command):
    """Return [(image_path, density_path), ...]; raise if any map is missing."""
    samples, missing = [], []
    for image_path in image_paths:
        density_path = density_dir / f"{image_path.stem}.h5"
        if density_path.is_file():
            samples.append((image_path, density_path))
        else:
            missing.append(density_path.name)
    if missing or not samples:
        raise FileNotFoundError(
            f"{len(missing)} density maps missing in {density_dir} "
            f"(found {len(samples)}). Run: {prepare_command}"
        )
    return samples


class DensityMapDataset(Dataset):
    """Images paired with .h5 density maps; subclasses fill self.samples."""

    def __init__(self, train, crop_size=None, flip_prob=0.5):
        """crop_size: (height, width) of the training crop; None crops 1/4 of the image."""
        self.train = train
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def _random_crop(self, image, density):
        height, width = density.shape
        crop_h, crop_w = self.crop_size if self.crop_size is not None else (height // 2, width // 2)

        # Pad images smaller than the crop so every crop has the same size
        pad_h, pad_w = max(crop_h - height, 0), max(crop_w - width, 0)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
            density = np.pad(density, ((0, pad_h), (0, pad_w)))
            height, width = density.shape

        top = int(torch.randint(0, height - crop_h + 1, (1,)))
        left = int(torch.randint(0, width - crop_w + 1, (1,)))
        return (
            image[top:top + crop_h, left:left + crop_w],
            density[top:top + crop_h, left:left + crop_w],
        )

    def __getitem__(self, index):
        image_path, density_path = self.samples[index]
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        with h5py.File(density_path, "r") as f:
            density = f["density"][:]
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        if self.train:
            # Same crop and flip for the image and its density map
            image, density = self._random_crop(image, density)
            if torch.rand(1).item() < self.flip_prob:
                image, density = image[:, ::-1], density[:, ::-1]

        density = np.ascontiguousarray(density)
        count = float(density.sum(dtype=np.float64))
        target = downsample_density(density)

        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        target = torch.from_numpy(target).unsqueeze(0)
        return image, target, torch.tensor(count, dtype=torch.float32)


class ShanghaiTechDataset(DensityMapDataset):
    def __init__(self, part="B", split="train", crop_size=None, flip_prob=0.5, density_root=None):
        """part: 'A' or 'B'. split: 'train' (augmented) or 'test' (full images).

        crop_size: (height, width) of the training crop. Defaults to 256x256
        for Part A and 1/4 of the image (half height, half width) for Part B.
        density_root: folder holding <part>/<split>/*.h5 (default: data/processed).
        """
        if part not in ("A", "B"):
            raise ValueError(f"part must be 'A' or 'B', got {part!r}")
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        super().__init__(
            train=split == "train",
            crop_size=crop_size if crop_size is not None else (PART_A_CROP if part == "A" else None),
            flip_prob=flip_prob,
        )
        self.part, self.split = part, split

        part_dir, split_dir = f"part_{part}", f"{split}_data"
        images, _ = index_files(find_dataset_root() / part_dir / split_dir / "images", IMAGE_PATTERN)
        density_dir = Path(density_root or PROCESSED_DIR) / part_dir / split_dir
        self.samples = pair_with_density_maps(
            [path for _, path in sorted(images.items())], density_dir, "python training/density_map.py",
        )


class RoboflowDataset(DensityMapDataset):
    def __init__(self, split="train", crop_size=None, flip_prob=0.5, density_root=None):
        """split: 'train' (augmented), 'valid' or 'test' (full 640x640 images).

        crop_size defaults to 1/4 of the image (320x320).
        density_root: folder holding <split>/*.h5 (default: data/processed/roboflow).
        """
        from training.prepare_roboflow import ROBOFLOW_DIR, ROBOFLOW_PROCESSED, ROBOFLOW_SPLITS

        if split not in ROBOFLOW_SPLITS:
            raise ValueError(f"split must be one of {ROBOFLOW_SPLITS}, got {split!r}")
        super().__init__(train=split == "train", crop_size=crop_size, flip_prob=flip_prob)
        self.split = split
        self.samples = pair_with_density_maps(
            sorted((ROBOFLOW_DIR / split).glob("*.jpg")),
            Path(density_root or ROBOFLOW_PROCESSED) / split,
            "python training/prepare_roboflow.py",
        )


class CrowdPseudoLabelDataset(DensityMapDataset):
    def __init__(self, crop_size=None, flip_prob=0.5):
        """The auto-labelled phone photos (training only); see training/pseudo_label_crowd.py."""
        from training.pseudo_label_crowd import CROWD_PROCESSED

        super().__init__(train=True, crop_size=crop_size, flip_prob=flip_prob)
        self.samples = pair_with_density_maps(
            sorted((CROWD_PROCESSED / "images").glob("*.jpg")), CROWD_PROCESSED,
            "python training/pseudo_label_crowd.py --checkpoint <part_A best_model.pth>",
        )


ALL_CROP = (256, 256)  # one crop size so images from every dataset fit in the same batch


def build_all_datasets(crop_size=None):
    """Train on ShanghaiTech A + B, Roboflow train and the pseudo-labelled photos (when present).

    Evaluate on ShanghaiTech A + B test and Roboflow valid, which all have real annotations.
    """
    from training.pseudo_label_crowd import CROWD_PROCESSED

    crop_size = crop_size or ALL_CROP
    train_sets = {
        "Part A": ShanghaiTechDataset("A", "train", crop_size=crop_size),
        "Part B": ShanghaiTechDataset("B", "train", crop_size=crop_size),
        "Roboflow": RoboflowDataset("train", crop_size=crop_size),
    }
    if any(CROWD_PROCESSED.glob("*.h5")):
        train_sets["crowd (pseudo)"] = CrowdPseudoLabelDataset(crop_size=crop_size)
    else:
        print("No pseudo-labelled crowd photos found; training without them.")
    eval_sets = {
        "Part A test": ShanghaiTechDataset("A", "test"),
        "Part B test": ShanghaiTechDataset("B", "test"),
        "Roboflow valid": RoboflowDataset("valid"),
    }

    for name, sets in (("train", train_sets), ("eval", eval_sets)):
        sizes = ", ".join(f"{key}: {len(d)}" for key, d in sets.items())
        print(f"All datasets {name}: {sum(map(len, sets.values()))} images ({sizes})")
    return ConcatDataset(list(train_sets.values())), ConcatDataset(list(eval_sets.values()))


def build_datasets(dataset="shanghaitech", part="B", crop_size=None, density_root=None, eval_split=None):
    """Return (train_set, eval_set). eval_split defaults to 'test' (ShanghaiTech) or 'valid' (Roboflow)."""
    if dataset == "all":
        return build_all_datasets(crop_size)
    if dataset == "roboflow":
        return (
            RoboflowDataset("train", crop_size=crop_size, density_root=density_root),
            RoboflowDataset(eval_split or "valid", density_root=density_root),
        )
    return (
        ShanghaiTechDataset(part, "train", crop_size=crop_size, density_root=density_root),
        ShanghaiTechDataset(part, eval_split or "test", density_root=density_root),
    )


def build_loaders(part="B", batch_size=8, test_batch_size=1, num_workers=0, crop_size=None, density_root=None,
                  dataset="shanghaitech", eval_split=None):
    """Return (train_loader, test_loader) for a dataset ('shanghaitech' part A/B, 'roboflow' or 'all').

    Test images are loaded at full size; Part A test images differ in size,
    so keep test_batch_size=1 there.
    """
    train_set, test_set = build_datasets(dataset, part, crop_size, density_root, eval_split)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def self_test(part, batch_size=4, dataset="shanghaitech"):
    """Load one batch per split, print shapes and check density sums."""
    torch.manual_seed(0)
    train_loader, test_loader = build_loaders(part=part, batch_size=batch_size, dataset=dataset)
    name = "Roboflow" if dataset == "roboflow" else f"Part {part}"
    print(f"=== {name}: {len(train_loader.dataset)} train / {len(test_loader.dataset)} eval images ===")
    ok = True

    for name, loader in (("train", train_loader), ("test", test_loader)):
        images, targets, counts = next(iter(loader))
        print(f"{name} batch: images {tuple(images.shape)}, density {tuple(targets.shape)}, counts {tuple(counts.shape)}")
        shape_ok = (
            targets.shape[2] == images.shape[2] // DOWNSAMPLE
            and targets.shape[3] == images.shape[3] // DOWNSAMPLE
        )
        ok &= shape_ok
        print(f"  [{'PASS' if shape_ok else 'FAIL'}] density is 1/{DOWNSAMPLE} of the image size")

        sums = targets.sum(dim=(1, 2, 3), dtype=torch.float64)
        for i, (density_sum, count) in enumerate(zip(sums.tolist(), counts.tolist())):
            close = abs(density_sum - count) <= 1e-2 + 1e-4 * count
            ok &= close
            print(f"  [{'PASS' if close else 'FAIL'}] sample {i}: density sum {density_sum:9.3f}, head count {count:9.3f}")

        if name == "test":
            # Full images: the sum must also match the annotated head count
            _, density_path = loader.dataset.samples[0]
            with h5py.File(density_path, "r") as f:
                annotated = int(f["density"].attrs["count"])
            close = abs(sums[0].item() - annotated) <= 1e-2 + 1e-4 * annotated
            ok &= close
            print(f"  [{'PASS' if close else 'FAIL'}] {density_path.stem}: annotated heads {annotated}, density sum {sums[0].item():.3f}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset self-test")
    parser.add_argument("--part", choices=["A", "B"], help="test one part only (default: both)")
    parser.add_argument("--dataset", choices=["shanghaitech", "roboflow"], default="shanghaitech")
    args = parser.parse_args()

    if args.dataset == "roboflow":
        results = [self_test(None, dataset="roboflow")]
    else:
        results = [self_test(part) for part in ([args.part] if args.part else ["A", "B"])]
    print("PASSED" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)
