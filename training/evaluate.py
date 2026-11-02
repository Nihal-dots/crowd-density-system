"""Evaluate trained CSRNet models on the ShanghaiTech test splits.

Loads best_model.pth for each part, runs the full test split and writes to
research/results/:
    eval_results.json        metrics, per-image predictions, worst images
    eval_scatter.png         predicted vs true counts
    eval_examples_<part>.png 6 examples: image | ground truth | prediction
    eval_worst_<part>.png    the 5 images with the largest count error

Every number comes from running the model; nothing is hard-coded. A part
without a checkpoint is skipped and left out of the results.

Usage (from the project root):
    python training/evaluate.py                       # both parts
    python training/evaluate.py --part B
    python training/evaluate.py --ckpt-b path/to/best_model.pth
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; works on Colab and headless machines
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.csrnet import CSRNet  # noqa: E402
from training.dataset import ShanghaiTechDataset  # noqa: E402
from training.inspect_dataset import RESULTS_DIR  # noqa: E402
from training.train import default_checkpoint_dir  # noqa: E402

NUM_EXAMPLES = 6
NUM_WORST = 5


def load_model(checkpoint_path, part, device):
    """Return (model, checkpoint metadata) for a checkpoint written by train.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trained_on = checkpoint.get("part")
    if trained_on is not None and trained_on != part:
        print(f"WARNING: {checkpoint_path} was trained on Part {trained_on}, evaluating on Part {part}")
    model = CSRNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {"path": str(checkpoint_path), "epoch": checkpoint.get("epoch"), "trained_on_part": trained_on}


@torch.no_grad()
def predict_counts(model, dataset, device, limit=None):
    """Run the model on the test split and return one record per image."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    records = []
    for index, (image, _, count) in enumerate(loader):
        if limit is not None and index >= limit:
            break
        predicted = model(image.to(device)).sum().item()
        true = count.item()
        records.append({
            "index": index,
            "image": dataset.samples[index][0].name,
            "true_count": round(true, 3),
            "predicted_count": round(predicted, 3),
            "error": round(predicted - true, 3),
        })
    return records


def compute_metrics(records):
    errors = np.array([r["predicted_count"] - r["true_count"] for r in records])
    return {
        "num_images": len(records),
        "mae": float(np.abs(errors).mean()),
        "mse": float(math.sqrt((errors**2).mean())),  # RMSE, called MSE in crowd-counting papers
        "mean_count_error": float(errors.mean()),  # signed: positive means over-counting
    }


@torch.no_grad()
def save_density_figure(model, dataset, records, device, out_path, title):
    """One row per record: image | ground-truth density | predicted density."""
    fig, axes = plt.subplots(len(records), 3, figsize=(15, 3.6 * len(records)), squeeze=False)
    for row, record in zip(axes, records):
        image_tensor, target, _ = dataset[record["index"]]
        predicted = model(image_tensor.unsqueeze(0).to(device))[0, 0].cpu().numpy()
        with Image.open(dataset.samples[record["index"]][0]) as img:
            row[0].imshow(img.convert("RGB"))
        row[0].set_title(record["image"])
        row[1].imshow(target[0].numpy(), cmap="jet")
        row[1].set_title(f"Ground truth: {record['true_count']:.0f}")
        row[2].imshow(predicted, cmap="jet")
        row[2].set_title(f"Predicted: {record['predicted_count']:.1f} (error {record['error']:+.1f})")
        for ax in row:
            ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))  # leave room for the suptitle
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_scatter(per_part_records, metrics, out_path):
    fig, axes = plt.subplots(1, len(per_part_records), figsize=(6.5 * len(per_part_records), 6), squeeze=False)
    for ax, (part, records) in zip(axes[0], per_part_records.items()):
        true = [r["true_count"] for r in records]
        predicted = [r["predicted_count"] for r in records]
        upper = max(max(true), max(predicted)) * 1.05
        ax.plot([0, upper], [0, upper], color="gray", linestyle="--", linewidth=1, label="perfect prediction")
        ax.scatter(true, predicted, s=14, alpha=0.6)
        ax.set_xlim(0, upper)
        ax.set_ylim(min(0, min(predicted)), upper)
        ax.set_aspect("equal")
        ax.set_xlabel("true count")
        ax.set_ylabel("predicted count")
        ax.set_title(f"Part {part}: MAE {metrics[part]['mae']:.2f}, MSE {metrics[part]['mse']:.2f} ({len(records)} images)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def pick_examples(records, n=NUM_EXAMPLES):
    """Spread the examples evenly from the sparsest to the densest image."""
    by_count = sorted(records, key=lambda r: r["true_count"])
    positions = np.linspace(0, len(by_count) - 1, min(n, len(by_count))).round().astype(int)
    return [by_count[i] for i in dict.fromkeys(positions.tolist())]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CSRNet on the ShanghaiTech test splits")
    parser.add_argument("--part", choices=["A", "B"], help="evaluate one part only (default: both)")
    parser.add_argument("--ckpt-a", type=Path, help="Part A checkpoint (default: best_model.pth in the training checkpoint directory)")
    parser.add_argument("--ckpt-b", type=Path, help="Part B checkpoint (default: best_model.pth in the training checkpoint directory)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, help="debug: evaluate only the first N test images (results are marked partial)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {"A": args.ckpt_a, "B": args.ckpt_b}

    results = {"device": str(device), "partial": args.limit is not None, "parts": {}}
    per_part_records, metrics = {}, {}

    for part in [args.part] if args.part else ["A", "B"]:
        checkpoint_path = overrides[part] or default_checkpoint_dir(part) / "best_model.pth"
        if not checkpoint_path.is_file():
            print(f"Part {part}: no checkpoint at {checkpoint_path}, skipping.")
            continue

        model, checkpoint_info = load_model(checkpoint_path, part, device)
        dataset = ShanghaiTechDataset(part, "test")
        print(f"\n=== Part {part} ===")
        print(f"Checkpoint: {checkpoint_path} (epoch {checkpoint_info['epoch']})")
        records = predict_counts(model, dataset, device, args.limit)
        part_metrics = compute_metrics(records)
        worst = sorted(records, key=lambda r: abs(r["error"]), reverse=True)[:NUM_WORST]

        print(f"Images           : {part_metrics['num_images']}" + (f" of {len(dataset)} (--limit)" if args.limit else ""))
        print(f"MAE              : {part_metrics['mae']:.2f}")
        print(f"MSE              : {part_metrics['mse']:.2f}")
        print(f"Mean count error : {part_metrics['mean_count_error']:+.2f}")
        print(f"Worst {len(worst)} images:")
        for r in worst:
            print(f"    {r['image']:<12} true {r['true_count']:8.1f}  predicted {r['predicted_count']:8.1f}  error {r['error']:+9.1f}")

        examples_path = args.out_dir / f"eval_examples_{part}.png"
        worst_path = args.out_dir / f"eval_worst_{part}.png"
        save_density_figure(model, dataset, pick_examples(records), device, examples_path, f"Part {part} - examples")
        save_density_figure(model, dataset, worst, device, worst_path, f"Part {part} - largest count errors")

        per_part_records[part], metrics[part] = records, part_metrics
        results["parts"][part] = {
            "checkpoint": checkpoint_info,
            **part_metrics,
            "worst_images": worst,
            "per_image": records,
        }

    if not per_part_records:
        print("\nNo checkpoints found, nothing evaluated. Train first: python training/train.py --part B")
        return 1

    scatter_path = args.out_dir / "eval_scatter.png"
    save_scatter(per_part_records, metrics, scatter_path)
    results_path = args.out_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved results : {results_path}")
    print(f"Saved figures : {scatter_path.name}, " + ", ".join(f"eval_examples_{p}.png, eval_worst_{p}.png" for p in per_part_records))
    if args.limit:
        print("NOTE: --limit was used, so these are partial results, not test-set metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
