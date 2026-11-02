"""Train CSRNet on ShanghaiTech Part A / Part B or the Roboflow crowd dataset.

Checkpoints (best_model.pth, last.pth and one every --save-every epochs) go to
Google Drive when it is mounted on Colab, so a disconnect does not lose
progress; rerun the same command with --resume to continue.

Usage (from the project root):
    python training/train.py --part B --batch 8
    python training/train.py --part B --batch 8 --resume
    python training/train.py --part B --epochs 2 --limit 4   # quick smoke test

    # Roboflow: run training/prepare_roboflow.py first; validates on the "valid" split
    python training/train.py --dataset roboflow --epochs 50 --init-from <part_A best_model.pth> --resume

    # All three datasets in one model (run training/pseudo_label_crowd.py first to include the crowd photos)
    python training/train.py --dataset all --epochs 30 --init-from <part_A best_model.pth> --resume
"""

import argparse
import csv
import itertools
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; works on Colab and headless machines
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.csrnet import CSRNet  # noqa: E402
from training.dataset import build_loaders  # noqa: E402
from training.inspect_dataset import RESULTS_DIR  # noqa: E402

COLAB_DRIVE = Path("/content/drive/MyDrive")
LOG_FIELDS = ["epoch", "train_loss", "mae", "mse", "time_s"]


def default_checkpoint_dir(part, dataset="shanghaitech"):
    """Google Drive on Colab (survives disconnects), experiments/ elsewhere."""
    folder = f"part_{part}" if dataset == "shanghaitech" else dataset
    if COLAB_DRIVE.is_dir():
        return COLAB_DRIVE / "crowd-density-system" / "checkpoints" / folder
    if Path("/content").is_dir():
        print("WARNING: Google Drive is not mounted; checkpoints will be lost when the runtime disconnects.")
    return PROJECT_ROOT / "experiments" / "checkpoints" / folder


def save_checkpoint(state, path):
    """Write to a temporary file first so an interrupted save cannot corrupt path."""
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, limit=None):
    """Return the mean training loss per image."""
    model.train()
    total_loss, num_images = 0.0, 0
    for images, targets, _ in itertools.islice(loader, limit):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        # Density values are tiny, so compute the loss in full precision
        loss = criterion(outputs.float(), targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_images += images.size(0)
    return total_loss / max(num_images, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp, limit=None):
    """Return (MAE, MSE) of the predicted counts; MSE is the root mean squared error."""
    model.eval()
    abs_error, squared_error, num_images = 0.0, 0.0, 0
    for images, _, counts in itertools.islice(loader, limit):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        predicted = outputs.float().sum(dim=(1, 2, 3)).cpu()
        diff = predicted - counts
        abs_error += diff.abs().sum().item()
        squared_error += (diff**2).sum().item()
        num_images += images.size(0)
    return abs_error / num_images, math.sqrt(squared_error / num_images)


def write_log(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history, path, name):
    """name labels the plots, e.g. 'Part B' or 'Roboflow'."""
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, [row["train_loss"] for row in history])
    ax_loss.set_title(f"{name} - training loss (per image)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("sum-reduced MSE loss")
    ax_loss.set_yscale("log")

    maes = [row["mae"] for row in history]
    best = min(range(len(maes)), key=maes.__getitem__)
    ax_mae.plot(epochs, maes)
    ax_mae.scatter([epochs[best]], [maes[best]], color="red", zorder=3, label=f"best {maes[best]:.2f} (epoch {epochs[best]})")
    ax_mae.set_title(f"{name} - eval MAE")
    ax_mae.set_xlabel("epoch")
    ax_mae.set_ylabel("MAE (heads)")
    ax_mae.legend()

    for ax in (ax_loss, ax_mae):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CSRNet on ShanghaiTech or the Roboflow crowd dataset")
    parser.add_argument("--dataset", choices=["shanghaitech", "roboflow", "all"], default="shanghaitech",
                        help="'all' = ShanghaiTech A + B, Roboflow and the pseudo-labelled crowd photos")
    parser.add_argument("--part", choices=["A", "B"], default="B", help="ShanghaiTech part (shanghaitech only)")
    parser.add_argument("--init-from", type=Path,
                        help="fine-tune: start from the weights in this checkpoint (e.g. a ShanghaiTech best_model.pth)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8, choices=range(1, 9), metavar="1-8")
    parser.add_argument("--resume", action="store_true", help="continue from last.pth in the checkpoint directory")
    parser.add_argument("--ckpt-dir", type=Path, help="checkpoint directory (default: Google Drive on Colab)")
    parser.add_argument("--save-every", type=int, default=10, help="keep a numbered checkpoint every N epochs")
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--limit", type=int, help="debug: use only N train batches and N test images per epoch")
    # Variants for ablation studies (see experiments/run_benchmarks.py)
    parser.add_argument("--dilation", type=int, default=2, help="back-end dilation rate (1 = plain convolutions)")
    parser.add_argument("--density-root", type=Path, help="folder with the density maps (default: data/processed)")
    parser.add_argument("--tag", help="name of a variant run; keeps its checkpoints, log and curves separate")
    return parser.parse_args(argv)


def run_training(args):
    """Train with the given arguments. Returns the best MAE/MSE and where things were saved."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    run_name = args.part if args.dataset == "shanghaitech" else args.dataset
    label = {"roboflow": "Roboflow", "all": "All datasets"}.get(args.dataset, f"Part {args.part}")
    suffix = f"{run_name}_{args.tag}" if args.tag else run_name

    ckpt_dir = args.ckpt_dir or default_checkpoint_dir(args.part, args.dataset) / (args.tag or "")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pth", ckpt_dir / "best_model.pth"
    log_path = RESULTS_DIR / f"train_log_{suffix}.csv"
    curves_path = RESULTS_DIR / f"training_curves_{suffix}.png"

    resuming = args.resume and last_path.is_file()
    if args.resume and not resuming:
        print(f"No checkpoint at {last_path}; starting from scratch.")

    train_loader, test_loader = build_loaders(
        part=args.part, batch_size=args.batch, num_workers=args.workers, density_root=args.density_root,
        dataset=args.dataset,
    )
    # When resuming or fine-tuning, a checkpoint supplies all weights, so skip the VGG download
    model = CSRNet(pretrained=not (resuming or args.init_from), backend_dilation=args.dilation).to(device)
    if args.init_from and not resuming:
        init = torch.load(args.init_from, map_location=device)
        model.load_state_dict(init["model"] if "model" in init else init)
        print(f"Fine-tuning from {args.init_from}")
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_mae, history = 1, float("inf"), []
    if resuming:
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_mae"]
        history = checkpoint["history"]
        print(f"Resumed from {last_path} (epoch {checkpoint['epoch']}, best MAE {best_mae:.2f})")

    print(
        f"{label} | device {device} | AMP {'on' if use_amp else 'off'} | "
        f"batch {args.batch} | lr {args.lr} | epochs {start_epoch}-{args.epochs}"
    )
    print(f"Checkpoints: {ckpt_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, args.limit)
        mae, mse = validate(model, test_loader, device, use_amp, args.limit)
        elapsed = time.time() - start

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "time_s": round(elapsed, 1),
        })
        is_best = mae < best_mae
        best_mae = min(mae, best_mae)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mae": best_mae,
            "mae": mae,
            "mse": mse,
            "history": history,
            "part": args.part,
            "dataset": args.dataset,
            "dilation": args.dilation,
        }
        save_checkpoint(state, last_path)
        if is_best:
            save_checkpoint(state, best_path)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(state, ckpt_dir / f"epoch_{epoch:03d}.pth")

        write_log(history, log_path)
        plot_curves(history, curves_path, label)

        print(
            f"Epoch {epoch:>3}/{args.epochs} | loss {train_loss:10.4f} | MAE {mae:7.2f} | "
            f"MSE {mse:7.2f} | best MAE {best_mae:7.2f} | {elapsed:6.1f}s{' *' if is_best else ''}"
        )

    print(f"\nBest MAE: {best_mae:.2f}")
    print(f"Log    : {log_path}")
    print(f"Curves : {curves_path}")
    best_row = min(history, key=lambda row: row["mae"]) if history else None
    return {
        "best_mae": best_row["mae"] if best_row else None,
        "best_mse": best_row["mse"] if best_row else None,
        "best_epoch": best_row["epoch"] if best_row else None,
        "epochs_completed": len(history),
        "checkpoint": str(best_path),
        "log": str(log_path),
    }


def main():
    run_training(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
