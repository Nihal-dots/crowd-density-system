"""Run the trained CSRNet on every annotated test split and cache the predictions.

Test splits: ShanghaiTech Part A test, Part B test and Roboflow test (Roboflow
valid was used to pick the best epoch, so test is the held-out split). Every
image is predicted at full resolution; nothing is hard-coded.

Writes research/results/eval_cache/<split>.json (per-image counts and latency)
and <split>_maps.npz (predicted density maps, float16) for
research/make_results_pdf.py. Splits already cached are skipped, so the
script can be rerun after an interruption.

Usage (from the project root):
    python research/evaluate_all.py --checkpoint experiments/checkpoints/all/best_model.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.csrnet import CSRNet  # noqa: E402
from training.dataset import RoboflowDataset, ShanghaiTechDataset  # noqa: E402
from training.inspect_dataset import RESULTS_DIR  # noqa: E402

CACHE_DIR = RESULTS_DIR / "eval_cache"
SPLITS = {
    "shanghaitech_A_test": lambda: ShanghaiTechDataset("A", "test"),
    "shanghaitech_B_test": lambda: ShanghaiTechDataset("B", "test"),
    "roboflow_test": lambda: RoboflowDataset("test"),
}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CSRNet(pretrained=False, backend_dilation=checkpoint.get("dilation", 2))
    model.load_state_dict(checkpoint["model"])
    info = {key: checkpoint.get(key) for key in ("epoch", "dataset", "mae", "mse")}
    info["path"] = str(checkpoint_path)
    return model.to(device).eval(), info


@torch.no_grad()
def evaluate_split(name, dataset, model, device):
    records, maps = [], {}
    for index in tqdm(range(len(dataset)), desc=name):
        image, _, _ = dataset[index]
        image_path, density_path = dataset.samples[index]
        with h5py.File(density_path, "r") as f:
            true_count = float(f["density"].attrs["count"])  # annotated heads

        start = time.perf_counter()
        density = model(image.unsqueeze(0).to(device)).clamp(min=0)[0, 0].cpu().numpy()
        latency_ms = (time.perf_counter() - start) * 1000.0

        predicted = float(density.sum(dtype=np.float64))
        records.append({
            "image": image_path.name,
            "height": int(image.shape[1]),
            "width": int(image.shape[2]),
            "true_count": true_count,
            "predicted_count": round(predicted, 3),
            "latency_ms": round(latency_ms, 1),
        })
        maps[image_path.stem] = density.astype(np.float16)
    return records, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "experiments" / "checkpoints" / "all" / "best_model.pth")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = load_model(args.checkpoint, device)
    info.update(device=str(device), threads=torch.get_num_threads(), torch_version=torch.__version__)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "checkpoint.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Model: {args.checkpoint} (epoch {info['epoch']}) on {device}, {info['threads']} threads")

    for name, make_dataset in SPLITS.items():
        out_path = CACHE_DIR / f"{name}.json"
        if out_path.is_file():
            print(f"{name}: cached, skipping")
            continue
        records, maps = evaluate_split(name, make_dataset(), model, device)
        np.savez_compressed(CACHE_DIR / f"{name}_maps.npz", **maps)
        out_path.write_text(json.dumps(records, indent=1), encoding="utf-8")
        errors = np.array([r["predicted_count"] - r["true_count"] for r in records])
        print(f"{name}: {len(records)} images, MAE {np.abs(errors).mean():.2f}, RMSE {np.sqrt((errors**2).mean()):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
