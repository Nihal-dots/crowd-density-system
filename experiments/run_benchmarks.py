"""Benchmarks and ablations for the paper.

Experiments (choose with --experiments, default: all):
    latency   inference latency / FPS at 480p, 720p, 1080p on CPU and GPU
    dilation  dilated back-end vs dilation=1, both trained on Part B
    kernels   geometry-adaptive vs fixed Gaussian kernels, both trained on Part A
    alerts    WARNING/CRITICAL events on the sample videos with and without hysteresis

Results are merged into research/results/benchmark.json, so experiments can be
run separately (e.g. CPU latency on a laptop, GPU latency and the training
ablations on Colab). One table and one chart per experiment are written as
PNG files next to it. Every number is measured by this script.

Usage (from the project root):
    python experiments/run_benchmarks.py --experiments latency
    python experiments/run_benchmarks.py --experiments dilation,kernels --ablation-epochs 100
    python experiments/run_benchmarks.py --experiments alerts --checkpoint path/to/best_model.pth
"""

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; works on Colab and headless machines
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.alerts.alert_engine import AlertEngine  # noqa: E402
from backend.alerts.zone_manager import CRITICAL, NORMAL, WARNING, ZoneManager  # noqa: E402
from backend.inference.predictor import CrowdPredictor  # noqa: E402
from backend.video.frame_extractor import VideoReader  # noqa: E402
from model.csrnet import CSRNet  # noqa: E402
from training import train as train_module  # noqa: E402
from training.density_map import PROCESSED_DIR, precompute_all  # noqa: E402
from training.inspect_dataset import EXPECTED_COUNTS, RESULTS_DIR  # noqa: E402

EXPERIMENTS = ["latency", "dilation", "kernels", "alerts"]
RESOLUTIONS = {"480p": (480, 854), "720p": (720, 1280), "1080p": (1080, 1920)}  # (height, width)
FIXED_KERNEL_ROOT = PROCESSED_DIR / "fixed_kernel"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

SERIES_COLORS = ["#2a78d6", "#eb6834"]
TEXT_COLOR, GRID_COLOR = "#0b0b0b", "#d9d8d4"


# ---------------------------------------------------------------- output helpers

def load_results(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def save_table(columns, rows, title, path):
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(columns)), 0.9 + 0.42 * (len(rows) + 1)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    table.auto_set_column_width(list(range(len(columns))))
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0efec")
    ax.set_title(title, fontsize=11, color=TEXT_COLOR, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_grouped_bars(groups, series, ylabel, title, path, errors=None, value_format="{:.1f}"):
    """Bar chart with one cluster per group and one bar per series (None values are skipped)."""
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(groups)), 4.5))
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series.items()):
        positions = np.arange(len(groups)) + (index - (len(series) - 1) / 2) * width
        present = [i for i, value in enumerate(values) if value is not None]
        heights = [values[i] for i in present]
        yerr = [errors[name][i] for i in present] if errors else None
        bars = ax.bar(
            positions[present], heights, width * 0.92, label=name, color=SERIES_COLORS[index % len(SERIES_COLORS)],
            yerr=yerr, capsize=3, error_kw={"ecolor": TEXT_COLOR, "elinewidth": 1},
        )
        ax.bar_label(bars, labels=[value_format.format(h) for h in heights], padding=3, fontsize=9, color=TEXT_COLOR)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if len(series) > 1:
        # Below the plot so it never covers a bar or its label
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(series))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 1. latency

def measure_latency(device, warmup, runs):
    model = CSRNet(pretrained=False).to(device).eval()  # weights do not affect speed
    results = {}
    for name, (height, width) in RESOLUTIONS.items():
        frame = torch.randn(1, 3, height, width, device=device)
        timings = []
        with torch.no_grad():
            for run in range(warmup + runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(frame)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if run >= warmup:
                    timings.append((time.perf_counter() - start) * 1000.0)
        mean, std = float(np.mean(timings)), float(np.std(timings))
        results[name] = {
            "width": width, "height": height, "mean_ms": mean, "std_ms": std,
            "fps": 1000.0 / mean, "fps_std": 1000.0 * std / mean**2,
        }
        print(f"  {device.type:<4} {name:<5} {mean:9.1f} ± {std:7.1f} ms   {1000.0 / mean:7.2f} FPS", flush=True)
    return results


def run_latency(args, results):
    print("\n=== 1. Latency / FPS ===")
    experiment = results.setdefault("latency", {"devices": {}})
    experiment["settings"] = {"warmup_runs": args.warmup, "timed_runs": args.runs, "batch_size": 1, "precision": "float32"}

    experiment["devices"]["cpu"] = {
        "hardware": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "results": measure_latency(torch.device("cpu"), args.warmup, args.runs),
    }
    if torch.cuda.is_available():
        experiment["devices"]["cuda"] = {
            "hardware": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "results": measure_latency(torch.device("cuda"), args.warmup, args.runs),
        }
    else:
        print("  cuda: not available on this machine (run again on a GPU machine to add it)")


def report_latency(results, out_dir):
    devices = results.get("latency", {}).get("devices", {})
    if not devices:
        return
    labels = {"cpu": "CPU", "cuda": "GPU"}
    rows = [
        [labels.get(device, device), info["hardware"], name, f"{r['width']}x{r['height']}",
         f"{r['mean_ms']:.1f} ± {r['std_ms']:.1f}", f"{r['fps']:.2f} ± {r['fps_std']:.2f}"]
        for device, info in devices.items() for name, r in info["results"].items()
    ]
    save_table(["Device", "Hardware", "Resolution", "Size", "Latency (ms)", "FPS"], rows,
               "CSRNet inference latency (mean ± std)", out_dir / "benchmark_latency_table.png")
    groups = list(RESOLUTIONS)
    save_grouped_bars(
        groups,
        {labels.get(d, d): [info["results"].get(g, {}).get("fps") for g in groups] for d, info in devices.items()},
        "frames per second", "CSRNet inference speed by resolution", out_dir / "benchmark_latency_chart.png",
        errors={labels.get(d, d): [info["results"].get(g, {}).get("fps_std", 0) for g in groups] for d, info in devices.items()},
        value_format="{:.2f}",
    )


# ---------------------------------------------------------------- 2 & 3. training ablations

def train_variant(part, tag, args, extra):
    """Train one ablation variant with train.py (resumes if interrupted) and return its metrics."""
    argv = [
        "--part", part, "--epochs", str(args.ablation_epochs), "--batch", str(args.batch),
        "--lr", str(args.lr), "--tag", tag, "--resume",
    ] + extra
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.ckpt_root:
        argv += ["--ckpt-dir", str(args.ckpt_root / f"part_{part}" / tag)]
    print(f"\n--- training Part {part} variant '{tag}' ---")
    summary = train_module.run_training(train_module.parse_args(argv))
    return {"mae": summary["best_mae"], "mse": summary["best_mse"], "best_epoch": summary["best_epoch"],
            "epochs_completed": summary["epochs_completed"], "checkpoint": summary["checkpoint"]}


def ablation_settings(args, part):
    return {"part": part, "epochs": args.ablation_epochs, "batch_size": args.batch, "lr": args.lr,
            "optimizer": "Adam", "weight_decay": 1e-4, "debug_limit": args.limit,
            "metric": "best test MAE over the run, with the MSE of that epoch"}


def run_dilation(args, results):
    print("\n=== 2. Ablation: dilated vs plain back-end (Part B) ===")
    results["dilation_ablation"] = {
        "settings": ablation_settings(args, "B"),
        "variants": {
            "Dilated back-end (rate 2)": train_variant("B", "dilation2", args, ["--dilation", "2"]),
            "Plain back-end (rate 1)": train_variant("B", "dilation1", args, ["--dilation", "1"]),
        },
    }


def run_kernels(args, results):
    print("\n=== 3. Ablation: adaptive vs fixed Gaussian kernels (Part A) ===")
    fixed_dir = FIXED_KERNEL_ROOT / "part_A"
    expected = sum(count for (part, _), count in EXPECTED_COUNTS.items() if part == "part_A")
    if len(list(fixed_dir.rglob("*.h5"))) < expected:
        print("Generating fixed-kernel density maps for Part A...")
        precompute_all(parts=["part_A"], kernel="fixed", out_root=FIXED_KERNEL_ROOT)
    results["kernel_ablation"] = {
        "settings": ablation_settings(args, "A"),
        "variants": {
            "Geometry-adaptive kernels": train_variant("A", "kernel_adaptive", args, []),
            "Fixed kernel (sigma 15)": train_variant("A", "kernel_fixed", args, ["--density-root", str(FIXED_KERNEL_ROOT)]),
        },
    }


def report_ablation(results, key, title, out_dir, stem):
    experiment = results.get(key)
    if not experiment:
        return
    variants, settings = experiment["variants"], experiment["settings"]
    subtitle = f"Part {settings['part']}, {settings['epochs']} epochs, batch {settings['batch_size']}"
    if settings.get("debug_limit"):
        subtitle += f" - DEBUG RUN (--limit {settings['debug_limit']}), not a real result"
    rows = [[name, f"{v['mae']:.2f}", f"{v['mse']:.2f}", str(v["best_epoch"])] for name, v in variants.items()]
    save_table(["Variant", "MAE", "MSE", "Best epoch"], rows, f"{title}\n{subtitle}", out_dir / f"{stem}_table.png")
    save_grouped_bars(
        ["MAE", "MSE"], {name: [v["mae"], v["mse"]] for name, v in variants.items()},
        "count error (lower is better)", f"{title}\n{subtitle}", out_dir / f"{stem}_chart.png",
    )


# ---------------------------------------------------------------- 4. alerts

def find_checkpoint():
    for root in (PROJECT_ROOT / "experiments" / "checkpoints", Path.home() / "crowd-density-checkpoints",
                 Path("/content/drive/MyDrive/crowd-density-system/checkpoints")):
        for part in ("part_B", "part_A"):
            if (root / part / "best_model.pth").is_file():
                return root / part / "best_model.pth"
    return None


def count_events_without_hysteresis(statuses):
    """An event every time the instantaneous status changes to WARNING or CRITICAL."""
    events, previous = {WARNING: 0, CRITICAL: 0}, NORMAL
    for status in statuses:
        if status != previous and status != NORMAL:
            events[status] += 1
        previous = status
    return events


def run_alerts(args, results):
    print("\n=== 4. Alert behaviour with and without hysteresis ===")
    checkpoint = args.checkpoint or find_checkpoint()
    videos = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)[:3]
    if checkpoint is None or not Path(checkpoint).is_file():
        print("  skipped: no trained checkpoint found (pass --checkpoint)")
        return
    if not videos:
        print(f"  skipped: no videos in {SAMPLES_DIR}")
        return

    predictor = CrowdPredictor(checkpoint, max_side=args.alert_max_side)
    base_zones = ZoneManager.from_json().zones
    experiment = {
        "settings": {
            "checkpoint": str(checkpoint), "checkpoint_epoch": predictor.epoch, "frame_step": args.alert_step,
            "max_side": args.alert_max_side, "consecutive_frames": args.consecutive_frames, "clear_ratio": 0.7,
            "thresholds": args.alert_thresholds,
            "threshold_note": (
                "each zone's threshold is the mean of its own count over the video, so the count crosses it"
                if args.alert_thresholds == "relative" else "thresholds taken from backend/alerts/zones.json"
            ),
            "without_hysteresis": "one event each time the per-frame status changes to WARNING or CRITICAL",
        },
        "videos": {},
    }

    for video in videos:
        with VideoReader(video, step=args.alert_step) as reader:
            densities = []
            for _, frame in reader:
                densities.append(predictor.predict_frame(frame)[1])
                print(f"\r  {video.name}: {len(densities)}/{len(reader)} frames", end="", flush=True)
        print()
        if not densities:
            print(f"  {video.name}: no frames could be read, skipped")
            continue

        counts = np.array([list(ZoneManager(base_zones).zone_counts(d).values()) for d in densities])
        zones = [
            dataclasses.replace(zone, threshold=float(max(counts[:, i].mean(), 1e-6)))
            if args.alert_thresholds == "relative" else zone
            for i, zone in enumerate(base_zones)
        ]
        manager = ZoneManager(zones)

        engine = AlertEngine(manager, consecutive_frames=args.consecutive_frames, clear_ratio=0.7)
        with_hysteresis = {WARNING: 0, CRITICAL: 0}
        for density in densities:
            for alert in engine.update(density)[1]:
                if alert.alert_level in with_hysteresis:
                    with_hysteresis[alert.alert_level] += 1

        without_hysteresis = {WARNING: 0, CRITICAL: 0}
        for i, zone in enumerate(zones):
            events = count_events_without_hysteresis([zone.status(count) for count in counts[:, i]])
            for level in events:
                without_hysteresis[level] += events[level]

        experiment["videos"][video.name] = {
            "frames_processed": len(densities),
            "total_count": {"min": float(counts.sum(1).min()), "mean": float(counts.sum(1).mean()), "max": float(counts.sum(1).max())},
            "zone_thresholds": {zone.id: zone.threshold for zone in zones},
            "with_hysteresis": with_hysteresis,
            "without_hysteresis": without_hysteresis,
        }
        print(f"  {video.name}: with hysteresis {with_hysteresis}, without {without_hysteresis}")
    results["alerts"] = experiment


def report_alerts(results, out_dir):
    experiment = results.get("alerts")
    if not experiment or not experiment["videos"]:
        return
    videos = experiment["videos"]
    rows = [
        [name, str(v["frames_processed"]),
         str(v["without_hysteresis"][WARNING]), str(v["without_hysteresis"][CRITICAL]),
         str(v["with_hysteresis"][WARNING]), str(v["with_hysteresis"][CRITICAL])]
        for name, v in videos.items()
    ]
    save_table(
        ["Video", "Frames", "WARNING\n(no hysteresis)", "CRITICAL\n(no hysteresis)", "WARNING\n(hysteresis)", "CRITICAL\n(hysteresis)"],
        rows, "Alert events with and without hysteresis", out_dir / "benchmark_alerts_table.png",
    )
    save_grouped_bars(
        [Path(name).stem for name in videos],
        {
            "Without hysteresis": [sum(v["without_hysteresis"].values()) for v in videos.values()],
            "With hysteresis": [sum(v["with_hysteresis"].values()) for v in videos.values()],
        },
        "WARNING + CRITICAL events", "Alert events with and without hysteresis",
        out_dir / "benchmark_alerts_chart.png", value_format="{:.0f}",
    )


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks and ablations")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS), help=f"comma-separated subset of: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--warmup", type=int, default=3, help="latency: untimed warm-up runs")
    parser.add_argument("--runs", type=int, default=20, help="latency: timed runs")
    parser.add_argument("--ablation-epochs", type=int, default=100, help="epochs for each ablation variant")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ckpt-root", type=Path, help="ablation checkpoints (default: as training/train.py)")
    parser.add_argument("--limit", type=int, help="debug: N train batches and N test images per epoch in the ablations")
    parser.add_argument("--checkpoint", type=Path, help="alerts: trained model (default: best_model.pth from training)")
    parser.add_argument("--alert-step", type=int, default=10, help="alerts: process every N-th frame")
    parser.add_argument("--alert-max-side", type=int, default=640, help="alerts: inference size (longer side)")
    parser.add_argument("--alert-thresholds", choices=["relative", "config"], default="relative")
    parser.add_argument("--consecutive-frames", type=int, default=3, help="alerts: hysteresis length N")
    args = parser.parse_args()

    selected = [name.strip() for name in args.experiments.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        parser.error(f"unknown experiments: {', '.join(unknown)}")

    results_path = args.out_dir / "benchmark.json"
    results = load_results(results_path)
    runners = {"latency": run_latency, "dilation": run_dilation, "kernels": run_kernels, "alerts": run_alerts}
    for name in selected:
        runners[name](args, results)
        save_results(results, results_path)  # keep finished experiments if a later one is interrupted

    report_latency(results, args.out_dir)
    report_ablation(results, "dilation_ablation", "Ablation: back-end dilation", args.out_dir, "benchmark_dilation")
    report_ablation(results, "kernel_ablation", "Ablation: density map kernel", args.out_dir, "benchmark_kernels")
    report_alerts(results, args.out_dir)

    print(f"\nResults : {results_path}")
    for figure in sorted(args.out_dir.glob("benchmark_*.png")):
        print(f"Figure  : {figure.name}")
    missing = [name for name, key in zip(EXPERIMENTS, ["latency", "dilation_ablation", "kernel_ablation", "alerts"]) if key not in results]
    if missing:
        print(f"Not yet run: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
