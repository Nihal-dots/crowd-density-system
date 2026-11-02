"""Annotate a video with crowd density heatmaps and counts.

Writes an annotated .mp4, a CSV with the count for every processed frame, and
a few preview frames.

Usage (from the project root):
    python backend/video/process_video.py data/samples/market.mp4 --checkpoint path/to/best_model.pth
    python backend/video/process_video.py 0 --checkpoint best_model.pth --max-frames 100   # webcam
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.predictor import CrowdPredictor  # noqa: E402
from backend.video.frame_extractor import VideoReader  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "videos"
PREVIEW_DIR = PROJECT_ROOT / "research" / "results" / "video_previews"


def process_video(source, predictor, output_path, csv_path, step=1, max_frames=None,
                  alpha=0.4, preview_dir=None, num_previews=3, name=None):
    """Run the predictor over a video source. Returns a summary dict."""
    name = name or Path(str(source)).stem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoReader(source, step=step, max_frames=max_frames) as reader:
        expected = len(reader)
        # Spread previews over the video; for streams of unknown length keep the first frames
        preview_at = (
            set(np.linspace(0, expected - 1, min(num_previews, expected)).round().astype(int).tolist())
            if expected else set(range(num_previews))
        )
        # Keep playback at real speed when frames are skipped
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            max(reader.fps / step, 1.0), (reader.width, reader.height),
        )
        if not writer.isOpened():
            raise IOError(f"Could not open video writer: {output_path}")

        counts, latencies, previews = [], [], []
        try:
            with open(csv_path, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["frame", "time_s", "count", "latency_ms"])
                for processed, (frame_index, frame) in enumerate(reader):
                    count, density, latency_ms = predictor.predict_frame(frame)
                    annotated = predictor.render_heatmap_overlay(frame, density, alpha)
                    predictor.draw_hud(annotated, count, 1000.0 / latency_ms)
                    writer.write(annotated)
                    csv_writer.writerow([frame_index, f"{frame_index / reader.fps:.3f}", f"{count:.2f}", f"{latency_ms:.1f}"])
                    f.flush()

                    counts.append(count)
                    latencies.append(latency_ms)
                    if preview_dir is not None and processed in preview_at:
                        preview_dir.mkdir(parents=True, exist_ok=True)
                        preview_path = preview_dir / f"{name}_frame_{frame_index:06d}.jpg"
                        cv2.imwrite(str(preview_path), annotated)
                        previews.append(preview_path)
                    print(f"  frame {frame_index:>6} ({processed + 1}/{expected or '?'}): count {count:8.1f}, {latency_ms:7.0f} ms", flush=True)
        finally:
            writer.release()

    if not counts:
        raise RuntimeError(f"No frames could be read from {source}")
    return {
        "source": str(source),
        "resolution": f"{reader.width}x{reader.height}",
        "frames_processed": len(counts),
        "min_count": min(counts),
        "mean_count": sum(counts) / len(counts),
        "max_count": max(counts),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "output": output_path,
        "csv": csv_path,
        "previews": previews,
    }


def main():
    parser = argparse.ArgumentParser(description="Annotate a video with crowd density and counts")
    parser.add_argument("source", help="video file, webcam index or RTSP/HTTP URL")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained CSRNet checkpoint (best_model.pth)")
    parser.add_argument("--output", type=Path, help="annotated video (default: data/processed/videos/<name>_annotated.mp4)")
    parser.add_argument("--csv", type=Path, help="count per frame (default: next to the output video)")
    parser.add_argument("--step", type=int, default=1, help="process every N-th frame")
    parser.add_argument("--max-frames", type=int, help="stop after N processed frames")
    parser.add_argument("--max-side", type=int, help="downscale frames so the longer side is at most this before inference")
    parser.add_argument("--alpha", type=float, default=0.4, help="heatmap opacity")
    parser.add_argument("--previews", type=int, default=3, help="number of preview frames to save")
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    args = parser.parse_args()

    is_file = Path(args.source).is_file()
    name = Path(args.source).stem if is_file else "stream"
    output_path = args.output or OUTPUT_DIR / f"{name}_annotated.mp4"
    csv_path = args.csv or output_path.with_name(f"{name}_counts.csv")

    predictor = CrowdPredictor(args.checkpoint, max_side=args.max_side)
    print(f"Model: {args.checkpoint} (epoch {predictor.epoch}) on {predictor.device}")
    print(f"Processing {args.source}")
    summary = process_video(
        args.source, predictor, output_path, csv_path, step=args.step, max_frames=args.max_frames,
        alpha=args.alpha, preview_dir=args.preview_dir if args.previews > 0 else None,
        num_previews=args.previews, name=name,
    )

    print(f"\nFrames processed : {summary['frames_processed']} ({summary['resolution']})")
    print(f"Count            : min {summary['min_count']:.1f}, mean {summary['mean_count']:.1f}, max {summary['max_count']:.1f}")
    print(f"Mean latency     : {summary['mean_latency_ms']:.0f} ms ({1000 / summary['mean_latency_ms']:.2f} FPS)")
    print(f"Annotated video  : {summary['output']}")
    print(f"Counts CSV       : {summary['csv']}")
    for preview in summary["previews"]:
        print(f"Preview          : {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
