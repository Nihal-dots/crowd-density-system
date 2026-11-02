"""Run CSRNet on video frames and draw the results."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.csrnet import CSRNet  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CrowdPredictor:
    def __init__(self, checkpoint_path, device=None, max_side=None):
        """checkpoint_path: a checkpoint from training/train.py or a plain state dict.

        max_side: if set, frames whose longer side exceeds it are downscaled
        before inference (faster on CPU; very small heads may be missed).
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_side = max_side

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.epoch = checkpoint.get("epoch") if "model" in checkpoint else None
        self.model = CSRNet(pretrained=False).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _preprocess(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        if self.max_side and max(height, width) > self.max_side:
            scale = self.max_side / max(height, width)
            frame_bgr = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_frame(self, frame_bgr):
        """Return (count, density_map, latency_ms) for one BGR frame.

        density_map is a float32 array at 1/8 of the inference resolution,
        clamped to be non-negative; count is its sum.
        """
        start = time.perf_counter()
        output = self.model(self._preprocess(frame_bgr))
        density = output.clamp(min=0)[0, 0].float().cpu().numpy()
        count = float(density.sum())
        latency_ms = (time.perf_counter() - start) * 1000.0
        return count, density, latency_ms

    @staticmethod
    def render_heatmap_overlay(frame, density, alpha=0.4):
        """Blend a JET heatmap of the density map onto the frame."""
        height, width = frame.shape[:2]
        heat = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR)
        peak = float(heat.max())
        heat = np.uint8(255 * heat / peak) if peak > 0 else np.zeros((height, width), np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0)

    @staticmethod
    def draw_hud(frame, count, fps):
        """Draw the crowd count and processing FPS in the top-left corner (in place)."""
        scale = max(frame.shape[0] / 720.0, 0.5)
        font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(int(2 * scale), 1)
        lines = [f"Count: {count:.0f}", f"FPS: {fps:.1f}"]
        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        pad = int(10 * scale)
        line_height = max(h for _, h in sizes) + pad
        box_w = max(w for w, _ in sizes) + 2 * pad
        box_h = line_height * len(lines) + pad

        cv2.rectangle(frame, (0, 0), (box_w, box_h), (0, 0, 0), thickness=-1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (pad, (i + 1) * line_height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame
