"""Read frames from a video file, webcam or network stream with OpenCV."""

from pathlib import Path

import cv2

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


class VideoReader:
    """Iterate over the frames of a video source.

    source     : video file path, webcam index (0 or "0"), or RTSP/HTTP URL
    step       : yield every step-th frame (1 = every frame)
    max_frames : stop after yielding this many frames (None = until the end)

    Yields (frame_index, frame_bgr), where frame_index is the position in the
    source. Use as a context manager, or call release() when done.
    """

    def __init__(self, source, step=1, max_frames=None):
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self.step = step
        self.max_frames = max_frames

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self.source, self.is_live = int(source), True
        elif str(source).lower().startswith(STREAM_PREFIXES):
            self.source, self.is_live = str(source), True
        else:
            if not Path(source).is_file():
                raise FileNotFoundError(f"Video file not found: {source}")
            self.source, self.is_live = str(source), False

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise IOError(f"Could not open video source: {source}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        # Webcams and some streams report 0 or nonsense; fall back to 25
        self.fps = fps if 1 <= fps <= 240 else 25.0
        # Unknown (0) for live sources
        self.frame_count = 0 if self.is_live else max(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

    def __len__(self):
        """Number of frames that will be yielded (0 if unknown, e.g. a stream)."""
        total = -(-self.frame_count // self.step)  # ceiling division
        if self.max_frames is not None:
            total = min(total, self.max_frames) if total else self.max_frames
        return total

    def __iter__(self):
        index, yielded = 0, 0
        while self.max_frames is None or yielded < self.max_frames:
            if index % self.step == 0:
                ok, frame = self.capture.read()
                if not ok:
                    break
                yield index, frame
                yielded += 1
            elif not self.capture.grab():  # skip without decoding
                break
            index += 1

    def release(self):
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
