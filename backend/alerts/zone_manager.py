"""Monitoring zones: regions of the frame with their own crowd threshold.

Zones use normalised coordinates (0-1, origin top-left), so the same config
works for any frame or density map resolution.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ZONES_PATH = Path(__file__).resolve().parent / "zones.json"

NORMAL, WARNING, CRITICAL = "NORMAL", "WARNING", "CRITICAL"


@dataclass(frozen=True)
class Zone:
    """A rectangle ([x1, y1, x2, y2]) or polygon ([[x, y], ...]) with a count threshold."""

    id: str
    name: str
    type: str
    coords: tuple
    threshold: float
    warn_ratio: float = 0.8

    def __post_init__(self):
        if self.type not in ("rectangle", "polygon"):
            raise ValueError(f"zone {self.id!r}: type must be 'rectangle' or 'polygon', got {self.type!r}")
        if self.threshold <= 0:
            raise ValueError(f"zone {self.id!r}: threshold must be positive")
        if not 0 < self.warn_ratio <= 1:
            raise ValueError(f"zone {self.id!r}: warn_ratio must be in (0, 1]")

        values = np.asarray(self.coords, dtype=float)
        if self.type == "rectangle":
            if values.shape != (4,):
                raise ValueError(f"zone {self.id!r}: rectangle coords must be [x1, y1, x2, y2]")
            x1, y1, x2, y2 = values
            if not (x1 < x2 and y1 < y2):
                raise ValueError(f"zone {self.id!r}: rectangle needs x1 < x2 and y1 < y2")
        elif values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
            raise ValueError(f"zone {self.id!r}: polygon coords must be at least 3 [x, y] points")
        if values.min() < 0 or values.max() > 1:
            raise ValueError(f"zone {self.id!r}: coords must be normalised to the range 0-1")

    def status(self, count):
        """NORMAL below warn_ratio * T, WARNING up to T, CRITICAL at T or above."""
        if count >= self.threshold:
            return CRITICAL
        if count >= self.warn_ratio * self.threshold:
            return WARNING
        return NORMAL

    def mask(self, height, width):
        """Boolean mask of the pixels whose centre lies inside the zone.

        Pixel centres are tested against half-open boundaries, so zones that
        share an edge never claim the same pixel.
        """
        xs = (np.arange(width) + 0.5) / width
        ys = (np.arange(height) + 0.5) / height
        grid_x, grid_y = np.meshgrid(xs, ys)

        if self.type == "rectangle":
            x1, y1, x2, y2 = self.coords
            return (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)

        # Even-odd ray casting against every polygon edge
        points = np.asarray(self.coords, dtype=float)
        inside = np.zeros((height, width), dtype=bool)
        for (xa, ya), (xb, yb) in zip(points, np.roll(points, -1, axis=0)):
            crosses = (ya > grid_y) != (yb > grid_y)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at_y = (xb - xa) * (grid_y - ya) / (yb - ya) + xa
            inside ^= crosses & (grid_x < x_at_y)
        return inside


class ZoneManager:
    def __init__(self, zones):
        self.zones = list(zones)
        ids = [zone.id for zone in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone ids must be unique")
        self._mask_cache = {}

    @classmethod
    def from_json(cls, path=DEFAULT_ZONES_PATH):
        """Load zones from a config file: {"zones": [{id, name, type, coords, threshold, warn_ratio?}, ...]}."""
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        zones = []
        for entry in config["zones"]:
            coords = entry["coords"]
            zones.append(Zone(
                id=str(entry["id"]),
                name=entry.get("name", str(entry["id"])),
                type=entry["type"],
                coords=tuple(tuple(p) if isinstance(p, list) else p for p in coords),
                threshold=float(entry["threshold"]),
                warn_ratio=float(entry.get("warn_ratio", 0.8)),
            ))
        return cls(zones)

    def masks(self, shape):
        """Zone masks for a density map shape (height, width), cached per shape."""
        shape = tuple(shape)
        if shape not in self._mask_cache:
            self._mask_cache[shape] = {zone.id: zone.mask(*shape) for zone in self.zones}
        return self._mask_cache[shape]

    def zone_counts(self, density):
        """Return {zone_id: estimated count}, the density map summed inside each zone."""
        masks = self.masks(density.shape)
        return {zone.id: float(density[masks[zone.id]].sum(dtype=np.float64)) for zone in self.zones}
