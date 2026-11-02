"""Turn per-zone crowd counts into alerts, with hysteresis to avoid flooding.

An alert is raised only after a zone has been at WARNING or CRITICAL level for
`consecutive_frames` frames in a row, and it stays active until the count
falls below `clear_ratio` * threshold. While an alert is active no duplicates
are produced; it can only escalate from WARNING to CRITICAL.
"""

from dataclasses import asdict, dataclass
from datetime import datetime

from backend.alerts.zone_manager import CRITICAL, NORMAL, WARNING

CLEARED = "CLEARED"


@dataclass
class Alert:
    timestamp: str
    video_source: str
    zone_id: str
    zone_name: str
    estimated_count: float
    density_val: float  # mean density map value per pixel inside the zone
    threshold: float
    alert_level: str  # WARNING, CRITICAL or CLEARED
    message: str

    def as_dict(self):
        return asdict(self)


class AlertEngine:
    def __init__(self, zone_manager, db=None, video_source="", consecutive_frames=3, clear_ratio=0.7):
        """db: optional AlertDatabase; new alerts are batch-inserted on every update."""
        if consecutive_frames < 1:
            raise ValueError("consecutive_frames must be >= 1")
        self.zone_manager = zone_manager
        self.db = db
        self.video_source = video_source
        self.consecutive_frames = consecutive_frames
        self.clear_ratio = clear_ratio
        # Per zone: frames in a row at >= WARNING, at CRITICAL, and the active alert level
        self._state = {
            zone.id: {"warn_streak": 0, "critical_streak": 0, "active": None}
            for zone in zone_manager.zones
        }

    def update(self, density, timestamp=None):
        """Process one density map. Returns (zone_states, new_alerts).

        zone_states: one dict per zone with its count, instantaneous status and
        the active alert level (None when no alert is active).
        """
        timestamp = timestamp or datetime.now().isoformat(timespec="milliseconds")
        counts = self.zone_manager.zone_counts(density)
        masks = self.zone_manager.masks(density.shape)
        zone_states, new_alerts = [], []

        for zone in self.zone_manager.zones:
            count = counts[zone.id]
            pixels = int(masks[zone.id].sum())
            density_val = count / pixels if pixels else 0.0
            status = zone.status(count)
            state = self._state[zone.id]

            state["warn_streak"] = state["warn_streak"] + 1 if status != NORMAL else 0
            state["critical_streak"] = state["critical_streak"] + 1 if status == CRITICAL else 0

            level = None
            if state["critical_streak"] >= self.consecutive_frames and state["active"] != CRITICAL:
                level = state["active"] = CRITICAL
            elif state["warn_streak"] >= self.consecutive_frames and state["active"] is None:
                level = state["active"] = WARNING
            elif state["active"] is not None and count < self.clear_ratio * zone.threshold:
                level, state["active"] = CLEARED, None

            if level is not None:
                new_alerts.append(Alert(
                    timestamp=timestamp,
                    video_source=self.video_source,
                    zone_id=zone.id,
                    zone_name=zone.name,
                    estimated_count=round(count, 2),
                    density_val=density_val,
                    threshold=zone.threshold,
                    alert_level=level,
                    message=f"{level}: {zone.name} estimated count {count:.0f} (threshold {zone.threshold:.0f})",
                ))

            zone_states.append({
                "zone_id": zone.id,
                "zone_name": zone.name,
                "count": count,
                "density_val": density_val,
                "threshold": zone.threshold,
                "status": status,
                "active_alert": state["active"],
            })

        if new_alerts and self.db is not None:
            self.db.insert_alerts(new_alerts)
        return zone_states, new_alerts
