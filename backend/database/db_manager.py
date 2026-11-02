"""SQLite storage for crowd alerts."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "alerts.db"

COLUMNS = (
    "timestamp", "video_source", "zone_id", "zone_name", "estimated_count",
    "density_val", "threshold", "alert_level", "message",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    video_source    TEXT,
    zone_id         TEXT NOT NULL,
    zone_name       TEXT,
    estimated_count REAL,
    density_val     REAL,
    threshold       REAL,
    alert_level     TEXT NOT NULL,
    message         TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_zone ON alerts(zone_id);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(alert_level);
"""


class AlertDatabase:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def insert_alerts(self, alerts):
        """Insert a batch of alerts (Alert objects or dicts) in one transaction."""
        rows = [
            tuple((alert if isinstance(alert, dict) else alert.as_dict())[column] for column in COLUMNS)
            for alert in alerts
        ]
        if not rows:
            return 0
        placeholders = ", ".join("?" for _ in COLUMNS)
        with self.connection:
            self.connection.executemany(
                f"INSERT INTO alerts ({', '.join(COLUMNS)}) VALUES ({placeholders})", rows
            )
        return len(rows)

    def insert_alert(self, alert):
        return self.insert_alerts([alert])

    def get_alerts(self, zone_id=None, alert_level=None, video_source=None, since=None, limit=None):
        """Return alerts as dicts, newest first, optionally filtered."""
        conditions, params = [], []
        for column, value in (("zone_id", zone_id), ("alert_level", alert_level), ("video_source", video_source)):
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(value)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        query = "SELECT * FROM alerts"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in self.connection.execute(query, params)]

    def summary_stats(self):
        """Totals, counts per level and per zone, and the time range of the alerts."""
        cursor = self.connection.cursor()
        total, first, last = cursor.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM alerts"
        ).fetchone()
        by_level = dict(cursor.execute(
            "SELECT alert_level, COUNT(*) FROM alerts GROUP BY alert_level"
        ).fetchall())
        by_zone = {
            row["zone_id"]: {
                "zone_name": row["zone_name"],
                "alerts": row["alerts"],
                "critical": row["critical"],
                "peak_count": row["peak_count"],
                "last_alert": row["last_alert"],
            }
            for row in cursor.execute(
                """SELECT zone_id, MAX(zone_name) AS zone_name, COUNT(*) AS alerts,
                          SUM(alert_level = 'CRITICAL') AS critical,
                          MAX(estimated_count) AS peak_count, MAX(timestamp) AS last_alert
                   FROM alerts GROUP BY zone_id"""
            )
        }
        return {
            "total_alerts": total,
            "first_alert": first,
            "last_alert": last,
            "by_level": by_level,
            "by_zone": by_zone,
        }

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
