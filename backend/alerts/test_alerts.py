"""Tests for zones, the alert engine and the alert database.

Usage (from the project root):
    python backend/alerts/test_alerts.py
(also collected by pytest)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.alerts.alert_engine import CLEARED, AlertEngine  # noqa: E402
from backend.alerts.zone_manager import CRITICAL, NORMAL, WARNING, Zone, ZoneManager  # noqa: E402
from backend.database.db_manager import AlertDatabase  # noqa: E402

DENSITY_SHAPE = (90, 120)


def partition_zones():
    """Two rectangles and two triangles that tile the whole frame."""
    return ZoneManager([
        Zone("top_left", "Top left", "rectangle", (0.0, 0.0, 0.5, 0.5), threshold=100),
        Zone("bottom_left", "Bottom left", "rectangle", (0.0, 0.5, 0.5, 1.0), threshold=100),
        Zone("upper_right", "Upper right", "polygon", ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0)), threshold=100),
        Zone("lower_right", "Lower right", "polygon", ((0.5, 0.0), (1.0, 1.0), (0.5, 1.0)), threshold=100),
    ])


def density_with_count(count, shape=DENSITY_SHAPE):
    """A uniform density map that sums to count."""
    return np.full(shape, count / (shape[0] * shape[1]), dtype=np.float32)


def test_zone_counts_sum_to_total():
    manager = partition_zones()
    rng = np.random.default_rng(0)
    worst = 0.0
    for shape in (DENSITY_SHAPE, (96, 128), (135, 76), (60, 80)):
        masks = manager.masks(shape)
        coverage = sum(mask.astype(int) for mask in masks.values())
        assert (coverage == 1).all(), f"zones overlap or leave gaps at {shape}"

        density = rng.random(shape, dtype=np.float32) * 0.05
        total = float(density.sum(dtype=np.float64))
        zone_total = sum(manager.zone_counts(density).values())
        worst = max(worst, abs(zone_total - total))
        assert abs(zone_total - total) < 1e-6, f"{zone_total} != {total} at {shape}"
    return f"max |sum of zones - total| = {worst:.1e} over 4 map sizes"


def test_status_thresholds():
    zone = Zone("z", "Zone", "rectangle", (0, 0, 1, 1), threshold=100, warn_ratio=0.8)
    cases = {0: NORMAL, 79.9: NORMAL, 80: WARNING, 99.9: WARNING, 100: CRITICAL, 250: CRITICAL}
    for count, expected in cases.items():
        assert zone.status(count) == expected, f"count {count}: {zone.status(count)} != {expected}"
    return f"{len(cases)} boundary cases"


def test_zones_json_and_validation():
    manager = ZoneManager.from_json()
    assert len(manager.zones) >= 1
    assert {zone.type for zone in manager.zones} == {"rectangle", "polygon"}
    assert all(zone.warn_ratio == 0.8 for zone in manager.zones)

    bad_configs = [
        dict(type="circle", coords=(0, 0, 1, 1), threshold=10),
        dict(type="rectangle", coords=(0, 0, 1.5, 1), threshold=10),
        dict(type="rectangle", coords=(0.5, 0, 0.2, 1), threshold=10),
        dict(type="polygon", coords=((0, 0), (1, 1)), threshold=10),
        dict(type="rectangle", coords=(0, 0, 1, 1), threshold=0),
    ]
    for config in bad_configs:
        try:
            Zone("bad", "Bad", **config)
        except ValueError:
            continue
        raise AssertionError(f"invalid zone accepted: {config}")
    return f"loaded {len(manager.zones)} zones from zones.json, rejected {len(bad_configs)} invalid configs"


def test_surge_logs_alerts():
    """Simulate a crowd surge in one zone and check what reaches the database."""
    zone = Zone("gate", "Gate", "rectangle", (0, 0, 1, 1), threshold=100)  # warn 80, clear below 70
    # Phases: count per frame, and the alert expected on that frame (None = no new alert)
    frames = [
        (40, None), (50, None),
        (120, None), (40, None),                    # one-frame spike: too short to alert
        (85, None), (90, None), (95, WARNING),      # 3 frames at warning level
        (105, None), (110, None), (130, CRITICAL),  # 3 frames at critical level
        (150, None), (140, None),                   # still critical: no duplicate alerts
        (90, None), (75, None), (72, None),         # below T but above 0.7T: stays active
        (69, CLEARED),                              # below 0.7T: clears
        (60, None), (40, None),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        with AlertDatabase(Path(tmp) / "alerts.db") as db:
            engine = AlertEngine(ZoneManager([zone]), db=db, video_source="surge_test", consecutive_frames=3)
            for index, (count, expected) in enumerate(frames):
                states, new_alerts = engine.update(density_with_count(count), timestamp=f"2026-01-01T00:00:{index:02d}")
                levels = [alert.alert_level for alert in new_alerts]
                assert levels == ([expected] if expected else []), f"frame {index} (count {count}): got {levels}, expected {expected}"
                assert states[0]["status"] == zone.status(states[0]["count"])

            logged = db.get_alerts()[::-1]  # oldest first
            assert [row["alert_level"] for row in logged] == [WARNING, CRITICAL, CLEARED]
            assert all(row["video_source"] == "surge_test" and row["zone_id"] == "gate" for row in logged)
            assert abs(logged[1]["estimated_count"] - 130) < 0.01 and logged[1]["threshold"] == 100
            assert logged[1]["message"].startswith("CRITICAL: Gate")
            assert [row["alert_level"] for row in db.get_alerts(alert_level=CRITICAL)] == [CRITICAL]

            stats = db.summary_stats()
            assert stats["total_alerts"] == 3
            assert stats["by_level"] == {WARNING: 1, CRITICAL: 1, CLEARED: 1}
            assert stats["by_zone"]["gate"]["critical"] == 1
            assert abs(stats["by_zone"]["gate"]["peak_count"] - 130) < 0.01
    return f"{len(frames)} frames -> 3 alerts logged (WARNING, CRITICAL, CLEARED), none for the one-frame spike"


def test_batch_insert():
    rows = [
        dict(timestamp=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}", video_source="batch", zone_id=f"z{i % 4}",
             zone_name=f"Zone {i % 4}", estimated_count=float(i), density_val=0.01, threshold=100.0,
             alert_level=WARNING if i % 2 else CRITICAL, message="test")
        for i in range(1000)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with AlertDatabase(Path(tmp) / "alerts.db") as db:
            assert db.insert_alerts(rows) == 1000
            stats = db.summary_stats()
            assert stats["total_alerts"] == 1000
            assert stats["by_level"] == {WARNING: 500, CRITICAL: 500}
            assert len(stats["by_zone"]) == 4 and stats["by_zone"]["z3"]["peak_count"] == 999
            assert len(db.get_alerts(zone_id="z1", limit=10)) == 10
            indexes = {row[1] for row in db.connection.execute("PRAGMA index_list(alerts)")}
            assert {"idx_alerts_timestamp", "idx_alerts_zone", "idx_alerts_level"} <= indexes
    return "1000 rows in one batch, 3 indexes present"


if __name__ == "__main__":
    tests = [
        test_zone_counts_sum_to_total,
        test_status_thresholds,
        test_zones_json_and_validation,
        test_surge_logs_alerts,
        test_batch_insert,
    ]
    failed = 0
    for test in tests:
        try:
            print(f"[PASS] {test.__name__}: {test()}")
        except AssertionError as error:
            failed += 1
            print(f"[FAIL] {test.__name__}: {error}")
    print("PASSED" if not failed else f"FAILED ({failed} of {len(tests)})")
    sys.exit(1 if failed else 0)
