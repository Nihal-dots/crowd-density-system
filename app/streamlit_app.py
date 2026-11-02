"""Crowd density dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import dataclasses
import os
import sys
import tempfile
import time
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.alerts.alert_engine import AlertEngine  # noqa: E402
from backend.alerts.zone_manager import CRITICAL, NORMAL, WARNING, ZoneManager  # noqa: E402
from backend.database.db_manager import DEFAULT_DB_PATH, AlertDatabase  # noqa: E402
from backend.inference.predictor import CrowdPredictor  # noqa: E402
from backend.video.frame_extractor import VideoReader  # noqa: E402

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]

MODE_VIDEO, MODE_LIVE, MODE_IMAGE = "Upload video", "Live feed (URL or webcam)", "Single image"

LEVEL_ORDER = {NORMAL: 0, WARNING: 1, CRITICAL: 2}
# Status is always shown as icon + label; the colour never carries it alone
LEVEL_STYLE = {
    NORMAL: {"icon": "✅", "hex": "#0ca30c", "text": "#ffffff", "bgr": (12, 163, 12)},
    WARNING: {"icon": "⚠️", "hex": "#fab219", "text": "#1a1a1a", "bgr": (25, 178, 250)},
    CRITICAL: {"icon": "🚨", "hex": "#d03b3b", "text": "#ffffff", "bgr": (59, 59, 208)},
}
HISTORY_POINTS = 600  # points kept for the count-over-time chart


def find_default_checkpoint():
    """First existing best_model.pth among the usual locations (combined 'all' model preferred, then Part B)."""
    candidates = [os.environ.get("CSRNET_CHECKPOINT")]
    for root in (
        PROJECT_ROOT / "experiments" / "checkpoints",
        Path.home() / "crowd-density-checkpoints",
        Path("/content/drive/MyDrive/crowd-density-system/checkpoints"),
    ):
        candidates += [root / folder / "best_model.pth" for folder in ("all", "part_B", "part_A")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return ""


@st.cache_resource(show_spinner="Loading CSRNet...")
def load_predictor(checkpoint_path, modified_time):
    """Cached model; modified_time reloads it when the checkpoint file changes."""
    return CrowdPredictor(checkpoint_path)


def badge(level):
    style = LEVEL_STYLE[level]
    return (
        f"<span style='background:{style['hex']};color:{style['text']};padding:2px 10px;"
        f"border-radius:12px;font-weight:600;white-space:nowrap'>{style['icon']} {level}</span>"
    )


def draw_zones(frame, zone_manager, zone_states):
    """Outline each zone on the frame in its status colour, with its name and count."""
    height, width = frame.shape[:2]
    scale = max(height / 720.0, 0.5)
    for zone, state in zip(zone_manager.zones, zone_states):
        if zone.type == "rectangle":
            x1, y1, x2, y2 = zone.coords
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            points = zone.coords
        pixels = np.array([[x * (width - 1), y * (height - 1)] for x, y in points], dtype=np.int32)
        color = LEVEL_STYLE[state["status"]]["bgr"]
        cv2.polylines(frame, [pixels], isClosed=True, color=color, thickness=max(int(2 * scale), 1))

        label = f"{zone.name}: {state['count']:.0f}/{zone.threshold:.0f}"
        cx, cy = pixels.mean(axis=0).astype(int)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(int(2 * scale), 1))
        origin = (int(np.clip(cx - text_w // 2, 0, max(width - text_w, 0))), int(cy))
        cv2.rectangle(frame, (origin[0] - 4, origin[1] - text_h - 6), (origin[0] + text_w + 4, origin[1] + 6), (0, 0, 0), -1)
        cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), max(int(2 * scale), 1), cv2.LINE_AA)
    return frame


def density_to_heatmap(density, size):
    """Raw density map as a JET image at the given (width, height)."""
    peak = float(density.max())
    scaled = np.uint8(255 * density / peak) if peak > 0 else np.zeros(density.shape, np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    return cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)


def count_chart(history):
    data = pd.DataFrame(history)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2, color="#2a78d6")
        .encode(
            x=alt.X("time_s:Q", title="time (s)"),
            y=alt.Y("count:Q", title="estimated count", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("time_s:Q", title="time (s)", format=".1f"),
                alt.Tooltip("count:Q", title="count", format=".0f"),
            ],
        )
        .properties(height=260)
    )


def alerts_dataframe(db, limit=500):
    return pd.DataFrame(db.get_alerts(limit=limit), columns=[
        "id", "timestamp", "video_source", "zone_id", "zone_name", "estimated_count",
        "density_val", "threshold", "alert_level", "message",
    ])


def build_layout():
    """Create the page placeholders, top to bottom."""
    layout = {"banner": st.empty(), "kpis": st.empty()}
    feed_column, heat_column = st.columns(2)
    feed_column.subheader("Annotated feed")
    heat_column.subheader("Density heatmap")
    layout["feed"], layout["heatmap"] = feed_column.empty(), heat_column.empty()
    st.subheader("Zone status")
    layout["zones"] = st.empty()
    st.subheader("Count over time")
    layout["chart"] = st.empty()
    st.subheader("Alert history")
    layout["alerts"] = st.empty()
    return layout


def render(layout, view, db):
    """Draw one processed frame (a dict built by process_frame) into the layout."""
    zone_states = view["zone_states"]
    overall = max((state["status"] for state in zone_states), key=LEVEL_ORDER.get, default=NORMAL)
    critical = [state["zone_name"] for state in zone_states if state["status"] == CRITICAL]

    if critical:
        layout["banner"].error(f"CRITICAL crowd level in: {', '.join(critical)}", icon="🚨")
    else:
        layout["banner"].empty()

    with layout["kpis"].container():
        columns = st.columns(5)
        columns[0].metric("Total count", f"{view['count']:.0f}")
        columns[1].metric("Alert level", f"{LEVEL_STYLE[overall]['icon']} {overall}")
        columns[2].metric("FPS", f"{view['fps']:.2f}")
        columns[3].metric("Latency", f"{view['latency_ms']:.0f} ms")
        columns[4].metric("Total alerts", db.summary_stats()["total_alerts"])

    layout["feed"].image(view["annotated"], channels="BGR", width="stretch")
    layout["heatmap"].image(view["heatmap"], channels="BGR", width="stretch")

    rows = "".join(
        f"<tr><td>{state['zone_name']}</td><td>{badge(state['status'])}</td>"
        f"<td style='text-align:right'>{state['count']:.0f}</td>"
        f"<td style='text-align:right'>{state['threshold']:.0f}</td>"
        f"<td style='text-align:right'>{100 * state['count'] / state['threshold']:.0f}%</td>"
        f"<td>{state['active_alert'] or '-'}</td></tr>"
        for state in zone_states
    )
    layout["zones"].markdown(
        "<table style='width:100%'><tr><th>Zone</th><th>Status</th><th style='text-align:right'>Count</th>"
        "<th style='text-align:right'>Threshold</th><th style='text-align:right'>Load</th><th>Active alert</th></tr>"
        f"{rows}</table>",
        unsafe_allow_html=True,
    )

    if len(view["history"]) > 1:
        layout["chart"].altair_chart(count_chart(view["history"]), width="stretch")
    else:
        layout["chart"].caption("The chart appears once two or more frames have been processed.")
    layout["alerts"].dataframe(alerts_dataframe(db), width="stretch", hide_index=True)


def process_frame(frame, time_s, predictor, zone_manager, engine, alpha, history, loop_started):
    count, density, latency_ms = predictor.predict_frame(frame)
    zone_states, _ = engine.update(density)
    annotated = predictor.render_heatmap_overlay(frame, density, alpha)
    draw_zones(annotated, zone_manager, zone_states)
    fps = 1.0 / max(time.perf_counter() - loop_started, 1e-6)
    predictor.draw_hud(annotated, count, fps)

    history.append({"time_s": round(time_s, 2), "count": round(count, 1)})
    del history[:-HISTORY_POINTS]
    return {
        "count": count,
        "latency_ms": latency_ms,
        "fps": fps,
        "zone_states": zone_states,
        "annotated": annotated,
        "heatmap": density_to_heatmap(density, (frame.shape[1], frame.shape[0])),
        "history": list(history),
    }


def sidebar():
    """Draw the sidebar and return the chosen settings."""
    st.sidebar.header("Input")
    mode = st.sidebar.radio("Input mode", [MODE_VIDEO, MODE_LIVE, MODE_IMAGE])
    source, source_name = None, ""
    if mode == MODE_VIDEO:
        uploaded = st.sidebar.file_uploader("Video file", type=VIDEO_TYPES)
        samples = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower().lstrip(".") in VIDEO_TYPES)
        sample = st.sidebar.selectbox("...or a sample video", ["(none)"] + samples) if samples else "(none)"
        if uploaded is not None:
            source, source_name = uploaded, uploaded.name
        elif sample != "(none)":
            source, source_name = str(SAMPLES_DIR / sample), sample
    elif mode == MODE_LIVE:
        url = st.sidebar.text_input("Stream URL or webcam index", value="0", help="e.g. rtsp://user:pass@host/stream, or 0 for the default webcam")
        source, source_name = url.strip() or None, url.strip()
    else:
        uploaded = st.sidebar.file_uploader("Image file", type=IMAGE_TYPES)
        if uploaded is not None:
            source, source_name = uploaded, uploaded.name

    st.sidebar.header("Model")
    checkpoint = st.sidebar.text_input("Checkpoint (best_model.pth)", value=find_default_checkpoint())
    max_side = st.sidebar.select_slider(
        "Inference size (longer side, px)", options=[480, 640, 960, 1280, "Original"], value=960,
        help="Smaller is faster on CPU but can miss small heads.",
    )

    st.sidebar.header("Display")
    alpha = st.sidebar.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05)
    frame_skip = st.sidebar.slider("Frame skip (process every N-th frame)", 1, 60, 10)
    max_frames = st.sidebar.number_input("Max frames to process (0 = no limit)", min_value=0, value=0, step=10)

    st.sidebar.header("Zone thresholds")
    zone_manager = ZoneManager.from_json()
    zones = []
    for zone in zone_manager.zones:
        threshold = st.sidebar.slider(
            zone.name, min_value=1, max_value=max(1000, int(zone.threshold * 3)),
            value=int(zone.threshold), key=f"threshold_{zone.id}",
        )
        zones.append(dataclasses.replace(zone, threshold=float(threshold)))

    return {
        "mode": mode, "source": source, "source_name": source_name, "checkpoint": checkpoint.strip().strip('"'),
        "max_side": None if max_side == "Original" else int(max_side), "alpha": alpha,
        "frame_skip": frame_skip, "max_frames": int(max_frames) or None, "zone_manager": ZoneManager(zones),
    }


def save_upload(uploaded):
    """Write an uploaded file to disk so OpenCV can open it."""
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getbuffer())
    return f.name


def run(settings, predictor, layout, db):
    zone_manager = settings["zone_manager"]
    history = []
    view = None

    if settings["mode"] == MODE_IMAGE:
        frame = cv2.imdecode(np.frombuffer(settings["source"].getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not read that image.")
            return
        # A single image has no following frames, so alert on the first one
        engine = AlertEngine(zone_manager, db=db, video_source=settings["source_name"], consecutive_frames=1)
        view = process_frame(frame, 0.0, predictor, zone_manager, engine, settings["alpha"], history, time.perf_counter())
        render(layout, view, db)
        st.session_state["last_view"] = view
        return

    temp_path = None
    source = settings["source"]
    if not isinstance(source, str):
        source = temp_path = save_upload(source)
    engine = AlertEngine(zone_manager, db=db, video_source=settings["source_name"])
    try:
        with VideoReader(source, step=settings["frame_skip"], max_frames=settings["max_frames"]) as reader:
            started = time.perf_counter()
            loop_started = started
            for frame_index, frame in reader:
                # Live sources have no reliable frame clock, so use wall time
                time_s = time.perf_counter() - started if reader.is_live else frame_index / reader.fps
                view = process_frame(frame, time_s, predictor, zone_manager, engine, settings["alpha"], history, loop_started)
                render(layout, view, db)
                st.session_state["last_view"] = view
                loop_started = time.perf_counter()
    except (IOError, FileNotFoundError) as error:
        st.error(str(error))
    finally:
        if temp_path:
            os.unlink(temp_path)
    if view is None:
        st.warning("No frames could be read from that source.")


def main():
    st.set_page_config(page_title="Crowd Density Monitor", page_icon="👥", layout="wide")
    st.title("Crowd Density Monitor")
    settings = sidebar()

    start_column, stop_column, _ = st.columns([1, 1, 6])
    start = start_column.button("Start", type="primary", disabled=settings["source"] is None, width="stretch")
    # Any click reruns the script, which ends a running loop
    stop_column.button("Stop", width="stretch")

    layout = build_layout()
    # CROWD_ALERTS_DB overrides the default alerts.db in the project root
    with AlertDatabase(os.environ.get("CROWD_ALERTS_DB") or DEFAULT_DB_PATH) as db:
        if start:
            if not Path(settings["checkpoint"]).is_file():
                st.error("Checkpoint not found. Enter the path to a trained best_model.pth in the sidebar.")
            else:
                predictor = load_predictor(settings["checkpoint"], Path(settings["checkpoint"]).stat().st_mtime)
                predictor.max_side = settings["max_side"]
                st.caption(f"Model: {settings['checkpoint']} (epoch {predictor.epoch}) on {predictor.device}")
                run(settings, predictor, layout, db)
        elif "last_view" in st.session_state:
            render(layout, st.session_state["last_view"], db)
        else:
            layout["feed"].info("Choose an input in the sidebar and press Start.")
            layout["alerts"].dataframe(alerts_dataframe(db), width="stretch", hide_index=True)

        export = alerts_dataframe(db, limit=None)
    st.download_button(
        "Download alert history (CSV)", export.to_csv(index=False).encode("utf-8"),
        file_name="alert_history.csv", mime="text/csv", disabled=export.empty,
    )


main()
