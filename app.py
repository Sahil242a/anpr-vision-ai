"""
ANPR Vision AI - Streamlit front end.

The UI is deliberately thin: it collects settings, calls :class:`ANPRPipeline`,
and renders results. No computer-vision logic lives in this file, which is what
lets the same pipeline be driven from a notebook, a test or a CLI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Make ``config`` and ``src`` importable when Streamlit runs from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import OUTPUT_DIR, AppConfig  # noqa: E402
from src.database.database import Database  # noqa: E402
from src.pipeline import ANPRPipeline  # noqa: E402
from src.utils.video import VideoError, decode_image_bytes, to_rgb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="ANPR Vision AI",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained instrument-panel palette: slate surfaces, one signal colour for
# readings, amber for "needs a human", red for rejected. Nothing decorative.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1400px;}
      div[data-testid="stMetric"] {
          background: #161a20; border: 1px solid #262c36;
          border-radius: 6px; padding: 14px 16px;
      }
      div[data-testid="stMetricValue"] {font-size: 1.6rem; font-variant-numeric: tabular-nums;}
      div[data-testid="stMetricLabel"] {color: #8b94a3; font-size: .78rem;}
      .plate-chip {
          display:inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 1.15rem; letter-spacing: .09em; padding: 6px 14px;
          border-radius: 4px; background:#11161c; border:1px solid #2c3441; margin-right:8px;
      }
      .ok    {border-left: 3px solid #35c47c;}
      .warn  {border-left: 3px solid #e0a63c;}
      .bad   {border-left: 3px solid #e0533c;}
      .subtle {color:#8b94a3; font-size:.86rem;}
      .stTabs [data-baseweb="tab"] {font-size: .95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_CLASS = {
    "VALID_FORMAT": "ok",
    "SUSPICIOUS_FORMAT": "warn",
    "LOW_CONFIDENCE": "bad",
    "NO_TEXT": "bad",
}


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_pipeline(vehicle_model: str, plate_model: str, device: str, tracker: str, lang: str):
    """Build the pipeline once per model configuration (weights are expensive)."""
    config = AppConfig()
    config.vehicle.model_path = vehicle_model
    config.plate.model_path = plate_model
    config.device = device
    config.tracker.backend = tracker
    config.ocr.lang = lang
    return ANPRPipeline(config)


@st.cache_resource(show_spinner=False)
def get_database(path: str) -> Database:
    from config.config import DatabaseConfig

    return Database(DatabaseConfig(path=path))


def transcode_h264(path: str) -> str:
    """Re-encode to H.264 so the browser can play the result inline.

    OpenCV writes mp4v, which Chrome and Safari refuse to play in a <video> tag.
    If ffmpeg is unavailable the original file is returned and the user can still
    download it.
    """
    if not shutil.which("ffmpeg"):
        return path
    out = str(Path(path).with_name(Path(path).stem + "_h264.mp4"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out],
            check=True, timeout=900,
        )
        return out if Path(out).exists() else path
    except Exception:  # noqa: BLE001
        return path


def status_chip(text: str, confidence: float, status: str) -> str:
    cls = STATUS_CLASS.get(status, "warn")
    return f'<span class="plate-chip {cls}">{text or "unreadable"} &nbsp; {confidence:.0%}</span>'


def readings_dataframe(readings) -> pd.DataFrame:
    rows = []
    for r in readings:
        rows.append(
            {
                "Plate": r.text or "—",
                "Status": r.status,
                "OCR": round(r.ocr.confidence, 3),
                "Combined": round(r.combined_confidence, 3),
                "Plate box conf": round(r.detection_confidence, 3),
                "Vehicle": getattr(r.vehicle, "class_name", "—"),
                "Best variant": r.ocr.variant or "—",
                "Rectified": "yes" if r.ocr.perspective_corrected else "no",
                "OCR ms": round(r.ocr.elapsed_ms, 1),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.markdown("### ANPR Vision AI")
st.sidebar.caption("Detection → tracking → OCR → validation → aggregation → storage")

default_config = AppConfig()

with st.sidebar.expander("Models and device", expanded=False):
    vehicle_model = st.text_input("Vehicle model (.pt)", value=default_config.vehicle.model_path)
    plate_model = st.text_input("Plate model (.pt)", value=default_config.plate.model_path)
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda:0"], index=0)
    tracker_backend = st.selectbox("Tracker", ["bytetrack", "botsort", "simple"], index=0)
    ocr_lang = st.selectbox("OCR language", ["en"], index=0)

pipeline = get_pipeline(vehicle_model, plate_model, device_choice, tracker_backend, ocr_lang)
db = get_database(pipeline.config.database.path)
pipeline.db = db
# Point the database at the live config object so the storage sliders below
# actually take effect - the cached Database was built with its own defaults.
db.cfg = pipeline.config.database

st.sidebar.markdown("#### Detection")
pipeline.config.vehicle.conf = st.sidebar.slider("Vehicle confidence", 0.05, 0.95, 0.35, 0.05)
pipeline.config.plate.conf = st.sidebar.slider("Plate confidence", 0.05, 0.95, 0.25, 0.05)
pipeline.config.vehicle.iou = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.50, 0.05)

st.sidebar.markdown("#### Performance")
pipeline.config.video.frame_skip = st.sidebar.slider("Frame skip", 0, 9, 2, 1)
pipeline.config.ocr.interval = st.sidebar.slider("OCR interval (frames per track)", 1, 30, 8, 1)
pipeline.config.video.resize_width = st.sidebar.select_slider(
    "Processing width", options=[0, 640, 960, 1280, 1920], value=1280,
    format_func=lambda v: "original" if v == 0 else f"{v}px",
)

st.sidebar.markdown("#### Storage")
pipeline.config.database.min_confidence_to_store = st.sidebar.slider(
    "Minimum confidence to store", 0.0, 1.0, 0.45, 0.05
)
pipeline.config.database.cooldown_seconds = st.sidebar.number_input(
    "Duplicate cooldown (seconds)", 0, 3600, 60, 10
)
pipeline.config.ocr.save_crops = st.sidebar.checkbox("Save plate crops for debugging", value=True)
pipeline.config.ocr.enable_perspective_correction = st.sidebar.checkbox(
    "Perspective correction", value=True
)

with st.sidebar.expander("Component status", expanded=True):
    report = pipeline.readiness()
    for name, info in report.items():
        icon = "●" if info.get("ok") else "○"
        label = name.replace("_", " ")
        if info.get("ok"):
            st.markdown(f"{icon} **{label}** — {info['detail']}")
        else:
            st.markdown(f"{icon} **{label}**")
            st.caption(info["detail"])

st.sidebar.caption(
    "Format validation is visual only. It does not verify vehicle registration, "
    "ownership, insurance or road-legality."
)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("ANPR Vision AI")
st.caption(
    "Automatic number plate recognition with vehicle tracking and temporal OCR aggregation."
)

tab_image, tab_video, tab_live, tab_dash, tab_search, tab_how = st.tabs(
    ["Image", "Video", "Live camera", "Dashboard", "Search", "How it works"]
)


# --------------------------------------------------------------------------- #
# Mode A - Image
# --------------------------------------------------------------------------- #

with tab_image:
    st.subheader("Single image")
    st.write("Detect vehicles and plates in one frame, then read and validate the text.")

    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_upload"
    )

    if uploaded is not None:
        try:
            image = decode_image_bytes(uploaded.getvalue())
        except VideoError as exc:
            st.error(str(exc))
            image = None

        if image is not None:
            with st.spinner("Running the pipeline…"):
                try:
                    result = pipeline.process_image(image, source=uploaded.name)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Processing failed: {exc}")
                    result = None

            if result is not None:
                for warning in result.warnings:
                    st.warning(warning)

                left, right = st.columns([3, 2])
                with left:
                    st.image(
                        to_rgb(result.annotated if result.annotated is not None else image),
                        caption="Annotated result",
                        use_container_width=True,
                    )
                with right:
                    st.metric("Vehicles detected", len(result.vehicles))
                    st.metric("Plates detected", len(result.plates))
                    readable = [r for r in result.plates if r.text]
                    st.metric("Plates read", len(readable))
                    if readable:
                        best = max(readable, key=lambda r: r.combined_confidence)
                        st.markdown("**Best reading**")
                        st.markdown(
                            status_chip(best.text, best.combined_confidence, best.status),
                            unsafe_allow_html=True,
                        )
                        v = best.ocr.validation
                        if v:
                            if v.state_name:
                                st.caption(f"State/UT prefix: {v.state_code} — {v.state_name}")
                            if v.pattern:
                                st.caption(f"Matched pattern: {v.pattern} (score {v.score:.2f})")
                            for note in v.notes:
                                st.caption(f"· {note}")

                if result.plates:
                    st.markdown("##### All plate readings")
                    st.dataframe(readings_dataframe(result.plates), use_container_width=True)

                    with st.expander("Per-variant OCR comparison"):
                        st.caption(
                            "Each plate is read through several preprocessing variants. "
                            "The winner is chosen by a score that blends OCR confidence "
                            "with how plausible the string is as a registration number."
                        )
                        for i, r in enumerate(result.plates, 1):
                            if not r.ocr.variant_results:
                                continue
                            st.markdown(f"**Plate {i}** — chosen: `{r.ocr.variant}`")
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "Variant": vr.variant,
                                            "Text": vr.text,
                                            "OCR confidence": round(vr.confidence, 3),
                                            "Combined score": round(vr.score, 3),
                                        }
                                        for vr in r.ocr.variant_results
                                    ]
                                ),
                                use_container_width=True,
                            )

                    crops = [r.crop_path for r in result.plates if r.crop_path]
                    if crops:
                        st.markdown("##### Saved crops")
                        st.image(crops, width=220)
                elif not result.warnings:
                    st.info(
                        "No plates were found. Try an image where the plate is larger "
                        "than roughly 60 px wide, or lower the plate confidence threshold."
                    )
    else:
        st.info("Upload an image to begin. Front or rear views with a readable plate work best.")


# --------------------------------------------------------------------------- #
# Mode B - Video
# --------------------------------------------------------------------------- #

with tab_video:
    st.subheader("Video")
    st.write(
        "Tracks each vehicle, reads its plate every few frames, and fuses the readings "
        "into one answer per vehicle."
    )

    video_file = st.file_uploader(
        "Upload a traffic video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
    )
    col_a, col_b = st.columns(2)
    with col_a:
        frame_limit = st.number_input(
            "Frame limit (0 = whole video)", min_value=0, max_value=100000, value=600, step=100
        )
    with col_b:
        write_output = st.checkbox("Write annotated video", value=True)

    if video_file is not None and st.button("Process video", type="primary"):
        suffix = Path(video_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.getvalue())
            temp_path = tmp.name

        progress = st.progress(0.0, text="Starting…")
        preview = st.empty()
        state = {"n": 0}

        def on_progress(fraction, frame_result):
            state["n"] += 1
            progress.progress(
                min(1.0, fraction), text=f"Processing frame {frame_result.frame_index}"
            )
            if state["n"] % 8 == 0 and frame_result.annotated is not None:
                preview.image(
                    to_rgb(frame_result.annotated), caption="Live preview",
                    use_container_width=True,
                )

        try:
            outcome = pipeline.process_video(
                temp_path,
                progress_callback=on_progress,
                write_video=write_output,
                max_frames=int(frame_limit),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Video processing failed: {exc}")
            outcome = None
        finally:
            progress.empty()
            preview.empty()
            Path(temp_path).unlink(missing_ok=True)

        if outcome is not None:
            st.session_state["last_video_result"] = outcome

    outcome = st.session_state.get("last_video_result")
    if outcome is not None:
        for warning in outcome.warnings:
            st.warning(warning)

        stats = outcome.stats
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Unique vehicles", stats.unique_vehicles)
        k2.metric("Plate detections", stats.plates_detected)
        k3.metric("Plates identified", len(outcome.plates))
        k4.metric("Average confidence", f"{stats.average_confidence:.0%}")

        perf = outcome.performance
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Processing FPS", f"{perf.get('fps_processed', 0):.1f}")
        p2.metric("Frames processed", perf.get("frames_processed", 0))
        p3.metric("Avg OCR time", f"{perf.get('avg_ocr_ms', 0):.0f} ms")
        p4.metric("Avg detect time", f"{perf.get('avg_vehicle_detection_ms', 0):.0f} ms")

        if outcome.plates:
            st.markdown("##### Identified vehicles")
            st.caption(
                "One row per tracked vehicle. Confidence is the fused value across "
                "all observations, not a single frame."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Track": p.track_id,
                            "Plate": p.text,
                            "Vehicle": p.vehicle_type,
                            "Confidence": round(p.confidence, 3),
                            "Observations": p.observations,
                            "Agreement": f"{p.agreement:.0%}",
                            "Method": p.method,
                            "Status": p.status,
                        }
                        for p in outcome.plates
                    ]
                ),
                use_container_width=True,
            )

            with st.expander("Aggregation detail — how each answer was chosen"):
                for p in outcome.plates:
                    st.markdown(
                        f"**Track #{p.track_id} → `{p.text}`** "
                        f"({p.confidence:.0%} from {p.observations} observations, "
                        f"{p.agreement:.0%} agreement, {p.method})"
                    )
                    obs = pipeline.aggregator.observations(p.track_id)
                    if obs:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Frame": o.frame_index,
                                        "Reading": o.text,
                                        "OCR confidence": round(o.confidence, 3),
                                        "Status": o.status,
                                        "Variant": o.variant,
                                    }
                                    for o in obs
                                ]
                            ),
                            use_container_width=True,
                        )

        if outcome.output_path and Path(outcome.output_path).exists():
            st.markdown("##### Annotated video")
            playable = transcode_h264(outcome.output_path)
            try:
                st.video(playable)
            except Exception:  # noqa: BLE001
                st.caption("Inline playback is unavailable for this codec — use the download below.")
            with open(outcome.output_path, "rb") as fh:
                st.download_button(
                    "Download annotated video",
                    fh.read(),
                    file_name=Path(outcome.output_path).name,
                    mime="video/mp4",
                )


# --------------------------------------------------------------------------- #
# Mode C - Live camera
# --------------------------------------------------------------------------- #

with tab_live:
    st.subheader("Live camera")
    st.write(
        "Reads from a locally attached camera. This only works when Streamlit runs on "
        "the same machine as the camera — a remotely hosted app cannot reach it."
    )

    cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    live_frames = st.slider("Frames to capture", 30, 900, 200, 10)
    start = st.button("Start camera", type="primary")

    if start:
        placeholder = st.empty()
        metrics = st.empty()
        try:
            captured = 0
            for result in pipeline.stream(int(cam_index), max_frames=int(live_frames)):
                captured += 1
                if result.annotated is not None:
                    placeholder.image(to_rgb(result.annotated), use_container_width=True)
                stats = getattr(pipeline, "last_stats", None)
                if stats:
                    c1, c2, c3 = metrics.columns(3)
                    c1.metric("Unique vehicles", pipeline.tracker.unique_vehicle_count)
                    c2.metric("Plate detections", stats.plates_detected)
                    c3.metric("Average confidence", f"{stats.average_confidence:.0%}")
            written = pipeline.flush_live_results(source=f"webcam:{cam_index}")
            st.success(f"Capture finished. {captured} frames processed, {written} plates stored.")
        except VideoError as exc:
            st.error(str(exc))
            st.info(
                "No camera is reachable from this process. Use the Image or Video tab "
                "instead — they run the identical pipeline."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Live capture stopped: {exc}")
            st.info("Use the Image or Video tab instead — they run the identical pipeline.")


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

with tab_dash:
    st.subheader("Dashboard")

    try:
        stats = db.get_statistics()
        recent = db.get_recent_detections(limit=200)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Database unavailable: {exc}")
        stats, recent = None, []

    if stats:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stored detections", stats["total_detections"])
        k2.metric("Unique plates", stats["unique_plates"])
        k3.metric("Avg OCR confidence", f"{stats['avg_ocr_confidence']:.0%}")
        valid = next(
            (r["count"] for r in stats["by_status"] if r["validation_status"] == "VALID_FORMAT"), 0
        )
        share = valid / stats["total_detections"] if stats["total_detections"] else 0
        k4.metric("Valid-format share", f"{share:.0%}")

        if stats["total_detections"] == 0:
            st.info("No detections stored yet. Process an image or a video to populate the dashboard.")
        else:
            import plotly.express as px

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Vehicle distribution")
                df_type = pd.DataFrame(stats["by_vehicle_type"])
                if not df_type.empty:
                    fig = px.bar(
                        df_type, x="vehicle_type", y="count",
                        color="vehicle_type", text="count",
                    )
                    fig.update_layout(
                        showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="", yaxis_title="detections",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("##### Validation outcome")
                df_status = pd.DataFrame(stats["by_status"])
                if not df_status.empty:
                    fig = px.pie(
                        df_status, names="validation_status", values="count", hole=0.55,
                        color="validation_status",
                        color_discrete_map={
                            "VALID_FORMAT": "#35c47c",
                            "SUSPICIOUS_FORMAT": "#e0a63c",
                            "LOW_CONFIDENCE": "#e0533c",
                            "NO_TEXT": "#6b7382",
                        },
                    )
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("##### Detections over time")
                df_day = pd.DataFrame(stats["by_day"]).sort_values("day") if stats["by_day"] else pd.DataFrame()
                if not df_day.empty:
                    fig = px.line(df_day, x="day", y="count", markers=True)
                    fig.update_layout(
                        height=300, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="", yaxis_title="detections",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with c4:
                st.markdown("##### OCR confidence distribution")
                if recent:
                    df_conf = pd.DataFrame(recent)
                    fig = px.histogram(df_conf, x="ocr_confidence", nbins=20)
                    fig.update_layout(
                        height=300, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="confidence", yaxis_title="detections", bargap=0.05,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Recent detections")
            if recent:
                df_recent = pd.DataFrame(recent)[
                    ["timestamp", "plate_number", "vehicle_type", "tracking_id",
                     "ocr_confidence", "validation_status", "observations", "source"]
                ]
                df_recent.columns = [
                    "Time", "Plate", "Vehicle", "Track", "Confidence", "Status", "Obs", "Source"
                ]
                st.dataframe(df_recent.head(50), use_container_width=True, height=380)
                st.download_button(
                    "Export detections as CSV",
                    pd.DataFrame(db.get_all()).to_csv(index=False).encode(),
                    file_name=f"anpr_detections_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv",
                )

        with st.expander("Danger zone"):
            if st.button("Delete all stored detections"):
                removed = db.delete_all()
                st.success(f"Removed {removed} rows.")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

with tab_search:
    st.subheader("Vehicle history")
    st.write("Look up every time a plate was seen.")

    query = st.text_input("Plate number", placeholder="UP32AB1234").strip().upper()
    col1, col2 = st.columns([1, 3])
    with col1:
        partial = st.checkbox("Partial match", value=True)

    if query:
        try:
            rows = db.query_by_plate(query, fuzzy=partial)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {exc}")
            rows = []

        if not rows:
            st.info(f"No stored detections for '{query}'.")
        else:
            st.success(f"{len(rows)} detection(s) found.")
            first, last = rows[-1]["timestamp"], rows[0]["timestamp"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Sightings", len(rows))
            m2.metric("First seen", first.replace("T", " "))
            m3.metric("Last seen", last.replace("T", " "))

            for row in rows[:40]:
                with st.container():
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown(
                            status_chip(
                                row["plate_number"], row["ocr_confidence"], row["validation_status"]
                            ),
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<span class='subtle'>{row['timestamp'].replace('T', ' · ')} — "
                            f"{row['vehicle_type']} — source: {row['source']} — "
                            f"{row['observations']} observation(s)</span>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        path = row.get("image_path")
                        if path and Path(path).exists():
                            st.image(path, width=240)
                    st.divider()

    st.markdown("##### Browse by date")
    d1, d2 = st.columns(2)
    start_date = d1.date_input("From", value=datetime.now().date() - timedelta(days=7))
    end_date = d2.date_input("To", value=datetime.now().date())
    if st.button("Show detections in range"):
        rows = db.query_by_date(start_date.isoformat(), end_date.isoformat())
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No detections in that range.")


# --------------------------------------------------------------------------- #
# How it works
# --------------------------------------------------------------------------- #

with tab_how:
    st.subheader("How it works")
    st.markdown(
        """
**Pipeline**

`frame → vehicle detection → tracking → plate detection → crop → perspective
correction → preprocessing variants → OCR → normalisation + validation →
temporal aggregation → duplicate-controlled storage → annotation`

**Why plates are searched inside vehicle boxes.** A plate can be 60 px wide in a
1080p frame. Cropping the vehicle first and running the plate model on that crop
gives the detector many more pixels on target at the same input size.

**Why OCR does not run every frame.** OCR is the most expensive stage by an order
of magnitude. With tracking, one vehicle needs only a handful of readings across
its time on screen; those are fused into a better answer than any single frame
would give. `Frame skip` and `OCR interval` in the sidebar control this trade-off
directly.

**Why readings are aggregated over time.** Each frame is a noisy measurement of a
constant string, taken at a different distance, angle and blur level. The errors
are largely independent between frames while the correct characters are not, so
weighted voting across observations converges on the right answer. When no single
string wins, the aggregator votes position by position, which can reconstruct a
plate that no individual frame read perfectly.

**What the statuses mean**

- `VALID_FORMAT` — matches a known Indian registration layout with a recognised
  state code and adequate confidence.
- `SUSPICIOUS_FORMAT` — readable, but the structure does not fit cleanly. Worth a
  human look.
- `LOW_CONFIDENCE` — the recogniser itself was unsure; treat as unverified.
- `NO_TEXT` — a plate was located but no characters could be read.

**Limits worth stating plainly.** Validation is visual only — it says a string
looks like a registration number, never that a vehicle is registered, insured or
legal. Position-aware character correction is a heuristic and can introduce its
own errors, which is why every correction is recorded and shown. Accuracy drops
sharply with motion blur, heavy rain, night glare and plates below roughly 60 px
wide.
        """
    )
