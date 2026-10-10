"""Foothold Safety Assistant - Shared frame pipeline.

This module is the single source of truth for turning a raw BGR frame into an
annotated frame. It exists so that the OpenCV desktop UI (``src/app.py``) and the
Qt/QML glass UI (``src/qt_app.py``) drive *exactly* the same processing code.

Pipeline order (matches docs/architecture.md section 4):
    1. compute_risk_grid()  -> 12x16 risk grid
    2. temporal smoothing   -> reduces flicker
    3. pick_next_step()     -> safest nearby footing patch
    4. draw_overlay()       -> heatmap + STEP HERE marker + safety banner
"""
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

try:
    from .overlay import draw_overlay
    from .postprocess import PipelineState, pick_next_step
    from .risk_engine import compute_risk_grid
except (ImportError, ValueError):
    from overlay import draw_overlay
    from postprocess import PipelineState, pick_next_step
    from risk_engine import compute_risk_grid

# Display width used by every UI. Kept here so both front-ends agree.
DISPLAY_WIDTH = 640


def resize_for_display(frame: np.ndarray, target_width: int = DISPLAY_WIDTH) -> np.ndarray:
    """Downscale a frame to the display width, preserving aspect ratio.

    The pipeline sizes its internal work off the incoming frame, so both UIs
    normalise here before calling process_frame().
    """
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    disp_h = int(target_width * h / max(w, 1))
    return cv2.resize(frame, (target_width, max(disp_h, 1)))


def process_frame(
    frame: np.ndarray,
    state: PipelineState,
    model=None,
) -> np.ndarray:
    """Run the full risk pipeline on one frame and return the annotated frame.

    Args:
        frame: BGR frame (already resized for display).
        state: PipelineState holding settings and temporal history.
        model: Optional override for the loaded model. If omitted, ``state.model``
            is used, which is what the Qt bridge populates when the user switches
            to the 'model' engine.

    Returns:
        Annotated BGR frame ready for display or storage.
    """
    state.update_fps()

    # 1. Compute terrain risk grid
    raw_grid = compute_risk_grid(
        frame,
        engine=state.engine,
        sensitivity=state.sensitivity,
        model=model if model is not None else state.model,
    )
    state.last_raw_grid = raw_grid

    # 2. Temporal smoothing (bypassed automatically for static images)
    smooth_grid = state.update_smoothing(raw_grid)

    # 3. Foothold selection in the lower-middle zone
    step_info: Tuple[int, int, float] = pick_next_step(smooth_grid)
    state.last_step = step_info

    # 4. Composite visual overlay.
    # FPS is wall-clock derived, so it is meaningless for a one-shot still image
    # (it would report ~1e6). Suppress it there rather than print nonsense.
    display_fps = 0.0 if state.is_static_image else state.fps
    annotated = draw_overlay(
        frame,
        smooth_grid,
        fps=display_fps,
        sensitivity=state.sensitivity,
        step_info=step_info,
    )
    state.last_overlay = annotated

    return annotated


def step_status(step_info: Optional[Tuple[int, int, float]], risk_red: float = 0.60) -> str:
    """Human-readable footing status for a UI banner.

    Returns 'safe' when a usable footing was found, 'stop' when every nearby
    patch is risky, or 'unknown' if no step has been computed yet.
    """
    if step_info is None:
        return "unknown"
    return "safe" if step_info[2] < risk_red else "stop"


def flag_frame_for_review(
    raw_frame: np.ndarray,
    overlay_frame: np.ndarray,
    source_type: str = "live",
    data_dir: Union[str, Path] = "data",
    location: str = "trail",
    weather: str = "unknown",
) -> str:
    """Save raw and overlay frames to data/review/ and append to metadata.csv.

    Args:
        raw_frame: Unmodified input frame (essential for future retraining).
        overlay_frame: Frame with visual heatmap and foothold suggestions.
        source_type: 'live' or 'upload'.
        data_dir: Path to the data directory.
        location: Name or ID of trail location.
        weather: Weather condition descriptor.

    Returns:
        Generated unique flag ID string.
    """
    data_path = Path(data_dir)
    review_dir = data_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    flag_id = f"flag_{timestamp_str}"
    date_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Save raw frame and overlay frame
    raw_path = review_dir / f"{flag_id}_raw.png"
    overlay_path = review_dir / f"{flag_id}_overlay.png"

    if not cv2.imwrite(str(raw_path), raw_frame):
        raise IOError(f"Could not write {raw_path}")
    if not cv2.imwrite(str(overlay_path), overlay_frame):
        raise IOError(f"Could not write {overlay_path}")

    # 2. Append row to metadata.csv
    # Format: id,source,date,weather,location,label,reviewed
    meta_path = data_path / "metadata.csv"
    file_exists = meta_path.exists() and meta_path.stat().st_size > 0

    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "source", "date", "weather", "location", "label", "reviewed"])
        writer.writerow([flag_id, source_type, date_formatted, weather, location, "flagged", "no"])

    print(f"\n[FLAGGED] Saved to data/review/ as '{flag_id}_raw.png' and '{flag_id}_overlay.png'")
    print(f"[FLAGGED] Appended row to {meta_path} (source: {source_type}, reviewed: no)")

    return flag_id
