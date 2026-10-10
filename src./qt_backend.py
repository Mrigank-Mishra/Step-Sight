"""Foothold Safety Assistant - Qt/QML backend bridge.

This module connects the glass QML interface (``src/web/*.qml``) to the same
processing pipeline the OpenCV UI uses (``src/pipeline.py``).

Design:
- A single QTimer drives frame acquisition and processing on the GUI thread.
  At 640px display width the heuristic engine has ample headroom for 10-15 FPS,
  which is the PRD target, and keeping it single-threaded means QML only ever
  touches already-computed state.
- Annotated frames are published through a QQuickImageProvider so QML can show
  them with a plain Image element. Python holds the only reference to the last
  frame, so the QImage stays alive until QML has uploaded its texture.

Exposed to QML as the context property ``backend``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Property, QObject, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtQml import QQmlImageProviderBase
from PySide6.QtQuick import QQuickImageProvider

try:
    from .input_source import InputSource
    from .pipeline import (
        DISPLAY_WIDTH,
        flag_frame_for_review,
        process_frame,
        resize_for_display,
        step_status,
    )
    from .postprocess import PipelineState
    from .risk_engine import RISK_RED, load_model
except (ImportError, ValueError):
    from input_source import InputSource
    from pipeline import (
        DISPLAY_WIDTH,
        flag_frame_for_review,
        process_frame,
        resize_for_display,
        step_status,
    )
    from postprocess import PipelineState
    from risk_engine import RISK_RED, load_model

# Project root (…/foothold), used to resolve data/ and models/ regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DATA_DIR = PROJECT_ROOT / "data"

# Display frame rate. The heuristic engine comfortably exceeds this; the model
# engine is heavier, so the timer simply runs slower in wall-clock terms.
TARGET_FPS = 30.0


def _to_local_path(url) -> str:
    """Normalise a QML file selection (QUrl, or already a path) to a local path."""
    if isinstance(url, QUrl):
        return url.toLocalFile() or url.toString()
    return str(url)


class FrameProvider(QQuickImageProvider):
    """Serves the most recently annotated frame to QML as ``image://frames/<id>``."""

    def __init__(self, backend: "Backend"):
        super().__init__(QQmlImageProviderBase.ImageType.Image)
        self._backend = backend

    def requestImage(self, image_id: str, size, requested_size):  # noqa: N802 (Qt API)
        image = self._backend.current_image()
        if image is None:
            image = QImage(2, 2, QImage.Format.Format_RGB888)
            image.fill(Qt.GlobalColor.black)
        if size is not None:
            size.setWidth(image.width())
            size.setHeight(image.height())
        return image


class Backend(QObject):
    """All application state and actions the QML layer can reach."""

    # --- Signals consumed by QML bindings ---
    frameChanged = Signal()
    statusChanged = Signal()
    messageChanged = Signal()
    engineChanged = Signal()
    sensitivityChanged = Signal()
    runningChanged = Signal()
    modeChanged = Signal()
    flagCountChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._source: Optional[InputSource] = None
        self._state: Optional[PipelineState] = None
        self._model = None

        self._frame_seq = 0
        self._last_image: Optional[QImage] = None
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_overlay_frame: Optional[np.ndarray] = None

        self._engine = "heuristic"
        self._sensitivity = 1.3
        self._running = False
        self._mode = "home"          # home | live | upload
        self._message = ""
        self._flag_count = 0

        # Which view to show at startup. Set from the CLI (--view) so the app can
        # open straight into a given screen, and so each view is testable.
        self._initial_view = "home"

        self._fps = 0.0
        self._source_label = ""
        self._source_kind = ""
        self._safe_to_step = True
        self._has_step = False

        # Static-image mode pauses the timer after the first processed frame.
        self._static_image_done = False

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / TARGET_FPS))
        self._timer.timeout.connect(self._on_tick)

        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(1500)
        self._settle_timer.timeout.connect(self._clear_message)

    # ------------------------------------------------------------------ #
    # Image plumbing
    # ------------------------------------------------------------------ #
    def current_image(self) -> Optional[QImage]:
        """Latest annotated frame (called by FrameProvider)."""
        return self._last_image

    def _publish(self, bgr: np.ndarray) -> None:
        """Convert a BGR frame to a QImage and notify QML."""
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)

        h, w = bgr.shape[:2]
        # Format_BGR888 maps 1:1 onto OpenCV's native channel order.
        image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888)
        # Copy so the QImage owns its buffer and does not alias the numpy array,
        # which is reused/released once this method returns.
        self._last_image = image.copy()
        self._last_overlay_frame = bgr

        self._frame_seq += 1
        self.frameChanged.emit()

    # ------------------------------------------------------------------ #
    # Read-only properties for QML
    # ------------------------------------------------------------------ #
    def _get_frame_seq(self) -> int:
        return self._frame_seq

    frameSeq = Property(int, _get_frame_seq, notify=frameChanged)

    def _get_engine(self) -> str:
        return self._engine

    engine = Property(str, _get_engine, notify=engineChanged)

    def _get_sensitivity(self) -> float:
        return self._sensitivity

    sensitivity = Property(float, _get_sensitivity, notify=sensitivityChanged)

    def _get_running(self) -> bool:
        return self._running

    running = Property(bool, _get_running, notify=runningChanged)

    def _get_mode(self) -> str:
        return self._mode

    mode = Property(str, _get_mode, notify=modeChanged)

    def _get_message(self) -> str:
        return self._message

    message = Property(str, _get_message, notify=messageChanged)

    def _get_flag_count(self) -> int:
        return self._flag_count

    flagCount = Property(int, _get_flag_count, notify=flagCountChanged)

    def _get_fps(self) -> float:
        return self._fps

    fps = Property(float, _get_fps, notify=statusChanged)

    def _get_source_label(self) -> str:
        return self._source_label

    sourceLabel = Property(str, _get_source_label, notify=statusChanged)

    def _get_source_kind(self) -> str:
        return self._source_kind

    sourceKind = Property(str, _get_source_kind, notify=statusChanged)

    def _get_safe_to_step(self) -> bool:
        return self._safe_to_step

    safeToStep = Property(bool, _get_safe_to_step, notify=statusChanged)

    def _get_has_step(self) -> bool:
        return self._has_step

    hasStep = Property(bool, _get_has_step, notify=statusChanged)

    def _get_initial_view(self) -> str:
        return self._initial_view

    initialView = Property(str, _get_initial_view, constant=True)

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #
    def _set_message(self, text: str, auto_clear: bool = True) -> None:
        self._message = text
        self.messageChanged.emit()
        if auto_clear and text:
            self._settle_timer.start()

    @Slot()
    def _clear_message(self) -> None:
        if self._message:
            self._message = ""
            self.messageChanged.emit()

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    @Slot(str)
    def setEngine(self, engine: str) -> None:  # noqa: N802 (QML-facing name)
        """Switch between the heuristic and trained-model risk engines."""
        engine = (engine or "").strip().lower()
        if engine not in ("heuristic", "model"):
            self._set_message(f"Unknown engine '{engine}'.")
            return
        if engine == self._engine:
            return

        if engine == "model":
            try:
                self._model = load_model(str(MODEL_PATH))
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self._set_message(f"Model unavailable: {exc}")
                return

        self._engine = engine
        self.engineChanged.emit()

        if self._state is not None:
            self._state.engine = engine
            self._state.model = self._model if engine == "model" else None
            self._state.reset_smoothing()

        self._set_message(f"Engine: {engine.upper()}")

    @Slot(float)
    def setSensitivity(self, value: float) -> None:  # noqa: N802
        value = float(min(max(value, 0.3), 3.0))
        if abs(value - self._sensitivity) < 1e-6:
            return
        self._sensitivity = value
        if self._state is not None:
            self._state.sensitivity = value
        self.sensitivityChanged.emit()

    @Slot()
    def increaseSensitivity(self) -> None:  # noqa: N802
        self.setSensitivity(round(self._sensitivity + 0.1, 2))

    @Slot()
    def decreaseSensitivity(self) -> None:  # noqa: N802
        self.setSensitivity(round(self._sensitivity - 0.1, 2))

    # ------------------------------------------------------------------ #
    # Source control
    # ------------------------------------------------------------------ #
    def _stop_source(self) -> None:
        self._timer.stop()
        if self._source is not None:
            self._source.release()
            self._source = None
        self._running = False
        self._runningChanged_emit()

    def _runningChanged_emit(self) -> None:
        self.runningChanged.emit()

    @Slot()
    def stop(self) -> None:
        """Return to the home screen and release the camera/file."""
        self._stop_source()
        self._mode = "home"
        self.modeChanged.emit()
        self.statusChanged.emit()

    @Slot(int)
    def startCamera(self, index: int) -> None:  # noqa: N802
        """Begin live processing from a webcam index."""
        self._start_source(str(int(index)), mode="live")

    @Slot()
    def startDefaultCamera(self) -> None:  # noqa: N802
        self.startCamera(0)

    @Slot(QUrl)
    def startFile(self, url) -> None:  # noqa: N802
        """Begin processing an image or video file chosen by the user.

        The QML FileDialog hands us a QUrl; the pipeline wants a filesystem path.
        """
        self._start_source(_to_local_path(url), mode="upload")

    def _start_source(self, source_val: str, mode: str) -> None:
        self._stop_source()
        self._clear_message()

        try:
            source = InputSource(source_val)
        except Exception as exc:  # noqa: BLE001 - shown in the UI banner
            self._mode = mode
            self.modeChanged.emit()
            self._set_message(str(exc).split("\n")[0], auto_clear=False)
            return

        self._source = source
        self._mode = mode
        self.modeChanged.emit()

        # A fresh state per source: smoothing history must not leak across runs.
        self._state = PipelineState(
            sensitivity=self._sensitivity,
            engine=self._engine,
            model=self._model if self._engine == "model" else None,
        )
        self._state.set_source(source.source_desc, is_static_image=source.is_image)

        self._source_label = Path(source.source_desc).name or source.source_desc
        self._source_kind = "Camera" if source.is_camera else ("Image" if source.is_image else "Video")
        self._static_image_done = False
        self._fps = 0.0
        self._safe_to_step = True
        self._has_step = False
        self._last_raw_frame = None
        self._last_overlay_frame = None

        self._running = True
        self.runningChanged.emit()
        self.statusChanged.emit()

        # A still image needs a single pass, not a 30 Hz loop.
        if source.is_image:
            self._on_tick()
        else:
            self._timer.start()

    # ------------------------------------------------------------------ #
    # Frame loop
    # ------------------------------------------------------------------ #
    def _on_tick(self) -> None:
        if self._source is None or self._state is None:
            self._timer.stop()
            return

        ok, frame = self._source.read()
        if not ok or frame is None:
            # Video ended (images never reach here - they are handled above).
            self._timer.stop()
            self._running = False
            self.runningChanged.emit()
            if self._source.is_video:
                self._set_message("Video finished.", auto_clear=False)
            self.statusChanged.emit()
            return

        try:
            display_frame = resize_for_display(frame, DISPLAY_WIDTH)
            annotated = process_frame(display_frame, self._state)
        except Exception as exc:  # noqa: BLE001 - keep the UI alive and report
            self._timer.stop()
            self._running = False
            self.runningChanged.emit()
            self._set_message(f"Processing error: {exc}", auto_clear=False)
            return

        self._last_raw_frame = display_frame
        self._publish(annotated)

        # Status for the QML overlay.
        # FPS is only meaningful for a live/video stream; the pipeline already
        # reports 0 for stills, and the chip hides itself below.
        self._fps = self._state.fps if not self._source.is_image else 0.0
        step_info: Optional[Tuple[int, int, float]] = self._state.last_step
        self._has_step = step_info is not None
        state = step_status(step_info, RISK_RED)
        self._safe_to_step = state != "stop"
        self.statusChanged.emit()

        if self._source.is_image and not self._static_image_done:
            self._static_image_done = True
            self._running = False
            self.runningChanged.emit()
            self._set_message("Image analysed.", auto_clear=False)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    @Slot(result=bool)
    def flagCurrent(self) -> bool:  # noqa: N802
        """Save the current raw + overlay frame for human review (PRD F5)."""
        if self._last_raw_frame is None or self._last_overlay_frame is None:
            self._set_message("Nothing to flag yet.")
            return False

        source_type = "live" if (self._source and self._source.is_camera) else "upload"
        try:
            flag_frame_for_review(
                raw_frame=self._last_raw_frame,
                overlay_frame=self._last_overlay_frame,
                source_type=source_type,
                data_dir=DATA_DIR,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_message(f"Flag failed: {exc}")
            return False

        self._flag_count += 1
        self.flagCountChanged.emit()
        self._set_message("Flagged as wrong - saved to data/review/")
        return True

    @Slot(result=str)
    def snapshot(self) -> str:
        """Save the current annotated frame and return the written path."""
        if self._last_overlay_frame is None:
            self._set_message("Nothing to save yet.")
            return ""

        out_dir = DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"snapshot_{int(time.time())}.png"
        if not cv2.imwrite(str(path), self._last_overlay_frame):
            self._set_message("Could not write snapshot.")
            return ""
        self._set_message(f"Snapshot saved: {path.name}")
        return str(path)
