import cv2
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot, Property, QThread, QTimer, Qt, QMutex, QMutexLocker
from PySide6.QtGui import QImage, QColor
from PySide6.QtQuick import QQuickImageProvider

from .input_source import InputSource
from .pipeline import process_frame, resize_for_display, flag_frame_for_review, DISPLAY_WIDTH
from .postprocess import PipelineState


class FrameProvider(QQuickImageProvider):
    """Provides OpenCV frames to QML via image://frames/"""
    def __init__(self):
        super().__init__(QQuickImageProvider.Image)
        self._image = QImage(DISPLAY_WIDTH, int(DISPLAY_WIDTH * 0.75), QImage.Format_RGB888)
        self._image.fill(QColor("black"))
        self._mutex = QMutex()

    def update_frame(self, frame_rgb):
        """Update the internal QImage with a new RGB frame."""
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        # QImage needs to be copied since the memory of frame_rgb might be overwritten
        new_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        with QMutexLocker(self._mutex):
            self._image = new_img

    def requestImage(self, id, size, requestedSize):
        with QMutexLocker(self._mutex):
            return self._image


class ProcessingThread(QThread):
    """Background thread to read and process frames without blocking the GUI."""
    frameReady = Signal(object, object) # frame_rgb, state_dict
    errorOccurred = Signal(str)
    finished_source = Signal()

    def __init__(self, source_val, engine="heuristic", sensitivity=1.3):
        super().__init__()
        self.source_val = source_val
        self.engine_type = engine
        self.sensitivity = sensitivity
        self.running = False
        
        self.state = PipelineState(sensitivity=self.sensitivity, engine=self.engine_type)
        self.source = None
        self.out_dir = Path("data")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_display_frame = None
        self.last_annotated = None
        self.flag_requested = False
        self.snapshot_requested = False

    def run(self):
        self.running = True
        try:
            self.source = InputSource(self.source_val)
        except Exception as e:
            self.errorOccurred.emit(f"Could not open camera/video: {e}")
            return

        self.state.set_source(self.source.source_desc, is_static_image=self.source.is_image)
        
        source_label = self.source.source_desc
        if self.source.is_camera:
            source_label = f"Camera {self.source_val}"
            
        frames_processed = 0
        start_time = time.time()

        while self.running:
            ok, frame = self.source.read()
            if not ok or frame is None:
                if self.source.is_video:
                    self.finished_source.emit()
                    # Replay logic for video
                    self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.state.reset_smoothing()
                    continue
                break
            
            frames_processed += 1

            display_frame = resize_for_display(frame, DISPLAY_WIDTH)
            annotated = process_frame(display_frame, self.state)
            
            self.last_display_frame = display_frame
            self.last_annotated = annotated
            
            if self.flag_requested:
                flag_frame_for_review(
                    raw_frame=display_frame,
                    overlay_frame=annotated,
                    source_type="live" if self.source.is_camera else "upload",
                    data_dir=self.out_dir,
                )
                self.flag_requested = False
                
            if self.snapshot_requested:
                snapshot_name = str(self.out_dir / f"snapshot_{int(time.time())}.png")
                cv2.imwrite(snapshot_name, annotated)
                self.snapshot_requested = False

            # Convert to RGB for Qt
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            elapsed = time.time() - start_time
            fps = frames_processed / elapsed if elapsed > 0 else 0.0

            has_step = self.state.last_step is not None
            safe_to_step = has_step and self.state.last_step[2] < 0.6  # arbitrary safe threshold

            # Package state info to send to UI
            state_info = {
                "hasStep": has_step,
                "safeToStep": safe_to_step,
                "fps": fps,
                "sourceLabel": source_label,
            }
            
            self.frameReady.emit(annotated_rgb, state_info)
            
            if self.source.is_image:
                # Still image, wait until stopped
                while self.running:
                    self.msleep(100)
            else:
                # Video/camera delay
                delay = 30 if self.source.is_video else 1
                self.msleep(delay)

        self.source.release()

    def stop(self):
        self.running = False


class PipelineBackend(QObject):
    """The QML backend bridge."""
    frameSeqChanged = Signal()
    hasStepChanged = Signal()
    safeToStepChanged = Signal()
    fpsChanged = Signal()
    sourceLabelChanged = Signal()
    messageChanged = Signal()
    flagCountChanged = Signal()
    sensitivityChanged = Signal()
    engineChanged = Signal()

    def __init__(self, provider: FrameProvider, parent=None):
        super().__init__(parent)
        self.provider = provider
        
        self._frameSeq = 0
        self._hasStep = False
        self._safeToStep = False
        self._fps = 0.0
        self._sourceLabel = ""
        self._message = ""
        self._flagCount = 0
        self._sensitivity = 1.3
        self._engine = "heuristic"
        
        self.thread: Optional[ProcessingThread] = None

    # --- QML Properties ---

    @Property(int, notify=frameSeqChanged)
    def frameSeq(self):
        return self._frameSeq

    @Property(bool, notify=hasStepChanged)
    def hasStep(self):
        return self._hasStep

    @Property(bool, notify=safeToStepChanged)
    def safeToStep(self):
        return self._safeToStep

    @Property(float, notify=fpsChanged)
    def fps(self):
        return self._fps

    @Property(str, notify=sourceLabelChanged)
    def sourceLabel(self):
        return self._sourceLabel

    @Property(str, notify=messageChanged)
    def message(self):
        return self._message
        
    @Property(int, notify=flagCountChanged)
    def flagCount(self):
        return self._flagCount

    @Property(float, notify=sensitivityChanged)
    def sensitivity(self):
        return self._sensitivity

    @Property(str, notify=engineChanged)
    def engine(self):
        return self._engine

    # --- QML Slots ---
    
    @Slot()
    def stop(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.thread = None
        self._fps = 0.0
        self.fpsChanged.emit()
        
    @Slot()
    def flagCurrent(self):
        if self.thread:
            self.thread.flag_requested = True
            self._flagCount += 1
            self.flagCountChanged.emit()
            self._message = "Frame flagged for review"
            self.messageChanged.emit()
            
    @Slot()
    def snapshot(self):
        if self.thread:
            self.thread.snapshot_requested = True
            self._message = "Snapshot saved"
            self.messageChanged.emit()

    @Slot(float)
    def setSensitivity(self, value: float):
        self._sensitivity = max(0.3, min(3.0, value))
        self.sensitivityChanged.emit()
        if self.thread:
            self.thread.state.sensitivity = self._sensitivity
            
    @Slot()
    def decreaseSensitivity(self):
        self.setSensitivity(self._sensitivity - 0.1)

    @Slot()
    def increaseSensitivity(self):
        self.setSensitivity(self._sensitivity + 0.1)

    @Slot(str)
    def setEngine(self, engine: str):
        if engine in ("heuristic", "model"):
            self._engine = engine
            self.engineChanged.emit()
            if self.thread:
                self.thread.state.engine = engine
                self.thread.state.reset_smoothing()

    def start_pipeline(self, source_val):
        self.stop()
        self._message = ""
        self.messageChanged.emit()
        self._flagCount = 0
        self.flagCountChanged.emit()
        
        self.thread = ProcessingThread(source_val, self._engine, self._sensitivity)
        self.thread.frameReady.connect(self._on_frame_ready)
        self.thread.errorOccurred.connect(self._on_error)
        self.thread.finished_source.connect(self._on_finished_source)
        self.thread.start()

    @Slot(object, object)
    def _on_frame_ready(self, frame_rgb, state_info):
        self.provider.update_frame(frame_rgb)
        
        self._hasStep = state_info["hasStep"]
        self._safeToStep = state_info["safeToStep"]
        self._fps = state_info["fps"]
        self._sourceLabel = state_info["sourceLabel"]
        self._frameSeq += 1
        
        self.hasStepChanged.emit()
        self.safeToStepChanged.emit()
        self.fpsChanged.emit()
        self.sourceLabelChanged.emit()
        self.frameSeqChanged.emit()

    @Slot(str)
    def _on_error(self, err_msg):
        self._message = err_msg
        self.messageChanged.emit()

    @Slot()
    def _on_finished_source(self):
        self._message = "End of video, replaying..."
        self.messageChanged.emit()
