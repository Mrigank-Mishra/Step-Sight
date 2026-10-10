import sys
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("QSG_RHI_BACKEND", "opengl")

from PySide6.QtCore import QUrl, QObject, Slot, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

# Add parent directory to path so imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui_backend import PipelineBackend, FrameProvider

def resource_path(rel: str) -> Path:
    """Works both in dev and inside a PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
        # We added data as src/web, so it's under base / 'src'
        return base / "src" / rel
    else:
        base = Path(__file__).parent
        return base / rel


def open_file_dialog() -> Optional[str]:
    """Open a native Windows file browser dialog to select an image or video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filetypes = [
            ("All Supported Media", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.mp4;*.avi;*.mov;*.mkv"),
            ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
            ("Videos", "*.mp4;*.avi;*.mov;*.mkv"),
            ("All Files", "*.*"),
        ]
        selected_path = filedialog.askopenfilename(title="Foothold - Select Terrain Image or Video", filetypes=filetypes)
        root.destroy()
        if selected_path and str(selected_path).strip():
            return str(selected_path).strip()
        return None
    except Exception as e:
        print(f"[WARNING] Could not open graphical file dialog: {e}")
        return None


def get_camera_index() -> Optional[int]:
    """Open a dialog to select the camera index."""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        cam_idx = simpledialog.askinteger(
            "Select Camera", 
            "Enter Camera Index:\n(0 = Default Webcam, 1 = External USB, etc.)", 
            initialvalue=0, minvalue=0, maxvalue=10
        )
        root.destroy()
        return cam_idx
    except Exception as e:
        print(f"[WARNING] Could not open dialog: {e}")
        return 0


class Launcher(QObject):
    uploadFileSelected = Signal(str)
    
    def __init__(self, backend: PipelineBackend, parent=None):
        super().__init__(parent)
        self.backend = backend

    @Slot()
    def startLive(self):
        print("[Launcher] Live mode requested")
        cam_idx = get_camera_index()
        if cam_idx is not None:
            self.backend.start_pipeline(cam_idx)
        else:
            # User cancelled, but UI transitions anyway. Just start default so it doesn't hang blank.
            self.backend.start_pipeline(0)

    @Slot()
    def startUpload(self):
        print("[Launcher] Upload mode requested")
        path = open_file_dialog()
        if path:
            self.uploadFileSelected.emit(path)
            self.backend.start_pipeline(path)


def main():
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication(sys.argv)

    engine = QQmlApplicationEngine()
    
    # Register Image Provider and Backend Context Property
    provider = FrameProvider()
    engine.addImageProvider("frames", provider)
    
    backend = PipelineBackend(provider)
    engine.rootContext().setContextProperty("backend", backend)
    
    launcher = Launcher(backend)
    engine.rootContext().setContextProperty("launcher", launcher)

    qml_file = resource_path("web/Main.qml")
    engine.load(QUrl.fromLocalFile(str(qml_file)))

    if not engine.rootObjects():
        sys.exit(-1)
        
    exit_code = app.exec()
    backend.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
