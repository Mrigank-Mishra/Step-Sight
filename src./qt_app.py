"""Foothold Safety Assistant - Qt/QML entry point (glass UI).

This is the primary front-end. It drives exactly the same pipeline as the
OpenCV desktop UI (``src/app.py``) via the shared module ``src/pipeline.py``.

Run:
    python src/qt_app.py                     # glass home screen
    python src/qt_app.py --view live         # straight to the camera
    python src/qt_app.py --view upload --source data/sample_trail.jpg
    python src/qt_app.py --engine model      # start on the trained YOLOv8 model

The OpenCV UI remains available as a dependency-light fallback:
    python src/app.py --source 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Qt Quick's software/OpenGL scene graph is more predictable than the default
# RHI backend on Windows machines with mixed GPU drivers. Must be set before
# the Qt platform plugins initialise.
os.environ.setdefault("QSG_RHI_BACKEND", "opengl")

# Allow running as `python src/qt_app.py` (script dir on the path) as well as
# `python -m src.qt_app`, without requiring a package install.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from PySide6.QtCore import QUrl  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine  # noqa: E402
from PySide6.QtQuickControls2 import QQuickStyle  # noqa: E402

from qt_backend import Backend, FrameProvider  # noqa: E402
from input_source import STREAM_SCHEMES  # noqa: E402

# QML lives beside the Python modules so the app works from any working dir.
QML_DIR = _THIS_DIR / "web"
QML_MAIN = QML_DIR / "Main.qml"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Foothold Safety Assistant - Qt/QML glass UI."
    )
    parser.add_argument(
        "--view",
        choices=["home", "live", "upload"],
        default="home",
        help="View to open at startup (default: home).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index or media file to load immediately. Implies --view "
             "upload for a file, or --view live for a camera index.",
    )
    parser.add_argument(
        "--engine",
        choices=["heuristic", "model"],
        default="heuristic",
        help="Risk engine to start with (default: heuristic).",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.3,
        help="Initial risk sensitivity multiplier (default: 1.3).",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    # 'Basic' keeps the slider and dialog chrome consistent with the glass theme
    # instead of pulling in a platform-specific native style.
    QQuickStyle.setStyle("Basic")

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Foothold Safety Assistant")
    app.setOrganizationName("Foothold")

    if not QML_MAIN.exists():
        print(f"[ERROR] QML entry point not found: {QML_MAIN}", file=sys.stderr)
        return 2

    # QGuiApplication is enough for a Qt Quick UI; the QML FileDialog uses the
    # Qt Quick dialogs implementation, so no QApplication/QtWidgets is needed.
    backend = Backend()
    backend.setSensitivity(args.sensitivity)
    if args.engine != "heuristic":
        backend.setEngine(args.engine)

    # --source overrides --view: a camera index or network URL means live,
    # a filesystem path means upload.
    view = args.view
    source = args.source
    is_url = source is not None and str(source).strip().lower().startswith(STREAM_SCHEMES)
    if source is not None:
        view = "live" if (str(source).strip().isdigit() or is_url) else "upload"
    backend._initial_view = view

    engine = QQmlApplicationEngine()
    engine.addImageProvider("frames", FrameProvider(backend))
    engine.rootContext().setContextProperty("backend", backend)

    engine.load(QUrl.fromLocalFile(str(QML_MAIN)))

    if not engine.rootObjects():
        print(
            "[ERROR] QML failed to load. Run with QT_LOGGING_RULES='qt.qml.*=true' "
            "for details.",
            file=sys.stderr,
        )
        return 1

    # A pre-selected source is applied once QML has finished wiring up.
    if source is not None:
        from PySide6.QtCore import QTimer

        if is_url:
            # Network stream: a phone camera or IP camera. Not a Path.
            url = str(source).strip()
            QTimer.singleShot(0, lambda: backend.startFile(url))
        elif view == "live":
            QTimer.singleShot(0, lambda: backend.startCamera(int(source)))
        else:
            resolved = Path(source)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            QTimer.singleShot(0, lambda: backend.startFile(str(resolved)))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
