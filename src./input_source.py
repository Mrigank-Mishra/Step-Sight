"""Foothold Safety Assistant - Unified Input Source.

Provides a single interface for reading frames from:
1. Live camera (webcam index: 0, 1, ...)
2. Network stream (phone camera over RTSP/HTTP, e.g. an iPhone running an
   RTSP server app)
3. Recorded video file (.mp4, .avi, .mov, etc.)
4. Single image file (.jpg, .png, etc.)
"""
import socket
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlsplit

import cv2
import numpy as np

# URL schemes OpenCV's FFMPEG backend can open directly. This is what lets a
# phone camera be used as a live source instead of a USB webcam.
STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://",
                  "udp://", "tcp://", "srt://")

DEFAULT_STREAM_PORT = {"rtsp": 554, "rtsps": 554, "rtmp": 1935, "http": 80,
                       "https": 443, "srt": 9000}


def _preflight(url: str, timeout: float = 3.0) -> Optional[str]:
    """Quickly check the host:port behind a stream URL before handing it to OpenCV.

    OpenCV/FFMPEG gives up only after its own ~30s timeout, so a single typo in an
    IP address costs half a minute of frozen UI. A 3-second TCP probe turns that
    into an instant, actionable error. Returns None when reachable, else a reason.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if not host:
            return None  # unusual URL; let OpenCV try
        port = parts.port
        if port is None:
            scheme = (parts.scheme or "").lower()
            if scheme == "rtsp":
                # A missing /live path is a common mistake; the port is still 554.
                port = DEFAULT_STREAM_PORT.get(scheme, 554)
            else:
                port = DEFAULT_STREAM_PORT.get(scheme)
        if port is None:
            return None

        with socket.create_connection((host, port), timeout=timeout):
            return None
    except socket.timeout:
        return f"No response from {host}:{port} within {timeout:.0f}s"
    except OSError as exc:
        return f"Cannot reach {host}:{port} ({exc.strerror or exc})"



class InputSource:
    """Unified reader for camera feeds, video files, and static images."""

    def __init__(self, source: Union[int, str] = 0, loop_image: bool = True):
        """Initialize the input source.

        Args:
            source: Camera index (int or digit string like '0') or file path.
            loop_image: If True and input is an image, read() will continually
                        return the image until released (ideal for interactive UI).
        """
        self.source_desc = str(source).strip()
        self.loop_image = loop_image
        self.cap: Optional[cv2.VideoCapture] = None
        self.image_frame: Optional[np.ndarray] = None
        self.is_image: bool = False
        self.is_camera: bool = False
        self.is_video: bool = False
        self.is_stream: bool = False
        self._has_read_image: bool = False
        # Consecutive failed reads tolerated before a network stream is declared
        # dead. Live streams drop frames; files never do.
        self._stream_misses: int = 0
        self.max_stream_misses: int = 30

        self._open()

    def _open(self) -> None:
        """Open the camera, video, or image file with clear error handling."""
        # Case 1: Webcam index (e.g. 0, 1, or '0', '1')
        if isinstance(self.source_desc, int) or self.source_desc.isdigit():
            cam_idx = int(self.source_desc)
            self.is_camera = True
            # Try DirectShow first on Windows (faster initialization, less lag)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera index {cam_idx}.\n"
                    "Helpful troubleshooting steps:\n"
                    "  1. Try another camera index (e.g. --source 1).\n"
                    "  2. Close any apps currently using the webcam (Zoom, Teams, Camera app).\n"
                    "  3. Check Windows Settings -> Privacy & security -> Camera permissions."
                )
            return

        # Case 2: Network stream (phone camera over RTSP/HTTP, IP camera, ...).
        # Must come before the file-path branch, otherwise Path() treats the URL
        # as a relative filename and raises FileNotFoundError.
        if self.source_desc.lower().startswith(STREAM_SCHEMES):
            self.is_stream = True
            self.is_video = True

            # Fail fast on an unreachable host instead of waiting ~30s for FFMPEG.
            reason = _preflight(self.source_desc)
            if reason:
                raise RuntimeError(
                    f"Could not reach the network stream.\n  {reason}\n"
                    f"  URL: {self.source_desc}\n"
                    "Helpful troubleshooting steps:\n"
                    "  1. The phone and this PC must be on the SAME Wi-Fi network.\n"
                    "  2. Start the RTSP/camera server app on the phone first, then retry.\n"
                    "  3. Double-check the IP, port and path (e.g. rtsp://192.168.1.23:8554/live).\n"
                    "  4. Allow the app through Windows Firewall when prompted.\n"
                    "  5. Try the stream in VLC first - if VLC cannot open it, neither can we."
                )

            # Force FFMPEG: it is the only backend that speaks RTSP/HTTP here.
            self.cap = cv2.VideoCapture(self.source_desc, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source_desc)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open network stream:\n  {self.source_desc}\n"
                    "The host answered, but no video came back.\n"
                    "  1. Confirm the stream path (many phone apps use /live or /video).\n"
                    "  2. Confirm the phone app is still broadcasting.\n"
                    "  3. Try the stream in VLC first - if VLC cannot open it, neither can we."
                )

            # Keep only the newest frame so the view stays live instead of
            # replaying a lagging backlog. Worth more than resolution here.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return

        # Case 3: File path (Image or Video)
        path = Path(self.source_desc)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path.resolve()}")

        ext = path.suffix.lower()
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

        if ext in image_extensions:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Could not decode image file: {path.resolve()}")
            self.is_image = True
            self.image_frame = img
        else:
            self.is_video = True
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open video file: {path.resolve()}.\n"
                    "Please check that the video file is valid and readable."
                )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame.

        Returns:
            (ok, frame): ok is True if frame was successfully read, False otherwise.
        """
        if self.is_image:
            if self._has_read_image and not self.loop_image:
                return False, None
            self._has_read_image = True
            return True, self.image_frame.copy()

        if self.cap is not None:
            ok, frame = self.cap.read()

            if not self.is_stream:
                return ok, frame

            # A live stream drops frames and briefly stalls, unlike a video file
            # that is genuinely over. Retry briefly rather than reporting the
            # stream as finished, which would flash "video ended" on every hiccup.
            if ok and frame is not None:
                self._stream_misses = 0
                return True, frame

            for _ in range(5):
                time.sleep(0.02)
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self._stream_misses = 0
                    return True, frame

            self._stream_misses += 1
            return False, None

        return False, None

    @property
    def fps(self) -> float:
        """Get video source frames per second (defaults to 20.0 for camera/images)."""
        if self.is_video and self.cap is not None:
            val = self.cap.get(cv2.CAP_PROP_FPS)
            if val > 0:
                return val
        return 20.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get source (width, height) dimensions."""
        if self.is_image and self.image_frame is not None:
            h, w = self.image_frame.shape[:2]
            return w, h
        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return 640, 480

    def release(self) -> None:
        """Release underlying video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
