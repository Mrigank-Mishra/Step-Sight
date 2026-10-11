"""Find which camera index your virtual camera (Camo, OBS, etc.) landed on.

OpenCV cannot list camera *names*, only try numeric indices. This probes a range
and reports what actually opens and delivers a frame, so you can pass the right
`--source <index>` to the app.

Why a grab test and not just isOpened(): virtual cameras (especially Camo) often
report themselves as open before any video is flowing. Reading a real frame is
the only reliable check.

Usage:
    python tools/list_cameras.py
    python tools/list_cameras.py --max 8
    python tools/list_cameras.py --save        # writes a frame per working camera
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# DirectShow is the backend virtual cameras register with on Windows.
BACKENDS = [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("Any", cv2.CAP_ANY)]


def probe(index: int, backend_flag: int, label: str, tries: int = 12, delay: float = 0.15):
    """Try to open an index and read one real frame. Returns (frame, w, h) or None."""
    cap = cv2.VideoCapture(index, backend_flag)
    if not cap.isOpened():
        cap.release()
        return None

    frame = None
    for _ in range(tries):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            frame = f
            break
        time.sleep(delay)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame is None:
        return None
    return frame, frame.shape[1], frame.shape[0]


def wait_for_video(index: int, seconds: float, backend_flag: int = cv2.CAP_DSHOW):
    """Give one index a longer grace period to start delivering frames.

    Virtual cameras (Camo, OBS) answer VideoCapture() immediately but may take
    several seconds before real video arrives, so a short probe reports them as
    blank.
    """
    cap = cv2.VideoCapture(index, backend_flag)
    if not cap.isOpened():
        print(f"  index {index}: could not open")
        return None

    print(f"  waiting up to {seconds:.0f}s for index {index} to deliver video...")
    deadline = time.time() + seconds
    attempt = 0
    while time.time() < deadline:
        ok, f = cap.read()
        attempt += 1
        if ok and f is not None and f.size > 0:
            blank = bool(np.allclose(f, f[0, 0], atol=2))
            w, h = f.shape[1], f.shape[0]
            if blank:
                print(f"    attempt {attempt}: frame received but BLANK {w}x{h} (still starting)")
                time.sleep(0.4)
                continue
            cap.release()
            print(f"    attempt {attempt}: GOT VIDEO {w}x{h}")
            return f, w, h
        time.sleep(0.3)

    cap.release()
    print(f"  index {index}: no usable video within {seconds:.0f}s")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="List cameras OpenCV can actually use.")
    ap.add_argument("--max", type=int, default=6, help="highest index to try (default 6)")
    ap.add_argument("--save", action="store_true", help="save a test frame per working camera")
    ap.add_argument("--index", type=int, default=None,
                    help="skip the sweep and wait on this index only (use for Camo/OBS)")
    ap.add_argument("--wait", type=float, default=15.0,
                    help="seconds to wait for one index with --index (default 15)")
    args = ap.parse_args()

    out_dir = Path("_camera_probe")
    if args.save:
        out_dir.mkdir(exist_ok=True)

    # Focused mode: virtual cameras need time, so wait on one index at length.
    if args.index is not None:
        print()
        print(f"Waiting on camera index {args.index}.")
        print("=" * 72)
        got = wait_for_video(args.index, args.wait)
        print("=" * 72)
        if not got:
            print()
            print("Still no video. Check, in order:")
            print("  1. Camo Studio (PC) is open AND shows the iPad connected with a live preview.")
            print("  2. The Camo app on the iPad is open and its screen is ON.")
            print("  3. Windows: Settings > Privacy & security > Camera -> both toggles ON.")
            print("  4. Camo Studio's own preview works - if it does not, the app has no video to give.")
            return 1
        frame, w, h = got
        if args.save:
            p = out_dir / f"camera_{args.index}.png"
            cv2.imwrite(str(p), frame)
            print(f"  saved {p}")
        print()
        print(f"Camera {args.index} works ({w}x{h}). Now run:")
        print(f'  & "C:\\Users\\mriga\\.venv\\Scripts\\python.exe" src\\qt_app.py --source {args.index}')
        return 0

    print()
    print("Probing camera indices. A camera only counts if it delivers a real frame.")
    print("(Virtual cameras like Camo need their app RUNNING and the iPad connected.)")
    print("Tip: if a camera opens but looks BLANK, give it longer with:")
    print("     python tools\\list_cameras.py --index <N> --wait 20")
    print("=" * 72)

    found: list[tuple[int, str, int, int]] = []

    for i in range(args.max + 1):
        result = None
        used = ""
        for label, flag in BACKENDS:
            result = probe(i, flag, label)
            if result:
                used = label
                break

        if result:
            frame, w, h = result
            # A completely black or single-colour frame usually means the virtual
            # camera is open but not yet receiving video from the device.
            blank = bool(np.allclose(frame, frame[0, 0], atol=2))
            note = "  <-- looks BLANK (device not sending video yet?)" if blank else ""
            print(f"  index {i}: OPEN via {used:<10} {w}x{h}{note}")
            found.append((i, used, w, h))
            if args.save:
                p = out_dir / f"camera_{i}.png"
                cv2.imwrite(str(p), frame)
                print(f"            saved {p}")
        else:
            print(f"  index {i}: no frame")

    print("=" * 72)

    if not found:
        print()
        print("No camera delivered a frame.")
        print()
        print("Most likely causes, in order:")
        print("  1. Camo Studio (PC app) is not running, or the iPad is not connected in it.")
        print("  2. Windows camera privacy: Settings > Privacy & security > Camera")
        print("     -> 'Camera access' ON and 'Let desktop apps access your camera' ON.")
        print("  3. This PC has no physical webcam, and no virtual camera is active.")
        return 1

    print()
    print("Working cameras:")
    for i, used, w, h in found:
        print(f"  --source {i}    ({used}, {w}x{h})")
    print()
    print("Run the app with one of those, for example:")
    print(f'  & "C:\\Users\\mriga\\.venv\\Scripts\\python.exe" src\\qt_app.py --source {found[0][0]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
