"""Foothold Safety Assistant - video/frame helper.

Public terrain datasets ship as *frame sequences* (PNG/JPG folders), not video
files. The app's video path needs an actual container, and the app's frame path
needs frames. This tool bridges both directions:

  frames -> mp4   : build a test video from a dataset's PNG/JPG sequence
  mp4    -> frames: extract 1 FPS frames for labeling (matches training/extract_frames.py)
  info            : report resolution, FPS and frame count of a video

Why this exists: RELLIS-3D and SANPO deliver per-frame images. Feeding those
folders to app.py --source is not possible, and the pipeline's temporal
smoothing / FPS behaviour only shows up on a real stream.

Usage:
    python tools/make_test_video.py info data/sample_trail.mp4
    python tools/make_test_video.py frames-to-video <frame_dir> -o out.mp4 --fps 10
    python tools/make_test_video.py frames-to-video <frame_dir> -o out.mp4 --limit 600 --resize 640
    python tools/make_test_video.py video-to-frames data/sample_trail.mp4 -o data/raw/demo
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(path: Path):
    """Sort frame_2 before frame_10 (plain string sort gets this wrong)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def collect_frames(frame_dir: Path, limit: int | None) -> list[Path]:
    files = [p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXT]
    if not files:
        # Datasets often nest one level deeper (e.g. <session>/image_data/*.png).
        files = [p for p in frame_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT]
    files.sort(key=natural_key)
    return files[:limit] if limit else files


def cmd_info(args) -> int:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"file       : {args.video}")
    print(f"resolution : {w}x{h}")
    print(f"fps        : {fps:.2f}")
    print(f"frames     : {n}")
    print(f"duration   : {(n / fps if fps else 0):.1f} s")
    print(f"size       : {args.video.stat().st_size / 1e6:.1f} MB")
    cap.release()
    return 0


def cmd_frames_to_video(args) -> int:
    frames = collect_frames(args.frames, args.limit)
    if not frames:
        print(f"ERROR: no images found under {args.frames}", file=sys.stderr)
        return 2

    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"ERROR: cannot decode {frames[0]}", file=sys.stderr)
        return 2
    h, w = first.shape[:2]
    if args.resize:
        scale = args.resize / w
        w, h = args.resize, int(h * scale)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), fourcc, float(args.fps), (w, h))
    if not writer.isOpened():
        print(f"ERROR: VideoWriter refused codec '{args.codec}'", file=sys.stderr)
        return 2

    written = 0
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        written += 1
        if written % 200 == 0:
            print(f"  ...{written}/{len(frames)}")

    writer.release()
    dur = written / float(args.fps) if args.fps else 0
    print(f"wrote {written} frames -> {args.out}")
    print(f"  resolution {w}x{h} @ {args.fps} fps  (~{dur:.1f} s)")
    print(f"  size       {args.out.stat().st_size / 1e6:.1f} MB")
    return 0


def cmd_video_to_frames(args) -> int:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        return 2

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Grab roughly one frame per second, mirroring training/extract_frames.py.
    step = max(int(round(src_fps / float(args.fps))), 1)
    args.out.mkdir(parents=True, exist_ok=True)

    stem = args.video.stem
    idx = read = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read % step == 0:
            out = args.out / f"{stem}_{idx:05d}.jpg"
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1
            idx += 1
        read += 1
    cap.release()
    print(f"extracted {saved} frames (1 per ~{step} source frames) -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build test videos from dataset frame sequences, or extract frames from video."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="report a video's properties")
    p.add_argument("video", type=Path)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("frames-to-video", help="assemble an image sequence into an mp4")
    p.add_argument("frames", type=Path, help="folder of PNG/JPG frames")
    p.add_argument("-o", "--out", type=Path, required=True, help="output .mp4")
    p.add_argument("--fps", type=float, default=10.0,
                   help="output FPS. Datasets are often captured slower than they "
                        "should play, so 8-15 is realistic (default: 10)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N frames")
    p.add_argument("--resize", type=int, default=None, help="scale width to N px")
    p.add_argument("--codec", default="mp4v", help="fourcc (default mp4v, broadly readable)")
    p.set_defaults(func=cmd_frames_to_video)

    p = sub.add_parser("video-to-frames", help="extract ~1 FPS frames from a video")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--fps", type=float, default=1.0, help="frames per second to keep (default 1)")
    p.set_defaults(func=cmd_video_to_frames)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
