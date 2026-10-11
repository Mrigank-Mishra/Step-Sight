"""Extract frames from trail videos at 1 frame per second (1 FPS).

Saves frames into data/raw/<trail_name>/ with standardized naming:
  <trail_name>_<date>_sec<second:04d>.jpg
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2


def extract_frames_from_video(
    video_path: Union[str, Path],
    trail_name: str,
    output_dir: Union[str, Path] = "data/raw",
    date_str: Optional[str] = None,
) -> int:
    """Extract 1 frame per second from a video file into the raw data folder.

    Args:
        video_path: Path to the input video file.
        trail_name: Identifier for the trail (e.g. 'trail_ridge', 'forest_path').
        output_dir: Root directory for raw images (default: data/raw).
        date_str: Optional date string (defaults to today's date YYYYMMDD).

    Returns:
        The total number of extracted frames saved.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found: {vpath.resolve()}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {vpath.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0  # Fallback if metadata is missing

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    clean_trail = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in trail_name)
    save_dir = Path(output_dir) / f"{clean_trail}_{date_str}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Extracting 1 FPS from '{vpath.name}' ---")
    print(f"Trail name  : {clean_trail}")
    print(f"Video FPS   : {fps:.2f}")
    print(f"Duration    : ~{duration_sec:.1f} seconds ({total_frames} frames)")
    print(f"Saving to   : {save_dir}")

    extracted_count = 0
    frame_idx = 0
    second_idx = 0
    interval = max(int(round(fps)), 1)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Extract only on 1-second intervals
        if frame_idx % interval == 0:
            out_filename = f"{clean_trail}_{date_str}_sec{second_idx:04d}.jpg"
            out_filepath = save_dir / out_filename
            cv2.imwrite(str(out_filepath), frame)
            extracted_count += 1
            second_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[SUCCESS] Extracted {extracted_count} frames (1 per second) to: {save_dir}\n")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract 1 frame per second from a trail video into data/raw/."
    )
    parser.add_argument("video", help="Path to video file (.mp4, .avi, etc.)")
    parser.add_argument(
        "--trail",
        required=True,
        help="Trail name or identifier (e.g. blue_trail, summit_ridge)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Target folder for extracted images (default: data/raw)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Capture date string (default: current date YYYYMMDD)",
    )

    args = parser.parse_args()
    extract_frames_from_video(
        video_path=args.video,
        trail_name=args.trail,
        output_dir=args.output_dir,
        date_str=args.date,
    )


if __name__ == "__main__":
    main()
