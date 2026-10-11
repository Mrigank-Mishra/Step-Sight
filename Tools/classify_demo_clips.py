"""Classify terrain footage for the hackathon demo.

For each video it reports which of the three risk states the pipeline actually
produces, so you can pick a clip that provably demonstrates a given rule instead
of guessing. States map to RULES.md 2:

  SAFE      risk < 0.35        -> green      (RULES 2.4: safe must be earned)
  UNCERTAIN 0.35 <= risk <= .60 -> yellow    (RULES 2.2: yellow must survive)
  STOP      best patch > 0.60  -> red banner (RULES 2.3: no safe spot -> STOP)

Usage:
    python tools/classify_demo_clips.py <video-or-folder> [...]
    python tools/classify_demo_clips.py "D:\\foothold_demo" --table
    python tools/classify_demo_clips.py clip.mp4 --frames 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline import process_frame, resize_for_display  # noqa: E402
from postprocess import PipelineState  # noqa: E402
from risk_engine import RISK_RED, SAFE_THRESHOLD  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def classify(path: Path, max_frames: int) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample evenly across the whole clip rather than just the opening seconds.
    step = max(total // max_frames, 1) if total else 1

    state = PipelineState(sensitivity=1.3, engine="heuristic")
    state.set_source(str(path), is_static_image=False)

    safe = unc = stop = 0
    best_risks: list[float] = []
    grid_means: list[float] = []
    grid_risky: list[int] = []
    read = used = 0

    while used < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if read % step == 0:
            annotated = process_frame(resize_for_display(frame, 640), state)
            assert annotated is not None
            info = state.last_step
            if info is not None:
                r = float(info[2])
                best_risks.append(r)
                if r > RISK_RED:
                    stop += 1
                elif r >= SAFE_THRESHOLD:
                    unc += 1
                else:
                    safe += 1
            # Heat-map density: how much of the frame reads as hazardous. This is
            # a different question from the best patch, and it is what drives the
            # STOP banner (RULES 2.3).
            if state.last_raw_grid is not None:
                g = state.last_raw_grid
                grid_means.append(float(g.mean()))
                grid_risky.append(int((g > RISK_RED).sum()))
            used += 1
        read += 1
    cap.release()

    if not best_risks:
        return None

    n = len(best_risks)
    return {
        "name": path.name,
        "path": path,
        "fps": fps,
        "total": total,
        "used": n,
        "safe_pct": 100.0 * safe / n,
        "unc_pct": 100.0 * unc / n,
        "stop_pct": 100.0 * stop / n,
        "risk_mean": float(np.mean(best_risks)),
        "risk_max": float(np.max(best_risks)),
        "hazard_pct": float(np.mean(grid_risky)) if grid_risky else 0.0,
        "grid_mean": float(np.mean(grid_means)) if grid_means else 0.0,
    }


def dominant(r: dict) -> str:
    if r["stop_pct"] >= 25:
        return "STOP"
    if r["unc_pct"] >= 40:
        return "UNCERTAIN"
    if r["safe_pct"] >= 70:
        return "SAFE"
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser(description="Which risk state does a clip produce?")
    ap.add_argument("targets", nargs="+", type=Path)
    ap.add_argument("--frames", type=int, default=40, help="frames to sample (default 40)")
    ap.add_argument("--table", action="store_true", help="compact output")
    args = ap.parse_args()

    files: list[Path] = []
    for t in args.targets:
        if t.is_dir():
            files += sorted(p for p in t.rglob("*") if p.suffix.lower() in VIDEO_EXT)
        elif t.exists():
            files.append(t)
    if not files:
        print("no videos found", file=sys.stderr)
        return 2

    rows = []
    for f in files:
        # Skip the giant ORAD/RELLIS artefacts if someone points this at D:\
        if f.stat().st_size < 20000:
            continue
        r = classify(f, args.frames)
        if r:
            rows.append(r)

    if not rows:
        print("could not classify anything", file=sys.stderr)
        return 2

    if args.table:
        print(f"{'state':<10} {'SAFE%':>6} {'UNC%':>6} {'STOP%':>6} {'mean':>6} {'HAZ%':>6}  file")
        print("-" * 92)
        for r in rows:
            print(f"{dominant(r):<10} {r['safe_pct']:>6.1f} {r['unc_pct']:>6.1f} "
                  f"{r['stop_pct']:>6.1f} {r['risk_mean']:>6.2f} {r['hazard_pct']:>6.1f}  {r['name']}")
        print("-" * 92)
        print("SAFE/UNC/STOP% = the STEP-HERE patch state (what the marker does)")
        print("HAZ%           = share of the frame above the red threshold (what STOP needs)")
        print(f"Thresholds: safe<{SAFE_THRESHOLD}, risky>{RISK_RED}")
        return 0

    for r in rows:
        print("=" * 74)
        print(f"{r['name']}")
        print(f"  video      : {r['total']} frames @ {r['fps']:.1f} fps (sampled {r['used']})")
        print(f"  step patch : SAFE {r['safe_pct']:.0f}%  UNCERTAIN {r['unc_pct']:.0f}%  "
              f"STOP {r['stop_pct']:.0f}%")
        print(f"  best risk  : mean {r['risk_mean']:.3f}  max {r['risk_max']:.3f}")
        print(f"  frame hazard: {r['hazard_pct']:.1f}% of cells over red  "
              f"(grid mean {r['grid_mean']:.3f})")
        print(f"  best for   : {dominant(r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
