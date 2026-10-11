"""Can the STOP banner be triggered on real footage, and at what sensitivity?

RULES.md 2.3 requires a "NO SAFE SPOT FOUND - STOP AND REASSESS" state. This
sweeps sensitivity (the +/- control in the UI) and reports the best-patch risk,
so we know whether the banner is reachable on a demo clip or needs a threshold
change.

Usage:
    python tools/stop_threshold_check.py "D:\\foothold_demo" --table
    python tools/stop_threshold_check.py data --engine heuristic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline import resize_for_display  # noqa: E402
from postprocess import PipelineState  # noqa: E402
from risk_engine import RISK_RED, compute_risk_grid  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // count, 1) if total else 1
    out, read = [], 0
    while len(out) < count:
        ok, f = cap.read()
        if not ok:
            break
        if read % step == 0:
            out.append(resize_for_display(f, 640))
        read += 1
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Where does STOP start firing?")
    ap.add_argument("targets", nargs="+", type=Path)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--engine", default="heuristic", choices=["heuristic", "model"])
    ap.add_argument("--sens", type=float, nargs="+",
                    default=[1.3, 1.6, 2.0, 2.5, 3.0])
    args = ap.parse_args()

    files: list[Path] = []
    for t in args.targets:
        if t.is_dir():
            files += sorted(p for p in t.rglob("*") if p.suffix.lower() in VIDEO_EXT)
        elif t.exists():
            files.append(t)
    files = [f for f in files if f.stat().st_size > 20000]
    if not files:
        print("no videos found", file=sys.stderr)
        return 2

    model = None
    if args.engine == "model":
        from risk_engine import load_model
        model = load_model("models/best.pt")

    print()
    print(f"STOP fires when the BEST patch risk > {RISK_RED}. engine={args.engine}")
    print("The value below is the mean best-patch risk across sampled frames.")
    print("=" * 96)
    header = f"{'clip':<30}" + "".join(f"{s:>9.1f}" for s in args.sens) + "   verdict"
    print(header)
    print("-" * 96)

    for f in files:
        frames = sample_frames(f, args.frames)
        if not frames:
            continue
        cells = []
        for s in args.sens:
            st = PipelineState(sensitivity=s, engine=args.engine, model=model)
            st.set_source(str(f), is_static_image=False)
            vals = []
            for fr in frames:
                g = compute_risk_grid(fr, engine=args.engine, sensitivity=s, model=model)
                st.smooth_grid = g
                from postprocess import pick_next_step
                vals.append(pick_next_step(g)[2])
            cells.append(float(np.mean(vals)))

        fires = [s for s, v in zip(args.sens, cells) if v > RISK_RED]
        if fires:
            verdict = f"STOP achievable at sens >= {min(fires):.1f}"
        else:
            verdict = "STOP NOT reachable"
        name = f.name if len(f.name) <= 29 else f.name[:26] + "..."
        print(f"{name:<30}" + "".join(f"{c:>9.3f}" for c in cells) + f"   {verdict}")

    print("-" * 96)
    print("If every clip says NOT reachable, the banner cannot be demoed on this")
    print("footage. Options: raise sensitivity further, or lower RISK_RED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
