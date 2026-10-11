"""How much of each frame is actually hazardous? Decides the STOP criterion.

pick_next_step() returns the MINIMUM risk over its search zone, which is biased
low, so 'best patch > red' almost never fires. This reports, per clip, the
composition of the look-ahead zone so a defensible STOP rule can be chosen.

Look-ahead zone matches postprocess.pick_next_step: rows 50-95%, cols 20-80%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline import resize_for_display  # noqa: E402
from risk_engine import (  # noqa: E402
    GRID_COLS, GRID_ROWS, RISK_RED, SAFE_THRESHOLD, compute_risk_grid,
)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SENS = 1.3


def zone_of(grid):
    r0, r1 = int(GRID_ROWS * 0.50), int(GRID_ROWS * 0.95)
    c0, c1 = int(GRID_COLS * 0.20), int(GRID_COLS * 0.80)
    return grid[r0:r1, c0:c1]


def frames_of(path, count=20):
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
    targets = [Path(a) for a in sys.argv[1:]] or [Path("data")]
    files = []
    for t in targets:
        if t.is_dir():
            files += sorted(p for p in t.rglob("*") if p.suffix.lower() in VIDEO_EXT)
        elif t.exists():
            files.append(t)
    files = [f for f in files if f.stat().st_size > 20000]

    print()
    print("Composition of the look-ahead zone (rows 50-95%, cols 20-80%)")
    print(f"safe < {SAFE_THRESHOLD}   |   risky > {RISK_RED}   |   sens {SENS}")
    print("=" * 104)
    print(f"{'clip':<30}{'best':>7}{'mean':>7}{'%safe':>8}{'%risky':>8}"
          f"{'min-cells-safe':>16}  STOP viable?")
    print("-" * 104)

    for f in files:
        frames = frames_of(f)
        if not frames:
            continue
        best, means, pct_safe, pct_risky, n_safe = [], [], [], [], []
        for fr in frames:
            g = compute_risk_grid(fr, engine="heuristic", sensitivity=SENS)
            z = zone_of(g)
            best.append(z.min())
            means.append(z.mean())
            pct_safe.append(100.0 * (z < SAFE_THRESHOLD).mean())
            pct_risky.append(100.0 * (z > RISK_RED).mean())
            n_safe.append(int((z < SAFE_THRESHOLD).sum()))

        name = f.name if len(f.name) <= 29 else f.name[:26] + "..."
        print(f"{name:<30}{np.mean(best):>7.3f}{np.mean(means):>7.3f}"
              f"{np.mean(pct_safe):>8.1f}{np.mean(pct_risky):>8.1f}"
              f"{min(n_safe):>16}  "
              f"{'YES' if max(best) > RISK_RED else 'no (best<red)'}")

    print("-" * 104)
    print("'min-cells-safe' = fewest safe cells seen in any sampled frame.")
    print("If that stays well above 0, a 'no safe spot' STOP rule would never fire")
    print("on this footage either - the hazard simply is not dense enough.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
