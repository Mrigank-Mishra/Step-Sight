"""Foothold Safety Assistant - Visualization & Risk Overlay.

Draws:
1. High-contrast semi-transparent heatmap (green = safe, yellow = uncertain, red = risky).
2. 'STEP HERE' target marker on the safest footing patch.
3. 'NO SAFE SPOT FOUND - STOP AND REASSESS' warning if the best patch is high-risk.
4. Top status text with FPS and sensitivity adjustment prompt.
5. Bottom mandatory safety disclaimer: 'PROTOTYPE - decision support only, not safety certified'.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from .postprocess import pick_next_step
    from .risk_engine import GRID_COLS, GRID_ROWS, RISK_RED
except (ImportError, ValueError):
    from postprocess import pick_next_step
    from risk_engine import GRID_COLS, GRID_ROWS, RISK_RED

ALPHA = 0.35  # Heatmap transparency blending weight


def risk_to_heatmap(grid: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Smoothly upscale the risk grid and map risk scores to BGR colors.

    Color mapping:
    - 0.00: Pure Green (0, 255, 0) - SAFE
    - 0.50: Pure Yellow (0, 255, 255) - UNCERTAIN
    - 1.00: Pure Red (0, 0, 255) - RISKY

    Args:
        grid: (GRID_ROWS, GRID_COLS) float32 array in [0, 1].
        out_w: Target width.
        out_h: Target height.

    Returns:
        BGR uint8 image of shape (out_h, out_w, 3).
    """
    # Smooth cubic interpolation to full frame size
    r = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    r = np.clip(r, 0.0, 1.0)

    # Calculate channel intensities
    red = np.clip(r * 2.0, 0.0, 1.0) * 255.0
    green = (1.0 - np.clip((r - 0.5) * 2.0, 0.0, 1.0)) * 255.0
    blue = np.zeros_like(r)

    return np.dstack([blue, green, red]).astype(np.uint8)


def draw_overlay(
    frame: np.ndarray,
    grid: np.ndarray,
    fps: float,
    sensitivity: float,
    step_info: Optional[Tuple[int, int, float]] = None,
) -> np.ndarray:
    """Composite the heatmap, footing marker, and status warnings onto the frame.

    Args:
        frame: Original BGR video/image frame.
        grid: Smoothed (GRID_ROWS, GRID_COLS) risk grid.
        fps: Current processing frames per second.
        sensitivity: Active sensitivity multiplier.
        step_info: Optional precomputed (row, col, risk) of the safest patch.

    Returns:
        BGR frame with blended annotations.
    """
    h, w = frame.shape[:2]

    # 1. Generate color heatmap and blend over the input frame
    heat = risk_to_heatmap(grid, w, h)
    out = cv2.addWeighted(frame, 1.0 - ALPHA, heat, ALPHA, 0)

    # 2. Safest patch identification
    if step_info is None:
        row, col, best_risk = pick_next_step(grid)
    else:
        row, col, best_risk = step_info

    cx = int((col + 0.5) * w / GRID_COLS)
    cy = int((row + 0.5) * h / GRID_ROWS)

    # 3. Draw foothold suggestion or safety stop alert
    if best_risk < RISK_RED:
        # Green / Safe or Moderate Foothold: draw circular bullseye target
        cv2.circle(out, (cx, cy), 22, (255, 255, 255), 3)
        cv2.circle(out, (cx, cy), 6, (255, 255, 255), -1)
        cv2.putText(
            out,
            "STEP HERE",
            (cx - 50, cy - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    else:
        # All nearby patches are risky: prominent safety warning banner
        cv2.putText(
            out,
            "NO SAFE SPOT FOUND - STOP AND REASSESS",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # 4. Status header (FPS, Sensitivity, Help)
    cv2.putText(
        out,
        f"FPS {fps:4.1f}  sens {sensitivity:.2f}  (+/- adjust, s snap, f flag, q quit)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    # 5. Mandatory safety disclaimer
    cv2.putText(
        out,
        "PROTOTYPE - decision support only, not safety certified",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return out
