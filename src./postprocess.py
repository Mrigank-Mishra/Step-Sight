"""Foothold Safety Assistant - Postprocessing & Pipeline State.

Handles:
1. Temporal smoothing to reduce flicker across consecutive video/camera frames.
2. Safest-patch selection: picks the lowest-risk 3x3 footing patch in the lower-middle zone.
3. PipelineState: maintains configuration, temporal state, and metrics across frames.
"""
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    from .risk_engine import GRID_COLS, GRID_ROWS
except (ImportError, ValueError):
    from risk_engine import GRID_COLS, GRID_ROWS

# Default smoothing factor: higher reacts faster to changes but flickers more
DEFAULT_SMOOTHING = 0.30
DEFAULT_SENSITIVITY = 1.3


def apply_temporal_smoothing(
    prev_grid: Optional[np.ndarray],
    current_grid: np.ndarray,
    factor: float = DEFAULT_SMOOTHING,
) -> np.ndarray:
    """Smooth the risk grid over time using exponential moving average.

    Args:
        prev_grid: Previous smoothed grid, or None if starting/reset.
        current_grid: Current raw risk grid (12, 16).
        factor: Smoothing weight for the new frame in (0, 1].

    Returns:
        Smoothed float32 grid of shape (12, 16).
    """
    if prev_grid is None:
        return current_grid.copy().astype(np.float32)
    smoothed = (1.0 - factor) * prev_grid + factor * current_grid
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def pick_next_step(grid: np.ndarray) -> Tuple[int, int, float]:
    """Find the best (lowest-risk) footing patch ahead of the walker.

    Uses a 3x3 local average to evaluate patch stability rather than a single cell.
    Assumes the camera points forward and down, searching the lower-middle zone:
    - Rows: 50% to 95% of grid height
    - Columns: 20% to 80% of grid width

    Args:
        grid: (GRID_ROWS, GRID_COLS) float32 risk array.

    Returns:
        (row, col, best_risk): The center row, column, and average risk score.
    """
    local = cv2.blur(grid, (3, 3))
    r0, r1 = int(GRID_ROWS * 0.50), int(GRID_ROWS * 0.95)
    c0, c1 = int(GRID_COLS * 0.20), int(GRID_COLS * 0.80)

    zone = local[r0:r1, c0:c1]
    i, j = np.unravel_index(np.argmin(zone), zone.shape)
    best_row = int(r0 + i)
    best_col = int(c0 + j)
    best_risk = float(zone[i, j])

    return best_row, best_col, best_risk


class PipelineState:
    """Encapsulates runtime state across frames."""

    def __init__(
        self,
        sensitivity: float = DEFAULT_SENSITIVITY,
        smoothing_factor: float = DEFAULT_SMOOTHING,
        engine: str = "heuristic",
        model: Optional[Any] = None,
        is_static_image: bool = False,
    ):
        """Initialize pipeline state.

        Args:
            sensitivity: Multiplier for risk engine sensitivity.
            smoothing_factor: Weight for temporal smoothing (0 = disabled, 1 = no smoothing).
            engine: 'heuristic' or 'model'.
            model: Optional model reference if engine='model'.
            is_static_image: If True, temporal smoothing is bypassed.
        """
        self.sensitivity: float = sensitivity
        self.smoothing_factor: float = smoothing_factor
        self.engine: str = engine
        self.model: Optional[Any] = model
        self.is_static_image: bool = is_static_image

        self.smooth_grid: Optional[np.ndarray] = None
        self.last_raw_grid: Optional[np.ndarray] = None
        self.last_step: Optional[Tuple[int, int, float]] = None
        self.last_overlay: Optional[np.ndarray] = None

        self.fps: float = 0.0
        self.last_time: float = time.time()
        self.current_source: Optional[str] = None

    def reset_smoothing(self) -> None:
        """Reset temporal smoothing state (clears previous grid history)."""
        self.smooth_grid = None

    def set_source(self, source_name: str, is_static_image: bool = False) -> None:
        """Set a new input source and automatically reset smoothing state.

        Args:
            source_name: Name or path of the input source.
            is_static_image: True if the source is a single still image.
        """
        if source_name != self.current_source:
            self.reset_smoothing()
            self.current_source = source_name
        self.is_static_image = is_static_image
        if self.is_static_image:
            self.reset_smoothing()

    def update_fps(self) -> float:
        """Update and return moving-average frames per second."""
        now = time.time()
        dt = max(now - self.last_time, 1e-6)
        instant_fps = 1.0 / dt
        if self.fps <= 0.0:
            self.fps = instant_fps
        else:
            self.fps = 0.9 * self.fps + 0.1 * instant_fps
        self.last_time = now
        return self.fps

    def update_smoothing(self, current_grid: np.ndarray) -> np.ndarray:
        """Update temporal smoothing with the latest frame's grid.

        If static image mode is active, smoothing is bypassed.
        """
        if self.is_static_image or self.smoothing_factor <= 0.0:
            self.smooth_grid = current_grid.copy()
        else:
            self.smooth_grid = apply_temporal_smoothing(
                self.smooth_grid,
                current_grid,
                factor=self.smoothing_factor,
            )
        return self.smooth_grid
