"""Automated test suite for Phase 1 refactored Foothold pipeline (using unittest)."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app import process_frame
from input_source import InputSource
from overlay import draw_overlay, risk_to_heatmap
from postprocess import PipelineState, apply_temporal_smoothing, pick_next_step
from risk_engine import (
    GRID_COLS,
    GRID_ROWS,
    PROC_H,
    PROC_W,
    RISK_RED,
    SAFE_THRESHOLD,
    cell_mean,
    compute_heuristic_risk_grid,
    compute_risk_grid,
)


class TestFootholdPhase1(unittest.TestCase):
    def test_grid_geometry(self):
        """Verify processing dimensions and grid divisions."""
        self.assertEqual(PROC_H % GRID_ROWS, 0)
        self.assertEqual(PROC_W % GRID_COLS, 0)
        self.assertEqual(GRID_ROWS, 12)
        self.assertEqual(GRID_COLS, 16)

    def test_cell_mean(self):
        """Verify cell_mean divides array into rows and cols accurately."""
        mask = np.ones((PROC_H, PROC_W), dtype=np.float32)
        means = cell_mean(mask, GRID_ROWS, GRID_COLS)
        self.assertEqual(means.shape, (GRID_ROWS, GRID_COLS))
        self.assertTrue(np.allclose(means, 1.0))

    def test_risk_engine_heuristic_output(self):
        """Verify risk engine returns (12, 16) float32 in [0, 1]."""
        test_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        grid = compute_heuristic_risk_grid(test_frame, sensitivity=1.3)

        self.assertEqual(grid.shape, (GRID_ROWS, GRID_COLS))
        self.assertEqual(grid.dtype, np.float32)
        self.assertTrue(np.all(grid >= 0.0))
        self.assertTrue(np.all(grid <= 1.0))

    def test_risk_engine_swappable_interface(self):
        """Verify swappable compute_risk_grid dispatcher."""
        test_frame = np.full((240, 320, 3), 100, dtype=np.uint8)

        # Heuristic engine works
        grid = compute_risk_grid(test_frame, engine="heuristic")
        self.assertEqual(grid.shape, (GRID_ROWS, GRID_COLS))

        # Model engine works (loads trained best.pt)
        grid_model = compute_risk_grid(test_frame, engine="model")
        self.assertEqual(grid_model.shape, (GRID_ROWS, GRID_COLS))

        # Model engine with invalid weights path raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            compute_risk_grid(test_frame, engine="model", model="non_existent_weights.pt")

        # Unknown engine raises ValueError
        with self.assertRaises(ValueError):
            compute_risk_grid(test_frame, engine="unknown_engine")

    def test_pick_next_step_bounds(self):
        """Verify safest-patch selection stays within defined ahead zone."""
        grid = np.full((GRID_ROWS, GRID_COLS), 0.5, dtype=np.float32)
        # Safe spot at row 8, col 8
        grid[7:10, 7:10] = 0.1

        row, col, risk = pick_next_step(grid)
        self.assertTrue(int(GRID_ROWS * 0.50) <= row <= int(GRID_ROWS * 0.95))
        self.assertTrue(int(GRID_COLS * 0.20) <= col <= int(GRID_COLS * 0.80))
        self.assertLess(risk, 0.5)

    def test_temporal_smoothing_and_reset(self):
        """Verify temporal smoothing and state resetting when source changes."""
        state = PipelineState(sensitivity=1.3, smoothing_factor=0.30)
        state.set_source("source_A")

        grid_1 = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        grid_2 = np.ones((GRID_ROWS, GRID_COLS), dtype=np.float32)

        # First update sets initial grid
        smoothed_1 = state.update_smoothing(grid_1)
        self.assertTrue(np.allclose(smoothed_1, 0.0))

        # Second update blends: (0.7 * 0.0 + 0.3 * 1.0 = 0.3)
        smoothed_2 = state.update_smoothing(grid_2)
        self.assertTrue(np.allclose(smoothed_2, 0.30))

        # Switch source -> resets smoothing
        state.set_source("source_B")
        self.assertIsNone(state.smooth_grid)

        # Next update on new source starts fresh with grid_2
        smoothed_3 = state.update_smoothing(grid_2)
        self.assertTrue(np.allclose(smoothed_3, 1.0))

    def test_static_image_skips_temporal_smoothing(self):
        """Verify static images do not accumulate prior temporal smoothing."""
        state = PipelineState(sensitivity=1.3, smoothing_factor=0.30)
        state.set_source("test.jpg", is_static_image=True)

        grid_1 = np.full((GRID_ROWS, GRID_COLS), 0.8, dtype=np.float32)
        smoothed = state.update_smoothing(grid_1)
        self.assertTrue(np.allclose(smoothed, 0.8))

    def test_heatmap_color_mapping(self):
        """Verify green -> yellow -> red color transitions in heatmap."""
        # 0.0 -> Green (B=0, G=255, R=0)
        grid_safe = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        heat_safe = risk_to_heatmap(grid_safe, out_w=64, out_h=48)
        self.assertEqual(heat_safe[24, 32, 1], 255)  # Green channel
        self.assertEqual(heat_safe[24, 32, 2], 0)    # Red channel

        # 1.0 -> Red (B=0, G=0, R=255)
        grid_risky = np.ones((GRID_ROWS, GRID_COLS), dtype=np.float32)
        heat_risky = risk_to_heatmap(grid_risky, out_w=64, out_h=48)
        self.assertEqual(heat_risky[24, 32, 2], 255)  # Red channel
        self.assertEqual(heat_risky[24, 32, 1], 0)    # Green channel

    def test_draw_overlay_safe_and_risky(self):
        """Verify overlay drawing handles both safe foothold and stop-and-reassess."""
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)

        # Safe grid
        safe_grid = np.full((GRID_ROWS, GRID_COLS), 0.1, dtype=np.float32)
        out_safe = draw_overlay(frame, safe_grid, fps=25.0, sensitivity=1.3)
        self.assertEqual(out_safe.shape, frame.shape)

        # Risky grid (triggers STOP AND REASSESS)
        risky_grid = np.full((GRID_ROWS, GRID_COLS), 0.9, dtype=np.float32)
        out_risky = draw_overlay(frame, risky_grid, fps=25.0, sensitivity=1.3)
        self.assertEqual(out_risky.shape, frame.shape)

    def test_process_frame_end_to_end(self):
        """Verify shared process_frame execution."""
        state = PipelineState()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        annotated = process_frame(frame, state)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertIsNotNone(state.last_raw_grid)
        self.assertIsNotNone(state.last_step)
        self.assertEqual(len(state.last_step), 3)

    def test_input_source_image_file(self):
        """Verify InputSource with a temporary static image file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test_terrain.png"
            sample_img = np.full((240, 320, 3), 150, dtype=np.uint8)
            cv2.imwrite(str(img_path), sample_img)

            with InputSource(str(img_path)) as src:
                self.assertTrue(src.is_image)
                self.assertFalse(src.is_video)
                self.assertFalse(src.is_camera)
                ok, frame = src.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (240, 320, 3))

    def test_input_source_missing_file(self):
        """Verify clear error when file does not exist."""
        with self.assertRaises(FileNotFoundError):
            InputSource("non_existent_file_12345.mp4")


if __name__ == "__main__":
    unittest.main()
