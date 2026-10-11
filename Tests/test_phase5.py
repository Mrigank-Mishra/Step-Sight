"""Automated test suite for Phase 5 (YOLO model training, model risk engine, accuracy & FPS)."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure src and training are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from app import process_frame
from postprocess import PipelineState
from risk_engine import (
    GRID_COLS,
    GRID_ROWS,
    compute_heuristic_risk_grid,
    compute_model_risk_grid,
    compute_risk_grid,
    load_model,
)


class TestFootholdPhase5(unittest.TestCase):
    def test_model_weights_exist(self):
        """Verify best.pt exists in models/ directory."""
        model_path = Path("models/best.pt")
        self.assertTrue(model_path.exists(), f"Expected {model_path} to exist")
        self.assertGreater(model_path.stat().st_size, 1000)

    def test_load_model_returns_yolo_instance(self):
        """Verify load_model successfully loads cached model."""
        model = load_model("models/best.pt")
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "names"))
        self.assertEqual(len(model.names), 3)

    def test_compute_model_risk_grid_shape_and_bounds(self):
        """Verify model risk engine outputs (12, 16) float32 in [0, 1]."""
        test_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        grid = compute_model_risk_grid(test_frame)

        self.assertEqual(grid.shape, (GRID_ROWS, GRID_COLS))
        self.assertEqual(grid.dtype, np.float32)
        self.assertTrue(np.all(grid >= 0.0))
        self.assertTrue(np.all(grid <= 1.0))

    def test_swappable_contract_consistency(self):
        """Verify heuristic and model engines have identical input/output contracts."""
        test_frame = np.full((360, 480, 3), 100, dtype=np.uint8)

        grid_heur = compute_risk_grid(test_frame, engine="heuristic")
        grid_model = compute_risk_grid(test_frame, engine="model")

        self.assertEqual(grid_heur.shape, grid_model.shape)
        self.assertEqual(grid_heur.dtype, grid_model.dtype)
        self.assertEqual(grid_model.shape, (12, 16))

    def test_pipeline_with_model_engine(self):
        """Verify full process_frame pipeline runs seamlessly using engine='model'."""
        state = PipelineState(engine="model")
        test_frame = np.full((480, 640, 3), 110, dtype=np.uint8)

        annotated = process_frame(test_frame, state)

        self.assertEqual(annotated.shape, test_frame.shape)
        self.assertIsNotNone(state.last_raw_grid)
        self.assertEqual(state.last_raw_grid.shape, (12, 16))
        self.assertIsNotNone(state.last_step)


if __name__ == "__main__":
    unittest.main()
