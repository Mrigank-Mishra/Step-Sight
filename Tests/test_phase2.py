"""Automated test suite for Phase 2 features (Live & Upload modes, export, error handling)."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app import open_file_dialog, run_pipeline
from input_source import InputSource
from postprocess import PipelineState


class TestFootholdPhase2(unittest.TestCase):
    def test_single_image_auto_save(self):
        """Verify single image mode automatically saves annotated result."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_img_path = Path(tmp_dir) / "test_rocky_ground.png"
            test_img = np.full((480, 640, 3), 110, dtype=np.uint8)
            cv2.imwrite(str(test_img_path), test_img)

            out_dir = Path(tmp_dir) / "output"
            saved_file = run_pipeline(
                source_val=str(test_img_path),
                engine="heuristic",
                sensitivity=1.3,
                output_dir=out_dir,
                max_frames=1,
            )

            expected_saved = out_dir / "annotated_test_rocky_ground.png"
            self.assertTrue(expected_saved.exists(), f"Expected {expected_saved} to exist")
            saved_img = cv2.imread(str(expected_saved))
            self.assertIsNotNone(saved_img)
            self.assertEqual(saved_img.shape[1], 640)

    def test_video_export(self):
        """Verify video mode with save_video=True exports a valid annotated MP4 file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate mini 5-frame video
            vid_path = Path(tmp_dir) / "mini_trail.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (320, 240))
            for _ in range(5):
                frame = np.full((240, 320, 3), 120, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            out_dir = Path(tmp_dir) / "output"
            saved_file = run_pipeline(
                source_val=str(vid_path),
                engine="heuristic",
                sensitivity=1.3,
                save_video=True,
                output_dir=out_dir,
                max_frames=5,
            )

            expected_vid = out_dir / "annotated_mini_trail.mp4"
            self.assertTrue(expected_vid.exists(), f"Expected {expected_vid} to exist")

            # Check that exported video can be opened and has frames
            cap = cv2.VideoCapture(str(expected_vid))
            self.assertTrue(cap.isOpened())
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape[1], 640)  # Standard display width
            cap.release()

    def test_camera_error_handling(self):
        """Verify non-existent camera index is caught cleanly without crash."""
        # Using an absurd camera index like 999
        res = run_pipeline(source_val=999)
        self.assertIsNone(res)

    def test_input_source_video_properties(self):
        """Verify InputSource exposes fps and frame_size correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            vid_path = Path(tmp_dir) / "test_props.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 15.0, (320, 240))
            for _ in range(3):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()

            source = InputSource(str(vid_path))
            self.assertTrue(source.is_video)
            self.assertEqual(source.frame_size, (320, 240))
            self.assertGreater(source.fps, 0.0)
            source.release()


if __name__ == "__main__":
    unittest.main()
