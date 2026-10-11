"""Automated test suite for Phase 3 features (Flag as wrong, metadata logging)."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app import flag_frame_for_review, run_pipeline


class TestFootholdPhase3(unittest.TestCase):
    def test_flag_frame_saves_images_and_metadata(self):
        """Verify flag_frame_for_review saves raw & overlay images and appends metadata.csv."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"

            raw_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
            overlay_frame = np.full((480, 640, 3), 200, dtype=np.uint8)

            flag_id = flag_frame_for_review(
                raw_frame=raw_frame,
                overlay_frame=overlay_frame,
                source_type="live",
                data_dir=data_dir,
                location="rocky_pass",
                weather="wet",
            )

            # 1. Verify saved images in review/ folder
            review_dir = data_dir / "review"
            raw_file = review_dir / f"{flag_id}_raw.png"
            overlay_file = review_dir / f"{flag_id}_overlay.png"

            self.assertTrue(raw_file.exists(), f"Raw image {raw_file} was not created")
            self.assertTrue(overlay_file.exists(), f"Overlay image {overlay_file} was not created")

            loaded_raw = cv2.imread(str(raw_file))
            loaded_overlay = cv2.imread(str(overlay_file))

            self.assertEqual(loaded_raw.shape, (480, 640, 3))
            self.assertEqual(loaded_overlay.shape, (480, 640, 3))
            self.assertTrue(np.all(loaded_raw == 100))
            self.assertTrue(np.all(loaded_overlay == 200))

            # 2. Verify metadata.csv contents
            meta_path = data_dir / "metadata.csv"
            self.assertTrue(meta_path.exists(), "metadata.csv was not created")

            with open(meta_path, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))

            # Header row + 1 data row
            self.assertGreaterEqual(len(reader), 2)
            header = reader[0]
            self.assertEqual(header, ["id", "source", "date", "weather", "location", "label", "reviewed"])

            data_row = reader[1]
            self.assertEqual(data_row[0], flag_id)
            self.assertEqual(data_row[1], "live")
            self.assertEqual(data_row[3], "wet")
            self.assertEqual(data_row[4], "rocky_pass")
            self.assertEqual(data_row[5], "flagged")
            self.assertEqual(data_row[6], "no")  # Safety rule: must be unreviewed

    def test_multiple_flags_append_sequentially(self):
        """Verify multiple flags add rows without overwriting previous entries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"

            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            id1 = flag_frame_for_review(dummy_frame, dummy_frame, source_type="live", data_dir=data_dir)
            id2 = flag_frame_for_review(dummy_frame, dummy_frame, source_type="upload", data_dir=data_dir)

            meta_path = data_dir / "metadata.csv"
            with open(meta_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))

            # Header + 2 rows = 3 rows
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1][0], id1)
            self.assertEqual(rows[1][1], "live")
            self.assertEqual(rows[2][0], id2)
            self.assertEqual(rows[2][1], "upload")


if __name__ == "__main__":
    unittest.main()
