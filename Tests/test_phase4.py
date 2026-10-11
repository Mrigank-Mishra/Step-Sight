"""Automated test suite for Phase 4 Data Tools (frame extraction, trail split, health check)."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure training and src are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from check_dataset import check_dataset_health
from extract_frames import extract_frames_from_video
from split_dataset import extract_trail_id, split_by_trail


class TestFootholdPhase4(unittest.TestCase):
    def test_extract_trail_id_patterns(self):
        """Verify trail ID parser extracts trail name correctly from filenames."""
        self.assertEqual(extract_trail_id("trail_ridge_20261005_sec0001.jpg"), "trail_ridge")
        self.assertEqual(extract_trail_id("rocky_pass_sec0012.png"), "rocky_pass")
        self.assertEqual(extract_trail_id("summit_frame0050.jpg"), "summit")
        self.assertEqual(extract_trail_id("trail1_0042.jpg"), "trail1")

    def test_extract_frames_at_1_fps(self):
        """Verify video frame extraction saves 1 frame per second."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            vid_path = tmp_path / "test_walk.mp4"

            # Create 3-second video @ 10 FPS (30 frames)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (160, 120))
            for i in range(30):
                img = np.full((120, 160, 3), i * 8, dtype=np.uint8)
                writer.write(img)
            writer.release()

            raw_out = tmp_path / "raw"
            count = extract_frames_from_video(
                video_path=vid_path,
                trail_name="ridge_walk",
                output_dir=raw_out,
                date_str="20261005",
            )

            # 30 frames @ 10 fps -> exactly 3 frames at 1 fps (frames 0, 10, 20)
            self.assertEqual(count, 3)

            trail_folder = raw_out / "ridge_walk_20261005"
            self.assertTrue(trail_folder.exists())
            saved_frames = list(trail_folder.glob("*.jpg"))
            self.assertEqual(len(saved_frames), 3)

            # Ensure saved frames are valid images
            sample = cv2.imread(str(saved_frames[0]))
            self.assertIsNotNone(sample)
            self.assertEqual(sample.shape, (120, 160, 3))

    def test_split_by_trail_zero_data_leakage(self):
        """Verify images of a trail exist exclusively in either train OR val, never both."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            labeled_dir = tmp_path / "labeled"

            # Generate synthetic labeled images for 4 distinct trails
            trails = ["trailA", "trailB", "trailC", "trailD"]
            classes = ["safe", "risky", "uncertain"]

            for cls in classes:
                (labeled_dir / cls).mkdir(parents=True, exist_ok=True)

            for t_idx, trail in enumerate(trails):
                for c_idx, cls in enumerate(classes):
                    for frame_idx in range(4):
                        filename = f"{trail}_sec{frame_idx:04d}.jpg"
                        img = np.zeros((32, 32, 3), dtype=np.uint8)
                        cv2.imwrite(str(labeled_dir / cls / filename), img)

            output_dir = tmp_path / "dataset"
            summary = split_by_trail(
                labeled_dir=labeled_dir,
                output_dir=output_dir,
                val_ratio=0.25,  # 1 trail val, 3 trails train
                seed=42,
            )

            # Verify train and val files
            train_files = list((output_dir / "train").glob("*/*.jpg"))
            val_files = list((output_dir / "val").glob("*/*.jpg"))

            self.assertGreater(len(train_files), 0)
            self.assertGreater(len(val_files), 0)

            # Extract trail IDs present in train and val sets
            train_trails = {extract_trail_id(f.name) for f in train_files}
            val_trails = {extract_trail_id(f.name) for f in val_files}

            # Crucial check: intersection MUST be empty (zero trail leakage)
            overlap = train_trails.intersection(val_trails)
            self.assertEqual(len(overlap), 0, f"Trail leakage detected! Overlapping trails: {overlap}")

    def test_check_dataset_health(self):
        """Verify check_dataset_health counts samples per class correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            for cls in ["safe", "risky", "uncertain"]:
                (data_dir / "labeled" / cls).mkdir(parents=True, exist_ok=True)

            # Create 3 safe images, 2 risky images, 1 uncertain image
            for i in range(3):
                cv2.imwrite(str(data_dir / "labeled" / "safe" / f"safe_{i}.jpg"), np.zeros((10, 10, 3)))
            for i in range(2):
                cv2.imwrite(str(data_dir / "labeled" / "risky" / f"risky_{i}.jpg"), np.zeros((10, 10, 3)))
            cv2.imwrite(str(data_dir / "labeled" / "uncertain" / "unc_0.jpg"), np.zeros((10, 10, 3)))

            counts = check_dataset_health(data_dir=str(data_dir))
            self.assertEqual(counts["safe"], 3)
            self.assertEqual(counts["risky"], 2)
            self.assertEqual(counts["uncertain"], 1)


if __name__ == "__main__":
    unittest.main()
