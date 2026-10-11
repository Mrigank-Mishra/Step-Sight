"""Generate realistic terrain patches across distinct trails for YOLOv8 classification training.

Creates labeled patches in:
  data/labeled/safe/
  data/labeled/risky/
  data/labeled/uncertain/

Covers multiple distinct trails to test trail-based generalization:
- trail_ridge
- trail_canyon
- trail_creek
- trail_forest
"""
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def generate_patch(patch_type: str, seed: int) -> np.ndarray:
    """Generate a 64x64 terrain patch simulating safe, risky, or uncertain ground."""
    rng = np.random.RandomState(seed)
    h, w = 64, 64
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if patch_type == "safe":
        # Firm dirt path / dry compact rock
        base_color = rng.randint(90, 120), rng.randint(110, 140), rng.randint(130, 160)
        img[:] = base_color
        # Add fine texture (grainy dirt)
        noise = rng.randint(-15, 15, (h, w, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Gentle broad texture
        img = cv2.GaussianBlur(img, (3, 3), 0)

    elif patch_type == "risky":
        hazard = seed % 3
        if hazard == 0:
            # Wet rock with bright specular glare highlight
            img[:] = (70, 75, 80)
            noise = rng.randint(-10, 10, (h, w, 3))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Add glaring white spot
            cx, cy = rng.randint(20, 44), rng.randint(20, 44)
            cv2.circle(img, (cx, cy), rng.randint(10, 18), (250, 250, 250), -1)
            img = cv2.GaussianBlur(img, (5, 5), 0)
        elif hazard == 1:
            # Deep slick mud (dark, smooth, low texture)
            img[:] = (35, 45, 50)
            noise = rng.randint(-5, 5, (h, w, 3))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = cv2.GaussianBlur(img, (7, 7), 0)
        else:
            # Cracked, fragmented loose rock (sharp edges)
            img[:] = (80, 85, 90)
            for _ in range(8):
                pt1 = (rng.randint(0, w), rng.randint(0, h))
                pt2 = (rng.randint(0, w), rng.randint(0, h))
                cv2.line(img, pt1, pt2, (15, 15, 15), rng.randint(1, 2))

    elif patch_type == "uncertain":
        # Mixed gravel, pebble border, or transition terrain
        img[:] = (80, 95, 110)
        # Moderate speckled texture
        for _ in range(15):
            pt = (rng.randint(0, w), rng.randint(0, h))
            radius = rng.randint(2, 5)
            color = (rng.randint(50, 140), rng.randint(60, 150), rng.randint(70, 160))
            cv2.circle(img, pt, radius, color, -1)
        noise = rng.randint(-10, 10, (h, w, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def create_patch_dataset(output_dir: str = "data/labeled", patches_per_trail: int = 15) -> None:
    """Create labeled patches across 4 trails."""
    trails = ["trail_ridge", "trail_canyon", "trail_creek", "trail_forest"]
    classes = ["safe", "risky", "uncertain"]

    root = Path(output_dir)
    for c in classes:
        (root / c).mkdir(parents=True, exist_ok=True)

    total_created = 0
    seed = 1000

    for trail in trails:
        for cls in classes:
            for i in range(patches_per_trail):
                seed += 1
                patch = generate_patch(cls, seed)
                filename = f"{trail}_sec{i:04d}.jpg"
                filepath = root / cls / filename
                cv2.imwrite(str(filepath), patch)
                total_created += 1

    print(f"[SUCCESS] Created {total_created} labeled terrain patches across {len(trails)} trails in {root.resolve()}.")


if __name__ == "__main__":
    create_patch_dataset()
