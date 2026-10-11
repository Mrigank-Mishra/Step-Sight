"""Split labeled images into train/val sets by TRAIL (not by random image).

Splitting by whole trails ensures the model is evaluated on completely unseen
terrain, preventing data leakage from consecutive video frames.

Input structure:
  data/labeled/
    safe/
    risky/
    uncertain/

Output structure (standard YOLO classification format):
  data/dataset/
    train/
      safe/
      risky/
      uncertain/
    val/
      safe/
      risky/
      uncertain/
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

CLASSES = ["safe", "risky", "uncertain"]


def extract_trail_id(filename: str) -> str:
    """Extract the trail/trip identifier from an image filename.

    Examples:
      'ridge_trail_20261005_sec0001.jpg' -> 'ridge_trail'
      'trail1_0042.jpg'                  -> 'trail1'
      'forest_walk_frame012.png'         -> 'forest_walk'
    """
    stem = Path(filename).stem

    # Pattern 1: <trail>_YYYYMMDD_sec<num>
    m = re.match(r"^(.+?)_\d{8}_sec\d+", stem)
    if m:
        return m.group(1)

    # Pattern 2: <trail>_sec<num> or <trail>_frame<num>
    m = re.match(r"^(.+?)_(?:sec|frame)\d+", stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Pattern 3: <trail>_<num> (e.g. trail1_0042)
    m = re.match(r"^(.+?)_\d+$", stem)
    if m:
        return m.group(1)

    # Fallback: take leading token before first underscore, or full stem
    parts = stem.split("_")
    return parts[0] if len(parts) > 1 else stem


def split_by_trail(
    labeled_dir: Union[str, Path] = "data/labeled",
    output_dir: Union[str, Path] = "data/dataset",
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """Group labeled images by trail and split into train/val partitions.

    Args:
        labeled_dir: Directory containing safe/, risky/, uncertain/ subfolders.
        output_dir: Target directory for train/ and val/ datasets.
        val_ratio: Desired fraction of trails in the validation set (default: 0.20).
        seed: Random seed for reproducible splitting.

    Returns:
        Dictionary with count summaries: {'train': {...}, 'val': {...}}.
    """
    random.seed(seed)
    lbl_path = Path(labeled_dir)
    out_path = Path(output_dir)

    # 1. Discover images and group by (trail, class)
    trail_files: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    total_found = 0

    for cls_name in CLASSES:
        cls_folder = lbl_path / cls_name
        if not cls_folder.exists():
            continue
        for img_file in cls_folder.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                trail_id = extract_trail_id(img_file.name)
                trail_files[trail_id].append((cls_name, img_file))
                total_found += 1

    if total_found == 0:
        print(f"[WARNING] No labeled images found in {lbl_path}. Please place images in safe/, risky/, uncertain/.")
        return {"train": {c: 0 for c in CLASSES}, "val": {c: 0 for c in CLASSES}}

    trails = sorted(list(trail_files.keys()))
    print(f"\n--- Splitting Dataset by Trail (Val ratio: {val_ratio:.0%}) ---")
    print(f"Total labeled images : {total_found}")
    print(f"Detected trails ({len(trails)}): {', '.join(trails)}")

    train_trails: Set[str] = set()
    val_trails: Set[str] = set()

    if len(trails) == 1:
        # Single trail fallback: split images chronologically/sequentially to avoid random frame leakage
        single_trail = trails[0]
        print(f"[NOTE] Only 1 trail '{single_trail}' detected. Partitioning chronologically to prevent frame leakage.")
        all_items = sorted(trail_files[single_trail], key=lambda x: x[1].name)
        split_idx = int(len(all_items) * (1.0 - val_ratio))
        train_items = all_items[:split_idx]
        val_items = all_items[split_idx:]
    else:
        # Multiple trails: shuffle trails and assign whole trails to train or val
        shuffled_trails = trails.copy()
        random.shuffle(shuffled_trails)

        n_val = max(1, int(round(len(shuffled_trails) * val_ratio)))
        val_trails = set(shuffled_trails[:n_val])
        train_trails = set(shuffled_trails[n_val:])

        print(f"Train trails ({len(train_trails)}) : {', '.join(sorted(train_trails))}")
        print(f"Val trails   ({len(val_trails)}) : {', '.join(sorted(val_trails))}")

        train_items = [item for t in train_trails for item in trail_files[t]]
        val_items = [item for t in val_trails for item in trail_files[t]]

    # 2. Clear & rebuild output directory
    for split in ("train", "val"):
        for cls_name in CLASSES:
            dest_dir = out_path / split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {
        "train": {c: 0 for c in CLASSES},
        "val": {c: 0 for c in CLASSES},
    }

    # 3. Copy files
    for cls_name, src_file in train_items:
        dest = out_path / "train" / cls_name / src_file.name
        shutil.copy2(src_file, dest)
        summary["train"][cls_name] += 1

    for cls_name, src_file in val_items:
        dest = out_path / "val" / cls_name / src_file.name
        shutil.copy2(src_file, dest)
        summary["val"][cls_name] += 1

    # 4. Print summary report
    print("\nDataset Split Summary:")
    print(f"{'Split':<8} | {'Safe':<8} | {'Risky':<8} | {'Uncertain':<10} | {'Total':<8}")
    print("-" * 52)
    for split in ("train", "val"):
        s = summary[split]["safe"]
        r = summary[split]["risky"]
        u = summary[split]["uncertain"]
        tot = s + r + u
        print(f"{split:<8} | {s:<8} | {r:<8} | {u:<10} | {tot:<8}")
    print(f"\n[SUCCESS] Train/val dataset created at: {out_path.resolve()}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Split data/labeled/ into train/ and val/ folders by TRAIL."
    )
    parser.add_argument(
        "--labeled-dir",
        default="data/labeled",
        help="Path to labeled data folder (default: data/labeled)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dataset",
        help="Path to output split dataset (default: data/dataset)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio for trails (default: 0.20 for 80/20 split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    split_by_trail(
        labeled_dir=args.labeled_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
