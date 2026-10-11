"""Dataset Health & Class Balance Checker for Foothold Safety Assistant.

Inspects:
1. data/labeled/ (safe, risky, uncertain)
2. data/review/ (flagged frames pending human review)
3. data/raw/ (extracted trail video frames)
4. data/dataset/ (train/val split status)
5. data/metadata.csv (catalog record count)
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

CLASSES = ["safe", "risky", "uncertain"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images_in_dir(directory: Path) -> int:
    """Count image files inside a directory."""
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def check_dataset_health(data_dir: str = "data") -> Dict[str, int]:
    """Inspect dataset folders, count samples per class, and display health diagnostics.

    Args:
        data_dir: Root directory of data folder (default: 'data').

    Returns:
        Dictionary of counts per labeled class.
    """
    root = Path(data_dir)
    labeled_dir = root / "labeled"
    review_dir = root / "review"
    raw_dir = root / "raw"
    meta_file = root / "metadata.csv"
    split_dir = root / "dataset"

    print("\n" + "=" * 54)
    print("       FOOTHOLD DATASET HEALTH CHECK")
    print("=" * 54)

    # 1. Labeled data counts
    print("\n[1] Labeled Training Patches (data/labeled/):")
    counts: Dict[str, int] = {}
    total_labeled = 0

    for cls in CLASSES:
        cls_folder = labeled_dir / cls
        cnt = count_images_in_dir(cls_folder)
        counts[cls] = cnt
        total_labeled += cnt

    for cls in CLASSES:
        cnt = counts[cls]
        pct = (cnt / total_labeled * 100.0) if total_labeled > 0 else 0.0
        bar = "#" * int(pct / 5)
        print(f"  - {cls:<10}: {cnt:>5} images ({pct:>5.1f}%)  [{bar:<20}]")

    print(f"  -----------------------------------------------")
    print(f"  Total Labeled : {total_labeled:>5} images")

    # 2. Flagged Review Queue
    raw_flags = sum(1 for f in review_dir.glob("*_raw.png")) if review_dir.exists() else 0
    overlay_flags = sum(1 for f in review_dir.glob("*_overlay.png")) if review_dir.exists() else 0
    print(f"\n[2] Human Review Queue (data/review/):")
    print(f"  - Flagged frames pending review: {raw_flags} frames")
    if raw_flags > 0:
        print(f"    (Tip: Review and sort these into data/labeled/ before retraining!)")

    # 3. Raw Data
    raw_images = sum(count_images_in_dir(sub) for sub in raw_dir.iterdir() if sub.is_dir()) if raw_dir.exists() else 0
    print(f"\n[3] Raw Video Frames (data/raw/):")
    print(f"  - Total extracted trail frames : {raw_images} frames")

    # 4. Train/Val Split status
    print(f"\n[4] Prepared YOLO Dataset (data/dataset/):")
    if split_dir.exists() and (split_dir / "train").exists():
        train_tot = sum(count_images_in_dir(split_dir / "train" / c) for c in CLASSES)
        val_tot = sum(count_images_in_dir(split_dir / "val" / c) for c in CLASSES)
        print(f"  - Train partition : {train_tot} images")
        print(f"  - Val partition   : {val_tot} images")
    else:
        print(f"  - Status: Not yet split (run 'python training/split_dataset.py')")

    # 5. Metadata Catalog
    meta_rows = 0
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
    print(f"\n[5] Catalog (data/metadata.csv):")
    print(f"  - Logged metadata entries: {meta_rows} records")

    # 6. Safety and Health Diagnostics
    print("\n" + "-" * 54)
    print("Health Diagnostics:")
    warnings: List[str] = []

    if total_labeled == 0:
        warnings.append("No labeled images found in data/labeled/.")
    else:
        for cls in CLASSES:
            if counts[cls] == 0:
                warnings.append(f"Class '{cls}' has 0 samples.")
            elif counts[cls] < 5:
                warnings.append(f"Class '{cls}' has very few samples ({counts[cls]}). Consider adding more.")

        # Check for severe imbalance
        if total_labeled >= 10:
            for cls in CLASSES:
                if counts[cls] / total_labeled > 0.85:
                    warnings.append(f"Severe imbalance: '{cls}' makes up over 85% of labeled data.")

    if warnings:
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("  [*] All classes populated and balanced! Ready for training.")

    print("=" * 54 + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Check dataset health and image counts per class.")
    parser.add_argument("--data-dir", default="data", help="Root data directory (default: data)")
    args = parser.parse_args()
    check_dataset_health(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
