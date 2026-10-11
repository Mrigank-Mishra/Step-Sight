"""Train YOLOv8n classification model on terrain patches and evaluate performance.

Features:
- Trains lightweight YOLOv8n-cls on safe/risky/uncertain patches.
- Evaluates per-class accuracy on the trail-split validation set.
- Exports best weights to models/best.pt.
- Measures and reports full-grid inference speed (FPS) on CPU.
"""
import argparse
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASSES = ["safe", "risky", "uncertain"]


def train_yolo_classifier(
    data_dir: str = "data/dataset",
    output_model_path: str = "models/best.pt",
    epochs: int = 5,
    imgsz: int = 64,
    batch_size: int = 16,
) -> Tuple[float, Dict[str, float], float]:
    """Train YOLOv8n-cls, save models/best.pt, and report accuracy and CPU FPS.

    Args:
        data_dir: Path to prepared YOLO dataset folder (with train/ and val/).
        output_model_path: Destination for best model weights.
        epochs: Number of training epochs.
        imgsz: Training image size (64x64 is fast and accurate on CPU).
        batch_size: Training batch size.

    Returns:
        (overall_accuracy, per_class_accuracy_dict, cpu_fps)
    """
    dataset_path = Path(data_dir).resolve()
    if not (dataset_path / "train").exists() or not (dataset_path / "val").exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please run 'python training/split_dataset.py' first."
        )

    print("\n" + "=" * 56)
    print("      FOOTHOLD YOLOv8n PATCH CLASSIFIER TRAINING")
    print("=" * 56)
    print(f"Dataset path : {dataset_path}")
    print(f"Epochs       : {epochs}")
    print(f"Image size   : {imgsz}x{imgsz}")
    print(f"Batch size   : {batch_size}")
    print(f"Device       : CPU (Intel/AMD)")
    print("=" * 56 + "\n")

    # 1. Load base pre-trained model
    model = YOLO("yolov8n-cls.pt")

    # 2. Train on the trail-split dataset
    results = model.train(
        data=str(dataset_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device="cpu",
        workers=0,  # Stable on Windows
        verbose=True,
    )

    # 3. Locate best.pt from training runs and copy to models/best.pt
    dest_path = Path(output_model_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(model.trainer.save_dir) / "weights"
    trained_best = weights_dir / "best.pt"
    if not trained_best.exists():
        trained_best = weights_dir / "last.pt"

    shutil.copy2(trained_best, dest_path)
    print(f"\n[SUCCESS] Exported best model weights to: {dest_path}")

    # 4. Load exported model for validation and per-class accuracy evaluation
    eval_model = YOLO(str(dest_path))
    val_dir = dataset_path / "val"

    class_names = eval_model.names  # e.g. {0: 'risky', 1: 'safe', 2: 'uncertain'}
    idx_to_name = {idx: name for idx, name in class_names.items()}
    name_to_idx = {name: idx for idx, name in class_names.items()}

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    val_images = []
    val_labels = []

    for cls_name in CLASSES:
        cls_folder = val_dir / cls_name
        if not cls_folder.exists():
            continue
        for img_p in cls_folder.glob("*.*"):
            if img_p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                val_images.append(str(img_p))
                val_labels.append(cls_name)

    if val_images:
        preds = eval_model(val_images, verbose=False)
        for pred, true_cls in zip(preds, val_labels):
            top1_idx = int(pred.probs.top1)
            pred_cls = idx_to_name[top1_idx]
            class_total[true_cls] += 1
            if pred_cls == true_cls:
                class_correct[true_cls] += 1

    total_samples = sum(class_total.values())
    total_correct = sum(class_correct.values())
    overall_acc = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0

    per_class_acc: Dict[str, float] = {}
    print("\n" + "=" * 56)
    print("             VALIDATION ACCURACY REPORT")
    print("=" * 56)
    print(f"{'Class':<12} | {'Correct':<8} | {'Total':<6} | {'Accuracy':<10}")
    print("-" * 46)

    for c in CLASSES:
        tot = class_total[c]
        cor = class_correct[c]
        acc = (cor / tot * 100.0) if tot > 0 else 0.0
        per_class_acc[c] = acc
        print(f"{c:<12} | {cor:<8} | {tot:<6} | {acc:>7.2f}%")

    print("-" * 46)
    print(f"Overall Top-1 Accuracy: {overall_acc:.2f}%\n")

    # 5. Measure CPU Inference Speed on Full 12x16 Grid (192 Patches)
    print("Measuring CPU full-frame inference speed (192 grid patches)...")
    pytorch_model = eval_model.model
    pytorch_model.eval()

    dummy_batch = torch.zeros((192, 3, imgsz, imgsz), dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = pytorch_model(dummy_batch[:16])

    # Timed runs
    num_runs = 10
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = pytorch_model(dummy_batch)
    elapsed = time.time() - start_time

    avg_frame_time = elapsed / num_runs
    fps = 1.0 / avg_frame_time

    print(f"Full-grid inference time : {avg_frame_time * 1000.0:.1f} ms per frame")
    print(f"CPU Processing Speed     : {fps:.1f} FPS")
    print("=" * 56 + "\n")

    return overall_acc, per_class_acc, fps


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n patch classification model for Foothold."
    )
    parser.add_argument(
        "--data-dir",
        default="data/dataset",
        help="Path to YOLO classification dataset (default: data/dataset)",
    )
    parser.add_argument(
        "--output",
        default="models/best.pt",
        help="Target output weights path (default: models/best.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=64,
        help="Input image size (default: 64)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )

    args = parser.parse_args()
    train_yolo_classifier(
        data_dir=args.data_dir,
        output_model_path=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
