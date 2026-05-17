"""
train.py
========
Training script for the conveyor belt damage detection system.

Designed to run on Google Colab or Kaggle with GPU.

This script:
  1. Prepares the dataset (train/val split)
  2. Trains a YOLOv8n-seg model for belt ROI segmentation
  3. Saves the trained model weights

Usage (local):
    python train.py --source_dir train/train --output_dir dataset --epochs 100

Usage (Google Colab):
    # Cell 1: Upload data
    # Upload your train.zip to Colab, then:
    # !unzip train.zip -d /content/data/

    # Cell 2: Install dependencies
    # !pip install ultralytics

    # Cell 3: Upload and run training
    # !python train.py --source_dir /content/data/train/train --epochs 100 --device 0

Usage (Kaggle):
    # Add dataset to your Kaggle notebook, then:
    # !python train.py --source_dir /kaggle/input/belt-dataset/train/train --epochs 100 --device 0
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 belt ROI segmentation model"
    )
    parser.add_argument(
        "--source_dir", type=str, default="train/train",
        help="Path to source dataset (containing train/images and train/labels)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset",
        help="Directory to store prepared dataset"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n-seg.pt",
        help="Base model (yolov8n-seg.pt, yolov8s-seg.pt, etc.)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (adjust based on GPU memory)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device: '0' for GPU, 'cpu' for CPU, '' for auto"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--project", type=str, default="runs/segment",
        help="Project directory for saving results"
    )
    parser.add_argument(
        "--name", type=str, default="belt_seg",
        help="Experiment name"
    )
    parser.add_argument(
        "--weights_dir", type=str, default="model_weights",
        help="Directory to copy final weights to"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Conveyor Belt ROI Segmentation — Training")
    print("=" * 60)
    
    # ---- Step 1: Prepare dataset ----
    print("\n[Step 1/3] Preparing dataset...")
    
    from prepare_dataset import prepare_dataset
    
    output_dir = Path(args.output_dir)
    if not (output_dir / "data.yaml").exists():
        data_yaml_path = prepare_dataset(
            args.source_dir,
            args.output_dir,
            args.val_split
        )
    else:
        data_yaml_path = str((output_dir / "data.yaml").resolve())
        print(f"  Dataset already prepared at {data_yaml_path}")
    
    # Verify data.yaml path format for cross-platform compatibility
    # On Colab/Kaggle (Linux), paths use forward slashes
    data_yaml_path = str(Path(data_yaml_path))
    
    # ---- Step 2: Train model ----
    print(f"\n[Step 2/3] Training {args.model}...")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device:    {args.device or 'auto'}")
    print(f"  Data:      {data_yaml_path}")
    
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Augmentation settings
        hsv_h=0.015,       # Hue augmentation
        hsv_s=0.5,         # Saturation augmentation
        hsv_v=0.4,         # Value/brightness augmentation (important for day/night)
        degrees=5.0,       # Rotation
        translate=0.1,     # Translation
        scale=0.3,         # Scale
        flipud=0.5,        # Vertical flip
        fliplr=0.5,        # Horizontal flip
        mosaic=0.5,        # Mosaic augmentation
        # Training params
        patience=20,       # Early stopping patience
        save=True,
        save_period=10,   # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        resume=args.resume,
    )
    
    # ---- Step 3: Save best weights ----
    print(f"\n[Step 3/3] Saving model weights...")
    
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best weights from training
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    last_weights = Path(args.project) / args.name / "weights" / "last.pt"
    
    if best_weights.exists():
        dest = weights_dir / "belt_seg_best.pt"
        shutil.copy2(best_weights, dest)
        print(f"  Best weights saved to: {dest}")
    else:
        print(f"  WARNING: best.pt not found at {best_weights}")
    
    if last_weights.exists():
        dest = weights_dir / "belt_seg_last.pt"
        shutil.copy2(last_weights, dest)
        print(f"  Last weights saved to: {dest}")
    
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Weights saved to: {weights_dir.resolve()}")
    print(f"  Results saved to: {Path(args.project) / args.name}")
    print("=" * 60)
    
    # Print validation metrics
    if hasattr(results, 'results_dict'):
        print("\nValidation metrics:")
        for key, val in results.results_dict.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
    
    return results


if __name__ == "__main__":
    main()
