"""
prepare_dataset.py
==================
Prepares the conveyor belt ROI dataset for YOLOv8 segmentation training.
- Reorganizes the directory structure for YOLOv8
- Splits data into train/val sets
- Creates a proper data.yaml config

Usage:
    python prepare_dataset.py --source_dir train/train --output_dir dataset --val_split 0.2
"""

import os
import sys
import random
import shutil
import argparse
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv8 training")
    parser.add_argument("--source_dir", type=str, default="train/train",
                        help="Path to source directory containing train/images and train/labels")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Output directory for the reorganized dataset")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def prepare_dataset(source_dir: str, output_dir: str, val_split: float = 0.2, seed: int = 42):
    """
    Reorganize dataset into YOLOv8, train/val structure.
    
    Args:
        source_dir: Path to source data (containing train/images and train/labels)
        output_dir: Output directory
        val_split: Fraction for validation
        seed: Random seed
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Locate images and labels
    images_dir = source_path / "train" / "images"
    labels_dir = source_path / "train" / "labels"
    
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        sys.exit(1)
    
    # Collect all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Filter: only keep images that have a corresponding label
    paired_files = []
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            paired_files.append((img_file, label_file))
        else:
            print(f"  WARNING: No label for {img_file.name}, skipping")
    
    print(f"Found {len(paired_files)} image-label pairs")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(paired_files)
    
    val_count = int(len(paired_files) * val_split)
    train_count = len(paired_files) - val_count
    
    train_pairs = paired_files[:train_count]
    val_pairs = paired_files[train_count:]
    
    print(f"Split: {train_count} train, {val_count} val")
    
    # Create output directories
    for split in ["train", "val"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_pairs(pairs, split_name):
        for img_file, label_file in pairs:
            shutil.copy2(img_file, output_path / split_name / "images" / img_file.name)
            shutil.copy2(label_file, output_path / split_name / "labels" / label_file.name)
    
    print("Copying training files...")
    copy_pairs(train_pairs, "train")
    print("Copying validation files...")
    copy_pairs(val_pairs, "val")
    
    # Create data.yaml
    data_yaml = {
        "path": str(output_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["belt_roi"]
    }
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\nDataset prepared successfully!")
    print(f"  Output: {output_path.resolve()}")
    print(f"  Config: {yaml_path.resolve()}")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    
    return str(yaml_path.resolve())


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.source_dir, args.output_dir, args.val_split, args.seed)
