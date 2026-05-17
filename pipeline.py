"""
pipeline.py
===========
Inference pipeline for conveyor belt damage detection.

Processes images from a directory,detects scratch and edge_damage defects,
and outputs annotated images + per-image JSON detection files.

Usage:
    python pipeline.py --image_dir <path_to_image_folder> --output_dir <folder>

Example:
    python pipeline.py --image_dir train/train/train/images --output_dir outputs
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from damage_detector import (
    BeltDamageDetector,
    draw_detections,
    detections_to_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conveyor Belt Damage Detection Pipeline"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to output directory for annotated images and JSON files"
    )
    parser.add_argument(
        "--model_path", type=str, default="model_weights/belt_seg_best.pt",
        help="Path to YOLOv8 segmentation model weights"
    )
    parser.add_argument(
        "--seg_conf", type=float, default=0.5,
        help="Confidence threshold for belt segmentation"
    )
    parser.add_argument(
        "--scratch_thresh", type=float, default=2.5,
        help="Threshold factor for scratch detection (higher = fewer detections)"
    )
    parser.add_argument(
        "--edge_thresh", type=float, default=3.0,
        help="Threshold factor for edge damage detection (higher = fewer detections)"
    )
    parser.add_argument(
        "--no_annotate", action="store_true",
        help="Skip generating annotated images (JSON only)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output for each image"
    )
    return parser.parse_args()


def process_images(
    image_dir: str,
    output_dir: str,
    model_path: str = "model_weights/belt_seg_best.pt",
    seg_conf: float = 0.5,
    scratch_thresh: float = 2.5,
    edge_thresh: float = 3.0,
    annotate: bool = True,
    verbose: bool = False,
):
    """
    Main processing function.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for output files
        model_path: Path to belt segmentation model
        seg_conf: Segmentation confidence threshold
        scratch_thresh: Scratch detection sensitivity
        edge_thresh: Edge damage detection sensitivity
        annotate: Whether to generate annotated images
        verbose: Verbose output
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    
    # Validate paths
    if not image_dir.exists():
        print(f"ERROR: Image directory does not exist: {image_dir}")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"ERROR: Model weights not found: {model_path}")
        print("  Please train the model first (see README.md)")
        print("  Or specify the correct path with --model_path")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_path}")
    print()
    
    # Initialize detector
    print("Loading model...")
    detector = BeltDamageDetector(
        model_path=model_path,
        seg_conf=seg_conf,
        scratch_params={"thresh_factor": scratch_thresh},
        edge_params={"deviation_threshold": edge_thresh},
    )
    print("Model loaded successfully.\n")
    
    # Process each image
    total_detections = 0
    total_scratches = 0
    total_edge_damages = 0
    start_time = time.time()
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = img_path.stem
        
        if verbose:
            print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        else:
            # Progress indicator
            print(f"\r  Processing: {idx}/{len(image_files)} ({img_name[:40]}...)", end="", flush=True)
        
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"\n  WARNING: Cannot read image: {img_path}")
                continue
            
            # Run detection
            detections = detector.detect(image)
            
            # Count by type
            n_scratches = sum(1 for d in detections if d.damage_type == "scratch")
            n_edge = sum(1 for d in detections if d.damage_type == "edge_damage")
            total_detections += len(detections)
            total_scratches += n_scratches
            total_edge_damages += n_edge
            
            if verbose:
                print(f"  Found {len(detections)} detections "
                      f"({n_scratches} scratch, {n_edge} edge_damage)")
            
            # Save annotated image
            if annotate and detections:
                annotated = draw_detections(image, detections)
                out_img_path = output_dir / f"{img_name}.jpg"
                cv2.imwrite(str(out_img_path), annotated)
            elif annotate:
                # Save original image if no detections (still expected output)
                out_img_path = output_dir / f"{img_name}.jpg"
                cv2.imwrite(str(out_img_path), image)
            
            # Save JSON
            json_data = detections_to_json(detections)
            out_json_path = output_dir / f"{img_name}.json"
            with open(out_json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        except Exception as e:
            print(f"\n  ERROR processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n\n{'=' * 60}")
    print(f"  Processing Complete!")
    print(f"{'=' * 60}")
    print(f"  Images processed:  {len(image_files)}")
    print(f"  Total detections:  {total_detections}")
    print(f"    - Scratches:     {total_scratches}")
    print(f"    - Edge damage:   {total_edge_damages}")
    print(f"  Time elapsed:      {elapsed:.1f}s ({elapsed/len(image_files):.2f}s/image)")
    print(f"  Output saved to:   {output_dir.resolve()}")
    print(f"{'=' * 60}")


def main():
    args = parse_args()
    
    process_images(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        seg_conf=args.seg_conf,
        scratch_thresh=args.scratch_thresh,
        edge_thresh=args.edge_thresh,
        annotate=not args.no_annotate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
