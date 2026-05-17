"""
damage_detector.py
==================
Core damage detection module for conveyor belt images.

Two-stage approach:
  Stage 1: Belt ROI extraction using YOLOv8 segmentation model
  Stage 2: Damage detection within belt ROI using computer vision:
    - ScratchDetector: Detects surface scratches via texture anomaly analysis
    - EdgeDamageDetector: Detects belt edge damage via contour irregularity analysis
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Data class for detections
# ---------------------------------------------------------------------------
class Detection:
    """Single damage detection result."""
    def __init__(self, bbox: Tuple[int, int, int, int], damage_type: str, confidence: float = 1.0):
        """
        Args:
            bbox: (x_min, y_min, x_max, y_max) in pixels
            damage_type: 'scratch' or 'edge_damage'
            confidence: detection confidence [0, 1]
        """
        self.bbox = bbox
        self.damage_type = damage_type
        self.confidence = confidence

    def __repr__(self):
        return f"Detection({self.damage_type}, bbox={self.bbox}, conf={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Scratch Detector
# ---------------------------------------------------------------------------
class ScratchDetector:
    """
    Detects surface scratches on the conveyor belt.
    
    Approach:
      1. Extract belt ROI and normalize lighting with CLAHE
      2. Compute difference between original and heavily smoothed version (anomaly map)
      3. Threshold to find anomalous bright/dark streaks
      4. Use morphological operations to connect scratch segments
      5. Filter connected components by shape (scratches are elongated)
      6. Return bounding boxes around detected scratches
    """
    
    def __init__(
        self,
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        blur_ksize: int = 31,
        thresh_factor: float = 2.5,
        min_area: int = 500,
        min_aspect_ratio: float = 3.0,
        morph_ksize: int = 5,
        dilate_iter: int = 2,
    ):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.blur_ksize = blur_ksize
        self.thresh_factor = thresh_factor
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        self.morph_ksize = morph_ksize
        self.dilate_iter = dilate_iter

    def detect(self, image: np.ndarray, belt_mask: np.ndarray) -> List[Detection]:
        """
        Detect scratches within the belt region.
        
        Args:
            image: BGR image (H, W, 3)
            belt_mask: Binary mask of belt region (H, W), 255=belt
            
        Returns:
            List of Detection objects for scratches
        """
        h, w = image.shape[:2]
        
        # Scale minimum area based on image resolution
        scale = (h * w) / (1920 * 1080)
        min_area = int(self.min_area * scale)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for lighting normalization
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
        enhanced = clahe.apply(gray)
        
        # Apply belt mask
        masked = cv2.bitwise_and(enhanced, enhanced, mask=belt_mask)
        
        # Compute anomaly map: difference from heavily smoothed version
        smoothed = cv2.GaussianBlur(masked, (self.blur_ksize, self.blur_ksize), 0)
        
        # Anomaly = absolute difference from smooth background
        diff = cv2.absdiff(masked, smoothed)
        
        # Only consider belt region
        diff = cv2.bitwise_and(diff, diff, mask=belt_mask)
        
        # Shrink belt mask to avoid edge artifacts (erode by a margin)
        margin = int(min(h, w) * 0.03)
        kernel_erode = np.ones((margin, margin), np.uint8)
        inner_mask = cv2.erode(belt_mask, kernel_erode, iterations=1)
        diff = cv2.bitwise_and(diff, diff, mask=inner_mask)
        
        # Adaptive thresholding based on statistics within belt
        belt_pixels = diff[inner_mask > 0]
        if len(belt_pixels) == 0:
            return []
        
        mean_val = np.mean(belt_pixels)
        std_val = np.std(belt_pixels)
        threshold = mean_val + self.thresh_factor * std_val
        threshold = max(threshold, 15)  # minimum threshold to avoid noise
        
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to connect scratch segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.dilate(binary, kernel, iterations=self.dilate_iter)
        
        # Find connected components
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Compute aspect ratio (max/min side)
            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
            
            # Scratches are elongated features
            if aspect_ratio >= self.min_aspect_ratio:
                # Compute a confidence based on anomaly intensity
                roi_diff = diff[y:y+bh, x:x+bw]
                roi_mask = belt_mask[y:y+bh, x:x+bw]
                if np.sum(roi_mask > 0) > 0:
                    intensity = np.mean(roi_diff[roi_mask > 0])
                    conf = min(1.0, intensity / (threshold * 2))
                else:
                    conf = 0.5
                
                bbox = (x, y, x + bw, y + bh)
                detections.append(Detection(bbox, "scratch", conf))
        
        # Non-maximum suppression
        detections = self._nms(detections, iou_thresh=0.3)
        
        return detections
    
    def _nms(self, detections: List[Detection], iou_thresh: float = 0.3) -> List[Detection]:
        """Simple NMS on detections."""
        if len(detections) == 0:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, det_i in enumerate(detections):
            if i in suppressed:
                continue
            keep.append(det_i)
            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue
                iou = self._compute_iou(det_i.bbox, detections[j].bbox)
                if iou > iou_thresh:
                    suppressed.add(j)
        
        return keep
    
    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two boxes (x_min, y_min, x_max, y_max)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Edge Damage Detector
# ---------------------------------------------------------------------------
class EdgeDamageDetector:
    """
    Detects edge damage on the conveyor belt.
    
    Approach:
      1. Extract belt edge contours from the segmentation mask
      2. Separate into left edge and right edge
      3. Fit a smooth reference curve to each edge
      4. Compute deviation of actual edge from smooth reference
      5. Regions with large deviations = edge damage
      6. Return bounding boxes around damaged edge regions
    """
    
    def __init__(
        self,
        smooth_window: int = 51,
        deviation_threshold: float = 3.0,
        min_damage_length: int = 20,
        edge_bbox_width: int = 80,
    ):
        self.smooth_window = smooth_window
        self.deviation_threshold = deviation_threshold
        self.min_damage_length = min_damage_length
        self.edge_bbox_width = edge_bbox_width
    
    def detect(self, image: np.ndarray, belt_mask: np.ndarray) -> List[Detection]:
        """
        Detect edge damage on the belt.
        
        Args:
            image: BGR image (H, W, 3)
            belt_mask: Binary mask of belt region (H, W), 255=belt
            
        Returns:
            List of Detection objects for edge damage
        """
        h, w = image.shape[:2]
        
        # Scale parameters based on image resolution
        scale_h = h / 1080
        scale_w = w / 1920
        scale = (scale_h + scale_w) / 2
        
        smooth_window = max(11, int(self.smooth_window * scale) | 1)  # ensure odd
        min_damage_length = max(5, int(self.min_damage_length * scale))
        edge_bbox_width = max(20, int(self.edge_bbox_width * scale))
        
        detections = []
        
        # Extract left and right edge profiles
        left_edge, right_edge = self._extract_edge_profiles(belt_mask)
        
        if left_edge is not None and len(left_edge) > smooth_window:
            left_detections = self._detect_edge_anomalies(
                left_edge, "left", h, w, smooth_window, min_damage_length, edge_bbox_width
            )
            detections.extend(left_detections)
        
        if right_edge is not None and len(right_edge) > smooth_window:
            right_detections = self._detect_edge_anomalies(
                right_edge, "right", h, w, smooth_window, min_damage_length, edge_bbox_width
            )
            detections.extend(right_detections)
        
        return detections
    
    def _extract_edge_profiles(self, mask: np.ndarray):
        """
        Extract left and right belt edge x-coordinates for each row of the mask.
        
        Returns:
            left_edge: array of (row, x_coord) for left edge
            right_edge: array of (row, x_coord) for right edge
        """
        h, w = mask.shape
        
        left_xs = []
        right_xs = []
        rows = []
        
        for y in range(h):
            row_pixels = np.where(mask[y] > 0)[0]
            if len(row_pixels) > 10:  # need enough belt pixels in this row
                left_xs.append(row_pixels[0])
                right_xs.append(row_pixels[-1])
                rows.append(y)
        
        if len(rows) < 20:
            return None, None
        
        rows = np.array(rows)
        left_xs = np.array(left_xs, dtype=float)
        right_xs = np.array(right_xs, dtype=float)
        
        left_edge = np.column_stack([rows, left_xs])
        right_edge = np.column_stack([rows, right_xs])
        
        return left_edge, right_edge
    
    def _detect_edge_anomalies(
        self, edge_profile, side, img_h, img_w,
        smooth_window, min_damage_length, edge_bbox_width
    ) -> List[Detection]:
        """
        Detect anomalies in a belt edge profile.
        
        Args:
            edge_profile: (N, 2) array of (row, x_coord)
            side: 'left' or 'right'
            img_h, img_w: image dimensions
        """
        rows = edge_profile[:, 0].astype(int)
        x_coords = edge_profile[:, 1]
        
        # Fit smooth reference using uniform filter (moving average)
        smooth_x = uniform_filter1d(x_coords, size=smooth_window)
        
        # Compute deviation from smooth reference
        deviation = x_coords - smooth_x
        
        # For left edge: inward deviation means edge is going right (damage = notch inward)
        # For right edge: inward deviation means edge is going left
        # We care about any large deviation (both inward and outward)
        abs_deviation = np.abs(deviation)
        
        # Compute statistics for adaptive thresholding
        mean_dev = np.mean(abs_deviation)
        std_dev = np.std(abs_deviation)
        threshold = mean_dev + self.deviation_threshold * std_dev
        threshold = max(threshold, 5.0)  # minimum threshold
        
        # Find regions where deviation exceeds threshold
        anomaly_mask = abs_deviation > threshold
        
        # Group consecutive anomalous points into segments
        detections = []
        segments = self._find_consecutive_segments(anomaly_mask, min_length=min_damage_length)
        
        for start_idx, end_idx in segments:
            # Get row range for this segment
            y_min = int(rows[start_idx])
            y_max = int(rows[end_idx])
            
            # Get x range based on both actual and smooth edge
            segment_x = x_coords[start_idx:end_idx + 1]
            segment_smooth = smooth_x[start_idx:end_idx + 1]
            
            x_min_actual = int(np.min(segment_x))
            x_max_actual = int(np.max(segment_x))
            x_min_smooth = int(np.min(segment_smooth))
            x_max_smooth = int(np.max(segment_smooth))
            
            # Create bbox that covers the damaged region
            x_min = max(0, min(x_min_actual, x_min_smooth) - edge_bbox_width // 4)
            x_max = min(img_w, max(x_max_actual, x_max_smooth) + edge_bbox_width // 4)
            
            # Ensure minimum width
            if (x_max - x_min) < edge_bbox_width // 2:
                center_x = (x_min + x_max) // 2
                x_min = max(0, center_x - edge_bbox_width // 2)
                x_max = min(img_w, center_x + edge_bbox_width // 2)
            
            # Ensure minimum height
            if (y_max - y_min) < 20:
                center_y = (y_min + y_max) // 2
                y_min = max(0, center_y - 20)
                y_max = min(img_h, center_y + 20)
            
            # Compute confidence based on deviation magnitude
            segment_dev = abs_deviation[start_idx:end_idx + 1]
            max_dev = np.max(segment_dev)
            conf = min(1.0, max_dev / (threshold * 3))
            
            bbox = (x_min, y_min, x_max, y_max)
            detections.append(Detection(bbox, "edge_damage", conf))
        
        return detections
    
    @staticmethod
    def _find_consecutive_segments(mask: np.ndarray, min_length: int = 10):
        """Find consecutive True segments in a boolean mask, with minimum length."""
        segments = []
        start = None
        
        for i in range(len(mask)):
            if mask[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if (i - start) >= min_length:
                        segments.append((start, i - 1))
                    start = None
        
        # Handle segment at the end
        if start is not None and (len(mask) - start) >= min_length:
            segments.append((start, len(mask) - 1))
        
        return segments


# ---------------------------------------------------------------------------
# Combined Belt Damage Detector
# ---------------------------------------------------------------------------
class BeltDamageDetector:
    """
    Complete belt damage detection pipeline.
    
    Stage 1: Belt ROI segmentation using YOLOv8
    Stage 2: Damage detection (scratch + edge damage) using CV
    """
    
    def __init__(
        self,
        model_path: str = "model_weights/belt_seg_best.pt",
        seg_conf: float = 0.5,
        scratch_params: Optional[Dict] = None,
        edge_params: Optional[Dict] = None,
    ):
        """
        Args:
            model_path: Path to YOLOv8 segmentation model weights
            seg_conf: Confidence threshold for belt segmentation
            scratch_params: Override parameters for ScratchDetector
            edge_params: Override parameters for EdgeDamageDetector
        """
        from ultralytics import YOLO
        
        self.model = YOLO(model_path)
        self.seg_conf = seg_conf
        
        self.scratch_detector = ScratchDetector(**(scratch_params or {}))
        self.edge_detector = EdgeDamageDetector(**(edge_params or {}))
    
    def get_belt_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get binary belt mask from the segmentation model.
        
        Args:
            image: BGR image
            
        Returns:
            Binary mask (H, W) with 255=belt, or None if no belt detected
        """
        results = self.model(image, conf=self.seg_conf, verbose=False)
        
        if results and results[0].masks is not None:
            masks = results[0].masks
            if len(masks) > 0:
                # Get the largest mask (main belt region)
                best_mask = None
                best_area = 0
                
                for mask in masks.data:
                    mask_np = mask.cpu().numpy()
                    area = np.sum(mask_np > 0.5)
                    if area > best_area:
                        best_area = area
                        best_mask = mask_np
                
                if best_mask is not None:
                    # Resize mask to image dimensions
                    h, w = image.shape[:2]
                    belt_mask = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    belt_mask = (belt_mask > 0.5).astype(np.uint8) * 255
                    return belt_mask
        
        return None
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run complete damage detection pipeline on an image.
        
        Args:
            image: BGR image
            
        Returns:
            List of Detection objects
        """
        # Stage 1: Get belt mask
        belt_mask = self.get_belt_mask(image)
        
        if belt_mask is None:
            print("  WARNING: No belt detected in image")
            return []
        
        # Stage 2: Detect damage within belt region
        all_detections = []
        
        # Detect scratches
        scratches = self.scratch_detector.detect(image, belt_mask)
        all_detections.extend(scratches)
        
        # Detect edge damage
        edge_damages = self.edge_detector.detect(image, belt_mask)
        all_detections.extend(edge_damages)
        
        return all_detections
    
    def detect_from_path(self, image_path: str) -> Tuple[np.ndarray, List[Detection]]:
        """
        Load image and run detection.
        
        Args:
            image_path: Path to image file
            
        Returns:
            (image, detections) tuple
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        detections = self.detect(image)
        return image, detections


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Draw bounding boxes on the image for each detection.
    
    Args:
        image: BGR image
        detections: List of Detection objects
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Color map for damage types
    colors = {
        "scratch": (0, 0, 255),       # Red (BGR)
        "edge_damage": (0, 165, 255), # Orange (BGR)
    }
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        color = colors.get(det.damage_type, (255, 255, 255))
        
        # Draw bounding box
        thickness = max(2, int(min(image.shape[:2]) * 0.003))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{det.damage_type} ({det.confidence:.2f})"
        font_scale = max(0.5, min(image.shape[:2]) * 0.0006)
        font_thickness = max(1, int(font_scale * 2))
        
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Background rectangle for text
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color, -1
        )
        
        # Text
        cv2.putText(
            annotated, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), font_thickness
        )
    
    return annotated


def detections_to_json(detections: List[Detection]) -> Dict:
    """
    Convert detections list to the required JSON format.
    
    Output format:
    {
        "1": {"bbox_coordinates": [x_min, y_min, x_max, y_max]},
        "2": {...}
    }
    """
    result = {}
    for i, det in enumerate(detections, start=1):
        result[str(i)] = {
            "bbox_coordinates": list(det.bbox)
        }
    return result
