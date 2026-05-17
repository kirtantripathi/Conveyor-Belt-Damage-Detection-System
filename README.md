# Conveyor Belt Damage Detection System

Automated detection of **scratch** and **edge damage** defects on conveyor belts using a two-stage approach:

1. **Belt ROI Segmentation** — YOLOv8n-seg model extracts the conveyor belt region
2. **Damage Detection** — Computer vision algorithms detect scratches and edge damage within the belt ROI (CLAHE normalization, Gaussian anomaly mapping, adaptive thresholding, morphological filtering, contour-based edge deviation analysis)

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Training (Google Colab / Kaggle)](#training)
- [Inference](#inference)
- [Output Format](#output-format)
- [Approach Details](#approach-details)

---

## Setup

### Local (for inference)

```bash
# Create virtual environment (optional)
python -m venv my_env
my_env\Scripts\activate    # Windows
# source my_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Required packages
- `ultralytics>=8.0.0` — YOLOv8 framework
- `opencv-python>=4.5.0` — Image processing
- `numpy>=1.20.0` — Numerical computing
- `scipy>=1.7.0` — Signal processing (edge analysis)
- `torch>=1.10.0` — PyTorch (CPU inference)

---

## Project Structure

```
Assignment/
├── pipeline.py                      # Main inference script (CLI)
├── damage_detector.py               # Core damage detection module
├── train.py                         # Training script (for Colab/Kaggle)
├── prepare_dataset.py               # Dataset preparation utility
├── conveyor_belt_training.ipynb     # All-in-one Kaggle/Colab notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── model_weights/                   # Trained model weights
│   └── belt_seg_best.pt             # Best YOLOv8 segmentation weights
├── outputs/                         # Inference output (359 images)
│   ├── <image_name>.jpg             # Annotated images with bounding boxes
│   └── <image_name>.json            # Detection coordinates (JSON)
└── train/                           # Training dataset (YOLO format)
    └── train/
        └── train/
            ├── images/              # 359 conveyor belt images
            └── labels/              # Belt ROI polygon annotations
```

---

## Training

Training is designed for **Google Colab** or **Kaggle** (free GPU). The model trains a YOLOv8n-seg segmentation model on belt ROI annotations.

### Recommended: Use the All-in-One Notebook

The easiest way to train is using **`conveyor_belt_training.ipynb`** — it handles everything (dataset prep, training, inference, visualization, download) in a single notebook. Just upload your dataset, enable GPU, and run all cells.

### Alternative: Manual Training Scripts

### Option A: Google Colab

1. **Open a new Colab notebook** at [colab.research.google.com](https://colab.research.google.com)

2. **Set GPU runtime**: `Runtime → Change runtime type → T4 GPU`

3. **Upload your data** (run in cells):

```python
# Cell 1: Upload the training data
# Zip your Assignment folder first, then upload
from google.colab import files
uploaded = files.upload()  # Upload train.zip
```

```bash
# Cell 2: Unzip
!unzip -q train.zip -d /content/data/
```

4. **Upload training scripts**:
```python
# Upload prepare_dataset.py, train.py
from google.colab import files
files.upload()  # Select prepare_dataset.py and train.py
```

5. **Install dependencies and train**:
```bash
# Cell 3: Install
!pip install ultralytics

# Cell 4: Train
!python train.py \
    --source_dir /content/data/train/train \
    --output_dir /content/dataset \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

6. **Download trained weights**:
```python
# Cell 5: Download weights
from google.colab import files
files.download('model_weights/belt_seg_best.pt')
```

### Option B: Kaggle

1. **Create a new notebook** on [kaggle.com](https://www.kaggle.com)

2. **Add your dataset**: Upload the training data as a Kaggle dataset

3. **Enable GPU**: `Settings → Accelerator → GPU T4 x2`

4. **Train**:
```bash
!pip install ultralytics
!python train.py \
    --source_dir /kaggle/input/your-dataset/train/train \
    --output_dir /kaggle/working/dataset \
    --epochs 100 \
    --batch 16 \
    --device 0
```

5. **Download weights** from `model_weights/belt_seg_best.pt` in the output

### After Training

Copy the downloaded `belt_seg_best.pt` to your local `model_weights/` directory:

```
Assignment/
└── model_weights/
    └── belt_seg_best.pt
```

---

## Inference

Run the pipeline on any folder of conveyor belt images:

```bash
python pipeline.py --image_dir <path_to_images> --output_dir <output_folder>
```

### Examples

```bash
# Run on training images
python pipeline.py --image_dir train/train/train/images --output_dir outputs

# Run with verbose output
python pipeline.py --image_dir test_images --output_dir results --verbose

# Adjust detection sensitivity
python pipeline.py --image_dir images --output_dir results \
    --scratch_thresh 2.0 \
    --edge_thresh 2.5

# Custom model path
python pipeline.py --image_dir images --output_dir results \
    --model_path path/to/weights.pt
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | *required* | Input image directory |
| `--output_dir` | *required* | Output directory |
| `--model_path` | `model_weights/belt_seg_best.pt` | Model weights path |
| `--seg_conf` | `0.5` | Belt segmentation confidence |
| `--scratch_thresh` | `2.5` | Scratch sensitivity (lower = more sensitive) |
| `--edge_thresh` | `3.0` | Edge damage sensitivity (lower = more sensitive) |
| `--verbose` | `false` | Detailed per-image output |
| `--no_annotate` | `false` | Skip annotated image generation |

---

## Output Format

For each image, the pipeline produces:

### 1. Annotated Image (`<original_image_name>.jpg`)

Original image with bounding boxes:
- **Red boxes**: Scratch defects
- **Orange boxes**: Edge damage defects
- Labels show damage type and confidence score

### 2. Detection JSON (`<original_image_name>.json`)

```json
{
  "1": {
    "bbox_coordinates": [x_min, y_min, x_max, y_max]
  },
  "2": {
    "bbox_coordinates": [x_min, y_min, x_max, y_max]
  }
}
```

- `bbox_coordinates`: Pixel coordinates `[x_min, y_min, x_max, y_max]`
- Detection index keys (`"1"`, `"2"`, ...) are identifiers only
- Empty JSON `{}` if no damage detected

---

## Approach Details

### Stage 1: Belt ROI Segmentation

**Model**: YOLOv8n-seg (nano variant for fast inference)

- Trained on 359 images with polygon annotations outlining the belt region
- Handles varying lighting conditions (day/night)
- Outputs a pixel-level mask of the conveyor belt

### Stage 2: Damage Detection

Since only belt ROI annotations exist (no damage labels), damage detection uses **computer vision techniques**:

#### Scratch Detection
1. Extract belt region using segmentation mask
2. Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
3. Compute anomaly map: difference between original and Gaussian-smoothed version
4. **Adaptive thresholding** based on local statistics (handles day/night automatically)
5. Morphological operations to connect scratch segments
6. Filter by **aspect ratio** (scratches are elongated) and **minimum area**
7. Non-maximum suppression to merge overlapping detections

#### Edge Damage Detection
1. Extract belt edge contour from segmentation mask (left and right edges)
2. Fit **smooth reference curve** using moving average filter
3. Compute **deviation** of actual edge from smooth reference
4. Regions with large deviations (> threshold × standard deviation) = edge damage
5. Group consecutive damaged points into segments
6. Generate bounding boxes around damaged edge regions

### Key Design Decisions

- **No pre-labeled damage data** → unsupervised CV approach instead of supervised detection
- **CLAHE normalization** → robust to day/night lighting variations
- **Adaptive thresholds** → statistics-based, avoids hardcoded values
- **YOLOv8n-seg** → fastest variant, suitable for CPU inference
- **NMS** → prevents duplicate detections

---

## Evaluation

Performance is measured using **mF1@0.5–0.95**:

- Bounding box matching via greedy IoU assignment
- F1 score computed at IoU thresholds 0.50 to 0.95 (step 0.05)
- Final metric = average F1 across all thresholds

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Model weights not found` | Download from Colab/Kaggle after training |
| `No belt detected` | Lower `--seg_conf` threshold |
| `Too many false positives` | Increase `--scratch_thresh` or `--edge_thresh` |
| `Missing detections` | Decrease `--scratch_thresh` or `--edge_thresh` |
| `Slow inference` | Images are 4K; resize input or accept ~2s/image on CPU |
