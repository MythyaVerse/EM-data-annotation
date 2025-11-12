# COCO Format Conversion Guide

Complete guide for converting your text detection annotations to COCO 1.0 format for training deep learning models.

## 🎯 What is COCO Format?

COCO (Common Objects in Context) is a standardized format used by most modern object detection frameworks:

- **Detectron2** (Facebook AI Research)
- **MMDetection** (OpenMMLab)
- **YOLOv5/YOLOv8** (with conversion)
- **Mask R-CNN**
- **RetinaNet**
- And many more...

## 📊 Format Comparison

### Your Detection JSON Format (Input)
```json
{
  "frame_number": 0,
  "detections": [
    {
      "box_id": 0,
      "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "confidence": 0.85
    }
  ],
  "total_detections": 5
}
```

### COCO Format (Output)
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "frame_00000.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
      "iscrowd": 0,
      "score": 0.85
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "text",
      "supercategory": "text"
    }
  ]
}
```

## 🚀 Quick Start

### Simple Conversion

```bash
# Convert original detection results to COCO format
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action to-coco
```

**Output:**
```
output/text_detection_updated/sample_video/coco_format/
└── annotations.json
```

### Convert Human-Corrected Annotations

After CVAT correction workflow:

```bash
# Convert corrected annotations to COCO format
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action to-coco \
  --use-corrected
```

This uses annotations from `cvat_export/corrected_detections/` instead of original detections.

### Convert with Train/Val Split

```bash
# Convert and split into training and validation sets
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action to-coco \
  --use-corrected \
  --split \
  --train-ratio 0.8
```

**Output:**
```
output/text_detection_updated/sample_video/coco_format/
├── annotations.json                      # Full dataset
└── splits/
    ├── train_annotations.json           # 80% for training
    └── val_annotations.json             # 20% for validation
```

## 📋 Complete Workflow Examples

### Example 1: Original Detections → COCO

```bash
# Step 1: Run detection
python run_text_detection.py

# Step 2: Convert to COCO
python -m cvat_integration.workflow \
  --video-name video1 \
  --action to-coco
```

### Example 2: Human-Corrected → COCO

```bash
# Step 1: Run detection
python run_text_detection.py

# Step 2: Upload to CVAT
python -m cvat_integration.workflow \
  --video-name video1 \
  --action upload

# Step 3: Human correction in CVAT
# (Open http://localhost:8080 and correct annotations)

# Step 4: Export from CVAT
python -m cvat_integration.workflow \
  --video-name video1 \
  --action export

# Step 5: Convert corrected to COCO
python -m cvat_integration.workflow \
  --video-name video1 \
  --action to-coco \
  --use-corrected
```

### Example 3: Full Pipeline with Train/Val Split

```bash
# Complete pipeline for model training
python run_text_detection.py

python -m cvat_integration.workflow --video-name video1 --action upload
# (Correct in CVAT)
python -m cvat_integration.workflow --video-name video1 --action export

python -m cvat_integration.workflow \
  --video-name video1 \
  --action to-coco \
  --use-corrected \
  --split \
  --train-ratio 0.8

# Now ready for training!
```

## 🛠️ Advanced Usage

### Using Module Directly

```bash
# Convert with custom options
python -m cvat_integration.coco_converter convert \
  --input-dir output/text_detection_updated/video1/frames_and_json \
  --output coco_annotations.json \
  --category-name text \
  --image-ext .jpg
```

### Split Existing COCO File

```bash
# Split an existing COCO annotation file
python -m cvat_integration.coco_converter split \
  --coco-file coco_annotations.json \
  --output-dir coco_splits \
  --train-ratio 0.8 \
  --seed 42
```

### Validate COCO Format

```bash
# Validate COCO format is correct
python -m cvat_integration.coco_converter validate \
  --coco-file coco_annotations.json
```

### Convert with Split in One Command

```bash
python -m cvat_integration.coco_converter convert \
  --input-dir frames_and_json/ \
  --output coco_annotations.json \
  --split \
  --train-ratio 0.8
```

## 📁 Dataset Structure for Training

After conversion, organize your dataset like this:

```
my_text_dataset/
├── images/
│   ├── frame_00000.jpg
│   ├── frame_00001.jpg
│   └── ...
└── annotations/
    ├── train_annotations.json
    └── val_annotations.json
```

**Copy images:**
```bash
# Windows
xcopy "output\text_detection_updated\video1\frames_and_json\*.jpg" "my_text_dataset\images\" /Y

# Linux/Mac
cp output/text_detection_updated/video1/frames_and_json/*.jpg my_text_dataset/images/
```

**Copy annotations:**
```bash
# Windows
copy "output\text_detection_updated\video1\coco_format\splits\*.json" "my_text_dataset\annotations\"

# Linux/Mac
cp output/text_detection_updated/video1/coco_format/splits/*.json my_text_dataset/annotations/
```

## 🔬 Using COCO Annotations for Training

### Detectron2 Example

```python
from detectron2.data.datasets import register_coco_instances

# Register dataset
register_coco_instances(
    "text_train",
    {},
    "my_text_dataset/annotations/train_annotations.json",
    "my_text_dataset/images"
)

register_coco_instances(
    "text_val",
    {},
    "my_text_dataset/annotations/val_annotations.json",
    "my_text_dataset/images"
)

# Use in config
cfg.DATASETS.TRAIN = ("text_train",)
cfg.DATASETS.TEST = ("text_val",)
```

### MMDetection Example

```python
# In config file (configs/my_config.py)
dataset_type = 'CocoDataset'
data_root = 'my_text_dataset/'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_annotations.json',
        img_prefix=data_root + 'images/',
        classes=('text',)
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_annotations.json',
        img_prefix=data_root + 'images/',
        classes=('text',)
    )
)
```

### PyTorch Example (Custom)

```python
from pycocotools.coco import COCO

# Load COCO annotations
coco_train = COCO('my_text_dataset/annotations/train_annotations.json')

# Get all image IDs
img_ids = coco_train.getImgIds()

# Load image info
img_info = coco_train.loadImgs(img_ids[0])[0]
print(f"Image: {img_info['file_name']}")

# Load annotations for an image
ann_ids = coco_train.getAnnIds(imgIds=img_info['id'])
annotations = coco_train.loadAnns(ann_ids)

for ann in annotations:
    bbox = ann['bbox']  # [x, y, width, height]
    segmentation = ann['segmentation']  # [[x1, y1, x2, y2, ...]]
    category = ann['category_id']
```

## 📊 Batch Processing Multiple Videos

```bash
# Process all videos in your dataset
for video in video1 video2 video3 video4 video5; do
  echo "Processing $video..."

  # Run detection (if not done already)
  python run_text_detection.py

  # Upload to CVAT
  python -m cvat_integration.workflow --video-name $video --action upload
done

# After human correction in CVAT...

for video in video1 video2 video3 video4 video5; do
  echo "Exporting $video..."

  # Export from CVAT
  python -m cvat_integration.workflow --video-name $video --action export

  # Convert to COCO
  python -m cvat_integration.workflow \
    --video-name $video \
    --action to-coco \
    --use-corrected
done

# Merge all COCO files (see below)
```

## 🔗 Merging Multiple COCO Files

If you have multiple videos and want one combined dataset:

```python
import json
import os

def merge_coco_files(coco_files, output_file):
    """Merge multiple COCO annotation files into one."""

    merged = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    ann_id_offset = 0

    for coco_file in coco_files:
        with open(coco_file, 'r') as f:
            data = json.load(f)

        # Set categories from first file
        if not merged["categories"]:
            merged["categories"] = data["categories"]

        # Add images with new IDs
        for img in data["images"]:
            old_id = img["id"]
            img["id"] = old_id + image_id_offset
            merged["images"].append(img)

        # Add annotations with new IDs
        for ann in data["annotations"]:
            ann["id"] = ann["id"] + ann_id_offset
            ann["image_id"] = ann["image_id"] + image_id_offset
            merged["annotations"].append(ann)

        # Update offsets
        if data["images"]:
            image_id_offset = merged["images"][-1]["id"] + 1
        if data["annotations"]:
            ann_id_offset = merged["annotations"][-1]["id"] + 1

    # Save merged file
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✓ Merged {len(coco_files)} files")
    print(f"  Total images: {len(merged['images'])}")
    print(f"  Total annotations: {len(merged['annotations'])}")
    print(f"  Output: {output_file}")

# Usage
coco_files = [
    "output/text_detection_updated/video1/coco_format/annotations.json",
    "output/text_detection_updated/video2/coco_format/annotations.json",
    "output/text_detection_updated/video3/coco_format/annotations.json",
]

merge_coco_files(coco_files, "merged_dataset.json")
```

## 🎯 COCO Format Details

### Bounding Box Format

COCO uses `[x, y, width, height]` format:
- `x, y`: Top-left corner coordinates
- `width, height`: Box dimensions

Your polygon coordinates are automatically converted to this format.

### Segmentation Format

Polygon segmentation: `[[x1, y1, x2, y2, x3, y3, x4, y4]]`
- Flattened list of polygon points
- Supports irregular quadrilaterals (your text boxes)

### Area Calculation

Area is computed using the shoelace formula for polygons:
```
area = 0.5 * |sum(x[i] * y[i+1] - x[i+1] * y[i])|
```

This gives accurate area for your rotated text boxes.

## ✅ Validation Checklist

After conversion, verify:

- [ ] `annotations.json` file created
- [ ] Number of images matches your frame count
- [ ] Number of annotations seems reasonable
- [ ] Validation passes (use validate command)
- [ ] Images exist in the source directory
- [ ] Train/val split is roughly correct ratio
- [ ] Category ID is consistent (usually 1 for single class)

## 🐛 Troubleshooting

### "No image files found"
- Ensure images are in the same directory as JSON files
- Check image extension matches (default: .jpg)
- Use `--image-ext .png` if needed

### "Area is zero"
- Polygon has fewer than 3 points
- Coordinates are invalid
- Check original detection JSON

### "Image dimensions are 0x0"
- Image file is corrupted
- PIL cannot read the image format
- Check image files manually

### Validation fails
- Missing required fields in JSON
- Invalid image or annotation IDs
- Category IDs don't match
- Use validate command for details

## 🎓 Best Practices

### 1. Always Use Corrected Annotations for Training

```bash
# ✓ GOOD: Use human-corrected data
python -m cvat_integration.workflow \
  --video-name video1 \
  --action to-coco \
  --use-corrected

# ✗ LESS IDEAL: Use raw detection output
python -m cvat_integration.workflow \
  --video-name video1 \
  --action to-coco
```

### 2. Split Consistently

Use the same seed for reproducible splits:
```bash
python -m cvat_integration.coco_converter split \
  --coco-file annotations.json \
  --output-dir splits \
  --seed 42
```

### 3. Validate Before Training

```bash
python -m cvat_integration.coco_converter validate \
  --coco-file annotations.json
```

### 4. Keep Original Files

COCO conversion is non-destructive. Original detection JSONs remain unchanged.

## 📚 Additional Resources

- **COCO Format Specification**: https://cocodataset.org/#format-data
- **Detectron2 Tutorial**: https://detectron2.readthedocs.io/
- **MMDetection**: https://mmdetection.readthedocs.io/
- **pycocotools**: https://github.com/cocodataset/cocoapi

## 🎉 Summary

You now have three annotation formats:

1. **Detection JSON** - Your original format
2. **CVAT Format** - For human annotation
3. **COCO Format** - For model training

This complete pipeline enables:
- Automated detection
- Human correction
- Standardized training data
- State-of-the-art model training

Happy training! 🚀
