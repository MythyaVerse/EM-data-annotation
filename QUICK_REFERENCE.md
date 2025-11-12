# Quick Reference - CVAT Integration & COCO Conversion

Fast command reference for common operations.

## 🚀 Complete Workflows

### Workflow 1: Detection Only
```bash
python run_text_detection.py
```

### Workflow 2: Detection + Human Correction
```bash
# 1. Detect
python run_text_detection.py

# 2. Upload to CVAT
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload

# 3. Correct in CVAT (web interface)
# Open: http://localhost:8080

# 4. Export corrections
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export
```

### Workflow 3: Detection + Correction + COCO
```bash
# 1. Detect
python run_text_detection.py

# 2. Upload
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload

# 3. Correct in CVAT
# ...

# 4. Export
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export

# 5. Convert to COCO
python -m cvat_integration.workflow --video-name VIDEO_NAME --action to-coco --use-corrected --split
```

## 📝 Common Commands

### CVAT Operations
```bash
# Upload to CVAT
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload

# Export from CVAT
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export

# Check status
python -m cvat_integration.workflow --video-name VIDEO_NAME --action status
```

### COCO Conversion
```bash
# Convert original detections
python -m cvat_integration.workflow --video-name VIDEO_NAME --action to-coco

# Convert corrected detections
python -m cvat_integration.workflow --video-name VIDEO_NAME --action to-coco --use-corrected

# Convert with train/val split
python -m cvat_integration.workflow --video-name VIDEO_NAME --action to-coco --use-corrected --split
```

### Direct Module Usage
```bash
# Format conversion
python -m cvat_integration.format_converter to-cvat --input-dir DIR --output FILE
python -m cvat_integration.format_converter from-cvat --cvat-file FILE --output DIR

# COCO conversion
python -m cvat_integration.coco_converter convert --input-dir DIR --output FILE
python -m cvat_integration.coco_converter split --coco-file FILE --output-dir DIR
python -m cvat_integration.coco_converter validate --coco-file FILE

# CVAT export
python -m cvat_integration.cvat_exporter list
python -m cvat_integration.cvat_exporter info --task-id ID
python -m cvat_integration.cvat_exporter export --task-id ID --output-dir DIR
```

## 🔧 CVAT Server Management

```bash
# Start CVAT
cd cvat_local/cvat
docker-compose up -d

# Stop CVAT
docker-compose down

# Restart CVAT
docker-compose restart

# View logs
docker-compose logs -f cvat_server

# Check status
docker ps | grep cvat
```

## 📂 Important File Locations

```
Data Annotation/
├── cvat_config.json                    # CVAT connection settings
├── run_text_detection.py               # Detection script
└── output/text_detection_updated/
    └── [VIDEO_NAME]/
        ├── frames_and_json/            # Original detections
        ├── cvat_annotations.json       # CVAT format
        ├── cvat_workflow_metadata.json # Workflow state
        ├── cvat_export/
        │   ├── corrected_detections/   # Human-corrected
        │   └── export_statistics.json  # Quality metrics
        └── coco_format/
            ├── annotations.json        # COCO format
            └── splits/
                ├── train_annotations.json
                └── val_annotations.json
```

## 🎯 Common Options

```bash
--video-name VIDEO_NAME      # Required: video folder name
--action ACTION              # Required: upload|export|status|to-coco
--config CONFIG_FILE         # CVAT config (default: cvat_config.json)
--output-dir DIR             # Output base dir (default: output/text_detection_updated)

# Upload options
--task-name NAME             # Custom task name
--use-rectangles             # Use rectangles instead of polygons

# Export options
--task-id ID                 # Specific task ID (auto-detected otherwise)

# COCO options
--use-corrected              # Use corrected detections
--split                      # Split into train/val
--train-ratio RATIO          # Train ratio (default: 0.8)
```

## 🔍 Troubleshooting Quick Fixes

### CVAT won't start
```bash
docker-compose down
docker-compose up -d
```

### Can't connect to CVAT
1. Check `cvat_config.json` credentials
2. Verify: http://localhost:8080 accessible
3. Check Docker: `docker ps | grep cvat`

### Import errors
```bash
pip install -r requirements_cvat.txt
```

### Task not found
```bash
# List all tasks
python -m cvat_integration.cvat_exporter list

# Get specific task info
python -m cvat_integration.cvat_exporter info --task-id ID
```

## 📊 Format Summary

| Format | Purpose | Location |
|--------|---------|----------|
| Detection JSON | Original format | `frames_and_json/` |
| CVAT Format | Human annotation | `cvat_annotations.json` |
| Corrected JSON | After CVAT | `cvat_export/corrected_detections/` |
| COCO Format | Model training | `coco_format/annotations.json` |

## ⚡ One-Liners

```bash
# Full pipeline in one go (after detection)
python -m cvat_integration.workflow --video-name V1 --action upload && \
  echo "Correct in CVAT then run:" && \
  echo "python -m cvat_integration.workflow --video-name V1 --action export && \\" && \
  echo "python -m cvat_integration.workflow --video-name V1 --action to-coco --use-corrected --split"

# Process multiple videos
for v in v1 v2 v3; do python -m cvat_integration.workflow --video-name $v --action upload; done

# Export all and convert to COCO
for v in v1 v2 v3; do \
  python -m cvat_integration.workflow --video-name $v --action export && \
  python -m cvat_integration.workflow --video-name $v --action to-coco --use-corrected; \
done

# Quick validation
python -m cvat_integration.coco_converter validate --coco-file path/to/annotations.json
```

## 📚 Documentation Files

- [CVAT_SETUP.md](CVAT_SETUP.md) - Detailed setup instructions
- [CVAT_QUICKSTART.md](CVAT_QUICKSTART.md) - Quick start guide
- [COCO_FORMAT_GUIDE.md](COCO_FORMAT_GUIDE.md) - COCO format details
- [README_CVAT_INTEGRATION.md](README_CVAT_INTEGRATION.md) - Complete reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - This file

## 🎓 Keyboard Shortcuts (CVAT)

| Key | Action |
|-----|--------|
| `N` | Next frame |
| `P` | Previous frame |
| `Ctrl+S` | Save |
| `Shift+N` | New polygon |
| `Esc` | Cancel |
| `Del` | Delete selected |
| `F1` | Help |

## 🔗 Quick Links

- CVAT Web: http://localhost:8080
- CVAT Docs: https://opencv.github.io/cvat/
- COCO Format: https://cocodataset.org/#format-data

## ✅ Pre-Flight Checklist

Before starting:
- [ ] CVAT server running (`docker ps | grep cvat`)
- [ ] `cvat_config.json` configured
- [ ] Detection completed
- [ ] `frames_and_json/` directory exists

After completion:
- [ ] Annotations saved in CVAT
- [ ] Export successful
- [ ] COCO format validated
- [ ] Ready for training

---

**Need help?** Check the detailed guides in the documentation files above.
