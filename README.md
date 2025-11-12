# EM Data Annotation - Text Detection with CVAT Integration

Complete system for text detection with human-in-the-loop annotation using CVAT and COCO format conversion.

## 🎯 Overview

This project provides:
- **Automated Text Detection**: Using DB (Differentiable Binarization) model
- **Human-in-the-Loop Correction**: Integration with CVAT for annotation correction
- **COCO Format Export**: Convert annotations to COCO 1.0 for model training
- **Quality Metrics**: Track improvements from human corrections

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/MythyaVerse/EM-data-annotation.git
cd EM-data-annotation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_cvat.txt
```

### 2. Configuration

```bash
# Copy config template
copy cvat_config.json.example cvat_config.json

# Edit cvat_config.json with your CVAT credentials
```

### 3. Run Detection

```bash
python run_text_detection.py
```

### 4. CVAT Workflow (Optional)

```bash
# Upload to CVAT
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload

# Correct annotations in CVAT web interface
# http://localhost:8080

# Export corrected annotations
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export
```

### 5. Convert to COCO Format

```bash
# Convert to COCO with train/val split
python -m cvat_integration.workflow --video-name VIDEO_NAME --action to-coco --use-corrected --split
```

## 📁 Project Structure

```
EM-data-annotation/
├── cvat_integration/              # CVAT integration package
│   ├── format_converter.py       # JSON ↔ CVAT format
│   ├── cvat_uploader.py          # Upload to CVAT
│   ├── cvat_exporter.py          # Export from CVAT
│   ├── coco_converter.py         # Convert to COCO format
│   └── workflow.py               # End-to-end orchestration
│
├── run_text_detection.py         # Main detection script
├── text_detection.py             # DB detection model
│
├── cvat_config.json.example      # Config template
├── requirements.txt              # Python dependencies
├── requirements_cvat.txt         # CVAT dependencies
│
└── docs/
    ├── CVAT_SETUP.md            # CVAT installation guide
    ├── CVAT_QUICKSTART.md       # Quick start guide
    ├── COCO_FORMAT_GUIDE.md     # COCO conversion guide
    ├── README_CVAT_INTEGRATION.md  # Complete reference
    └── QUICK_REFERENCE.md       # Command cheat sheet
```

## 📚 Documentation

- **[CVAT_SETUP.md](CVAT_SETUP.md)** - Complete CVAT installation and setup
- **[CVAT_QUICKSTART.md](CVAT_QUICKSTART.md)** - Quick start with examples
- **[COCO_FORMAT_GUIDE.md](COCO_FORMAT_GUIDE.md)** - COCO format conversion guide
- **[README_CVAT_INTEGRATION.md](README_CVAT_INTEGRATION.md)** - Full integration reference
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference

## 🔧 Features

### Text Detection
- DB (Differentiable Binarization) model
- Frame-by-frame video processing
- JSON output with polygon coordinates
- Confidence scores for each detection

### CVAT Integration
- Automatic upload of detection results
- Pre-annotations for faster human correction
- Export corrected annotations
- Quality metrics and comparison

### COCO Format Support
- Convert to COCO 1.0 format
- Train/validation split
- Compatible with Detectron2, MMDetection
- Polygon segmentation support

## 🎯 Workflow Examples

### Basic Detection
```bash
python run_text_detection.py
```

### Detection + Human Correction + COCO
```bash
# 1. Detect
python run_text_detection.py

# 2. Upload to CVAT
python -m cvat_integration.workflow --video-name video1 --action upload

# 3. Correct in CVAT (web interface)

# 4. Export corrections
python -m cvat_integration.workflow --video-name video1 --action export

# 5. Convert to COCO
python -m cvat_integration.workflow --video-name video1 --action to-coco --use-corrected --split
```

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- PyTorch
- CVAT SDK
- Pillow
- NumPy

For CVAT server:
- Docker
- Docker Compose

## 📊 Output Formats

### Detection JSON
```json
{
  "frame_number": 0,
  "detections": [
    {
      "box_id": 0,
      "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "confidence": 0.85
    }
  ]
}
```

### COCO Format
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
      "area": 1234.5
    }
  ],
  "categories": [{"id": 1, "name": "text"}]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of the EM Data Annotation initiative.

## 🆘 Support

For issues or questions:
1. Check the documentation in the `docs/` folder
2. Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common commands
3. Open an issue on GitHub

## 🙏 Acknowledgments

- CVAT Team for the annotation tool
- DB Text Detection model
- COCO Dataset format

## 📞 Contact

Project maintained by MythyaVerse Team

---

**Getting Started**: See [CVAT_QUICKSTART.md](CVAT_QUICKSTART.md) for a complete walkthrough.
