# Text Detection Pipeline + CVAT Integration

Complete human-in-the-loop system for text detection with annotation correction using CVAT.

## 🎯 Overview

This system integrates your text detection pipeline with CVAT (Computer Vision Annotation Tool) to enable:

1. **Automated Detection**: Process videos with DB text detection model
2. **Human Review**: Import detections into CVAT for correction
3. **Quality Improvement**: Export corrected annotations
4. **Iterative Learning**: Use corrected data for model improvement

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Video Input                                                    │
│      ↓                                                          │
│  Text Detection (run_text_detection.py)                        │
│      ↓                                                          │
│  Detection Results (frames + JSON)                             │
│      ↓                                                          │
│  Format Conversion (detection → CVAT)                          │
│      ↓                                                          │
│  CVAT Upload (create task + import pre-annotations)            │
│      ↓                                                          │
│  Human Correction (add/delete/adjust boxes in CVAT)            │
│      ↓                                                          │
│  Export from CVAT                                               │
│      ↓                                                          │
│  Format Conversion (CVAT → detection)                          │
│      ↓                                                          │
│  Corrected Annotations (same format as input)                  │
│      ↓                                                          │
│  Model Training / Evaluation                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Data Annotation/
├── run_text_detection.py              # Your detection pipeline
├── text_detection.py                  # DB detection model
│
├── cvat_integration/                  # CVAT integration package
│   ├── __init__.py
│   ├── format_converter.py           # JSON ↔ CVAT format
│   ├── cvat_uploader.py              # Upload to CVAT
│   ├── cvat_exporter.py              # Export from CVAT
│   └── workflow.py                   # End-to-end orchestration
│
├── cvat_config.json                   # CVAT connection settings
├── requirements_cvat.txt              # Python dependencies
│
├── CVAT_SETUP.md                      # Detailed setup guide
├── CVAT_QUICKSTART.md                 # Quick start guide
└── README_CVAT_INTEGRATION.md         # This file
```

## 🚀 Installation

### 1. Install CVAT Server (One-time setup)

**Option A: Docker Compose (Recommended)**

```bash
# Clone CVAT repository
git clone https://github.com/cvat-ai/cvat
cd cvat

# Start CVAT services
docker-compose up -d

# Create admin user
docker exec -it cvat_server bash -ic 'python3 manage.py createsuperuser'
```

**Option B: Use Existing CVAT Instance**

If you have CVAT running elsewhere, just configure the connection in `cvat_config.json`.

### 2. Install Python Dependencies

```bash
# Install CVAT SDK
pip install -r requirements_cvat.txt

# Verify installation
python -c "import cvat_sdk; print('✓ CVAT SDK ready')"
```

### 3. Configure Connection

Edit `cvat_config.json`:
```json
{
  "cvat_host": "http://localhost:8080",
  "cvat_username": "admin",
  "cvat_password": "your_password",
  "project_name": "Text Detection Correction"
}
```

### 4. Verify Setup

```bash
# Check CVAT is accessible
curl http://localhost:8080

# Or open in browser
# http://localhost:8080
```

## 📖 Usage Guide

### Basic Workflow (3 Commands)

```bash
# 1. Run detection
python run_text_detection.py

# 2. Upload to CVAT
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload

# 3. (After human correction in CVAT web interface)
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export
```

### Detailed Steps

#### Step 1: Run Text Detection

```bash
python run_text_detection.py
```

**Output:**
```
output/text_detection_updated/
└── [video_name]/
    ├── annotated_video_detection.mp4
    ├── detection_statistics.json
    └── frames_and_json/
        ├── frame_00000.jpg
        ├── frame_00000.json
        ├── frame_00001.jpg
        ├── frame_00001.json
        └── ...
```

Each JSON contains:
```json
{
  "frame_number": 0,
  "detections": [
    {
      "box_id": 0,
      "coordinates": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "confidence": 0.85
    }
  ],
  "total_detections": 5
}
```

#### Step 2: Upload to CVAT

```bash
python -m cvat_integration.workflow \
  --video-name [VIDEO_NAME] \
  --action upload
```

**What it does:**
1. ✓ Converts detection JSON to CVAT format
2. ✓ Creates task in CVAT
3. ✓ Uploads frame images
4. ✓ Imports pre-annotations (your detections)

**Output:**
```
Task ID: 123
Task URL: http://localhost:8080/tasks/123
✓ Ready for annotation
```

#### Step 3: Human Correction (CVAT Web Interface)

1. Open task URL in browser: `http://localhost:8080/tasks/123`
2. Click "Job #1" to start annotating
3. Review and correct detections:
   - **Add** missing text boxes
   - **Delete** false positives
   - **Adjust** box boundaries
   - **Merge/Split** as needed
4. Save frequently (`Ctrl+S`)

**CVAT Shortcuts:**
- `N` - Next frame
- `P` - Previous frame
- `Ctrl+S` - Save
- `Shift+N` - Draw new polygon
- `Esc` - Cancel current action

#### Step 4: Export Corrected Annotations

```bash
python -m cvat_integration.workflow \
  --video-name [VIDEO_NAME] \
  --action export
```

**What it does:**
1. ✓ Exports annotations from CVAT
2. ✓ Converts back to detection JSON format
3. ✓ Compares with original detections
4. ✓ Generates quality metrics

**Output:**
```
output/text_detection_updated/[video_name]/cvat_export/
├── cvat_annotations.json              # Raw CVAT format
├── corrected_detections/              # Converted to detection format
│   ├── frame_00000.json
│   ├── frame_00001.json
│   └── ...
└── export_statistics.json             # Quality metrics
```

**Quality Metrics:**
```json
{
  "comparison": {
    "original_detections": 856,
    "corrected_detections": 892,
    "added_detections": 48,
    "removed_detections": 12,
    "change_percentage": 4.2
  }
}
```

## 🛠️ Advanced Features

### Check Workflow Status

```bash
python -m cvat_integration.workflow \
  --video-name VIDEO_NAME \
  --action status
```

Shows:
- Current workflow stage
- Task ID and URL
- Live status from CVAT
- Next steps

### Custom Task Names

```bash
python -m cvat_integration.workflow \
  --video-name VIDEO_NAME \
  --action upload \
  --task-name "Production Dataset - Batch 1"
```

### Rectangle Format (instead of polygons)

```bash
python -m cvat_integration.workflow \
  --video-name VIDEO_NAME \
  --action upload \
  --use-rectangles
```

### Export with Specific Task ID

```bash
python -m cvat_integration.workflow \
  --video-name VIDEO_NAME \
  --action export \
  --task-id 123
```

## 🔧 Module-Level Usage

### Format Converter

```bash
# Detection → CVAT (polygons)
python -m cvat_integration.format_converter to-cvat \
  --input-dir frames_and_json/ \
  --output cvat_annotations.json

# Detection → CVAT (rectangles)
python -m cvat_integration.format_converter to-cvat-rect \
  --input-dir frames_and_json/ \
  --output cvat_annotations.json

# CVAT → Detection
python -m cvat_integration.format_converter from-cvat \
  --cvat-file cvat_annotations.json \
  --output corrected_detections/
```

### CVAT Uploader

```bash
python -m cvat_integration.cvat_uploader \
  --task-name "My Task" \
  --frames-dir frames_and_json/ \
  --annotations cvat_annotations.json \
  --project-name "My Project"
```

### CVAT Exporter

```bash
# List tasks
python -m cvat_integration.cvat_exporter list

# Get task info
python -m cvat_integration.cvat_exporter info --task-id 123

# Export task
python -m cvat_integration.cvat_exporter export \
  --task-id 123 \
  --output-dir corrected/ \
  --compare-with frames_and_json/
```

## 📊 Use Cases

### 1. Model Training Data Creation

```bash
# Process multiple videos
for video in video1 video2 video3; do
  python run_text_detection.py  # Process video
  python -m cvat_integration.workflow --video-name $video --action upload
done

# Annotators correct in CVAT...

# Export all corrected data
for video in video1 video2 video3; do
  python -m cvat_integration.workflow --video-name $video --action export
done

# Use corrected_detections/ for training
```

### 2. Quality Assurance

```bash
# Run detection on test set
python run_text_detection.py

# Upload to CVAT for expert review
python -m cvat_integration.workflow --video-name test_video --action upload

# Experts review and correct
# ...

# Export and compare
python -m cvat_integration.workflow --video-name test_video --action export

# Check export_statistics.json for quality metrics
```

### 3. Active Learning

```bash
# 1. Detect on unlabeled data
python run_text_detection.py

# 2. Filter low-confidence detections (modify workflow.py)
# 3. Upload uncertain cases to CVAT
python -m cvat_integration.workflow --video-name uncertain_cases --action upload

# 4. Human labeling of difficult cases
# 5. Export and add to training set
python -m cvat_integration.workflow --video-name uncertain_cases --action export
```

## 🎯 Best Practices

### 1. Annotation Guidelines

Create clear guidelines:
- What counts as "text"? (printed, handwritten, logos?)
- Box placement: Tight vs. breathing room
- When to merge adjacent text
- How to handle rotated/curved text

### 2. Quality Control

- Review random sample before full annotation
- Have senior annotator spot-check
- Use comparison metrics to identify issues
- Track inter-annotator agreement

### 3. Efficient Annotation

- Use keyboard shortcuts extensively
- Batch similar frames together
- Start with high-confidence detections
- Flag unclear cases for review

### 4. Iterative Improvement

```
Round 1: Detect → Correct → Export
         ↓
Round 2: Retrain → Better Detection → Correct → Export
         ↓
Round 3: Fine-tune → Even Better → ...
```

## 🐛 Troubleshooting

### CVAT Connection Issues

```bash
# Check CVAT is running
docker ps | grep cvat

# View logs
docker-compose logs -f cvat_server

# Restart
docker-compose restart

# Check port
netstat -an | grep 8080
```

### Import Errors

```bash
# Reinstall dependencies
pip uninstall cvat-sdk cvat-cli
pip install -r requirements_cvat.txt

# Verify
python -c "from cvat_sdk import make_client"
```

### Authentication Errors

- Check `cvat_config.json` credentials
- Try logging in via web interface first
- Ensure admin user was created properly

### Task Creation Fails

- Check disk space: `docker system df`
- Verify images are valid: `file frames_and_json/*.jpg`
- Try smaller batch first (10 frames)
- Check CVAT server logs

### Export Fails

- Ensure task exists: visit task URL
- Check annotations were saved in CVAT
- Verify network connection to CVAT
- Check CVAT user has export permissions

## 📚 Documentation

- **[CVAT_SETUP.md](CVAT_SETUP.md)** - Detailed setup instructions
- **[CVAT_QUICKSTART.md](CVAT_QUICKSTART.md)** - Quick start guide with examples
- **[README_CVAT_INTEGRATION.md](README_CVAT_INTEGRATION.md)** - This file

## 🤝 Contributing

To extend this integration:

1. **Add new format**: Extend `format_converter.py`
2. **Custom workflows**: Create new methods in `workflow.py`
3. **Validation**: Add checks in `cvat_uploader.py`
4. **Metrics**: Enhance comparison in `cvat_exporter.py`

## 📝 License

This integration layer is part of your text detection pipeline project.

## 🆘 Support

Issues or questions:
1. Check troubleshooting section above
2. Review CVAT documentation: https://opencv.github.io/cvat/
3. Check CVAT SDK docs: https://github.com/cvat-ai/cvat/tree/develop/cvat-sdk

## 🎉 Getting Started

Ready to begin? Follow these steps:

1. ✅ Read [CVAT_SETUP.md](CVAT_SETUP.md) for installation
2. ✅ Follow [CVAT_QUICKSTART.md](CVAT_QUICKSTART.md) for first workflow
3. ✅ Process your first video end-to-end
4. ✅ Train your annotation team
5. ✅ Scale up to full dataset

Happy annotating! 🚀
