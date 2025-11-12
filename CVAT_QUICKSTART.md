# CVAT Integration - Quick Start Guide

This guide will walk you through the complete workflow from text detection to human-corrected annotations.

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Install CVAT SDK
pip install cvat-sdk

# Verify installation
python -c "import cvat_sdk; print('CVAT SDK installed successfully')"
```

### 2. Start CVAT Server

```bash
# Option A: Using existing CVAT installation
cd cvat_local/cvat
docker-compose up -d

# Option B: Quick Docker command (if you have the compose file)
docker-compose -f docker-compose.yml up -d

# Verify CVAT is running
# Open browser: http://localhost:8080
```

### 3. Configure Connection

Edit `cvat_config.json`:
```json
{
  "cvat_host": "http://localhost:8080",
  "cvat_username": "admin",
  "cvat_password": "your_password_here",
  "project_name": "Text Detection Correction"
}
```

### 4. Run Complete Workflow

```bash
# Step 1: Run text detection (generates frames + JSON)
python run_text_detection.py

# Step 2: Upload to CVAT for correction
python -m cvat_integration.workflow --video-name [VIDEO_NAME] --action upload

# Step 3: Open CVAT in browser and correct annotations
# http://localhost:8080/tasks/[TASK_ID]

# Step 4: Export corrected annotations
python -m cvat_integration.workflow --video-name [VIDEO_NAME] --action export
```

## 📋 Complete Example Walkthrough

### Example: Processing "sample_video.mp4"

#### Step 1: Run Detection
```bash
python run_text_detection.py
```

This creates:
```
output/text_detection_updated/sample_video/
├── annotated_video_detection.mp4
├── frames_and_json/
│   ├── frame_00000.jpg
│   ├── frame_00000.json
│   ├── frame_00001.jpg
│   ├── frame_00001.json
│   └── ...
└── detection_statistics.json
```

#### Step 2: Upload to CVAT
```bash
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action upload
```

Output:
```
======================================================================
UPLOAD WORKFLOW COMPLETE ✓
======================================================================
Task ID: 123
Task Name: text_detection_sample_video
URL: http://localhost:8080/tasks/123

Next Steps:
1. Open URL in browser: http://localhost:8080/tasks/123
2. Review and correct annotations in CVAT
3. Save changes
4. Run export workflow
======================================================================
```

#### Step 3: Human Correction in CVAT

1. Open the URL: http://localhost:8080/tasks/123
2. Click "Job #1" to start annotating
3. Review each frame:
   - **Add missing text** regions (if detector missed any)
   - **Delete false positives** (non-text detections)
   - **Adjust boxes** to fit text better
   - **Merge/split** boxes as needed
4. Use keyboard shortcuts for efficiency:
   - `N` - Next frame
   - `P` - Previous frame
   - `Ctrl+S` - Save
   - `N` key - Draw new polygon/rectangle
5. Click "Save" frequently!

#### Step 4: Export Corrected Annotations
```bash
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action export
```

Output:
```
======================================================================
EXPORT WORKFLOW COMPLETE ✓
======================================================================
Task ID: 123
Corrected detections: output/text_detection_updated/sample_video/cvat_export/corrected_detections

Quality Metrics:
  Frames compared: 150
  Original detections: 856
  Corrected detections: 892
  Change: +4.2%
  Added: 48
  Removed: 12

Next Steps:
1. Review corrected detections in: output/.../corrected_detections
2. Use corrected data for model training/evaluation
======================================================================
```

#### Step 5: Use Corrected Data

The corrected annotations are now in the same format as your original detections:
```
output/text_detection_updated/sample_video/cvat_export/corrected_detections/
├── frame_00000.json
├── frame_00001.json
└── ...
```

Each JSON file has the same structure:
```json
{
  "frame_number": 0,
  "detections": [
    {
      "box_id": 0,
      "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "confidence": 0.95
    }
  ],
  "total_detections": 5
}
```

## 🔄 Workflow Commands Reference

### Check Status
```bash
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action status
```

### Upload with Custom Task Name
```bash
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action upload \
  --task-name "My Custom Annotation Task"
```

### Upload with Rectangle Format (instead of polygons)
```bash
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action upload \
  --use-rectangles
```

### Export with Specific Task ID
```bash
python -m cvat_integration.workflow \
  --video-name sample_video \
  --action export \
  --task-id 123
```

## 🛠️ Advanced Usage

### Manual Format Conversion Only

Convert detection JSON to CVAT format without uploading:
```bash
python -m cvat_integration.format_converter to-cvat \
  --input-dir output/text_detection_updated/sample_video/frames_and_json \
  --output cvat_annotations.json
```

Convert CVAT back to detection format:
```bash
python -m cvat_integration.format_converter from-cvat \
  --cvat-file cvat_annotations.json \
  --output corrected_detections/
```

### Manual Upload to CVAT

```bash
python -m cvat_integration.cvat_uploader \
  --task-name "My Task" \
  --frames-dir output/text_detection_updated/sample_video/frames_and_json \
  --annotations cvat_annotations.json
```

### Manual Export from CVAT

List available tasks:
```bash
python -m cvat_integration.cvat_exporter list
```

Export specific task:
```bash
python -m cvat_integration.cvat_exporter export \
  --task-id 123 \
  --output-dir corrected_annotations/
```

Get task info:
```bash
python -m cvat_integration.cvat_exporter info --task-id 123
```

## 📊 Quality Metrics

After export, you'll get a comparison report in `export_statistics.json`:

```json
{
  "task_id": 123,
  "num_frames": 150,
  "output_dir": "corrected_detections",
  "comparison": {
    "frames_compared": 150,
    "original_detections": 856,
    "corrected_detections": 892,
    "added_detections": 48,
    "removed_detections": 12,
    "change_percentage": 4.2
  }
}
```

This helps you understand:
- How many detections were missed by the model (added)
- How many false positives the model had (removed)
- Overall improvement percentage

## 🎯 Best Practices

### 1. Batch Processing Multiple Videos

```bash
# Process all videos
for video in video1 video2 video3; do
  echo "Processing $video..."
  python -m cvat_integration.workflow --video-name $video --action upload
done

# Later, export all
for video in video1 video2 video3; do
  echo "Exporting $video..."
  python -m cvat_integration.workflow --video-name $video --action export
done
```

### 2. Annotation Guidelines for Team

Create clear guidelines for your annotation team:
- **Text definition**: What counts as "text"? (handwritten? logos? numbers?)
- **Box placement**: Tight fit vs. breathing room
- **Merged text**: When to merge vs. split boxes
- **Low confidence**: How to handle unclear text

### 3. Quality Control

Before exporting, have a reviewer check:
- Random sample of frames (e.g., every 10th frame)
- Frames with most corrections
- Edge cases (rotated text, small text, overlapping text)

### 4. Iterative Improvement

```
Detection → CVAT → Corrected Data → Retrain Model → Better Detection → ...
```

Use corrected data to:
1. Fine-tune your detection model
2. Adjust detection thresholds
3. Identify systematic errors
4. Build a gold-standard dataset

## 🐛 Troubleshooting

### CVAT Connection Issues

```bash
# Check if CVAT is running
docker ps | grep cvat

# Check CVAT logs
docker-compose logs -f cvat_server

# Restart CVAT
docker-compose restart
```

### Authentication Errors

- Verify `cvat_config.json` has correct credentials
- Try logging into CVAT web interface manually
- Check if password needs to be changed

### Import Errors

```bash
# Reinstall CVAT SDK
pip uninstall cvat-sdk
pip install cvat-sdk

# Verify installation
python -c "from cvat_sdk import make_client; print('OK')"
```

### Task Creation Fails

- Check available disk space in Docker volumes
- Verify image files are valid (not corrupted)
- Try with smaller batch first (e.g., first 10 frames)

## 📚 Additional Resources

- CVAT Documentation: https://opencv.github.io/cvat/
- CVAT Shortcuts: Press `F1` in CVAT interface
- Video Tutorial: See `CVAT_SETUP.md` for links

## 🎓 Training Your Team

### For Annotators

1. Show them this quick start guide
2. Walk through one video together
3. Emphasize keyboard shortcuts for efficiency
4. Set up quality control checkpoints

### For ML Engineers

1. Review format converter code to understand data flow
2. Integrate corrected data into training pipeline
3. Monitor quality metrics over time
4. Adjust detection model based on systematic errors

## ✅ Checklist

Before starting annotation:
- [ ] CVAT server is running (http://localhost:8080 accessible)
- [ ] `cvat_config.json` configured with correct credentials
- [ ] Detection pipeline has completed successfully
- [ ] Output directory exists and contains frames + JSON
- [ ] Team is trained on annotation guidelines

After completing annotation:
- [ ] All frames reviewed and saved in CVAT
- [ ] Quality control check completed
- [ ] Export workflow executed successfully
- [ ] Corrected annotations reviewed
- [ ] Data ready for training/evaluation

## 🚀 Next Steps

Now that you have human-corrected annotations:

1. **Train a better model**: Use corrected data for fine-tuning
2. **Evaluate performance**: Compare model output vs. human annotations
3. **Build gold standard**: Accumulate corrected annotations over time
4. **Active learning**: Focus annotation on uncertain/difficult cases
5. **Continuous improvement**: Iterate detection → correction → training

Happy annotating! 🎉
