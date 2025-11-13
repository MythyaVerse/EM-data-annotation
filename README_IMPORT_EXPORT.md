# CVAT Import/Export Guide

Complete guide for importing data to CVAT and exporting annotations in multiple formats.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Importing Data to CVAT (Upload)](#importing-data-to-cvat-upload)
- [Exporting Data from CVAT](#exporting-data-from-cvat)
- [Directory Structure](#directory-structure)
- [Export Formats](#export-formats)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites
1. CVAT server running at `http://localhost:8080`
2. Video processed with text detection pipeline
3. Python virtual environment activated

### Basic Workflow
```bash
# 1. Upload to CVAT
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload

# 2. Annotate in CVAT web interface (http://localhost:8080)

# 3. Export from CVAT
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export --task-id TASK_ID
```

---

## 📤 Importing Data to CVAT (Upload)

### Command Syntax

```bash
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload [OPTIONS]
```

### Required Arguments
- `--video-name`: Name of the video (must match detection output directory name)
- `--action upload`: Specifies upload operation

### Optional Arguments
- `--task-name`: Custom task name (default: `text_detection_{video_name}`)
- `--rectangles`: Use bounding boxes instead of polygons

### Examples

#### Basic Upload
```bash
python -m cvat_integration.workflow --video-name triangles_15s --action upload
```

#### Custom Task Name
```bash
python -m cvat_integration.workflow --video-name triangles_15s --action upload --task-name "Triangles Video Round 1"
```

#### Upload with Rectangles (Bounding Boxes)
```bash
python -m cvat_integration.workflow --video-name triangles_15s --action upload --rectangles
```

### What Happens During Upload

1. **Detection Conversion** (Step 1/3)
   - Reads detection JSONs from: `output/text_detection_updated/{video_name}/frames_and_json/`
   - Converts to CVAT format
   - Saves to: `output/text_detection_updated/{video_name}/cvat_annotations.json`

2. **Upload to CVAT** (Step 2/3)
   - Connects to CVAT server
   - Creates/finds project: "Text Detection Correction"
   - Creates task with name: `text_detection_{video_name}`
   - Uploads original frames from: `output/text_detection_updated/{video_name}/original_frames/`
   - Imports pre-annotations

3. **Save Metadata** (Step 3/3)
   - Saves workflow metadata to: `output/text_detection_updated/{video_name}/cvat_workflow_metadata.json`
   - Records task ID and URL for future reference

### Upload Output Example

```
======================================================================
CVAT UPLOAD WORKFLOW
======================================================================

[STEP 1/3] Converting detections to CVAT format...
Found 83 JSON files in output/text_detection_updated\triangles_15s\frames_and_json
✓ Converted 454 detections from 83 frames
✓ Saved CVAT annotations to: output/text_detection_updated\triangles_15s\cvat_annotations.json

[STEP 2/3] Uploading to CVAT...
Connecting to CVAT at http://localhost:8080...
✓ Connected to CVAT successfully
Looking for project: Text Detection Correction...
✓ Found existing project: Text Detection Correction (ID: 1)
Creating task: text_detection_triangles_15s
  Images: 83 frames
✓ Task created: text_detection_triangles_15s (ID: 8)
  URL: http://localhost:8080/tasks/8
  Uploading frames...
  Waiting for task initialization...
  ✓ Task ready

[STEP 3/3] Saving workflow metadata...
✓ Workflow metadata saved

======================================================================
UPLOAD WORKFLOW COMPLETE ✓
======================================================================
Task ID: 8
Task URL: http://localhost:8080/tasks/8
Project: Text Detection Correction (ID: 1)

Next Steps:
1. Open CVAT in browser: http://localhost:8080/tasks/8
2. Review and correct annotations
3. Export corrected data
4. Run export workflow:
   python -m cvat_integration.workflow --video-name triangles_15s --action export
======================================================================
```

---

## 📥 Exporting Data from CVAT

### Command Syntax

```bash
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export [OPTIONS]
```

### Required Arguments
- `--video-name`: Name of the video
- `--action export`: Specifies export operation

### Optional Arguments
- `--task-id`: CVAT task ID (auto-detected from metadata if not provided)

### Examples

#### Auto-Detected Export (Using Metadata)
```bash
python -m cvat_integration.workflow --video-name triangles_15s --action export
```

#### Manual Task ID
```bash
python -m cvat_integration.workflow --video-name triangles_15s --action export --task-id 8
```

### What Happens During Export

1. **Export Annotations** (Step 1/2)
   - Connects to CVAT server
   - Retrieves task information
   - Exports annotations in two formats:
     - **CVAT Format**: XML file with full annotation data
     - **COCO Format**: JSON file ready for model training

2. **Save Metadata** (Step 2/2)
   - Creates export directory: `exports/{video_name}_{task_id}/`
   - Saves export metadata: `exports/{video_name}_{task_id}/export_metadata.json`
   - Updates original workflow metadata

### Export Output Example

```
======================================================================
CVAT EXPORT WORKFLOW
======================================================================
Connecting to CVAT at http://localhost:8080...
✓ Connected to CVAT successfully

📁 Export directory: exports\triangles_15s_8

[STEP 1/2] Exporting annotations in CVAT and COCO formats...

  Exporting CVAT format...
Exporting annotations from task 8...
  Task: text_detection_triangles_15s
  Frames: 83
✓ Annotations exported to: exports\triangles_15s_8\cvat\annotations.xml

  Exporting COCO format...
Exporting annotations from task 8...
  Task: text_detection_triangles_15s
  Frames: 83
✓ Annotations exported to: exports\triangles_15s_8\coco\annotations.json

[STEP 2/2] Saving export metadata...

======================================================================
EXPORT WORKFLOW COMPLETE ✓
======================================================================
Task ID: 8
Task Name: text_detection_triangles_15s
Frames: 83

📁 Exports saved to: exports\triangles_15s_8
  ├── cvat/annotations.xml
  └── coco/annotations.json

✨ Next Steps:
1. Use COCO format for model training: exports\triangles_15s_8\coco/annotations.json
2. Use CVAT format for re-import or backup: exports\triangles_15s_8\cvat/annotations.xml
======================================================================
```

---

## 📁 Directory Structure

### Before Upload
```
Data Annotation/
├── output/
│   └── text_detection_updated/
│       └── {video_name}/
│           ├── frames_and_json/        # Detection JSONs (read for upload)
│           │   ├── frame_00000.jpg
│           │   ├── frame_00000.json
│           │   └── ...
│           ├── original_frames/        # Clean frames (uploaded to CVAT)
│           │   ├── frame_00000.jpg
│           │   └── ...
│           └── cvat_annotations.json   # Generated during upload
```

### After Upload
```
Data Annotation/
├── output/
│   └── text_detection_updated/
│       └── {video_name}/
│           ├── cvat_annotations.json
│           └── cvat_workflow_metadata.json  # ← Created (contains task ID)
```

### After Export
```
Data Annotation/
├── exports/
│   └── {video_name}_{task_id}/
│       ├── cvat/
│       │   └── annotations.xml         # CVAT format (XML)
│       ├── coco/
│       │   └── annotations.json        # COCO format (JSON)
│       └── export_metadata.json        # Export info
```

---

## 🎨 Export Formats

### CVAT Format (XML)
- **Location**: `exports/{video_name}_{task_id}/cvat/annotations.xml`
- **Use Case**:
  - Re-import to CVAT
  - Backup annotations
  - Share with other CVAT users
- **Format**: XML with full annotation metadata including attributes

### COCO Format (JSON)
- **Location**: `exports/{video_name}_{task_id}/coco/annotations.json`
- **Use Case**:
  - Model training (Detectron2, MMDetection, etc.)
  - Data analysis
  - Integration with other tools
- **Format**: Standard COCO 1.0 JSON format
- **Structure**:
  ```json
  {
    "images": [...],
    "annotations": [...],
    "categories": [{"id": 1, "name": "text"}]
  }
  ```

---

## 💡 Common Use Cases

### Use Case 1: First Time Upload
```bash
# Process video
python run_text_detection.py --video data/videos/my_video.mp4

# Upload to CVAT
python -m cvat_integration.workflow --video-name my_video --action upload

# Open in browser: http://localhost:8080
# Correct annotations in CVAT GUI

# Export corrected data
python -m cvat_integration.workflow --video-name my_video --action export
```

### Use Case 2: Multiple Videos
```bash
# Upload multiple videos
python -m cvat_integration.workflow --video-name video1 --action upload
python -m cvat_integration.workflow --video-name video2 --action upload
python -m cvat_integration.workflow --video-name video3 --action upload

# Annotate all in CVAT...

# Export all (auto-detects task IDs from metadata)
python -m cvat_integration.workflow --video-name video1 --action export
python -m cvat_integration.workflow --video-name video2 --action export
python -m cvat_integration.workflow --video-name video3 --action export
```

### Use Case 3: Re-export After More Corrections
```bash
# Export again with same task ID (overwrites previous export)
python -m cvat_integration.workflow --video-name my_video --action export --task-id 8
```

### Use Case 4: Export Old Task (No Metadata)
```bash
# If you lost metadata or want to export an old task
python -m cvat_integration.workflow --video-name my_video --action export --task-id 5
```

---

## 🔧 Troubleshooting

### Error: "Workflow metadata not found"
**Problem**: Trying to export without metadata file
```
❌ Error: Workflow metadata not found: output/text_detection_updated\my_video\cvat_workflow_metadata.json
Please provide --task-id manually
```

**Solution**: Provide task ID manually
```bash
python -m cvat_integration.workflow --video-name my_video --action export --task-id 8
```

### Error: "frames_and_json directory not found"
**Problem**: Detection output doesn't exist or wrong video name

**Solution**:
1. Check video name matches detection output:
   ```bash
   ls output/text_detection_updated/
   ```
2. Run detection first:
   ```bash
   python run_text_detection.py --video data/videos/my_video.mp4
   ```

### Error: "original_frames directory not found"
**Problem**: Text detection didn't save original frames

**Solution**: Re-run detection with `save_original_frames=True` (should be default)

### Error: "Cannot connect to CVAT"
**Problem**: CVAT server not running

**Solution**: Start CVAT server
```bash
cd cvat_local/cvat
docker compose up -d
```

### Error: "Task ID not found"
**Problem**: Invalid or deleted task

**Solution**:
1. Check CVAT web interface for correct task ID
2. List available tasks:
   ```bash
   python -m cvat_integration.workflow --action status
   ```

---

## 📊 Image Dimensions

### Current Pipeline Settings

| Stage | Dimensions | Notes |
|-------|-----------|-------|
| **Original Video** | 1920×1080 | Your input resolution |
| **Model Inference** | **736×736** | ONNX model processing size |
| **Saved Frames** | 1920×1080 | Original quality preserved |
| **CVAT Upload** | 1920×1080 | Full resolution for annotation |
| **Exported Data** | 1920×1080 | Coordinates at original scale |

**Key Point**: The 736×736 is only used internally for model inference. All saved files and CVAT annotations use the original 1920×1080 resolution.

---

## 🔗 Related Documentation

- [Main CVAT Integration README](README_CVAT_INTEGRATION.md) - Complete workflow overview
- [CVAT Configuration](cvat_config.json) - Server and authentication settings
- [Text Detection README](README.md) - Detection pipeline documentation

---

## 📝 Notes

- **Task IDs**: Sequential integers starting from 1 (e.g., 1, 2, 3, ...)
- **Project**: All tasks are created under "Text Detection Correction" project
- **Original Frames**: Always uploaded from `original_frames/` directory (no bounding boxes)
- **Annotations**: Pre-annotations from detection are imported automatically during upload
- **Export**: Generates both CVAT (XML) and COCO (JSON) formats simultaneously
- **Metadata**: Workflow metadata enables auto-detection of task IDs for export

---

## ⚡ Quick Reference

```bash
# Upload
python -m cvat_integration.workflow --video-name VIDEO_NAME --action upload

# Export (auto-detect task ID)
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export

# Export (manual task ID)
python -m cvat_integration.workflow --video-name VIDEO_NAME --action export --task-id TASK_ID

# Check status
python -m cvat_integration.workflow --action status
```

---

**Last Updated**: 2025
**Pipeline Version**: 1.0
