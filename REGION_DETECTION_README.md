# Region Detection - Video Annotation Pipeline

Automated video annotation tool using PaddleOCR's PP-StructureV3 to detect and label document layout elements (text, tables, figures, equations, etc.) in videos. Exports annotations in X-AnyLabeling format for easy correction and refinement.

## Features

- **Automated Layout Detection**: Detects 20+ document element types including text, paragraphs, titles, tables, figures, equations, and more
- **Frame Extraction**: Configurable FPS-based frame extraction from videos
- **OCR Content Recognition**: Extracts text content from detected regions
- **Formula Recognition**: Detects and recognizes mathematical equations with LaTeX output
- **Table Recognition**: Identifies table structures and content
- **Visual Annotations**: Creates annotated video with bounding boxes and labels
- **X-AnyLabeling Export**: Exports frames + annotations in X-AnyLabeling format for manual correction
- **Multiple Output Formats**: JSON annotations, classes file, and annotated video

## Detected Region Types

The pipeline can detect and classify the following layout elements:

### Text Elements
- `text` - General text
- `paragraph` - Text paragraphs
- `title` - Document titles
- `header` - Page headers
- `footer` - Page footers
- `reference` - References/citations
- `page_number` - Page numbers
- `footnote` - Footnotes
- `code` - Code blocks
- `list` - Lists

### Visual Elements
- `figure` / `image` - Images and figures
- `figure_caption` - Figure captions
- `chart` - Charts and graphs
- `table` - Tables
- `table_caption` - Table captions

### Mathematical Elements
- `equation` / `formula` - Mathematical equations

### Special Elements
- `seal` / `stamp` - Seals and stamps
- `unknown` - Unclassified elements
- `paragraph_title` - Paragraph titles

## Requirements

### System Requirements
- GPU (NVIDIA CUDA-enabled recommended for faster processing)
- 8GB+ RAM recommended
- Python 3.8+

### Dependencies

```bash
pip install paddlepaddle-gpu  # or paddlepaddle for CPU
pip install paddleocr
pip install opencv-python
pip install numpy
```

**Note**: For GPU support, ensure CUDA and cuDNN are properly installed.

## Installation

1. **Clone or download** this repository

2. **Install dependencies**:
   ```bash
   pip install paddlepaddle-gpu paddleocr opencv-python numpy
   ```

3. **Create required directories**:
   ```bash
   mkdir data
   mkdir output_video
   ```

4. **Place your videos** in the `data/` directory

## Usage

### Basic Usage

Process a video with default settings (1 FPS extraction):

```bash
python region_detection.py --video your_video.mp4
```

### Custom FPS Extraction

Extract frames at 2 frames per second:

```bash
python region_detection.py --video your_video.mp4 --fps 2
```

### Command Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--video` | `-v` | Yes | - | Video filename (e.g., `triangles.mp4`) |
| `--fps` | `-f` | No | 1 | Frames per second to extract |

## Input/Output Structure

### Input
```
data/
├── triangles.mp4
├── lecture_video.mp4
└── document_scan.mp4
```

### Output

For example, running: `python region_detection.py --video triangles.mp4`

```
output_video/
└── triangles/                      # Video name (without extension)
    ├── annotated_video.mp4        # Video with visual annotations
    ├── video_annotations.json     # Complete annotations in JSON format
    ├── classes.txt                # List of all detectable classes
    └── export/                    # X-AnyLabeling compatible directory
        ├── frame_0001.jpg         # Original extracted frame
        ├── frame_0001.json        # X-AnyLabeling annotation file
        ├── frame_0002.jpg
        ├── frame_0002.json
        └── ...
```

## Output Files Explained

### 1. `annotated_video.mp4`
Video with bounding boxes and labels drawn on each frame. Different colors represent different element types.

### 2. `video_annotations.json`
Complete annotations in structured JSON format:
```json
{
  "video_path": "data/triangles.mp4",
  "video_properties": {
    "width": 1920,
    "height": 1080,
    "original_fps": 30.0,
    "duration_seconds": 60.0,
    "total_frames": 1800
  },
  "processing_info": {
    "extraction_fps": 1,
    "frames_processed": 60,
    "processing_time_seconds": 120.5
  },
  "frames": [
    {
      "frame_number": 1,
      "timestamp": 0.0,
      "num_annotations": 15,
      "annotations": [
        {
          "id": 0,
          "label": "title",
          "bbox": {
            "x_min": 100.0,
            "y_min": 50.0,
            "x_max": 800.0,
            "y_max": 120.0,
            "width": 700.0,
            "height": 70.0
          },
          "confidence": 0.95,
          "content": "Introduction to Geometry"
        }
      ]
    }
  ]
}
```

### 3. `classes.txt`
List of all detectable class names (one per line), useful for training custom models.

### 4. `export/` Directory
Contains original frames and X-AnyLabeling compatible JSON files for manual annotation refinement.

## Using with X-AnyLabeling

X-AnyLabeling is a free annotation tool that allows you to correct and refine the automated annotations.

### Steps:

1. **Install X-AnyLabeling**:
   ```bash
   pip install x-anylabeling
   # or download from: https://github.com/CVHub520/X-AnyLabeling
   ```

2. **Open the export directory**:
   - Launch X-AnyLabeling
   - Go to **File > Open Dir**
   - Browse to `output_video/<video_name>/export/`

3. **Refine annotations**:
   - Frames and annotations load automatically
   - Correct bounding boxes, labels, or text content
   - Add missing annotations
   - Delete incorrect detections

4. **Export corrections**:
   - Save changes to update JSON files
   - Use for training or downstream tasks

## Configuration

### Extraction Strategy

Edit the `EXTRACTION_CONFIG` in [region_detection.py](region_detection.py:29-34) to control which PP-Structure data sources to use:

```python
EXTRACTION_CONFIG = {
    'use_parsing_res_list': True,      # High-level parsed blocks (RECOMMENDED)
    'use_layout_det_res': False,       # Raw layout detections (may have duplicates)
    'use_overall_ocr_res': False,      # Individual OCR text items (very granular)
    'use_formula_res_list': True,      # Formula detections with LaTeX
}
```

### PP-StructureV3 Pipeline Settings

Edit the pipeline initialization in [region_detection.py](region_detection.py:493-501):

```python
pipeline = PPStructureV3(
    device="gpu:0",                         # Use "cpu" for CPU inference
    use_doc_orientation_classify=False,     # Document orientation detection
    use_doc_unwarping=False,                # Document unwarping
    use_table_recognition=True,             # Table structure recognition
    use_formula_recognition=True,           # Formula recognition
    use_chart_recognition=False,            # Chart recognition
    use_seal_recognition=False,             # Seal/stamp recognition
)
```

### Label Mapping

Customize label names by editing `LABEL_MAPPING` in [region_detection.py](region_detection.py:37-68).

### Visualization Colors

Customize bounding box colors by editing `LABEL_COLORS` in [region_detection.py](region_detection.py:71-89) (BGR format for OpenCV).

## Performance Tips

1. **GPU Acceleration**: Use GPU for significantly faster processing:
   ```python
   device="gpu:0"  # Uses first GPU
   ```

2. **Adjust FPS**: Lower FPS for faster processing, higher for more frames:
   ```bash
   python region_detection.py --video video.mp4 --fps 0.5  # 1 frame every 2 seconds
   ```

3. **Disable Unused Features**: Turn off unnecessary PP-Structure features to speed up processing

4. **Batch Processing**: Process multiple videos using a shell script:
   ```bash
   for video in data/*.mp4; do
       python region_detection.py --video $(basename $video)
   done
   ```

## Troubleshooting

### Issue: "Video file not found"
**Solution**: Ensure your video is in the `data/` directory with the correct filename.

### Issue: "No annotations extracted"
**Solutions**:
- Check if the video frame contains detectable document content
- Enable more extraction sources in `EXTRACTION_CONFIG`
- Verify the video quality is sufficient for OCR

### Issue: GPU memory error
**Solutions**:
- Use CPU instead: change `device="gpu:0"` to `device="cpu"`
- Process at lower FPS
- Reduce video resolution before processing

### Issue: Slow processing
**Solutions**:
- Use GPU acceleration
- Reduce extraction FPS
- Disable unused PP-Structure features (table recognition, formula recognition, etc.)

## Example Workflows

### 1. Quick Video Preview
```bash
# Extract 1 frame every 5 seconds for quick overview
python region_detection.py --video lecture.mp4 --fps 0.2
```

### 2. High-Quality Annotation
```bash
# Extract 2 frames per second for detailed annotation
python region_detection.py --video document.mp4 --fps 2
```

### 3. Manual Refinement
```bash
# Process video, then refine in X-AnyLabeling
python region_detection.py --video slides.mp4 --fps 1
x-anylabeling output_video/slides/export/
```

## Technical Details

### Architecture
- **Layout Detection**: PP-StructureV3 layout analysis model
- **OCR Engine**: PaddleOCR text recognition
- **Formula Recognition**: LaTeX formula recognition
- **Table Recognition**: Table structure analysis
- **Export Format**: X-AnyLabeling v5.5.0 compatible JSON

### Annotation Format
Each frame annotation includes:
- **Bounding box**: x_min, y_min, x_max, y_max coordinates
- **Label**: Element type (text, table, figure, etc.)
- **Content**: OCR-extracted text or LaTeX formula
- **Confidence**: Model confidence score (0.0 - 1.0)
- **Metadata**: Frame number, timestamp

## Output Statistics

After processing, the tool displays:
- Total frames processed
- Total annotations detected
- Average annotations per frame
- Processing time and speed
- Breakdown by element type
- File paths for all outputs

## License

This project uses PaddleOCR and PP-StructureV3, which are licensed under Apache License 2.0.

## Citation

If you use this tool in your research, please cite PaddleOCR:

```bibtex
@misc{paddleocr,
    title={PaddleOCR: Awesome multilingual OCR toolkits},
    author={PaddlePaddle Authors},
    year={2020},
    url={https://github.com/PaddlePaddle/PaddleOCR}
}
```

## Support

For issues, questions, or feature requests:
- PaddleOCR Documentation: https://github.com/PaddlePaddle/PaddleOCR
- X-AnyLabeling: https://github.com/CVHub520/X-AnyLabeling

## Version History

- **v1.0.0** (Current)
  - Initial release
  - PP-StructureV3 integration
  - X-AnyLabeling export support
  - Multi-format output (video, JSON, classes.txt)
  - 20+ document element types
  - GPU acceleration support
