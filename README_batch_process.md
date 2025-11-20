# PP-StructureV3 Batch Video Annotation Pipeline

Automated video annotation tool that extracts frames from videos, detects document layout elements using PaddleOCR's PP-StructureV3, and exports annotations in X-AnyLabeling format for manual correction and refinement.

## Features

- Batch processes all videos in the `/data` directory
- Detects 11 document layout classes: title, text, paragraph, paragraph_title, list, equation, table, image, figure, logo, unknown
- Exports frame-by-frame annotations in X-AnyLabeling JSON format
- Creates annotated videos with bounding boxes
- Supports multiple video formats: mp4, avi, mov, mkv, flv, wmv, webm

## Prerequisites

- Python 3.8+
- PaddleOCR with PP-StructureV3
- CUDA-compatible GPU (configured for `gpu:0`)
- OpenCV, NumPy

## Installation

```bash
# Install dependencies
pip install paddleocr opencv-python numpy

# Ensure PaddlePaddle GPU version is installed
# See Paddle_Setup_Guide.md for detailed setup
```

## Usage

1. Place your videos in the `/data` directory:
```bash
# Ensure /data directory exists and copy your videos there
# The script expects videos to be in /data/
```

2. Run the batch processor:
```bash
python batch_process_videos.py
```

3. Find outputs in `./output_video/[video_name]/`

## Configuration

Edit the following in the script:

- **Input directory** (line 789): `data_dir = "/data"` - Change to your video source directory
- **Output directory** (line 790): `output_base_dir = "./output_video"` - Change output location
- **FPS extraction**: Use `--fps` argument (default: 1 frame per second)
- `EXTRACTION_CONFIG` - Choose which PP-Structure data sources to use
- `LABEL_MAPPING` - Customize label mappings

## Output Structure

For each video, the script creates:

```
output_video/
└── [video_name]/
    ├── export/                    # X-AnyLabeling compatible
    │   ├── frame_0001.jpg
    │   ├── frame_0001.json
    │   ├── frame_0002.jpg
    │   ├── frame_0002.json
    │   └── ...
    ├── annotated_video.mp4        # Visualization with bboxes
    ├── video_annotations.json     # Complete annotation data
    └── classes.txt                # List of label classes
```

## Using Annotations in X-AnyLabeling

1. Open X-AnyLabeling
2. Select: **File > Open Dir**
3. Browse to: `output_video/[video_name]/export/`
4. Annotations load automatically - review and correct as needed

## Label Classes

The pipeline detects and maps to these 11 final classes:

- `title` - Document titles
- `text` - General text, headers, footers, references
- `paragraph` - Text paragraphs
- `paragraph_title` - Paragraph headings
- `list` - List elements
- `equation` - Mathematical formulas
- `table` - Tables
- `image` - Images
- `figure` - Charts and figures
- `logo` - Logos and seals
- `unknown` - Unclassified elements

## Performance

Processing time varies by video resolution and content complexity:
- Typical: 2-5 seconds per frame
- GPU acceleration required for reasonable performance

## Troubleshooting

- **Data directory not found**: Ensure `/data` directory exists and contains videos, or modify `data_dir` on line 789
- **No GPU detected**: Modify `device="gpu:0"` to `device="cpu"` in [batch_process_videos.py:507](batch_process_videos.py#L507)
- **Out of memory**: Use `--fps` argument with lower value (e.g., `--fps 0.5` for 1 frame every 2 seconds)
- **No annotations**: Enable debug mode by setting `debug=True` in extraction

## Related Files

- [Paddle_Setup_Guide.md](Paddle_Setup_Guide.md) - PaddleOCR installation instructions
- `classes.txt` - Generated class list for training
