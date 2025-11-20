"""
PP-StructureV3 Batch Video Annotation Pipeline with X-AnyLabeling Export

This script processes ALL videos in the data/ directory sequentially with PaddleOCR's
PP-StructureV3, extracts frames, detects layout elements, and exports annotations in
X-AnyLabeling format for easy correction and refinement.

Usage:
    python batch_process_videos.py
"""

import os
import json
import cv2
import numpy as np
import base64
import time
from pathlib import Path
from typing import Dict, Any
from paddleocr import PPStructureV3


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default FPS for batch processing
DEFAULT_FPS = 5

# Supported video formats
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# Extraction strategy: which PP-Structure data sources to use
EXTRACTION_CONFIG = {
    'use_parsing_res_list': True,      # High-level parsed blocks (RECOMMENDED)
    'use_layout_det_res': False,       # Raw layout detections (may have duplicates)
    'use_overall_ocr_res': False,      # Individual OCR text items (very granular)
    'use_formula_res_list': True,      # Formula detections with LaTeX
}

# Comprehensive label mapping: ALL PP-Structure labels → Final 11 classes
LABEL_MAPPING = {
    # Title elements
    'title': 'title',
    'document_title': 'title',       # Document title → title
    'paragraph_title': 'paragraph_title',  # Paragraph title → paragraph_title

    # Text elements - map to final classes
    'text': 'text',
    'paragraph': 'paragraph',
    'abstract': 'text',              # Abstract → text
    'header': 'text',                # Header → text
    'footer': 'text',                # Footer → text
    'reference': 'text',             # Reference → text
    'references': 'text',            # References → text
    'page_number': 'text',           # Page number → text
    'footnote': 'text',              # Footnote → text
    'footnotes': 'text',             # Footnotes → text
    'code': 'text',                  # Code → text
    'algorithm': 'text',             # Algorithm → text
    'sidebar_text': 'text',          # Sidebar text → text
    'figure_table_title': 'text',    # Figure/table title → text
    'formula_number': 'text',        # Formula number → text

    # List elements
    'list': 'list',
    'lists': 'list',                 # Lists → list

    # Visual elements - map to final classes
    'figure': 'figure',
    'image': 'image',
    'figure_caption': 'text',        # Figure caption → text
    'chart': 'figure',               # Chart → figure
    'table': 'table',
    'table_caption': 'text',         # Table caption → text
    'seal': 'logo',                  # Seal → logo

    # Mathematical elements - map to equation
    'equation': 'equation',
    'formula': 'equation',           # Formula → equation

    # Fallback
    'unknown': 'unknown',
}

# Color scheme for visualization (BGR format for OpenCV) - Final 11 classes
LABEL_COLORS = {
    'title': (255, 0, 0),          # Blue
    'text': (0, 255, 0),           # Green
    'paragraph': (0, 200, 0),      # Light Green
    'paragraph_title': (0, 180, 0),# Dark Green
    'list': (50, 200, 50),         # Bright Green
    'equation': (0, 255, 255),     # Yellow
    'table': (0, 165, 255),        # Orange
    'image': (255, 50, 255),       # Light Magenta
    'figure': (255, 0, 255),       # Magenta
    'logo': (128, 0, 128),         # Purple
    'unknown': (128, 128, 128),    # Medium Gray
}


# ============================================================================
# X-ANYLABELING CONVERTER CLASS
# ============================================================================

class PaddleOCRToXAnyLabeling:
    """Convert PaddleOCR annotations to X-AnyLabeling format."""

    def __init__(self, include_image_data=False):
        """
        Initialize converter.

        Args:
            include_image_data: If True, embeds base64-encoded image data in JSON.
        """
        self.include_image_data = include_image_data
        self.version = "5.5.0"

    def normalize_bbox(self, bbox):
        """Normalize bbox to [x1, y1, x2, y2] format."""
        if isinstance(bbox, dict):
            if all(k in bbox for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                return [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']]
        elif isinstance(bbox, (list, tuple)):
            if len(bbox) == 4:
                return list(bbox)
            elif len(bbox) == 8:
                x_coords = [bbox[i] for i in range(0, 8, 2)]
                y_coords = [bbox[i] for i in range(1, 8, 2)]
                return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        raise ValueError(f"Unsupported bbox format: {bbox}")

    def bbox_to_polygon(self, bbox):
        """Convert bounding box to 4-point polygon format."""
        x1, y1, x2, y2 = bbox
        return [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)]
        ]

    def create_shape(self, annotation: Dict[str, Any], shape_type: str = "rectangle") -> Dict[str, Any]:
        """Create a shape object in X-AnyLabeling format."""
        bbox = self.normalize_bbox(annotation['bbox'])

        if shape_type == "rectangle":
            points = [
                [float(bbox[0]), float(bbox[1])],
                [float(bbox[2]), float(bbox[3])]
            ]
        else:  # polygon
            points = self.bbox_to_polygon(bbox)

        shape = {
            "label": annotation.get('label', 'text'),
            "text": annotation.get('content', ''),
            "points": points,
            "group_id": None,
            "description": f"Confidence: {annotation.get('confidence', 0.0):.2f}",
            "difficult": False,
            "shape_type": shape_type,
            "flags": {},
            "attributes": {}
        }

        return shape

    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not encode image {image_path}: {e}")
            return None

    def convert_frame_to_xanylabeling(
        self,
        frame_data: Dict[str, Any],
        image_path: str,
        image_height: int,
        image_width: int,
        shape_type: str = "rectangle"
    ) -> Dict[str, Any]:
        """Convert a single frame's annotations to X-AnyLabeling format."""
        shapes = []
        for ann in frame_data.get('annotations', []):
            try:
                shape = self.create_shape(ann, shape_type=shape_type)
                shapes.append(shape)
            except Exception as e:
                print(f"Warning: Could not convert annotation: {e}")
                continue

        xanylabeling_data = {
            "version": self.version,
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }

        if self.include_image_data and os.path.exists(image_path):
            xanylabeling_data["imageData"] = self.image_to_base64(image_path)

        return xanylabeling_data


# ============================================================================
# PP-STRUCTURE ANNOTATION EXTRACTION
# ============================================================================

def extract_annotations(results, label_mapping, extraction_config, debug=False):
    """Extract structured annotations from PP-StructureV3 results."""
    annotations = []

    if not results or len(results) == 0:
        if debug:
            print("    DEBUG: No results returned")
        return annotations

    for page_idx, page_result in enumerate(results):
        if debug:
            print(f"\n    DEBUG: Processing page {page_idx}")
            print(f"    DEBUG: Result type: {type(page_result)}")

        try:
            res_data = page_result['res'] if 'res' in page_result else page_result
        except:
            res_data = page_result

        def get_value(obj, *keys):
            """Try to get value from object attribute or dict key."""
            for key in keys:
                try:
                    if isinstance(obj, dict) and key in obj:
                        return obj[key]
                    if hasattr(obj, key):
                        return getattr(obj, key)
                except:
                    continue
            return None

        # Source 1: parsing_res_list
        if extraction_config.get('use_parsing_res_list', True):
            parsing_list = get_value(res_data, 'parsing_res_list')
            if parsing_list:
                if debug:
                    print(f"    DEBUG: Found parsing_res_list with {len(parsing_list)} items")

                for item in parsing_list:
                    bbox = get_value(item, 'block_bbox', 'bbox', 'coordinate')
                    if not bbox:
                        continue

                    raw_label = get_value(item, 'block_label', 'label', 'type')
                    raw_label = str(raw_label).lower() if raw_label else 'text'
                    mapped_label = label_mapping.get(raw_label, raw_label)

                    content = get_value(item, 'block_content', 'content', 'text')
                    content = str(content).strip() if content else ''

                    annotations.append({
                        'label': mapped_label,
                        'bbox': bbox,
                        'confidence': 0.95,
                        'content': content,
                        'original_label': raw_label,
                        'source': 'parsing_res_list'
                    })

        # Source 2: layout_det_res
        if extraction_config.get('use_layout_det_res', False):
            layout_det = get_value(res_data, 'layout_det_res')
            if layout_det:
                boxes = get_value(layout_det, 'boxes')
                if boxes:
                    for item in boxes:
                        bbox = get_value(item, 'coordinate', 'bbox', 'box')
                        if not bbox:
                            continue

                        raw_label = get_value(item, 'label', 'type')
                        raw_label = str(raw_label).lower() if raw_label else 'text'
                        mapped_label = label_mapping.get(raw_label, raw_label)
                        confidence = get_value(item, 'score', 'confidence') or 0.5

                        is_duplicate = False
                        for existing in annotations:
                            if existing['source'] == 'parsing_res_list':
                                if boxes_overlap(existing['bbox'], bbox, threshold=0.7):
                                    is_duplicate = True
                                    break

                        if not is_duplicate:
                            annotations.append({
                                'label': mapped_label,
                                'bbox': bbox,
                                'confidence': float(confidence),
                                'content': '',
                                'original_label': raw_label,
                                'source': 'layout_det_res'
                            })

        # Source 3: overall_ocr_res
        if extraction_config.get('use_overall_ocr_res', False):
            ocr_res = get_value(res_data, 'overall_ocr_res')
            if ocr_res:
                rec_texts = get_value(ocr_res, 'rec_texts') or []
                rec_boxes = get_value(ocr_res, 'rec_boxes') or []
                rec_scores = get_value(ocr_res, 'rec_scores') or []

                for text, box, score in zip(rec_texts, rec_boxes, rec_scores):
                    if not text or box is None:
                        continue

                    is_duplicate = False
                    for existing in annotations:
                        if existing['source'] in ['parsing_res_list', 'layout_det_res']:
                            if boxes_overlap(existing['bbox'], box, threshold=0.8):
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        annotations.append({
                            'label': 'text',
                            'bbox': list(box) if hasattr(box, '__iter__') else box,
                            'confidence': float(score),
                            'content': str(text),
                            'original_label': 'ocr_text',
                            'source': 'overall_ocr_res'
                        })

        # Source 4: formula_res_list
        if extraction_config.get('use_formula_res_list', True):
            formula_list = get_value(res_data, 'formula_res_list')
            if formula_list:
                for item in formula_list:
                    bbox = get_value(item, 'dt_polys', 'bbox', 'coordinate')
                    if not bbox:
                        continue

                    formula_text = get_value(item, 'rec_formula', 'formula', 'text') or ''

                    is_duplicate = False
                    for existing in annotations:
                        if existing['label'] == 'equation' and existing['source'] == 'parsing_res_list':
                            if boxes_overlap(existing['bbox'], bbox, threshold=0.7):
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        annotations.append({
                            'label': 'equation',
                            'bbox': bbox,
                            'confidence': 0.90,
                            'content': str(formula_text),
                            'original_label': 'formula',
                            'source': 'formula_res_list'
                        })

    return annotations


def boxes_overlap(box1, box2, threshold=0.7):
    """Check if two boxes overlap significantly (IoU > threshold)."""
    try:
        b1 = normalize_bbox(box1)
        b2 = normalize_bbox(box2)

        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        if x2 < x1 or y2 < y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

        iou = intersection / (area1 + area2 - intersection)
        return iou > threshold
    except:
        return False


def normalize_bbox(bbox):
    """Normalize bbox to [x_min, y_min, x_max, y_max] format."""
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()

    if isinstance(bbox, list) and len(bbox) > 0:
        if isinstance(bbox[0], (list, tuple)):
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
        elif len(bbox) == 4:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    return bbox


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_annotations_on_frame(frame, annotations, label_colors):
    """Draw bounding boxes on frame with labels."""
    annotated_frame = frame.copy()

    for ann in annotations:
        label = ann['label']
        bbox = normalize_bbox(ann['bbox'])
        confidence = ann['confidence']
        color = label_colors.get(label, (128, 128, 128))

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        cv2.putText(
            annotated_frame,
            label_text,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return annotated_frame


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def process_video(video_path, output_dir, fps_extract=5):
    """
    Process video: extract frames, run PP-Structure, export to X-AnyLabeling format.

    Args:
        video_path: Path to input video
        output_dir: Directory to save outputs
        fps_extract: Extract frames at this FPS
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create export directory for frames and X-AnyLabeling JSONs
    export_dir = os.path.join(output_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps

    print(f"\nVideo Information:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extracting at: {fps_extract} FPS")

    frame_interval = int(original_fps / fps_extract)
    expected_frames = int(duration * fps_extract)
    print(f"  Expected extracted frames: {expected_frames}")

    # Initialize PP-StructureV3
    print("\nInitializing PP-StructureV3 pipeline...")
    pipeline = PPStructureV3(
        device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_table_recognition=True,
        use_formula_recognition=True,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )

    # Initialize X-AnyLabeling converter
    converter = PaddleOCRToXAnyLabeling(include_image_data=False)

    # Prepare output video writer
    output_video_path = os.path.join(output_dir, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps_extract, (width, height))

    # Storage for all annotations
    all_frame_annotations = []

    # Process video
    print("\nProcessing video frames...")
    frame_count = 0
    processed_count = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            processed_count += 1
            timestamp = frame_count / original_fps

            print(f"\n[Frame {processed_count}/{expected_frames}] Time: {timestamp:.2f}s")

            # Save original frame to export directory
            frame_filename = f"frame_{processed_count:04d}.jpg"
            frame_export_path = os.path.join(export_dir, frame_filename)
            cv2.imwrite(frame_export_path, frame)

            # Save frame temporarily for PP-Structure processing
            temp_frame_path = os.path.join(output_dir, "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)

            try:
                results = pipeline.predict(temp_frame_path)

                debug_mode = (processed_count == 1)
                annotations = extract_annotations(results, LABEL_MAPPING, EXTRACTION_CONFIG, debug=debug_mode)

                if len(annotations) == 0 and debug_mode:
                    print("    WARNING: No annotations extracted!")

                print(f"  Detected {len(annotations)} elements")

                # Count by label
                label_counts = {}
                for ann in annotations:
                    label = ann['label']
                    label_counts[label] = label_counts.get(label, 0) + 1

                if label_counts:
                    print("  ", end="")
                    for label, count in sorted(label_counts.items()):
                        print(f"{label}:{count} ", end="")
                    print()

                # Draw annotations on frame
                annotated_frame = draw_annotations_on_frame(frame, annotations, LABEL_COLORS)
                out.write(annotated_frame)

                # Store annotations with metadata
                frame_data = {
                    'frame_number': processed_count,
                    'timestamp': timestamp,
                    'num_annotations': len(annotations),
                    'annotations': []
                }

                for idx, ann in enumerate(annotations):
                    bbox = normalize_bbox(ann['bbox'])
                    x1, y1, x2, y2 = bbox

                    frame_data['annotations'].append({
                        'id': idx,
                        'label': ann['label'],
                        'bbox': {
                            'x_min': float(x1),
                            'y_min': float(y1),
                            'x_max': float(x2),
                            'y_max': float(y2),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1)
                        },
                        'confidence': ann['confidence'],
                        'content': ann['content']
                    })

                all_frame_annotations.append(frame_data)

                # Convert to X-AnyLabeling format and save
                xanylabeling_data = converter.convert_frame_to_xanylabeling(
                    frame_data,
                    frame_export_path,
                    height,
                    width,
                    shape_type="rectangle"
                )

                json_filename = f"frame_{processed_count:04d}.json"
                json_export_path = os.path.join(export_dir, json_filename)

                with open(json_export_path, 'w', encoding='utf-8') as f:
                    json.dump(xanylabeling_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"  Error processing frame: {str(e)}")
                out.write(frame)

        frame_count += 1

    # Cleanup
    cap.release()
    out.release()

    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

    elapsed_time = time.time() - start_time

    # Save main annotations JSON
    annotations_json_path = os.path.join(output_dir, "video_annotations.json")
    print(f"\nSaving annotations to: {annotations_json_path}")

    output_data = {
        'video_path': video_path,
        'video_properties': {
            'width': width,
            'height': height,
            'original_fps': original_fps,
            'duration_seconds': duration,
            'total_frames': total_frames
        },
        'processing_info': {
            'extraction_fps': fps_extract,
            'frames_processed': processed_count,
            'processing_time_seconds': elapsed_time
        },
        'frames': all_frame_annotations
    }

    with open(annotations_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save classes.txt - Final 11 classes
    classes_txt_path = os.path.join(output_dir, "classes.txt")
    classes = [
        "title",
        "text",
        "paragraph",
        "paragraph_title",
        "list",
        "equation",
        "table",
        "image",
        "figure",
        "logo",
        "unknown"
    ]
    with open(classes_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(classes))

    # Generate summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)

    total_annotations = sum(frame['num_annotations'] for frame in all_frame_annotations)
    avg_annotations = total_annotations / max(processed_count, 1)

    print(f"\nProcessing Statistics:")
    print(f"  Frames processed: {processed_count}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Average annotations per frame: {avg_annotations:.1f}")
    print(f"  Processing time: {elapsed_time:.1f} seconds")
    print(f"  Time per frame: {elapsed_time/max(processed_count, 1):.2f} seconds")

    # Count all labels
    all_label_counts = {}
    for frame_data in all_frame_annotations:
        for ann in frame_data['annotations']:
            label = ann['label']
            all_label_counts[label] = all_label_counts.get(label, 0) + 1

    if all_label_counts:
        print("\nAnnotations by label:")
        for label, count in sorted(all_label_counts.items()):
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"  {label:15s}: {count:5d} ({percentage:5.1f}%)")

    print(f"\nOutput files:")
    print(f"  - Annotated video: {output_video_path}")
    print(f"  - Annotations JSON: {annotations_json_path}")
    print(f"  - Classes file: {classes_txt_path}")
    print(f"  - X-AnyLabeling export: {export_dir}/")
    print(f"    └─ {processed_count} frames + JSON annotations")

    print(f"\nTo use in X-AnyLabeling:")
    print(f"1. Open X-AnyLabeling")
    print(f"2. Select: File > Open Dir")
    print(f"3. Browse to: {export_dir}")
    print(f"4. Annotations will be loaded automatically!")
    print("="*70 + "\n")

    return output_video_path, annotations_json_path


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def find_videos_in_directory(directory):
    """Find all video files in the specified directory."""
    video_files = []

    if not os.path.exists(directory):
        return video_files

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                video_files.append(file)

    return sorted(video_files)


def batch_process_all_videos(data_dir="data", output_base_dir="./output_video", fps_extract=DEFAULT_FPS):
    """
    Process all videos in the data directory sequentially.

    Args:
        data_dir: Directory containing video files
        output_base_dir: Base directory for outputs
        fps_extract: FPS for frame extraction
    """
    # Find all videos
    video_files = find_videos_in_directory(data_dir)

    if not video_files:
        print(f"\nNo video files found in '{data_dir}/' directory")
        print(f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        return

    print("="*70)
    print("PP-STRUCTUREV3 BATCH VIDEO ANNOTATION PIPELINE")
    print("WITH X-ANYLABELING EXPORT")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Extraction FPS: {fps_extract}")
    print(f"\nFound {len(video_files)} video(s) to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")
    print()

    # Process each video
    results = []
    successful = 0
    failed = 0

    for i, video_file in enumerate(video_files, 1):
        print("\n" + "="*70)
        print(f"PROCESSING VIDEO {i}/{len(video_files)}: {video_file}")
        print("="*70)

        video_name = Path(video_file).stem
        video_path = os.path.join(data_dir, video_file)
        output_dir = os.path.join(output_base_dir, video_name)

        try:
            video_output, annotations_output = process_video(
                video_path=video_path,
                output_dir=output_dir,
                fps_extract=fps_extract
            )

            results.append({
                'video_file': video_file,
                'status': 'success',
                'output_dir': output_dir
            })
            successful += 1
            print(f"\n✓ Successfully processed: {video_file}")

        except Exception as e:
            results.append({
                'video_file': video_file,
                'status': 'failed',
                'error': str(e)
            })
            failed += 1
            print(f"\n✗ Failed to process {video_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"\nTotal videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        print(f"\nSuccessfully processed videos:")
        for result in results:
            if result['status'] == 'success':
                print(f"  ✓ {result['video_file']}")
                print(f"    Output: {result['output_dir']}")

    if failed > 0:
        print(f"\nFailed videos:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  ✗ {result['video_file']}")
                print(f"    Error: {result['error']}")

    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for batch processing."""
    data_dir = "data"
    output_base_dir = "./output_video"

    # Validate data directory exists
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print(f"Please create the '{data_dir}/' directory and place your videos there")
        return 1

    try:
        batch_process_all_videos(
            data_dir=data_dir,
            output_base_dir=output_base_dir,
            fps_extract=DEFAULT_FPS
        )
        return 0

    except Exception as e:
        print(f"\nError during batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
