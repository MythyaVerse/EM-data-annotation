"""
Video Text Detection Pipeline - Detection Only (Updated for scale parameter)

This script processes video files using ONLY text detection:
1. Manual ONNX Runtime text detection (DB model) - UPDATED VERSION
2. Bounding box generation with confidence scores
3. Frame sampling at 5 FPS
4. NO OCR, NO Classification - Pure detection only

UPDATED: Uses scale parameter (1.0/255) instead of std normalization

FEATURES:
- Custom DB text detection (full control over detection pipeline)
- Configurable FPS sampling (default: 5 FPS)
- Comprehensive output (JSON metadata with bounding boxes + annotated video)
- Visual debugging (boxes with detection confidence)

Author: Detection Pipeline (Updated)
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple
from text_detection import load_db_model_manual, detect_text_manual


# ============================================================================
# CONFIGURATION
# ============================================================================

class DetectionConfig:
    """Configuration for detection-only pipeline"""

    def __init__(self):
        # ===== INPUT/OUTPUT PATHS =====
        self.video_path = "data/videos"  # Path to video or folder
        self.model_path = "model/DB_TD500_resnet50.onnx"
        self.output_dir = "output_detection_only"

        # ===== DETECTION PARAMETERS (UPDATED: using scale instead of std) =====
        self.binary_threshold = 0.3
        self.polygon_threshold = 0.65  # Increased from 0.5 to reduce noise
        self.input_size = (320, 320)  # Model input size
        self.mean = (122.67891434, 116.66876762, 104.00698793)
        self.scale = 1.0 / 255.0  # UPDATED: Scale factor instead of std
        self.swap_rb = True

        # ===== VIDEO SAMPLING =====
        self.target_fps = 5  # Process video at 5 FPS (sample frames)
        self.max_frames = None  # Set to None for all frames, or number for limit

        # ===== VISUALIZATION =====
        self.box_color = (0, 255, 0)  # Green boxes (BGR format)
        self.box_thickness = 2
        self.show_confidence = True  # Draw confidence scores on boxes
        self.confidence_threshold = 0.0  # Minimum confidence to display (0.0 = all)

        # ===== OUTPUT OPTIONS =====
        self.save_annotated_video = True
        self.save_frames = True  # Save individual frames
        self.save_json = True  # Save detection metadata


# ============================================================================
# DETECTION-ONLY VIDEO PROCESSOR
# ============================================================================

class VideoTextDetector:
    """Video processing pipeline with text detection only"""

    def __init__(self, config: DetectionConfig):
        """
        Initialize text detector

        Args:
            config: Configuration object with all settings
        """
        self.config = config
        self.onnx_session = None
        self.detection_config = None
        self.frames_processed = 0

        # Statistics tracking
        self.total_detections = 0
        self.confidence_scores = []

        # Create output directories
        self._setup_directories()

    def _setup_directories(self):
        """Create output directory structure"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        if self.config.save_frames or self.config.save_json:
            self.frames_json_dir = os.path.join(self.config.output_dir, "frames_and_json")
            if not os.path.exists(self.frames_json_dir):
                os.makedirs(self.frames_json_dir)
            print(f"✓ Frames/JSON directory: {self.frames_json_dir}")

        print(f"✓ Output directory: {self.config.output_dir}")

    def load_model(self):
        """Load ONNX model for text detection"""
        print("\n" + "="*70)
        print("LOADING DETECTION MODEL (UPDATED VERSION)")
        print("="*70)

        # Load text detection model
        self.onnx_session = load_db_model_manual(self.config.model_path)

        # Configure detection parameters (UPDATED: using scale instead of std)
        self.detection_config = {
            'binary_threshold': self.config.binary_threshold,
            'polygon_threshold': self.config.polygon_threshold,
            'input_size': self.config.input_size,
            'mean': self.config.mean,
            'scale': self.config.scale,  # UPDATED: scale parameter
            'swap_rb': self.config.swap_rb
        }

        print(f"✓ Detection model loaded: {self.config.model_path}")
        print(f"✓ Binary threshold: {self.config.binary_threshold}")
        print(f"✓ Polygon threshold: {self.config.polygon_threshold}")
        print(f"✓ Input size: {self.config.input_size}")
        print(f"✓ Scale factor: {self.config.scale} (UPDATED)")

    def process_video(self):
        """
        Main video processing pipeline - Detection Only

        Pipeline:
        1. Opens video file
        2. Calculates frame sampling interval for target FPS
        3. Creates video writer for output (if enabled)
        4. Processes sampled frames:
           - Detect text using manual implementation
           - Draw bounding boxes with confidence scores
           - Save frame and JSON metadata (if enabled)
           - Write to output video (if enabled)
        5. Closes video files
        6. Prints detection statistics
        """
        print("\n" + "="*70)
        print("VIDEO PROCESSING (DETECTION ONLY - UPDATED)")
        print("="*70)

        # STEP 1: Open video file
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.config.video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame sampling interval for target FPS
        frame_interval = max(1, int(original_fps / self.config.target_fps))
        effective_fps = original_fps / frame_interval

        print(f"\n✓ Video opened: {self.config.video_path}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  Original FPS: {original_fps:.2f}")
        print(f"  Target FPS: {self.config.target_fps}")
        print(f"  Frame interval: {frame_interval} (process every {frame_interval} frames)")
        print(f"  Effective FPS: {effective_fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames to process: ~{total_frames // frame_interval}")

        # STEP 2: Create video writer for output (if enabled)
        out = None
        output_video_path = None
        if self.config.save_annotated_video:
            output_video_path = os.path.join(self.config.output_dir, "annotated_video_detection.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, effective_fps, (frame_width, frame_height))
            print(f"✓ Output video: {output_video_path} (at {effective_fps:.2f} FPS)")

        # STEP 3: Process frames with sampling
        print(f"\n{'='*70}")
        print("PROCESSING FRAMES")
        print(f"{'='*70}\n")

        frame_count = 0
        processed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we should process this frame (sampling logic)
            should_process = (frame_count % frame_interval == 0)

            if should_process:
                # Check frame limit
                if self.config.max_frames and processed_count >= self.config.max_frames:
                    print(f"\n✓ Reached processed frame limit: {self.config.max_frames}")
                    break

                # Process frame (detection only)
                annotated_frame, metadata = self._process_frame(frame, frame_count)

                # Write to output video (if enabled)
                if out is not None:
                    out.write(annotated_frame)

                # Save frame and metadata (if enabled)
                if self.config.save_frames or self.config.save_json:
                    self._save_frame_outputs(annotated_frame, metadata, processed_count)

                processed_count += 1

                # Progress update
                if processed_count % 20 == 0:
                    print(f"Processed {processed_count} frames (original frame {frame_count})...")

            frame_count += 1

        # STEP 4: Cleanup
        cap.release()
        if out is not None:
            out.release()

        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"\n✓ Total video frames: {frame_count}")
        print(f"✓ Frames processed (sampled): {processed_count}")
        print(f"✓ Sampling ratio: 1 in {frame_interval} frames")
        if self.config.save_annotated_video and output_video_path:
            print(f"✓ Annotated video: {output_video_path}")
        if self.config.save_frames or self.config.save_json:
            print(f"✓ Frames and JSON: {self.frames_json_dir}")

        # STEP 5: Set frames processed and print detection statistics
        self.frames_processed = processed_count
        self._print_statistics()

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame - detection only

        Args:
            frame: Input frame (BGR format)
            frame_number: Frame index

        Returns:
            annotated_frame: Frame with bounding boxes drawn
            metadata: Dictionary with detection information
        """
        # Create copy for annotation
        annotated_frame = frame.copy()

        # DETECT TEXT USING MANUAL IMPLEMENTATION (UPDATED VERSION)
        boxes, confidences = detect_text_manual(
            frame,
            self.onnx_session,
            self.detection_config
        )

        # Prepare metadata
        metadata = {
            "frame_number": frame_number,
            "detections": []
        }

        # Draw bounding boxes
        num_displayed = 0
        if boxes is not None and len(boxes) > 0:
            for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                # Filter by confidence threshold
                if confidence < self.config.confidence_threshold:
                    continue

                num_displayed += 1
                self.total_detections += 1
                self.confidence_scores.append(confidence)

                # Draw polygon
                cv2.polylines(
                    annotated_frame,
                    [box.astype(np.int32)],
                    isClosed=True,
                    color=self.config.box_color,
                    thickness=self.config.box_thickness
                )

                # Draw confidence label (if enabled)
                if self.config.show_confidence:
                    text_pos = (int(box[0][0]), int(box[0][1]) - 5)
                    label_text = f"{confidence:.2f}"

                    # Add background for better readability
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (int(box[0][0]), int(box[0][1]) - text_height - baseline - 5),
                        (int(box[0][0]) + text_width, int(box[0][1])),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        annotated_frame,
                        label_text,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.config.box_color,
                        1
                    )

                # Add to metadata
                detection_info = {
                    "box_id": i,
                    "coordinates": box.tolist(),
                    "confidence": float(confidence)
                }
                metadata["detections"].append(detection_info)

        metadata["total_detections"] = len(boxes) if boxes is not None else 0
        metadata["displayed_detections"] = num_displayed

        return annotated_frame, metadata

    def _save_frame_outputs(self, frame: np.ndarray, metadata: Dict[str, Any], frame_number: int):
        """
        Save frame image and JSON metadata

        Args:
            frame: Annotated frame
            metadata: Detection metadata
            frame_number: Frame index
        """
        # Generate filenames
        base_filename = f"frame_{frame_number:05d}"

        # Save frame image (if enabled)
        if self.config.save_frames:
            image_path = os.path.join(self.frames_json_dir, f"{base_filename}.jpg")
            cv2.imwrite(image_path, frame)
            metadata["image_path"] = image_path

        # Save JSON metadata (if enabled)
        if self.config.save_json:
            json_path = os.path.join(self.frames_json_dir, f"{base_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=4)

    def _print_statistics(self):
        """Print detection statistics"""
        print("\n" + "="*70)
        print("DETECTION STATISTICS")
        print("="*70)

        if self.total_detections == 0:
            print("\nNo text detected in video")
            return

        print(f"\nTotal text regions detected: {self.total_detections}")
        print(f"Average detections per frame: {self.total_detections / self.frames_processed:.1f}")

        # Confidence statistics
        if self.confidence_scores:
            avg_confidence = np.mean(self.confidence_scores)
            median_confidence = np.median(self.confidence_scores)
            min_confidence = np.min(self.confidence_scores)
            max_confidence = np.max(self.confidence_scores)

            print(f"\nDetection Confidence Statistics:")
            print(f"  Average:  {avg_confidence:.3f}")
            print(f"  Median:   {median_confidence:.3f}")
            print(f"  Min:      {min_confidence:.3f}")
            print(f"  Max:      {max_confidence:.3f}")

            # Confidence distribution
            high_conf = sum(1 for c in self.confidence_scores if c > 0.7)
            med_conf = sum(1 for c in self.confidence_scores if 0.5 <= c <= 0.7)
            low_conf = sum(1 for c in self.confidence_scores if c < 0.5)

            print(f"\nConfidence Distribution:")
            print(f"  High (>0.7):     {high_conf:>5} ({high_conf/len(self.confidence_scores)*100:>5.1f}%)")
            print(f"  Medium (0.5-0.7): {med_conf:>5} ({med_conf/len(self.confidence_scores)*100:>5.1f}%)")
            print(f"  Low (<0.5):      {low_conf:>5} ({low_conf/len(self.confidence_scores)*100:>5.1f}%)")

        # Save statistics to JSON
        stats_path = os.path.join(self.config.output_dir, "detection_statistics.json")
        stats_data = {
            "total_frames_processed": self.frames_processed,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": self.total_detections / self.frames_processed if self.frames_processed > 0 else 0,
            "confidence_statistics": {
                "average": float(np.mean(self.confidence_scores)) if self.confidence_scores else 0,
                "median": float(np.median(self.confidence_scores)) if self.confidence_scores else 0,
                "min": float(np.min(self.confidence_scores)) if self.confidence_scores else 0,
                "max": float(np.max(self.confidence_scores)) if self.confidence_scores else 0,
                "std": float(np.std(self.confidence_scores)) if self.confidence_scores else 0
            },
            "confidence_distribution": {
                "high_above_0.7": sum(1 for c in self.confidence_scores if c > 0.7),
                "medium_0.5_to_0.7": sum(1 for c in self.confidence_scores if 0.5 <= c <= 0.7),
                "low_below_0.5": sum(1 for c in self.confidence_scores if c < 0.5)
            },
            "configuration": {
                "target_fps": self.config.target_fps,
                "binary_threshold": self.config.binary_threshold,
                "polygon_threshold": self.config.polygon_threshold,
                "input_size": self.config.input_size,
                "scale": self.config.scale,  # UPDATED: scale instead of std
                "confidence_threshold": self.config.confidence_threshold
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=4)

        print(f"\n✓ Statistics saved to: {stats_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function with batch video processing

    This function processes all videos in a folder with:
    - Manual DB text detection ONLY (UPDATED VERSION with scale parameter)
    - Bounding box generation with confidence scores
    - 5 FPS frame sampling
    - Annotated video output
    - JSON metadata with detection results

    Usage:
        python run_text_detection_updated.py
    """
    print("\n" + "="*70)
    print("VIDEO TEXT DETECTION PIPELINE (UPDATED VERSION)")
    print("Using scale parameter (1.0/255) instead of std normalization")
    print("="*70)

    # ====== USER INPUTS ======
    input_folder = "data/videos"       # Path to folder containing multiple videos
    model_path = "model/DB_TD500_resnet50.onnx"
    base_output_dir = "output/text_detection_updated"
    max_frames = None  # Set to None for all frames, or e.g. 100 for testing
    target_fps = 5     # Process video at 5 FPS (sample every N frames)

    # ====== FIND ALL VIDEO FILES ======
    supported_exts = (".mp4", ".avi", ".mov", ".mkv")

    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        print(f"   Please update the input_folder path in the script")
        return

    video_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(supported_exts)
    ]

    if not video_files:
        print(f"❌ No videos found in: {input_folder}")
        return

    print(f"\n✓ Found {len(video_files)} videos in: {input_folder}")
    print("=" * 70)

    # ====== PROCESS EACH VIDEO ======
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing: {os.path.basename(video_path)}")

        # Create configuration for this video
        config = DetectionConfig()
        config.video_path = video_path
        config.model_path = model_path
        config.target_fps = target_fps  # Set target FPS for sampling
        config.max_frames = max_frames

        # Create unique output folder per video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        config.output_dir = os.path.join(base_output_dir, video_name)

        # Create processor and run pipeline
        processor = VideoTextDetector(config)
        processor.load_model()
        processor.process_video()

        print(f"\n✅ Completed: {os.path.basename(video_path)}")
        print(f"   Output saved in: {config.output_dir}")
        print("-" * 70)

    print("\n" + "="*70)
    print("ALL VIDEOS PROCESSED ✅")
    print("="*70)


if __name__ == "__main__":
    main()
