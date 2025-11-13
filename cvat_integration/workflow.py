"""
CVAT Integration Workflow - End-to-End Orchestration

This module provides a complete workflow for integrating text detection
pipeline with CVAT for human-in-the-loop annotation correction.

Author: CVAT Integration Team
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

from .format_converter import DetectionToCVATConverter, CVATToDetectionConverter
from .cvat_uploader import CVATUploader
from .cvat_exporter import CVATExporter
from .coco_converter import DetectionToCOCOConverter


class CVATWorkflow:
    """
    Orchestrates the complete CVAT integration workflow.

    Workflow Steps:
    1. Convert detection JSON to CVAT format
    2. Upload to CVAT (create task + upload frames + import annotations)
    3. Human correction in CVAT (external step)
    4. Export corrected annotations
    5. Convert back to detection format
    6. Generate quality metrics
    """

    def __init__(self, config_file: str = "cvat_config.json"):
        """
        Initialize workflow orchestrator.

        Args:
            config_file: Path to CVAT configuration file
        """
        self.config_file = config_file
        self.converter = DetectionToCVATConverter()
        self.uploader = CVATUploader(config_file)
        self.exporter = CVATExporter(config_file)

    def upload_workflow(
        self,
        video_name: str,
        detection_output_dir: str,
        task_name: Optional[str] = None,
        use_rectangles: bool = False
    ) -> Dict[str, Any]:
        """
        Complete upload workflow: Convert → Upload → Create Task.

        Args:
            video_name: Name of the video being processed
            detection_output_dir: Directory containing detection output
            task_name: Custom task name (optional)
            use_rectangles: Use rectangle format instead of polygons

        Returns:
            Upload result dictionary
        """
        print("\n" + "="*70)
        print("CVAT UPLOAD WORKFLOW")
        print("="*70)

        # Validate input directories
        frames_json_dir = os.path.join(detection_output_dir, "frames_and_json")
        if not os.path.exists(frames_json_dir):
            raise FileNotFoundError(f"frames_and_json directory not found: {frames_json_dir}")

        original_frames_dir = os.path.join(detection_output_dir, "original_frames")
        if not os.path.exists(original_frames_dir):
            raise FileNotFoundError(f"original_frames directory not found: {original_frames_dir}")

        # Step 1: Convert to CVAT format
        print("\n[STEP 1/3] Converting detections to CVAT format...")

        cvat_annotations_file = os.path.join(detection_output_dir, "cvat_annotations.json")

        if use_rectangles:
            self.converter.convert_to_rectangle_format(
                frames_json_dir,
                cvat_annotations_file
            )
        else:
            self.converter.convert_batch(
                frames_json_dir,
                cvat_annotations_file
            )

        # Step 2: Prepare task name
        if task_name is None:
            task_name = f"text_detection_{video_name}"

        # Step 3: Upload to CVAT
        print("\n[STEP 2/3] Uploading to CVAT...")

        upload_result = self.uploader.create_complete_task(
            task_name=task_name,
            frames_dir=original_frames_dir,
            annotations_file=cvat_annotations_file
        )

        # Step 3: Save workflow metadata
        print("\n[STEP 3/3] Saving workflow metadata...")

        workflow_metadata = {
            "video_name": video_name,
            "detection_output_dir": detection_output_dir,
            "task_id": upload_result["task_id"],
            "task_name": upload_result["task_name"],
            "task_url": upload_result["url"],
            "cvat_annotations_file": cvat_annotations_file,
            "workflow_stage": "uploaded_to_cvat",
            "next_step": "Human correction in CVAT"
        }

        metadata_file = os.path.join(detection_output_dir, "cvat_workflow_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(workflow_metadata, f, indent=2)

        print(f"✓ Workflow metadata saved: {metadata_file}")

        # Print summary
        print("\n" + "="*70)
        print("UPLOAD WORKFLOW COMPLETE ✓")
        print("="*70)
        print(f"Task ID: {upload_result['task_id']}")
        print(f"Task Name: {upload_result['task_name']}")
        print(f"URL: {upload_result['url']}")
        print(f"\nNext Steps:")
        print(f"1. Open URL in browser: {upload_result['url']}")
        print(f"2. Review and correct annotations in CVAT")
        print(f"3. Save changes")
        print(f"4. Run export workflow:")
        print(f"   python -m cvat_integration.workflow --video-name {video_name} --action export")
        print("="*70)

        return upload_result

    def export_workflow(
        self,
        video_name: str,
        detection_output_dir: str,
        task_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete export workflow: Export → Convert → Compare.

        Args:
            video_name: Name of the video being processed
            detection_output_dir: Directory containing original detection output
            task_id: Task ID (auto-detected if not provided)

        Returns:
            Export result dictionary
        """
        print("\n" + "="*70)
        print("CVAT EXPORT WORKFLOW")
        print("="*70)

        # Load workflow metadata to get task ID
        if task_id is None:
            metadata_file = os.path.join(detection_output_dir, "cvat_workflow_metadata.json")
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(
                    f"Workflow metadata not found: {metadata_file}\n"
                    "Please provide --task-id manually"
                )

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            task_id = metadata["task_id"]
            print(f"✓ Detected task ID from metadata: {task_id}")

        # Get task name for directory structure
        task = self.exporter.client.tasks.retrieve(task_id) if self.exporter.client else None
        if not task:
            self.exporter.connect()
            task = self.exporter.client.tasks.retrieve(task_id)

        # Create export directory: exports/{video_name}_{task_id}/
        export_base_dir = os.path.join("exports", f"{video_name}_{task_id}")
        cvat_dir = os.path.join(export_base_dir, "cvat")
        coco_dir = os.path.join(export_base_dir, "coco")

        os.makedirs(export_base_dir, exist_ok=True)

        print(f"\n📁 Export directory: {export_base_dir}")

        # Step 1: Export in both formats
        print("\n[STEP 1/2] Exporting annotations in CVAT and COCO formats...")

        export_stats = self.exporter.export_dual_format(
            task_id=task_id,
            cvat_output_dir=cvat_dir,
            coco_output_dir=coco_dir
        )

        # Step 2: Save export metadata
        print("\n[STEP 2/2] Saving export metadata...")

        export_metadata = {
            "video_name": video_name,
            "task_id": task_id,
            "task_name": export_stats["task_name"],
            "export_base_dir": export_base_dir,
            "cvat_export": export_stats["cvat_export"],
            "coco_export": export_stats["coco_export"],
            "num_frames": export_stats["num_frames"],
            "workflow_stage": "exported_from_cvat"
        }

        export_metadata_file = os.path.join(export_base_dir, "export_metadata.json")
        with open(export_metadata_file, 'w') as f:
            json.dump(export_metadata, f, indent=2)

        # Update original workflow metadata if it exists
        workflow_metadata_file = os.path.join(detection_output_dir, "cvat_workflow_metadata.json")
        if os.path.exists(workflow_metadata_file):
            with open(workflow_metadata_file, 'r') as f:
                metadata = json.load(f)

            metadata["workflow_stage"] = "exported_from_cvat"
            metadata["export_dir"] = export_base_dir
            metadata["last_export"] = export_metadata

            with open(workflow_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("EXPORT WORKFLOW COMPLETE ✓")
        print("="*70)
        print(f"Task ID: {task_id}")
        print(f"Task Name: {export_stats['task_name']}")
        print(f"Frames: {export_stats['num_frames']}")
        print(f"\n📁 Exports saved to: {export_base_dir}")
        print(f"  ├── cvat/annotations.xml")
        print(f"  └── coco/annotations.json")

        print(f"\n✨ Next Steps:")
        print(f"1. Use COCO format for model training: {coco_dir}/annotations.json")
        print(f"2. Use CVAT format for re-import or backup: {cvat_dir}/annotations.xml")
        print("="*70)

        return export_metadata

    def status_workflow(
        self,
        video_name: str,
        detection_output_dir: str
    ):
        """
        Check workflow status for a video.

        Args:
            video_name: Name of the video
            detection_output_dir: Directory containing detection output
        """
        print("\n" + "="*70)
        print(f"WORKFLOW STATUS: {video_name}")
        print("="*70)

        # Check for metadata
        metadata_file = os.path.join(detection_output_dir, "cvat_workflow_metadata.json")

        if not os.path.exists(metadata_file):
            print("\nStatus: NOT STARTED")
            print("\nTo start workflow:")
            print(f"  python -m cvat_integration.workflow --video-name {video_name} --action upload")
            print("="*70)
            return

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Display status
        print(f"\nCurrent Stage: {metadata['workflow_stage'].upper()}")
        print(f"\nTask Information:")
        print(f"  Task ID: {metadata['task_id']}")
        print(f"  Task Name: {metadata['task_name']}")
        print(f"  Task URL: {metadata['task_url']}")

        # Get live task info from CVAT
        try:
            task_info = self.exporter.get_task_info(metadata['task_id'])
            print(f"\nLive Status from CVAT:")
            print(f"  Status: {task_info['status']}")
            print(f"  Frames: {task_info['size']}")
            print(f"  Annotations: {task_info['num_annotations']}")
        except Exception as e:
            print(f"\nWarning: Could not fetch live status: {e}")

        # Suggest next step
        print(f"\nNext Step:")
        if metadata['workflow_stage'] == 'uploaded_to_cvat':
            print(f"  1. Complete annotation in CVAT: {metadata['task_url']}")
            print(f"  2. Run export:")
            print(f"     python -m cvat_integration.workflow --video-name {video_name} --action export")
        elif metadata['workflow_stage'] == 'exported_from_cvat':
            print(f"  Workflow complete!")
            print(f"  Corrected detections: {metadata['corrected_detections_dir']}")

        print("="*70)

    def convert_to_coco_workflow(
        self,
        video_name: str,
        detection_output_dir: str,
        use_corrected: bool = False,
        split_train_val: bool = False,
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        Convert detection JSONs to COCO 1.0 format.

        Args:
            video_name: Name of the video
            detection_output_dir: Directory containing detection output
            use_corrected: Use corrected detections from CVAT export (default: False)
            split_train_val: Split into train/val sets (default: False)
            train_ratio: Ratio for train split if splitting (default: 0.8)

        Returns:
            Conversion result dictionary
        """
        print("\n" + "="*70)
        print("COCO FORMAT CONVERSION")
        print("="*70)

        # Determine input directory
        if use_corrected:
            # Use corrected detections from CVAT export
            input_dir = os.path.join(detection_output_dir, "cvat_export", "corrected_detections")
            if not os.path.exists(input_dir):
                raise FileNotFoundError(
                    f"Corrected detections not found: {input_dir}\n"
                    "Please run export workflow first"
                )
            print(f"✓ Using corrected detections from CVAT")
        else:
            # Use original detections
            input_dir = os.path.join(detection_output_dir, "frames_and_json")
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Detection output not found: {input_dir}")
            print(f"✓ Using original detections")

        # Create output directory
        coco_output_dir = os.path.join(detection_output_dir, "coco_format")
        os.makedirs(coco_output_dir, exist_ok=True)

        # Convert to COCO format
        print(f"\nConverting to COCO 1.0 format...")
        coco_converter = DetectionToCOCOConverter(category_name="text")

        coco_file = os.path.join(coco_output_dir, "annotations.json")
        coco_data = coco_converter.convert_batch(
            input_dir=input_dir,
            output_file=coco_file,
            image_extension=".jpg"
        )

        # Validate
        print(f"\nValidating COCO format...")
        is_valid = coco_converter.validate_coco_format(coco_file)

        result = {
            "video_name": video_name,
            "input_dir": input_dir,
            "coco_file": coco_file,
            "num_images": len(coco_data["images"]),
            "num_annotations": len(coco_data["annotations"]),
            "is_valid": is_valid,
            "used_corrected": use_corrected
        }

        # Split into train/val if requested
        if split_train_val:
            print(f"\nSplitting into train/val sets...")
            split_dir = os.path.join(coco_output_dir, "splits")
            coco_converter.split_train_val(
                coco_file=coco_file,
                output_dir=split_dir,
                train_ratio=train_ratio
            )
            result["split_dir"] = split_dir
            result["train_file"] = os.path.join(split_dir, "train_annotations.json")
            result["val_file"] = os.path.join(split_dir, "val_annotations.json")

        # Print summary
        print("\n" + "="*70)
        print("COCO CONVERSION COMPLETE ✓")
        print("="*70)
        print(f"Images: {result['num_images']}")
        print(f"Annotations: {result['num_annotations']}")
        print(f"COCO file: {result['coco_file']}")

        if split_train_val:
            print(f"\nTrain/Val Split:")
            print(f"  Train: {result['train_file']}")
            print(f"  Val: {result['val_file']}")

        print(f"\nNext Steps:")
        print(f"1. Use COCO annotations for training (Detectron2, MMDetection, etc.)")
        print(f"2. Copy images to match COCO dataset structure")
        print("="*70)

        return result


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description="CVAT Integration Workflow for Text Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload detection results to CVAT
  python -m cvat_integration.workflow --video-name video1 --action upload

  # Export corrected annotations from CVAT
  python -m cvat_integration.workflow --video-name video1 --action export

  # Check workflow status
  python -m cvat_integration.workflow --video-name video1 --action status

  # Convert to COCO format (original detections)
  python -m cvat_integration.workflow --video-name video1 --action to-coco

  # Convert to COCO format (corrected detections from CVAT)
  python -m cvat_integration.workflow --video-name video1 --action to-coco --use-corrected

  # Convert to COCO and split into train/val
  python -m cvat_integration.workflow --video-name video1 --action to-coco --split --train-ratio 0.8

  # Upload with custom task name
  python -m cvat_integration.workflow --video-name video1 --action upload --task-name "My Custom Task"

  # Export with specific task ID
  python -m cvat_integration.workflow --video-name video1 --action export --task-id 123
        """
    )

    parser.add_argument(
        "--video-name",
        required=True,
        help="Name of the video (folder name in output directory)"
    )

    parser.add_argument(
        "--action",
        choices=["upload", "export", "status", "to-coco"],
        required=True,
        help="Workflow action to perform"
    )

    parser.add_argument(
        "--output-dir",
        default="output/text_detection_updated",
        help="Base output directory (default: output/text_detection_updated)"
    )

    parser.add_argument(
        "--config",
        default="cvat_config.json",
        help="Path to CVAT configuration file"
    )

    parser.add_argument(
        "--task-name",
        help="Custom task name for CVAT (optional)"
    )

    parser.add_argument(
        "--task-id",
        type=int,
        help="Task ID for export (auto-detected if not provided)"
    )

    parser.add_argument(
        "--use-rectangles",
        action="store_true",
        help="Use rectangle format instead of polygons"
    )

    parser.add_argument(
        "--use-corrected",
        action="store_true",
        help="Use corrected detections from CVAT export (for to-coco action)"
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val sets (for to-coco action)"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for split (default: 0.8, for to-coco action)"
    )

    args = parser.parse_args()

    # Build detection output directory path
    detection_output_dir = os.path.join(args.output_dir, args.video_name)

    if not os.path.exists(detection_output_dir):
        print(f"Error: Detection output directory not found: {detection_output_dir}")
        print(f"\nPlease run text detection first:")
        print(f"  python run_text_detection.py")
        sys.exit(1)

    # Create workflow orchestrator
    workflow = CVATWorkflow(config_file=args.config)

    # Execute action
    try:
        if args.action == "upload":
            workflow.upload_workflow(
                video_name=args.video_name,
                detection_output_dir=detection_output_dir,
                task_name=args.task_name,
                use_rectangles=args.use_rectangles
            )

        elif args.action == "export":
            workflow.export_workflow(
                video_name=args.video_name,
                detection_output_dir=detection_output_dir,
                task_id=args.task_id
            )

        elif args.action == "status":
            workflow.status_workflow(
                video_name=args.video_name,
                detection_output_dir=detection_output_dir
            )

        elif args.action == "to-coco":
            workflow.convert_to_coco_workflow(
                video_name=args.video_name,
                detection_output_dir=detection_output_dir,
                use_corrected=args.use_corrected,
                split_train_val=args.split,
                train_ratio=args.train_ratio
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
