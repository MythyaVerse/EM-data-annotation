"""
CVAT Exporter: Export Corrected Annotations from CVAT

This module downloads human-corrected annotations from CVAT and converts
them back to the detection JSON format for training or evaluation.

Author: CVAT Integration Team
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import zipfile

try:
    from cvat_sdk import make_client
    from cvat_sdk.core.proxies.tasks import Task
except ImportError:
    print("WARNING: cvat-sdk not installed. Install with: pip install cvat-sdk")
    make_client = None

from .format_converter import CVATToDetectionConverter


class CVATExporter:
    """
    Exports corrected annotations from CVAT back to detection format.

    This class:
    1. Connects to CVAT server
    2. Retrieves task annotations
    3. Downloads in CVAT format
    4. Converts back to detection JSON format
    5. Generates quality metrics
    """

    def __init__(self, config_file: str = "cvat_config.json"):
        """
        Initialize CVAT exporter.

        Args:
            config_file: Path to CVAT configuration JSON
        """
        if make_client is None:
            raise ImportError("cvat-sdk is required. Install with: pip install cvat-sdk")

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize CVAT client
        self.client = None

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load CVAT configuration from JSON file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ["cvat_host", "cvat_username", "cvat_password"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        return config

    def connect(self):
        """Connect to CVAT server."""
        print(f"Connecting to CVAT at {self.config['cvat_host']}...")

        try:
            self.client = make_client(
                host=self.config['cvat_host'],
                credentials=(
                    self.config['cvat_username'],
                    self.config['cvat_password']
                )
            )
            print("✓ Connected to CVAT successfully")

        except Exception as e:
            raise ConnectionError(f"Failed to connect to CVAT: {e}")

    def list_tasks(self, project_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available tasks.

        Args:
            project_name: Filter by project name (optional)

        Returns:
            List of task information dictionaries
        """
        if not self.client:
            raise RuntimeError("Not connected to CVAT. Call connect() first.")

        tasks = self.client.tasks.list()
        task_list = []

        for task in tasks:
            # Filter by project if specified
            if project_name:
                try:
                    project = self.client.projects.retrieve(task.project_id)
                    if project.name != project_name:
                        continue
                except:
                    continue

            task_info = {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "size": task.size,
                "project_id": task.project_id if hasattr(task, 'project_id') else None
            }
            task_list.append(task_info)

        return task_list

    def export_task_annotations(
        self,
        task_id: int,
        output_file: str,
        format_name: str = "CVAT for images 1.1"
    ):
        """
        Export annotations from a CVAT task.

        Args:
            task_id: Task ID to export
            output_file: Path to save exported annotations
            format_name: Export format (default: "CVAT for images 1.1")
        """
        if not self.client:
            raise RuntimeError("Not connected to CVAT. Call connect() first.")

        print(f"Exporting annotations from task {task_id}...")

        # Get task
        task = self.client.tasks.retrieve(task_id)

        print(f"  Task: {task.name}")
        print(f"  Frames: {task.size}")

        # Export annotations
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Download annotations
            task.export_dataset(
                format_name=format_name,
                filename=tmp_path
            )

            # Extract ZIP to get annotations.json
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                # Look for annotations.json in the ZIP
                json_file = None
                for name in zip_ref.namelist():
                    if name.endswith('annotations.json'):
                        json_file = name
                        break

                if not json_file:
                    raise ValueError("annotations.json not found in exported archive")

                # Extract annotations.json
                with zip_ref.open(json_file) as source:
                    annotations_data = json.load(source)

                # Save to output file
                with open(output_file, 'w') as f:
                    json.dump(annotations_data, f, indent=2)

            print(f"✓ Annotations exported to: {output_file}")

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def export_and_convert(
        self,
        task_id: int,
        output_dir: str,
        compare_with_original: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export annotations and convert to detection JSON format.

        Args:
            task_id: Task ID to export
            output_dir: Directory to save converted detection JSONs
            compare_with_original: Path to original detections for comparison (optional)

        Returns:
            Export statistics and comparison results
        """
        # Connect if not already connected
        if not self.client:
            self.connect()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Export CVAT annotations
        cvat_annotation_file = os.path.join(output_dir, "cvat_annotations.json")
        self.export_task_annotations(task_id, cvat_annotation_file)

        # Convert back to detection format
        print("\nConverting to detection JSON format...")
        converter = CVATToDetectionConverter()

        corrected_dir = os.path.join(output_dir, "corrected_detections")
        num_frames = converter.convert_cvat_to_detection(
            cvat_annotation_file,
            corrected_dir
        )

        # Gather statistics
        stats = {
            "task_id": task_id,
            "num_frames": num_frames,
            "output_dir": corrected_dir,
            "cvat_annotations": cvat_annotation_file
        }

        # Compare with original if provided
        if compare_with_original and os.path.exists(compare_with_original):
            print("\nComparing with original detections...")
            comparison = self._compare_annotations(compare_with_original, corrected_dir)
            stats["comparison"] = comparison

        # Save statistics
        stats_file = os.path.join(output_dir, "export_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Export complete!")
        print(f"  Corrected detections: {corrected_dir}")
        print(f"  Statistics: {stats_file}")

        return stats

    def _compare_annotations(
        self,
        original_dir: str,
        corrected_dir: str
    ) -> Dict[str, Any]:
        """
        Compare original and corrected annotations.

        Args:
            original_dir: Directory with original detection JSONs
            corrected_dir: Directory with corrected detection JSONs

        Returns:
            Comparison statistics
        """
        comparison = {
            "frames_compared": 0,
            "original_detections": 0,
            "corrected_detections": 0,
            "added_detections": 0,
            "removed_detections": 0,
            "modified_detections": 0
        }

        # Get all JSON files in corrected directory
        corrected_files = [f for f in os.listdir(corrected_dir) if f.endswith('.json')]

        for json_file in corrected_files:
            original_path = os.path.join(original_dir, json_file)
            corrected_path = os.path.join(corrected_dir, json_file)

            # Skip if original doesn't exist
            if not os.path.exists(original_path):
                continue

            comparison["frames_compared"] += 1

            # Load both files
            with open(original_path, 'r') as f:
                original_data = json.load(f)

            with open(corrected_path, 'r') as f:
                corrected_data = json.load(f)

            # Count detections
            original_count = len(original_data.get("detections", []))
            corrected_count = len(corrected_data.get("detections", []))

            comparison["original_detections"] += original_count
            comparison["corrected_detections"] += corrected_count

            # Estimate changes (simplified - real comparison would need IoU matching)
            if corrected_count > original_count:
                comparison["added_detections"] += (corrected_count - original_count)
            elif corrected_count < original_count:
                comparison["removed_detections"] += (original_count - corrected_count)

        # Calculate percentages
        if comparison["original_detections"] > 0:
            comparison["change_percentage"] = (
                (comparison["corrected_detections"] - comparison["original_detections"]) /
                comparison["original_detections"] * 100
            )
        else:
            comparison["change_percentage"] = 0.0

        print(f"  Frames compared: {comparison['frames_compared']}")
        print(f"  Original detections: {comparison['original_detections']}")
        print(f"  Corrected detections: {comparison['corrected_detections']}")
        print(f"  Change: {comparison['change_percentage']:.1f}%")

        return comparison

    def get_task_info(self, task_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a task.

        Args:
            task_id: Task ID

        Returns:
            Task information dictionary
        """
        if not self.client:
            self.connect()

        task = self.client.tasks.retrieve(task_id)

        # Get annotation statistics
        annotations = task.get_annotations()
        num_shapes = len(annotations.shapes) if hasattr(annotations, 'shapes') else 0

        info = {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "size": task.size,
            "num_annotations": num_shapes,
            "url": f"{self.config['cvat_host']}/tasks/{task.id}"
        }

        return info


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for CVAT exporter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export corrected annotations from CVAT"
    )

    parser.add_argument(
        "--config",
        default="cvat_config.json",
        help="Path to CVAT configuration file"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List tasks command
    list_parser = subparsers.add_parser("list", help="List available tasks")
    list_parser.add_argument(
        "--project",
        help="Filter by project name"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export task annotations")
    export_parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task ID to export"
    )
    export_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for converted detections"
    )
    export_parser.add_argument(
        "--compare-with",
        help="Path to original detections for comparison (optional)"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Get task information")
    info_parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task ID"
    )

    args = parser.parse_args()

    # Create exporter
    exporter = CVATExporter(config_file=args.config)

    if args.command == "list":
        exporter.connect()
        tasks = exporter.list_tasks(project_name=args.project)

        print("\n" + "="*70)
        print("AVAILABLE TASKS")
        print("="*70)

        if not tasks:
            print("No tasks found")
        else:
            for task in tasks:
                print(f"\nTask ID: {task['id']}")
                print(f"  Name: {task['name']}")
                print(f"  Status: {task['status']}")
                print(f"  Frames: {task['size']}")

        print("="*70)

    elif args.command == "export":
        stats = exporter.export_and_convert(
            task_id=args.task_id,
            output_dir=args.output_dir,
            compare_with_original=args.compare_with
        )

        print("\n" + "="*70)
        print("EXPORT COMPLETE")
        print("="*70)
        print(f"Task ID: {stats['task_id']}")
        print(f"Frames: {stats['num_frames']}")
        print(f"Output: {stats['output_dir']}")

        if "comparison" in stats:
            comp = stats["comparison"]
            print(f"\nComparison with original:")
            print(f"  Original detections: {comp['original_detections']}")
            print(f"  Corrected detections: {comp['corrected_detections']}")
            print(f"  Change: {comp['change_percentage']:.1f}%")

        print("="*70)

    elif args.command == "info":
        info = exporter.get_task_info(args.task_id)

        print("\n" + "="*70)
        print("TASK INFORMATION")
        print("="*70)
        print(f"Task ID: {info['id']}")
        print(f"Name: {info['name']}")
        print(f"Status: {info['status']}")
        print(f"Frames: {info['size']}")
        print(f"Annotations: {info['num_annotations']}")
        print(f"URL: {info['url']}")
        print("="*70)


if __name__ == "__main__":
    main()
