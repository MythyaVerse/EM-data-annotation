"""
CVAT Uploader: Upload Tasks and Annotations to CVAT

This module handles creating tasks in CVAT and uploading frames with
pre-annotations from text detection pipeline.

Author: CVAT Integration Team
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from cvat_sdk import make_client
    from cvat_sdk.core.proxies.tasks import Task, ResourceType
    from cvat_sdk.core.proxies.projects import Project
except ImportError:
    print("WARNING: cvat-sdk not installed. Install with: pip install cvat-sdk")
    make_client = None
    ResourceType = None
    Task = None
    Project = None


class CVATUploader:
    """
    Uploads detection results to CVAT for human annotation correction.

    This class:
    1. Creates or retrieves CVAT project
    2. Creates annotation tasks
    3. Uploads frame images
    4. Imports pre-annotations from detection results
    """

    def __init__(self, config_file: str = "cvat_config.json"):
        """
        Initialize CVAT uploader.

        Args:
            config_file: Path to CVAT configuration JSON
        """
        if make_client is None:
            raise ImportError("cvat-sdk is required. Install with: pip install cvat-sdk")

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize CVAT client
        self.client = None
        self.project = None

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

    def get_or_create_project(
        self,
        project_name: Optional[str] = None
    ) -> "Project":
        """
        Get existing project or create new one.

        Args:
            project_name: Name of the project (default: from config)

        Returns:
            CVAT Project object
        """
        if not self.client:
            raise RuntimeError("Not connected to CVAT. Call connect() first.")

        project_name = project_name or self.config.get("project_name", "Text Detection Correction")

        print(f"Looking for project: {project_name}...")

        # Search for existing project
        projects = self.client.projects.list()
        for proj in projects:
            if proj.name == project_name:
                print(f"✓ Found existing project: {proj.name} (ID: {proj.id})")
                self.project = proj
                return proj

        # Create new project
        print(f"Creating new project: {project_name}...")

        project = self.client.projects.create(
            spec={
                "name": project_name,
                "labels": [
                    {
                        "name": "text",
                        "color": "#00ff00",  # Green
                        "attributes": [
                            {
                                "name": "confidence",
                                "mutable": False,
                                "input_type": "number",
                                "default_value": "0.0",
                                "values": ["0.0", "1.0"]
                            },
                            {
                                "name": "detection_id",
                                "mutable": False,
                                "input_type": "number",
                                "default_value": "0",
                                "values": []
                            }
                        ]
                    }
                ]
            }
        )

        print(f"✓ Created project: {project.name} (ID: {project.id})")
        self.project = project
        return project

    def create_task(
        self,
        task_name: str,
        frames_dir: str,
        project_id: Optional[int] = None
    ) -> "Task":
        """
        Create a new annotation task in CVAT.

        Args:
            task_name: Name for the task
            frames_dir: Directory containing frame images
            project_id: Project ID to associate task with (optional)

        Returns:
            CVAT Task object
        """
        if not self.client:
            raise RuntimeError("Not connected to CVAT. Call connect() first.")

        # Get image files
        image_files = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not image_files:
            raise ValueError(f"No image files found in {frames_dir}")

        print(f"Creating task: {task_name}")
        print(f"  Images: {len(image_files)} frames")

        # Use project from parameter or stored project
        if project_id is None and self.project:
            project_id = self.project.id

        # Create task specification
        task_spec = {
            "name": task_name
        }

        # Add project association OR labels (not both)
        if project_id:
            # Task inherits labels from project
            task_spec["project_id"] = project_id
        else:
            # Task defines its own labels
            task_spec["labels"] = [
                {
                    "name": "text",
                    "color": "#00ff00",
                    "attributes": [
                        {
                            "name": "confidence",
                            "mutable": False,
                            "input_type": "number",
                            "default_value": "0.0",
                            "values": ["0.0", "1.0"]
                        },
                        {
                            "name": "detection_id",
                            "mutable": False,
                            "input_type": "number",
                            "default_value": "0",
                            "values": []
                        }
                    ]
                }
            ]

        # Create task (without data first to avoid SDK bug)
        task = self.client.tasks.create(spec=task_spec)

        print(f"✓ Task created: {task.name} (ID: {task.id})")
        print(f"  URL: {self.config['cvat_host']}/tasks/{task.id}")

        # Upload data to the task
        print("  Uploading frames...")
        task.upload_data(
            resource_type=ResourceType.LOCAL,
            resources=image_files,
            params={
                "image_quality": 95,
                "sorting_method": "natural"  # Sort frames naturally (0, 1, 2, ... not 0, 1, 10, 11, ...)
            }
        )

        # Wait for task to be ready
        print("  Waiting for task initialization...")
        self._wait_for_task_ready(task.id)

        return task

    def _wait_for_task_ready(self, task_id: int, timeout: int = 300):
        """
        Wait for task to be ready for annotation.

        Args:
            task_id: Task ID
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout:
            task = self.client.tasks.retrieve(task_id)

            # Show status changes
            if task.status != last_status:
                elapsed = time.time() - start_time
                print(f"  Task status: {task.status} (elapsed: {elapsed:.1f}s)")
                last_status = task.status

            # Task is ready when it's in annotation, validation, or completed status
            if task.status in ["annotation", "validation", "completed"]:
                print("  ✓ Task ready")
                return

            elif task.status == "failed":
                raise RuntimeError(f"Task creation failed: {task.id}")

            time.sleep(2)

        raise TimeoutError(f"Task initialization timeout after {timeout}s")

    def upload_annotations(
        self,
        task_id: int,
        annotations_file: str
    ):
        """
        Upload pre-annotations to a CVAT task.

        Args:
            task_id: Task ID
            annotations_file: Path to CVAT format annotation JSON
        """
        print(f"Uploading annotations to task {task_id}...")

        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Get task
        task = self.client.tasks.retrieve(task_id)

        # Get label mapping from task (label name -> label ID)
        labels = task.get_labels()
        label_map = {label.name: label.id for label in labels}

        # Build attribute mapping (label_id -> {attribute_name -> attribute_id})
        attr_map = {}
        for label in labels:
            attr_map[label.id] = {attr.name: attr.id for attr in label.attributes}

        # Convert label names to label IDs and fix attributes format in shapes
        for shape in annotations.get("shapes", []):
            if "label" in shape and shape["label"] in label_map:
                label_id = label_map[shape["label"]]
                shape["label_id"] = label_id

                # Convert attributes from dict to list format
                if "attributes" in shape and isinstance(shape["attributes"], dict):
                    attr_list = []
                    for attr_name, attr_value in shape["attributes"].items():
                        if attr_name in attr_map.get(label_id, {}):
                            attr_list.append({
                                "spec_id": attr_map[label_id][attr_name],
                                "value": str(attr_value)
                            })
                    shape["attributes"] = attr_list
                elif "attributes" not in shape:
                    shape["attributes"] = []

            elif "label" in shape:
                raise ValueError(f"Unknown label: {shape['label']}. Available labels: {list(label_map.keys())}")

        # Upload annotations using CVAT JSON format directly via API
        # The annotations JSON is already in CVAT format, so we can patch it directly
        task.update_annotations(annotations)

        num_shapes = len(annotations.get("shapes", []))
        print(f"✓ Uploaded {num_shapes} pre-annotations")
        print(f"  Ready for human correction at: {self.config['cvat_host']}/tasks/{task_id}")

    def create_complete_task(
        self,
        task_name: str,
        frames_dir: str,
        annotations_file: str,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete workflow: Create task and upload annotations.

        Args:
            task_name: Name for the task
            frames_dir: Directory containing frame images
            annotations_file: Path to CVAT format annotation JSON
            project_name: Project name (optional)

        Returns:
            Dictionary with task information
        """
        # Connect to CVAT
        self.connect()

        # Get or create project
        project = self.get_or_create_project(project_name)

        # Create task
        task = self.create_task(
            task_name=task_name,
            frames_dir=frames_dir,
            project_id=project.id
        )

        # Upload annotations
        self.upload_annotations(task.id, annotations_file)

        # Return task info
        return {
            "task_id": task.id,
            "task_name": task.name,
            "project_id": project.id,
            "project_name": project.name,
            "url": f"{self.config['cvat_host']}/tasks/{task.id}",
            "num_frames": task.size
        }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for CVAT uploader."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload detection results to CVAT for annotation correction"
    )

    parser.add_argument(
        "--config",
        default="cvat_config.json",
        help="Path to CVAT configuration file"
    )

    parser.add_argument(
        "--task-name",
        required=True,
        help="Name for the CVAT task"
    )

    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing frame images"
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to CVAT format annotation JSON"
    )

    parser.add_argument(
        "--project-name",
        help="CVAT project name (optional)"
    )

    args = parser.parse_args()

    # Create uploader
    uploader = CVATUploader(config_file=args.config)

    # Upload complete task
    result = uploader.create_complete_task(
        task_name=args.task_name,
        frames_dir=args.frames_dir,
        annotations_file=args.annotations,
        project_name=args.project_name
    )

    print("\n" + "="*70)
    print("UPLOAD COMPLETE")
    print("="*70)
    print(f"Task ID: {result['task_id']}")
    print(f"Task Name: {result['task_name']}")
    print(f"Project: {result['project_name']}")
    print(f"Frames: {result['num_frames']}")
    print(f"URL: {result['url']}")
    print("\nNext steps:")
    print("1. Open the URL in your browser")
    print("2. Review and correct annotations")
    print("3. Save changes in CVAT")
    print("4. Export corrected annotations using cvat_exporter.py")
    print("="*70)


if __name__ == "__main__":
    main()
