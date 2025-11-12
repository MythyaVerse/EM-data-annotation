"""
Format Converter: Detection JSON → CVAT Format

This module converts text detection output JSON files to CVAT-compatible format
for importing pre-annotations into CVAT tasks.

Author: CVAT Integration Team
"""

import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np


class DetectionToCVATConverter:
    """
    Converts text detection JSON format to CVAT annotation format.

    Detection JSON Format (input):
    {
        "frame_number": 0,
        "detections": [
            {
                "box_id": 0,
                "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "confidence": 0.85
            }
        ],
        "total_detections": 5,
        "displayed_detections": 5
    }

    CVAT Format (output):
    {
        "version": 0,
        "tags": [],
        "shapes": [
            {
                "type": "polygon",
                "occluded": false,
                "points": [x1, y1, x2, y2, x3, y3, x4, y4],
                "label": "text",
                "attributes": {
                    "confidence": 0.85
                },
                "frame": 0
            }
        ],
        "tracks": []
    }
    """

    def __init__(self, label_name: str = "text"):
        """
        Initialize converter.

        Args:
            label_name: Label to use for text regions in CVAT (default: "text")
        """
        self.label_name = label_name

    def convert_single_frame(
        self,
        detection_json: Dict[str, Any],
        frame_index: int = None
    ) -> List[Dict[str, Any]]:
        """
        Convert a single frame's detection JSON to CVAT shapes.

        Args:
            detection_json: Detection output for one frame
            frame_index: Frame index (overrides frame_number in JSON if provided)

        Returns:
            List of CVAT shape dictionaries
        """
        shapes = []

        # Get frame number
        frame_num = frame_index if frame_index is not None else detection_json.get("frame_number", 0)

        # Process each detection
        for detection in detection_json.get("detections", []):
            # Extract polygon coordinates
            coordinates = detection.get("coordinates", [])
            if not coordinates or len(coordinates) < 3:
                continue  # Skip invalid polygons

            # Convert [[x1, y1], [x2, y2], ...] to [x1, y1, x2, y2, ...]
            points = []
            for point in coordinates:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.extend([float(point[0]), float(point[1])])

            if len(points) < 6:  # Need at least 3 points for a polygon
                continue

            # Get confidence score
            confidence = detection.get("confidence", 0.0)

            # Create CVAT shape
            shape = {
                "type": "polygon",
                "occluded": False,
                "z_order": 0,
                "rotation": 0.0,
                "points": points,
                "label": self.label_name,
                "group": None,
                "source": "auto",  # Mark as auto-generated
                "attributes": {
                    "confidence": round(confidence, 3),
                    "detection_id": detection.get("box_id", 0)
                },
                "frame": frame_num
            }

            shapes.append(shape)

        return shapes

    def convert_batch(
        self,
        input_dir: str,
        output_file: str,
        frame_pattern: str = "frame_{:05d}.json"
    ) -> Dict[str, Any]:
        """
        Convert multiple detection JSON files to a single CVAT annotation file.

        Args:
            input_dir: Directory containing detection JSON files
            output_file: Path to save CVAT annotation JSON
            frame_pattern: Filename pattern for detection JSONs

        Returns:
            CVAT annotation dictionary
        """
        all_shapes = []

        # Find all JSON files
        json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

        print(f"Found {len(json_files)} JSON files in {input_dir}")

        for json_file in json_files:
            json_path = os.path.join(input_dir, json_file)

            try:
                # Load detection JSON
                with open(json_path, 'r') as f:
                    detection_json = json.load(f)

                # Convert to CVAT shapes
                shapes = self.convert_single_frame(detection_json)
                all_shapes.extend(shapes)

            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue

        # Create CVAT annotation structure
        cvat_annotation = {
            "version": 0,
            "tags": [],
            "shapes": all_shapes,
            "tracks": []
        }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(cvat_annotation, f, indent=2)

        print(f"✓ Converted {len(all_shapes)} detections from {len(json_files)} frames")
        print(f"✓ Saved CVAT annotations to: {output_file}")

        return cvat_annotation

    def polygon_to_bbox(self, points: List[float]) -> Tuple[float, float, float, float]:
        """
        Convert polygon points to bounding box (for CVAT rectangle format if needed).

        Args:
            points: Flat list [x1, y1, x2, y2, x3, y3, x4, y4]

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x_coords = points[0::2]  # Every other element starting from 0
        y_coords = points[1::2]  # Every other element starting from 1

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        return (x_min, y_min, x_max, y_max)

    def convert_to_rectangle_format(
        self,
        input_dir: str,
        output_file: str
    ) -> Dict[str, Any]:
        """
        Convert detections to CVAT rectangle format (alternative to polygons).

        This is useful if you want axis-aligned bounding boxes instead of polygons.

        Args:
            input_dir: Directory containing detection JSON files
            output_file: Path to save CVAT annotation JSON

        Returns:
            CVAT annotation dictionary
        """
        all_shapes = []

        # Find all JSON files
        json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

        print(f"Found {len(json_files)} JSON files in {input_dir}")

        for json_file in json_files:
            json_path = os.path.join(input_dir, json_file)

            try:
                # Load detection JSON
                with open(json_path, 'r') as f:
                    detection_json = json.load(f)

                frame_num = detection_json.get("frame_number", 0)

                # Process each detection
                for detection in detection_json.get("detections", []):
                    coordinates = detection.get("coordinates", [])
                    if not coordinates:
                        continue

                    # Convert to flat points
                    points = []
                    for point in coordinates:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            points.extend([float(point[0]), float(point[1])])

                    if len(points) < 6:
                        continue

                    # Convert polygon to bounding box
                    x_min, y_min, x_max, y_max = self.polygon_to_bbox(points)
                    width = x_max - x_min
                    height = y_max - y_min

                    # Get confidence
                    confidence = detection.get("confidence", 0.0)

                    # Create CVAT rectangle shape
                    shape = {
                        "type": "rectangle",
                        "occluded": False,
                        "z_order": 0,
                        "rotation": 0.0,
                        "points": [x_min, y_min, x_max, y_max],
                        "label": self.label_name,
                        "group": None,
                        "source": "auto",
                        "attributes": {
                            "confidence": round(confidence, 3),
                            "detection_id": detection.get("box_id", 0)
                        },
                        "frame": frame_num
                    }

                    all_shapes.append(shape)

            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue

        # Create CVAT annotation structure
        cvat_annotation = {
            "version": 0,
            "tags": [],
            "shapes": all_shapes,
            "tracks": []
        }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(cvat_annotation, f, indent=2)

        print(f"✓ Converted {len(all_shapes)} detections to rectangles")
        print(f"✓ Saved CVAT annotations to: {output_file}")

        return cvat_annotation


class CVATToDetectionConverter:
    """
    Converts CVAT exported annotations back to detection JSON format.

    This is used after human correction to get the improved annotations
    back into your original format for model training or evaluation.
    """

    def __init__(self):
        pass

    def convert_cvat_to_detection(
        self,
        cvat_annotation_file: str,
        output_dir: str,
        frame_pattern: str = "frame_{:05d}.json"
    ) -> int:
        """
        Convert CVAT annotation JSON back to detection format.

        Args:
            cvat_annotation_file: Path to CVAT annotation JSON
            output_dir: Directory to save detection JSON files
            frame_pattern: Filename pattern for output JSONs

        Returns:
            Number of frames processed
        """
        # Load CVAT annotations
        with open(cvat_annotation_file, 'r') as f:
            cvat_data = json.load(f)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Group shapes by frame
        frames_data = {}

        for shape in cvat_data.get("shapes", []):
            frame_num = shape.get("frame", 0)

            if frame_num not in frames_data:
                frames_data[frame_num] = {
                    "frame_number": frame_num,
                    "detections": [],
                    "total_detections": 0,
                    "displayed_detections": 0
                }

            # Convert CVAT points back to coordinate format
            points = shape.get("points", [])

            if shape["type"] == "polygon":
                # Convert [x1, y1, x2, y2, ...] back to [[x1, y1], [x2, y2], ...]
                coordinates = []
                for i in range(0, len(points), 2):
                    if i + 1 < len(points):
                        coordinates.append([points[i], points[i + 1]])

            elif shape["type"] == "rectangle":
                # Rectangle: [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = points[:4]
                coordinates = [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ]
            else:
                continue  # Skip other shape types

            # Get attributes
            attributes = shape.get("attributes", {})
            confidence = attributes.get("confidence", 1.0)  # Default to 1.0 for human-corrected

            # Create detection entry
            detection = {
                "box_id": len(frames_data[frame_num]["detections"]),
                "coordinates": coordinates,
                "confidence": float(confidence)
            }

            frames_data[frame_num]["detections"].append(detection)

        # Save each frame's data
        for frame_num, frame_data in frames_data.items():
            frame_data["total_detections"] = len(frame_data["detections"])
            frame_data["displayed_detections"] = len(frame_data["detections"])

            # Generate filename
            output_file = os.path.join(output_dir, frame_pattern.format(frame_num))

            with open(output_file, 'w') as f:
                json.dump(frame_data, f, indent=4)

        print(f"✓ Converted {len(frames_data)} frames back to detection format")
        print(f"✓ Saved to: {output_dir}")

        return len(frames_data)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for format conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert between detection JSON and CVAT annotation formats"
    )

    parser.add_argument(
        "mode",
        choices=["to-cvat", "from-cvat", "to-cvat-rect"],
        help="Conversion mode"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing detection JSONs (for to-cvat mode)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output file (for to-cvat) or directory (for from-cvat)"
    )

    parser.add_argument(
        "--label",
        default="text",
        help="Label name for CVAT annotations (default: text)"
    )

    parser.add_argument(
        "--cvat-file",
        help="CVAT annotation file (for from-cvat mode)"
    )

    args = parser.parse_args()

    if args.mode == "to-cvat":
        print("Converting detection JSON to CVAT format (polygons)...")
        converter = DetectionToCVATConverter(label_name=args.label)
        converter.convert_batch(args.input_dir, args.output)

    elif args.mode == "to-cvat-rect":
        print("Converting detection JSON to CVAT format (rectangles)...")
        converter = DetectionToCVATConverter(label_name=args.label)
        converter.convert_to_rectangle_format(args.input_dir, args.output)

    elif args.mode == "from-cvat":
        if not args.cvat_file:
            print("Error: --cvat-file required for from-cvat mode")
            return
        print("Converting CVAT annotations back to detection JSON...")
        converter = CVATToDetectionConverter()
        converter.convert_cvat_to_detection(args.cvat_file, args.output)

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
