"""
COCO Format Converter: Detection JSON → COCO 1.0 Format

This module converts text detection JSON format to COCO 1.0 format
for training object detection models (Detectron2, MMDetection, etc.).

Author: CVAT Integration Team
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image
import numpy as np


class DetectionToCOCOConverter:
    """
    Converts text detection JSON format to COCO 1.0 format.

    Detection JSON Format (input):
    {
        "frame_number": 0,
        "detections": [
            {
                "box_id": 0,
                "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "confidence": 0.85
            }
        ]
    }

    COCO Format (output):
    {
        "images": [...],
        "annotations": [...],
        "categories": [{"id": 1, "name": "text", "supercategory": "text"}]
    }
    """

    def __init__(self, category_name: str = "text", category_id: int = 1):
        """
        Initialize converter.

        Args:
            category_name: Name for the text category
            category_id: ID for the text category (default: 1)
        """
        self.category_name = category_name
        self.category_id = category_id
        self.annotation_id_counter = 1

    def polygon_to_bbox(
        self,
        coordinates: List[List[float]]
    ) -> Tuple[float, float, float, float]:
        """
        Convert polygon coordinates to bounding box [x, y, width, height].

        Args:
            coordinates: Polygon coordinates [[x1, y1], [x2, y2], ...]

        Returns:
            Tuple of (x, y, width, height) in COCO format
        """
        if not coordinates:
            return (0, 0, 0, 0)

        x_coords = [point[0] for point in coordinates]
        y_coords = [point[1] for point in coordinates]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        width = x_max - x_min
        height = y_max - y_min

        return (x_min, y_min, width, height)

    def polygon_to_segmentation(
        self,
        coordinates: List[List[float]]
    ) -> List[List[float]]:
        """
        Convert polygon coordinates to COCO segmentation format.

        Args:
            coordinates: Polygon coordinates [[x1, y1], [x2, y2], ...]

        Returns:
            Segmentation in COCO format [[x1, y1, x2, y2, x3, y3, x4, y4]]
        """
        if not coordinates:
            return [[]]

        # Flatten coordinates
        flat_coords = []
        for point in coordinates:
            flat_coords.extend([float(point[0]), float(point[1])])

        return [flat_coords]

    def calculate_area(self, coordinates: List[List[float]]) -> float:
        """
        Calculate polygon area using shoelace formula.

        Args:
            coordinates: Polygon coordinates [[x1, y1], [x2, y2], ...]

        Returns:
            Area of the polygon
        """
        if len(coordinates) < 3:
            return 0.0

        area = 0.0
        n = len(coordinates)

        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]

        area = abs(area) / 2.0
        return area

    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get image width and height.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (width, height)
        """
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}")
            return (0, 0)

    def convert_single_frame(
        self,
        detection_json: Dict[str, Any],
        image_path: str,
        image_id: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convert a single frame's detection JSON to COCO format.

        Args:
            detection_json: Detection output for one frame
            image_path: Path to the corresponding image file
            image_id: Unique image ID

        Returns:
            Tuple of (image_info, list_of_annotations)
        """
        # Get image dimensions
        width, height = self.get_image_dimensions(image_path)

        # Create image info
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height,
            "date_captured": "",
            "license": 0,
            "coco_url": "",
            "flickr_url": ""
        }

        # Create annotations
        annotations = []

        for detection in detection_json.get("detections", []):
            coordinates = detection.get("coordinates", [])

            if not coordinates or len(coordinates) < 3:
                continue

            # Convert to COCO formats
            bbox = self.polygon_to_bbox(coordinates)
            segmentation = self.polygon_to_segmentation(coordinates)
            area = self.calculate_area(coordinates)

            # Get confidence (optional, not standard COCO but useful)
            confidence = detection.get("confidence", 1.0)

            # Create annotation
            annotation = {
                "id": self.annotation_id_counter,
                "image_id": image_id,
                "category_id": self.category_id,
                "bbox": list(bbox),
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0,
                "score": confidence  # Non-standard but useful
            }

            annotations.append(annotation)
            self.annotation_id_counter += 1

        return image_info, annotations

    def convert_batch(
        self,
        input_dir: str,
        output_file: str,
        image_extension: str = ".jpg"
    ) -> Dict[str, Any]:
        """
        Convert multiple detection JSON files to a single COCO annotation file.

        Args:
            input_dir: Directory containing detection JSON files and images
            output_file: Path to save COCO annotation JSON
            image_extension: Image file extension (default: .jpg)

        Returns:
            COCO annotation dictionary
        """
        print(f"Converting detection JSONs to COCO 1.0 format...")
        print(f"Input directory: {input_dir}")

        # Initialize COCO structure
        coco_data = {
            "info": {
                "description": "Text Detection Dataset",
                "url": "",
                "version": "1.0",
                "year": 2025,
                "contributor": "Text Detection Pipeline",
                "date_created": ""
            },
            "licenses": [
                {
                    "id": 0,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": self.category_id,
                    "name": self.category_name,
                    "supercategory": self.category_name
                }
            ]
        }

        # Find all JSON files
        json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

        print(f"Found {len(json_files)} JSON files")

        image_id = 1
        total_annotations = 0

        for json_file in json_files:
            json_path = os.path.join(input_dir, json_file)

            # Get corresponding image file
            base_name = os.path.splitext(json_file)[0]
            image_file = base_name + image_extension
            image_path = os.path.join(input_dir, image_file)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found for {json_file}, skipping")
                continue

            try:
                # Load detection JSON
                with open(json_path, 'r') as f:
                    detection_json = json.load(f)

                # Convert to COCO format
                image_info, annotations = self.convert_single_frame(
                    detection_json,
                    image_path,
                    image_id
                )

                # Add to COCO data
                coco_data["images"].append(image_info)
                coco_data["annotations"].extend(annotations)

                total_annotations += len(annotations)
                image_id += 1

            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue

        # Save COCO annotations
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"\n✓ Conversion complete!")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  Categories: {len(coco_data['categories'])}")
        print(f"  Output: {output_file}")

        return coco_data

    def split_train_val(
        self,
        coco_file: str,
        output_dir: str,
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        """
        Split COCO dataset into train and validation sets.

        Args:
            coco_file: Path to COCO annotation file
            output_dir: Directory to save train/val splits
            train_ratio: Ratio of training data (default: 0.8)
            seed: Random seed for reproducibility
        """
        print(f"\nSplitting dataset into train/val...")

        # Load COCO data
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # Set random seed
        np.random.seed(seed)

        # Shuffle images
        images = coco_data["images"]
        np.random.shuffle(images)

        # Split
        num_train = int(len(images) * train_ratio)
        train_images = images[:num_train]
        val_images = images[num_train:]

        # Get image IDs
        train_image_ids = {img["id"] for img in train_images}
        val_image_ids = {img["id"] for img in val_images}

        # Split annotations
        train_annotations = [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] in train_image_ids
        ]
        val_annotations = [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] in val_image_ids
        ]

        # Create train dataset
        train_data = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "images": train_images,
            "annotations": train_annotations,
            "categories": coco_data["categories"]
        }

        # Create val dataset
        val_data = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "images": val_images,
            "annotations": val_annotations,
            "categories": coco_data["categories"]
        }

        # Save splits
        os.makedirs(output_dir, exist_ok=True)

        train_file = os.path.join(output_dir, "train_annotations.json")
        val_file = os.path.join(output_dir, "val_annotations.json")

        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)

        with open(val_file, 'w') as f:
            json.dump(val_data, f, indent=2)

        print(f"✓ Split complete!")
        print(f"  Train: {len(train_images)} images, {len(train_annotations)} annotations")
        print(f"  Val: {len(val_images)} images, {len(val_annotations)} annotations")
        print(f"  Train file: {train_file}")
        print(f"  Val file: {val_file}")

    def validate_coco_format(self, coco_file: str) -> bool:
        """
        Validate COCO format JSON.

        Args:
            coco_file: Path to COCO annotation file

        Returns:
            True if valid, False otherwise
        """
        print(f"\nValidating COCO format: {coco_file}")

        try:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)

            # Check required keys
            required_keys = ["images", "annotations", "categories"]
            for key in required_keys:
                if key not in coco_data:
                    print(f"✗ Missing required key: {key}")
                    return False

            # Check images
            if not coco_data["images"]:
                print(f"✗ No images found")
                return False

            for img in coco_data["images"]:
                if not all(k in img for k in ["id", "file_name", "width", "height"]):
                    print(f"✗ Invalid image entry: {img}")
                    return False

            # Check annotations
            for ann in coco_data["annotations"]:
                if not all(k in ann for k in ["id", "image_id", "category_id", "bbox"]):
                    print(f"✗ Invalid annotation entry: {ann}")
                    return False

            # Check categories
            if not coco_data["categories"]:
                print(f"✗ No categories found")
                return False

            for cat in coco_data["categories"]:
                if not all(k in cat for k in ["id", "name"]):
                    print(f"✗ Invalid category entry: {cat}")
                    return False

            print(f"✓ COCO format is valid")
            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
            print(f"  Categories: {len(coco_data['categories'])}")

            return True

        except Exception as e:
            print(f"✗ Validation error: {e}")
            return False


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for COCO conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert detection JSON to COCO 1.0 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert detection JSONs to COCO format
  python -m cvat_integration.coco_converter convert \\
    --input-dir output/text_detection_updated/video1/frames_and_json \\
    --output coco_annotations.json

  # Convert and split into train/val
  python -m cvat_integration.coco_converter convert \\
    --input-dir output/text_detection_updated/video1/frames_and_json \\
    --output coco_annotations.json \\
    --split

  # Split existing COCO file
  python -m cvat_integration.coco_converter split \\
    --coco-file coco_annotations.json \\
    --output-dir coco_splits \\
    --train-ratio 0.8

  # Validate COCO format
  python -m cvat_integration.coco_converter validate \\
    --coco-file coco_annotations.json
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert to COCO format")
    convert_parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing detection JSONs and images"
    )
    convert_parser.add_argument(
        "--output",
        required=True,
        help="Output COCO annotation file"
    )
    convert_parser.add_argument(
        "--category-name",
        default="text",
        help="Category name (default: text)"
    )
    convert_parser.add_argument(
        "--image-ext",
        default=".jpg",
        help="Image file extension (default: .jpg)"
    )
    convert_parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val after conversion"
    )
    convert_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for split (default: 0.8)"
    )

    # Split command
    split_parser = subparsers.add_parser("split", help="Split COCO dataset")
    split_parser.add_argument(
        "--coco-file",
        required=True,
        help="COCO annotation file to split"
    )
    split_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for splits"
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio (default: 0.8)"
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate COCO format")
    validate_parser.add_argument(
        "--coco-file",
        required=True,
        help="COCO annotation file to validate"
    )

    args = parser.parse_args()

    # Create converter
    converter = DetectionToCOCOConverter(
        category_name=getattr(args, 'category_name', 'text')
    )

    if args.command == "convert":
        # Convert to COCO
        coco_data = converter.convert_batch(
            input_dir=args.input_dir,
            output_file=args.output,
            image_extension=args.image_ext
        )

        # Validate
        converter.validate_coco_format(args.output)

        # Split if requested
        if args.split:
            output_dir = os.path.dirname(args.output)
            split_dir = os.path.join(output_dir, "splits")
            converter.split_train_val(
                coco_file=args.output,
                output_dir=split_dir,
                train_ratio=args.train_ratio
            )

    elif args.command == "split":
        converter.split_train_val(
            coco_file=args.coco_file,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed
        )

    elif args.command == "validate":
        converter.validate_coco_format(args.coco_file)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
