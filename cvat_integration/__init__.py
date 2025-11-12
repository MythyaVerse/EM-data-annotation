"""
CVAT Integration Package

This package provides tools to integrate text detection pipeline with CVAT
for human-in-the-loop annotation correction and improvement.

Modules:
    - format_converter: Convert detection JSON to CVAT format
    - cvat_uploader: Upload tasks and annotations to CVAT
    - cvat_exporter: Export corrected annotations from CVAT
    - coco_converter: Convert detection JSON to COCO format
    - workflow: End-to-end orchestration
"""

__version__ = "1.0.0"
__author__ = "Text Detection Pipeline"

from . import format_converter
from . import cvat_uploader
from . import cvat_exporter
from . import coco_converter
from . import workflow

__all__ = [
    'format_converter',
    'cvat_uploader',
    'cvat_exporter',
    'coco_converter',
    'workflow'
]
