"""
Segmentation evaluation utilities for CARLA
"""

import numpy as np


class SegmentationEvaluator:
    """Evaluator for semantic segmentation"""

    def __init__(self, num_classes=13):
        self.num_classes = num_classes

    def _convert_carla_seg_to_classes(self, seg_image):
        """
        Convert CARLA's segmentation to class indices.

        CARLA semantic segmentation can be either:
        1. Raw label IDs: [H, W] array with values 0-27
        2. RGB colors: [H, W, 3] array with specific colors per class

        Args:
            seg_image: numpy array [H, W] or [H, W, 3] with segmentation from CARLA

        Returns:
            class_image: numpy array [H, W] with class indices
        """
        # If already in label ID format (2D array), return as-is
        if seg_image.ndim == 2:
            return seg_image.astype(np.uint8)

        # Otherwise, convert from RGB colors to class IDs
        # CARLA semantic segmentation color mapping
        # Based on CARLA documentation: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
        color_map = {
            (0, 0, 0): 0,        # Unlabeled
            (70, 70, 70): 1,     # Building
            (100, 40, 40): 2,    # Fence
            (55, 90, 80): 3,     # Other
            (220, 20, 60): 4,    # Pedestrian
            (153, 153, 153): 5,  # Pole
            (157, 234, 50): 6,   # RoadLine
            (128, 64, 128): 7,   # Road
            (244, 35, 232): 8,   # Sidewalk
            (107, 142, 35): 9,   # Vegetation
            (0, 0, 142): 10,     # Vehicles
            (102, 102, 156): 11, # Wall
            (220, 220, 0): 12    # TrafficSign
        }

        class_image = np.zeros((seg_image.shape[0], seg_image.shape[1]), dtype=np.uint8)

        for color, class_id in color_map.items():
            # Find all pixels matching this color
            mask = np.all(seg_image == color, axis=-1)
            class_image[mask] = class_id

        return class_image
