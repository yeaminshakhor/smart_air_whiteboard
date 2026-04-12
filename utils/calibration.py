import json
import os
from typing import Tuple, Optional, Any, List
import numpy as np


# NOTE: Ensure that recalibrate_mapping (i.e., _calculate_mapping) is called whenever the canvas size changes in the main application logic.


import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any




import cv2
import numpy as np


_log = logging.getLogger(__name__)

class CalibrationData:
    """Stores and applies hand-to-canvas calibration."""

    def __init__(self, save_path: str = "data/calibration.json") -> None:
        self.save_path = save_path
        self.corners = []
        self.is_calibrated = False
        self.mapping_matrix: Optional[np.ndarray] = None
        self.load()

    def add_corner(self, hand_pos: Tuple[int, int], corner_name: str, canvas_size: Tuple[int, int]) -> None:
        self.corners = [c for c in self.corners if c.get("corner") != corner_name]
        self.corners.append(
            {
                "hand": [int(hand_pos[0]), int(hand_pos[1])],
                "corner": corner_name,
                "timestamp": time.time(),
            }
        )

        expected = ["top-left", "top-right", "bottom-left", "bottom-right"]
        if all(any(c.get("corner") == name for c in self.corners) for name in expected):
            self._calculate_mapping(canvas_size)
            self.save()

    def _calculate_mapping(self, canvas_size: Tuple[int, int]) -> None:
        corner_lookup = {c["corner"]: c["hand"] for c in self.corners}
        hand_points = np.float32(
            [
                corner_lookup["top-left"],
                corner_lookup["top-right"],
                corner_lookup["bottom-left"],
                corner_lookup["bottom-right"],
            ]
        )

        width, height = canvas_size
        screen_corners = np.float32(
            [
                [0, 0],
                [width - 1, 0],
                [0, height - 1],
                [width - 1, height - 1],
            ]
        )

        self.mapping_matrix = cv2.getPerspectiveTransform(hand_points, screen_corners)
        self.is_calibrated = True

    def map_to_canvas(self, hand_pos: Tuple[int, int]) -> Tuple[int, int]:
        if not self.is_calibrated or self.mapping_matrix is None:
            return hand_pos

        hand_point = np.array([[[float(hand_pos[0]), float(hand_pos[1])]]], dtype=np.float32)
        canvas_point = cv2.perspectiveTransform(hand_point, self.mapping_matrix)
        return int(canvas_point[0][0][0]), int(canvas_point[0][0][1])

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        payload = {
            "corners": self.corners,
            "is_calibrated": self.is_calibrated,
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self) -> None:
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("Could not load calibration data %s: %s", self.save_path, exc)
            return
        self.corners = list(data.get("corners", []))
        self.is_calibrated = bool(data.get("is_calibrated", False))
        if self.is_calibrated and len(self.corners) >= 4:
            # Default to current app canvas size; caller can recalculate later if needed.
            self._calculate_mapping((1280, 720))

    def reset(self) -> None:
        self.corners = []
        self.is_calibrated = False
        self.mapping_matrix = None
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
