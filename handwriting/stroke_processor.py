"""
Stroke Collection
"""

import os
import sys

try:
    import config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
import cv2
import numpy as np
import time
import math
from typing import List, Tuple

from utils.geometry import interpolate_stroke

class StrokeProcessor:
    def __init__(self, completion_timeout_ms: int = 800):
        self.current_stroke: List[Tuple[int, int]] = []
        self.completion_timeout_ms = completion_timeout_ms
        self.last_point_ms = 0
        self.index_active = False
        import logging; logging.getLogger(__name__).info("Initializing StrokeProcessor")

    def add_point(self, x: int, y: int):
        """Add to current stroke"""
        self.current_stroke.append((x, y))
        self.last_point_ms = int(time.time() * 1000)

    def set_index_activity(self, is_active: bool):
        """Track when index-finger writing gesture is active."""
        self.index_active = is_active
        if is_active:
            self.last_point_ms = int(time.time() * 1000)

    def is_stroke_complete(self) -> bool:
        """Stroke completes when there are points and no index activity for timeout."""
        if not self.current_stroke:
            return False
        if self.index_active:
            return False

        now_ms = int(time.time() * 1000)
        return (now_ms - self.last_point_ms) > self.completion_timeout_ms

    def get_current_stroke(self) -> List[Tuple[int, int]]:
        """Return buffered points"""
        return self.current_stroke

    def clear_current_stroke(self):
        """Reset buffer"""
        self.current_stroke = []
        self.index_active = False
        self.last_point_ms = 0

    def preprocess_for_recognition(self, stroke: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Normalize and resample"""
        if not stroke:
            return []
        if len(stroke) == 1:
            return stroke

        xs = [p[0] for p in stroke]
        ys = [p[1] for p in stroke]
        cx = float(sum(xs)) / len(xs)
        cy = float(sum(ys)) / len(ys)

        centered = [(float(x) - cx, float(y) - cy) for x, y in stroke]
        max_dist = max(math.hypot(x, y) for x, y in centered)
        if max_dist <= 0.0:
            normalized = [(0.0, 0.0) for _ in centered]
        else:
            normalized = [(x / max_dist, y / max_dist) for x, y in centered]

        quantized = [(int(x * 1000), int(y * 1000)) for x, y in normalized]
        # Use interpolate_stroke to densify the stroke for recognition
        return interpolate_stroke(quantized, min_dist=32.0/32)  # ~1 pixel per point for 32 points

    def extract_features(self, stroke: List[Tuple[int, int]]) -> List:
        """For ML input"""
        processed = self.preprocess_for_recognition(stroke)
        if len(processed) < 3:
            return []

        points = [(p[0] / 1000.0, p[1] / 1000.0) for p in processed]
        directions = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            directions.append(math.atan2(dy, dx))

        hist, _ = np.histogram(directions, bins=8, range=(-math.pi, math.pi), density=False)
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()

        curvatures = []
        for i in range(1, len(directions)):
            delta = directions[i] - directions[i - 1]
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            curvatures.append(abs(delta))
        avg_curvature = float(np.mean(curvatures)) if curvatures else 0.0

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        aspect = float(width) / float(max(1e-6, height))

        center_x = float(sum(xs)) / len(xs)
        center_y = float(sum(ys)) / len(ys)
        start_dir = math.atan2(points[0][1] - center_y, points[0][0] - center_x)
        end_dir = math.atan2(points[-1][1] - center_y, points[-1][0] - center_x)

        return [aspect, avg_curvature, math.cos(start_dir), math.sin(start_dir), math.cos(end_dir), math.sin(end_dir)] + hist.tolist()

    def stroke_to_image(self, stroke: List[Tuple[int, int]], size: int = 28, padding: int = 4) -> np.ndarray:
        """Render stroke points into a normalized grayscale image for recognition."""
        if not stroke:
            return np.zeros((size, size), dtype=np.uint8)

        xs = [p[0] for p in stroke]
        ys = [p[1] for p in stroke]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        draw_side = max(width, height)
        draw_side = max(1, draw_side)

        temp_side = draw_side + 2 * padding
        temp = np.zeros((temp_side, temp_side), dtype=np.uint8)

        x_offset = (temp_side - width) // 2
        y_offset = (temp_side - height) // 2

        shifted = [
            (int(p[0] - min_x + x_offset), int(p[1] - min_y + y_offset))
            for p in stroke
        ]

        for i in range(1, len(shifted)):
            cv2.line(temp, shifted[i - 1], shifted[i], 255, 2)

        return cv2.resize(temp, (size, size), interpolation=cv2.INTER_AREA)

