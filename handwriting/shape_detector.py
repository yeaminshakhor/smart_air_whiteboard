import math
from typing import List, Optional, Tuple


class ShapeDetector:
    """Detects simple geometric shapes from stroke points."""

    def __init__(self):
        self.min_points = 12

    def detect(self, points: List[Tuple[int, int]]) -> Optional[str]:
        if not points or len(points) < self.min_points:
            return None

        normalized = self._normalize(points)

        if self._is_line(normalized):
            return "line"

        if self._is_closed(normalized):
            if self._is_circle(normalized):
                return "circle"
            if self._is_rectangle(normalized):
                return "rectangle"

        return None

    def _normalize(self, points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max(1.0, float(max_x - min_x))
        height = max(1.0, float(max_y - min_y))
        scale = max(width, height)

        cx = (max_x + min_x) / 2.0
        cy = (max_y + min_y) / 2.0

        return [((x - cx) / scale, (y - cy) / scale) for x, y in points]

    def _path_length(self, points: List[Tuple[float, float]]) -> float:
        total = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            total += math.hypot(dx, dy)
        return total

    def _is_closed(self, points: List[Tuple[float, float]], threshold: float = 0.15) -> bool:
        start = points[0]
        end = points[-1]
        gap = math.hypot(end[0] - start[0], end[1] - start[1])
        return gap <= threshold

    def _is_line(self, points: List[Tuple[float, float]]) -> bool:
        start = points[0]
        end = points[-1]
        chord = math.hypot(end[0] - start[0], end[1] - start[1])
        if chord == 0.0:
            return False

        length = self._path_length(points)
        return (length / chord) <= 1.12

    def _is_circle(self, points: List[Tuple[float, float]]) -> bool:
        if not self._is_closed(points, threshold=0.18):
            return False

        radii = [math.hypot(x, y) for x, y in points]
        avg_radius = sum(radii) / len(radii)
        if avg_radius <= 0.05:
            return False

        variance = sum((r - avg_radius) ** 2 for r in radii) / len(radii)
        return variance <= 0.012

    def _is_rectangle(self, points: List[Tuple[float, float]]) -> bool:
        if not self._is_closed(points, threshold=0.2):
            return False

        corners = self._count_corners(points)
        if corners < 3 or corners > 6:
            return False

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width <= 0.1 or height <= 0.1:
            return False

        perimeter = self._path_length(points)
        bbox_perimeter = 2.0 * (width + height)
        return abs(perimeter - bbox_perimeter) <= 0.55

    def _count_corners(self, points: List[Tuple[float, float]]) -> int:
        if len(points) < 5:
            return 0

        count = 0
        for i in range(2, len(points) - 2):
            a = points[i - 2]
            b = points[i]
            c = points[i + 2]
            angle = self._angle(a, b, c)
            if 55.0 <= angle <= 135.0:
                count += 1

        return count

    def _angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])

        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 == 0.0 or n2 == 0.0:
            return 0.0

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
        return math.degrees(math.acos(cos_theta))
