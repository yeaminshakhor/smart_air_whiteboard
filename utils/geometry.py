import math
from typing import List, Tuple

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def interpolate_points(p1: Tuple[int, int], p2: Tuple[int, int], num_points: int) -> List[Tuple[int, int]]:
    """Linearly interpolate between two points, returning a list of points including endpoints."""
    if num_points < 2:
        return [p1, p2]
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = int(round(lerp(p1[0], p2[0], t)))
        y = int(round(lerp(p1[1], p2[1], t)))
        points.append((x, y))
    return points

def interpolate_stroke(stroke: List[Tuple[int, int]], min_dist: float = 2.0) -> List[Tuple[int, int]]:
    """Interpolates a stroke so that consecutive points are no more than min_dist apart."""
    if not stroke:
        return []
    result = [stroke[0]]
    for i in range(1, len(stroke)):
        prev = result[-1]
        curr = stroke[i]
        dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        if dist > min_dist:
            num_points = int(math.ceil(dist / min_dist))
            interp = interpolate_points(prev, curr, num_points + 1)[1:]
            result.extend(interp)
        else:
            result.append(curr)
    return result
