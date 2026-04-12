"""
Coordinate Conversion
"""


import config
from typing import Tuple
from .calibration import CalibrationData

_calibration = None
_calibration_mode = False

def _get_calibration() -> CalibrationData:
    global _calibration
    if _calibration is None:
        _calibration = CalibrationData()
    return _calibration

def normalized_to_pixel(x: float, y: float, frame_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    """Convert MediaPipe coords"""
    return (int(x * frame_shape[1]), int(y * frame_shape[0]))

def hand_to_canvas(
    hand_x: int,
    hand_y: int,
    frame_shape: Tuple[int, int, int],
    canvas_size: Tuple[int, int],
    roi_scale: float = 0.6,
    sensitivity: float = 1.0,
    active_zone_size: int = 0,
) -> Tuple[int, int]:
    """Map hand coordinates from a central camera ROI to full canvas coordinates."""
    frame_h, frame_w = frame_shape[:2]
    canvas_w, canvas_h = canvas_size

    if active_zone_size and active_zone_size > 0:
        roi_w = max(1, min(frame_w, int(active_zone_size)))
        roi_h = max(1, min(frame_h, int(active_zone_size)))
    else:
        roi_w = max(1, int(frame_w * roi_scale))
        roi_h = max(1, int(frame_h * roi_scale))
    roi_x0 = (frame_w - roi_w) // 2
    roi_y0 = (frame_h - roi_h) // 2
    roi_x1 = roi_x0 + roi_w
    roi_y1 = roi_y0 + roi_h

    clamped_x = max(roi_x0, min(roi_x1, hand_x))
    clamped_y = max(roi_y0, min(roi_y1, hand_y))

    norm_x = (clamped_x - roi_x0) / max(1, (roi_x1 - roi_x0))
    norm_y = (clamped_y - roi_y0) / max(1, (roi_y1 - roi_y0))

    # Expand movement around center when sensitivity > 1.
    norm_x = 0.5 + (norm_x - 0.5) * sensitivity
    norm_y = 0.5 + (norm_y - 0.5) * sensitivity
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))

    canvas_x = int(norm_x * (canvas_w - 1))
    canvas_y = int(norm_y * (canvas_h - 1))
    calibration = _get_calibration()
    if calibration.is_calibrated:
        return calibration.map_to_canvas((hand_x, hand_y))
    return (canvas_x, canvas_y)

def set_calibration_mode(enabled: bool) -> None:
    global _calibration_mode
    _calibration_mode = bool(enabled)


def get_calibration_status() -> dict:
    calibration = _get_calibration()
    return {
        "is_calibrated": calibration.is_calibrated,
        "corners_collected": len(calibration.corners),
        "corners": [c.get("corner") for c in calibration.corners],
        "mode": _calibration_mode,
    }


def add_calibration_point(hand_pos: Tuple[int, int], corner: str, canvas_size: Tuple[int, int]) -> dict:
    calibration = _get_calibration()
    calibration.add_corner(hand_pos, corner, canvas_size)
    return get_calibration_status()


def reset_calibration() -> None:
    calibration = _get_calibration()
    calibration.reset()


def recalibrate_mapping(canvas_size: Tuple[int, int]) -> None:
    calibration = _get_calibration()
    if len(calibration.corners) >= 4:
        calibration._calculate_mapping(canvas_size)


def apply_deadzone(hand_pos: Tuple[int, int]) -> Tuple[int, int]:
    """Ignore small movements"""
    # Reserved for future coordinate mapping logic
    return hand_pos

