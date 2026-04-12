import cv2
import numpy as np
from typing import Tuple


def render_text(canvas: np.ndarray, text: str, pos: Tuple[int, int], font_scale: float, color: Tuple[int, int, int, int]) -> None:
    """Render text onto a BGRA canvas using OpenCV."""
    if canvas is None or text is None:
        return

    bgr_color = (int(color[0]), int(color[1]), int(color[2])) if len(color) >= 3 else (255, 255, 255)
    thickness = max(1, int(round(max(0.5, font_scale) * 1.2)))
    cv2.putText(canvas, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), bgr_color, thickness, cv2.LINE_AA)

    if canvas.shape[2] == 4:
        mask = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_BGR2GRAY)
        canvas[:, :, 3] = np.maximum(canvas[:, :, 3], (mask > 0).astype(np.uint8) * 255)
