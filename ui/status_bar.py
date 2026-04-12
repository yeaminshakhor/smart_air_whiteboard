"""
Status bar UI drawing utilities for the smart air whiteboard.
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
from typing import Tuple

def draw_status_bar(frame: np.ndarray, text: str, position: Tuple[int, int]):
    """
    Draw a status bar with the given text at the given position on the frame.
    Args:
        frame (np.ndarray): The image frame to draw on.
        text (str): The text to display.
        position (Tuple[int, int]): The (x, y) position for the text.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
