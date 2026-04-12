from typing import Tuple, List, Optional, Any, Dict
import cv2
import numpy as np
from core.canvas_engine import CanvasEngine
from ui import UIManager
from managers.state_manager import StateManager
from utils.coordinate_mapper import hand_to_canvas
import math

def smooth_points(points, factor):
    if len(points) < 2:
        return points
    smoothed = [points[0]]
    for p in points[1:]:
        sx = smoothed[-1][0] * factor + p[0] * (1 - factor)
        sy = smoothed[-1][1] * factor + p[1] * (1 - factor)
        smoothed.append((sx, sy))
    return smoothed

class GestureController:
    def __init__(self, canvas: CanvasEngine, ui: UIManager, state: StateManager, stroke_processor, handwriting_recognizer, shape_detector, emoji_manager, config):
        self.canvas = canvas
        self.ui = ui
        self.state = state
        self.stroke_processor = stroke_processor
        self.handwriting_recognizer = handwriting_recognizer
        self.shape_detector = shape_detector
        self.emoji_manager = emoji_manager
        self.config = config
        self.pointer_path = []
        self.last_index_ms = 0
        self.open_hand_start_ms = 0
        self.draw_grace_ms = config.get('gesture', 'DRAW_GRACE_MS', 180)
        self.open_hand_hold_ms = config.get('gesture', 'OPEN_HAND_HOLD_MS', 220)
        self.ui_select_cooldown_ms = config.get('gesture', 'UI_SELECT_COOLDOWN_MS', 450)
        self.stroke_points = []

    def update_pointer(self, landmarks, frame_shape, canvas_size, roi_scale, sensitivity, active_zone_size):
        if not landmarks:
            self.pointer_path = []
            return None
        raw_index_pos = landmarks[8]
        index_pos = hand_to_canvas(
            raw_index_pos[0],
            raw_index_pos[1],
            frame_shape,
            canvas_size,
            roi_scale=roi_scale,
            sensitivity=sensitivity,
            active_zone_size=active_zone_size,
        )
        self.pointer_path.append(index_pos)
        self.pointer_path = self.pointer_path[-6:]
        smoothed = smooth_points(self.pointer_path, factor=0.7)
        return smoothed[-1] if smoothed else index_pos

    def handle_gesture(self, gesture: str, pos: Tuple[int, int], landmarks: List[Tuple[int, int]], frame_shape, canvas_size):
        mode = self.state.get("mode")
        prev_pos = self.state.get("prev_pos")
        self.state.set("last_position", pos)

        effective_gesture = self._effective_gesture(gesture, landmarks)
        if effective_gesture == "index_finger":
            ui_action = self.ui.handle_click(pos, select_enabled=True)
            self._apply_ui_action(ui_action, pos)
            self.state.set("prev_pos", pos)
            return

        # Other gestures
        # TODO: thumbs nav, fist grab etc.

    def _effective_gesture(self, gesture, landmarks):
        # Port from main.py _effective_gesture
        # ... (implement)
        return gesture

    def _apply_ui_action(self, ui_action, cursor_pos):
        # Port from main.py _apply_ui_action
        if ui_action is None:
            return
        # ... (mode, color, size, opacity, paste, image)
        pass

    # _recognize_and_draw_char, _draw_detected_shape etc. move here too

