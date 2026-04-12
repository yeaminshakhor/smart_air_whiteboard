import os

import config

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import math
import subprocess
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.camera_thread import CameraThread
from core.canvas_engine import CanvasEngine
from core.config_manager import ConfigManager
from core.gesture_recognizer import GestureRecognizer
from core.hand_tracker import HandTracker
from handwriting.character_recognizer import CharacterRecognizer
from handwriting.shape_detector import ShapeDetector
from handwriting.stroke_processor import StrokeProcessor
from managers.clipboard_manager import ClipboardManager
from managers.emoji_manager import EmojiManager
from managers.page_manager import PageManager
from managers.state_manager import StateManager
from ui.ui_panels import UIManager
from utils.coordinate_mapper import (
    add_calibration_point,
    get_calibration_status,
    hand_to_canvas,
    recalibrate_mapping,
    reset_calibration,
    set_calibration_mode,
)
from utils.geometry import interpolate_stroke


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def smooth_points(points, factor):
    if len(points) < 2:
        return points
    smoothed = [points[0]]
    for p in points[1:]:
        sx = smoothed[-1][0] * factor + p[0] * (1 - factor)
        sy = smoothed[-1][1] * factor + p[1] * (1 - factor)
        smoothed.append((sx, sy))
    return smoothed


def get_mode(self):
    return self.state.get("mode", "drawing")


def get_brush_size(self):
    return self.state.get("brush_size", 8)


def get_current_color(self):
    color = self.state.get("color", (255, 255, 255))
    opacity = self.state.get("brush_opacity", 1.0)
    alpha = int(np.clip(opacity, 0.1, 1.0) * 255)
    return (int(color[0]), int(color[1]), int(color[2]), alpha)


def get_draw_grace_ms(self):
    return self.config.get("gesture", "DRAW_GRACE_MS", 100)


class GestureWhiteboard:
    def __init__(self, width=1280, height=720):
        self.config = ConfigManager()

        self.width = self.config.get("canvas", "WIDTH", width)
        self.height = self.config.get("canvas", "HEIGHT", height)
        cam_w = self.config.get("camera", "CAMERA_WIDTH", width)
        cam_h = self.config.get("camera", "CAMERA_HEIGHT", height)

        self.prev_pos = None

        self.hand_tracker = HandTracker(
            model_asset_path="data/models/hand_landmarker.task",
            max_hands=self.config.get("hand_tracking", "MAX_HANDS", 1),
            detection_confidence=self.config.get("hand_tracking", "DETECTION_CONFIDENCE", 0.8),
        )
        self.camera_thread = CameraThread(0, cam_w, cam_h, self.hand_tracker)
        self.camera_thread.start()

        self.gesture_recognizer = GestureRecognizer(
            debounce_frames=2,
            smoothing_window=3,
            scroll_threshold_px=30,
            two_finger_distance_ratio=0.08,
            min_frame_confidence=0.55,
            smoothing_min_confidence=0.5,
        )

        self.handwriting_recognizer = CharacterRecognizer()
        self.shape_detector = ShapeDetector()
        stroke_complete_ms = self.config.get("recognition", "STROKE_COMPLETE_MS", 800)
        self.stroke_processor = StrokeProcessor(completion_timeout_ms=stroke_complete_ms)

        self.roi_scale = self.config.get("hand_tracking", "ROI_SCALE", 1.0)
        self.pointer_sensitivity = self.config.get("hand_tracking", "SENSITIVITY", 1.0)
        self.active_zone_size = self.config.get("hand_tracking", "ACTIVE_ZONE_SIZE", 0)

        self.canvas = CanvasEngine(self.width, self.height)
        self.pages = PageManager(self.width, self.height)
        self.clipboard = ClipboardManager()
        self.emoji_manager = EmojiManager("data/assets/emojis.json")
        self.state = StateManager()
        self.ui = UIManager(self.width, self.height, self.clipboard)

        self.window_name = "Gesture Whiteboard"
        self.window_fullscreen = True
        self._bg_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stroke_points = []

        self.page_nav_cooldown_ms = self.config.get("gesture", "PAGE_NAV_COOLDOWN_MS", 900)
        self.nav_stationary_px = self.config.get("gesture", "NAV_STATIONARY_PX", 18)
        self.copy_cooldown_ms = self.config.get("gesture", "COPY_COOLDOWN_MS", 700)
        self.thumb_hold_ms = self.config.get("gesture", "THUMB_HOLD_MS", 500)
        self.enable_thumb_nav = True
        self.fist_hold_ms = self.config.get("gesture", "FIST_HOLD_MS", 250)
        self.thumb_hold_gesture = None
        self.thumb_hold_start_ms = 0
        self.fist_hold_start_ms = 0
        self.fist_hold_pos = None
        self.text_buffer = ""
        self.running = True
        self.open_hand_hold_ms = self.config.get("gesture", "OPEN_HAND_HOLD_MS", 220)
        self.last_index_ms = 0
        self.open_hand_start_ms = 0
        self.pointer_path = []
        self.latest_hand_pixel = None
        self.calibration_mode = False
        self.calibration_step = 0
        self.calibration_corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
        self.intent_hold_start_ms: Dict[str, int] = {}

        self.performance_state = {
            "no_hand_frames": 0,
            "skip_counter": 0,
            "hand_moving": False,
            "last_hand_pos": None,
            "motion_history": deque(maxlen=self.config.get("performance", "MOTION_HISTORY_FRAMES", 5)),
        }
        self.no_hand_skip_max = self.config.get("performance", "NO_HAND_SKIP_MAX", 5)
        self.static_hand_skip = max(1, self.config.get("performance", "STATIC_HAND_SKIP", 2))
        self.motion_threshold = self.config.get("performance", "MOTION_THRESHOLD", 15)
        self.motion_required_frames = max(1, self.config.get("performance", "MOTION_REQUIRED_FRAMES", 3))

        self.intent_thresholds_ms = {
            "erase": self.config.get("gesture", "ERASE_HOLD_MS", 200),
            "copy": self.config.get("gesture", "COPY_HOLD_MS", 300),
            "navigate": self.config.get("gesture", "NAVIGATE_HOLD_MS", 400),
            "scroll": self.config.get("gesture", "SCROLL_HOLD_MS", 180),
            "ui_select": self.config.get("gesture", "UI_SELECT_HOLD_MS", 120),
        }
        self.ui_select_cooldown_ms = self.config.get("gesture", "UI_SELECT_COOLDOWN_MS", 450)

        self.debug_overlay = self.state.get("debug_overlay", True)
        self.debug_state = {
            "raw_gesture": "unknown",
            "effective_gesture": "unknown",
            "confidence": 0.0,
            "mode": self.state.get("mode", "drawing"),
            "brush_size": 0,
            "hand_detected": False,
            "processed": False,
            "handedness": None,
        }

        recalibrate_mapping((self.width, self.height))

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        if self.window_fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            self._main_loop()
        finally:
            self.cleanup()

    def _main_loop(self) -> None:
        while self.running:
            frame, landmarks, handedness = self.camera_thread.get_state()
            if frame is None:
                cv2.waitKey(1)
                continue

            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            self.ui.update()
            hand_detected = landmarks is not None
            hand_pos = landmarks[8] if hand_detected else None

            should_process = self._should_process_frame(hand_detected, hand_pos)

            self.debug_state["hand_detected"] = hand_detected
            self.debug_state["processed"] = bool(hand_detected and should_process)
            self.debug_state["handedness"] = handedness

            if landmarks:
                self.latest_hand_pixel = landmarks[8]

            if landmarks and should_process:
                gesture, confidence = self.gesture_recognizer.recognize(landmarks, handedness)
                self.debug_state["raw_gesture"] = gesture
                self.debug_state["confidence"] = float(confidence)

                raw_index_pos = landmarks[8]
                index_pos = hand_to_canvas(
                    raw_index_pos[0],
                    raw_index_pos[1],
                    frame.shape,
                    (self.width, self.height),
                    roi_scale=self.roi_scale,
                    sensitivity=self.pointer_sensitivity,
                    active_zone_size=self.active_zone_size,
                )
                self.pointer_path.append(index_pos)
                self.pointer_path = self.pointer_path[-6:]
                smoothed = smooth_points(self.pointer_path, factor=0.7)
                if smoothed:
                    index_pos = smoothed[-1]

                recognizer_debug = self.gesture_recognizer.get_debug_snapshot()
                effective_gesture = self._effective_gesture(gesture, landmarks, recognizer_debug)
                self.debug_state["effective_gesture"] = effective_gesture
                self.debug_state.update(recognizer_debug)

                brush_size = get_brush_size(self) if effective_gesture in ("index_finger", "open_hand") else 0
                self.debug_state["brush_size"] = int(brush_size)

                self.stroke_processor.set_index_activity(get_mode(self) == "writing" and effective_gesture == "index_finger")
                self._handle_gesture(effective_gesture, index_pos, landmarks)

                if get_mode(self) == "writing" and self.stroke_processor.is_stroke_complete():
                    self._recognize_and_draw_char()

                self.prev_pos = index_pos

            elif not landmarks:
                if self.prev_pos is not None and get_mode(self) in ("drawing", "erasing"):
                    self.canvas.record_history()
                self.state.set("prev_pos", None)
                self.prev_pos = None
                self.pointer_path = []
                self.state.set("last_position", None)
                self.stroke_processor.set_index_activity(False)
                if get_mode(self) == "writing" and self.stroke_processor.is_stroke_complete():
                    self._recognize_and_draw_char()
                    self.canvas.record_history()
                elif self.state.get("grabbing"):
                    self._drop_grab((0, 0), dropped_without_target=True)
                self.debug_state.update({
                    "raw_gesture": "unknown",
                    "effective_gesture": "unknown",
                    "confidence": 0.0,
                    "brush_size": 0,
                })

            self.pages.update_current_page(self.canvas.get_canvas_with_items())
            self.debug_state["mode"] = get_mode(self)
            display_frame = self._render_display(frame)
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            self._handle_keypress(key)

    def _handle_keypress(self, key: int) -> None:
        if key == ord("q"):
            self.pages.save_all_pages()
            self.running = False
        elif key == ord("c"):
            self._confirm_clear()
        elif key == ord("u"):
            self.canvas.undo()
        elif key == ord("s"):
            self.pages.save_page(self.pages.current_index)
        elif key == ord("n"):
            self.pages.next_page()
            self.canvas.load_page(self.pages.get_current_page())
        elif key == ord("p"):
            self.pages.prev_page()
            self.canvas.load_page(self.pages.get_current_page())
        elif key == ord("x"):
            self.ui.show_clipboard()
        elif key == ord("m"):
            emoji = self.emoji_manager.render_keyword("SMILE", size=96)
            self.canvas.add_item({"type": "image", "content": emoji, "pos": (self.width // 2, self.height // 2)})
        elif key == ord("k"):
            self.calibration_mode = not self.calibration_mode
            set_calibration_mode(self.calibration_mode)
            if self.calibration_mode:
                reset_calibration()
                self.calibration_step = 0
        elif key == ord("g"):
            self.debug_overlay = not self.debug_overlay
        elif key == ord("f"):
            self.window_fullscreen = not self.window_fullscreen
            mode = cv2.WINDOW_FULLSCREEN if self.window_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, mode)
            if not self.window_fullscreen:
                cv2.resizeWindow(self.window_name, self.width, self.height)
        elif key == 13 and self.calibration_mode:
            if self.latest_hand_pixel is not None and self.calibration_step < len(self.calibration_corners):
                add_calibration_point(
                    self.latest_hand_pixel,
                    self.calibration_corners[self.calibration_step],
                    (self.width, self.height),
                )
                self.calibration_step += 1
                if self.calibration_step >= len(self.calibration_corners):
                    self.calibration_mode = False
                    set_calibration_mode(False)

        if key != -1 and key != ord("c") and getattr(self, "pending_clear", False):
            self.pending_clear = False

    def _is_hand_moving(self, current_pos: Optional[Tuple[int, int]], threshold: int = 15) -> bool:
        if current_pos is None:
            return False
        last_pos = self.performance_state["last_hand_pos"]
        if last_pos is None:
            self.performance_state["last_hand_pos"] = current_pos
            self.performance_state["motion_history"].append(False)
            return False
        delta = math.hypot(current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])
        moving = delta > threshold
        self.performance_state["last_hand_pos"] = current_pos
        self.performance_state["motion_history"].append(moving)
        history = list(self.performance_state["motion_history"])
        required = min(self.motion_required_frames, len(history))
        return sum(1 for m in history if m) >= required

    def _should_process_frame(self, hand_detected: bool, hand_pos: Optional[Tuple[int, int]] = None) -> bool:
        if not hand_detected:
            self.performance_state["no_hand_frames"] += 1
            skip_rate = min(self.no_hand_skip_max, self.performance_state["no_hand_frames"] // 10)
            self.performance_state["skip_counter"] = (self.performance_state["skip_counter"] + 1) % (skip_rate + 1)
            return self.performance_state["skip_counter"] == 0

        self.performance_state["no_hand_frames"] = 0
        return True

    def _set_mode(self, mode: str) -> None:
        if mode == get_mode(self):
            return
        if self.state.get("prev_pos") is not None and get_mode(self) in {"drawing", "erasing"}:
            self.canvas.record_history()
        self.state.set("mode", mode)

    def _handle_gesture(
        self,
        gesture: str,
        pos: Tuple[int, int],
        landmarks: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        if gesture == "index_finger":
            if self.ui.is_point_over_ui(pos[0], pos[1]):
                ui_action = self.ui.handle_click(pos, select_enabled=True)
                self._apply_ui_action(ui_action, pos)
                return

            self._set_mode("drawing")
            prev_pos = self.state.get("prev_pos")
            brush_size = get_brush_size(self)

            if prev_pos is not None:
                points = interpolate_stroke([prev_pos, pos], min_dist=2.0)
                for i in range(1, len(points)):
                    self.canvas.draw_line(points[i - 1], points[i], self._current_draw_color(), max(2, int(brush_size)))
            self.state.set("prev_pos", pos)
            return

        if gesture == "open_hand":
            self._set_mode("erasing")
            prev_pos = self.state.get("prev_pos")
            brush_size = get_brush_size(self)

            erase_radius = max(6, int(brush_size * 1.3))
            if prev_pos is not None:
                points = interpolate_stroke([prev_pos, pos], min_dist=max(1.0, brush_size * 0.5))
                for i in range(1, len(points)):
                    self.canvas.erase_stroke(points[i - 1], points[i], radius=erase_radius)
            else:
                self.canvas.erase(pos, radius=erase_radius)
            self.state.set("prev_pos", pos)
            return

        if gesture == "four_fingers":
            self._set_mode("writing")
            self.stroke_points.append(pos)
            self.stroke_points = self.stroke_points[-512:]
            self.stroke_processor.add_point(pos[0], pos[1])
            self.state.set("prev_pos", pos)
            return

        if gesture in ("thumbs_up", "thumbs_down") and self.enable_thumb_nav:
            if self._can_navigate_page(gesture, pos):
                if gesture == "thumbs_up":
                    self.pages.prev_page()
                else:
                    self.pages.next_page()
                self.canvas.load_page(self.pages.get_current_page())
            return

        if gesture == "fist":
            if self.state.get("grabbing"):
                self._move_active_grab(pos)
            elif self._can_start_grab(pos):
                self._start_grab(pos)
            return

        if get_mode(self) not in ("drawing", "erasing", "writing"):
            self.state.set("prev_pos", None)

    def _apply_ui_action(self, ui_action: Any, cursor_pos: Tuple[int, int]) -> None:
        if ui_action is None:
            return

        if not isinstance(ui_action, tuple):
            if isinstance(ui_action, dict) and ui_action.get("type") == "image":
                content = ui_action.get("content")
                if content is not None:
                    h, w = content.shape[:2]
                    self.canvas.add_item(
                        {
                            "type": "image",
                            "content": content.copy(),
                            "pos": (cursor_pos[0] - w // 2, cursor_pos[1] - h // 2),
                        }
                    )
                    self._set_mode("drawing")
            return

        tag = ui_action[0]

        if tag == "mode":
            self._set_mode(str(ui_action[1]))

        elif tag == "color":
            self.state.set("color", ui_action[1])

        elif tag == "size":
            current = self.state.get("brush_size", 8)
            self.state.set("brush_size", int(np.clip(current + int(ui_action[1]), 2, 48)))

        elif tag == "opacity":
            current = self.state.get("brush_opacity", 1.0)
            self.state.set("brush_opacity", float(np.clip(current + float(ui_action[1]), 0.1, 1.0)))

        elif tag == "action" and ui_action[1] == "paste":
            self._paste_image_from_system_clipboard(cursor_pos)

    def _current_draw_color(self) -> Tuple[int, int, int, int]:
        return get_current_color(self)

    def _effective_gesture(
        self,
        gesture: str,
        landmarks: Optional[List[Tuple[int, int]]] = None,
        debug_snapshot: Optional[dict] = None,
    ) -> str:
        now_ms = int(time.time() * 1000)

        if gesture == "index_finger":
            self.last_index_ms = now_ms

        if gesture == "unknown" and (self._is_index_pose(landmarks) or self._is_index_curl_pose(debug_snapshot)):
            self.last_index_ms = now_ms
            gesture = "index_finger"

        if gesture == "open_hand":
            if self.open_hand_start_ms == 0:
                self.open_hand_start_ms = now_ms
                return "unknown"
            if now_ms - self.open_hand_start_ms < self.open_hand_hold_ms:
                return "unknown"
        else:
            self.open_hand_start_ms = 0

        if (
            gesture == "unknown"
            and get_mode(self) in ("drawing", "erasing", "writing")
            and self.state.get("prev_pos") is not None
            and now_ms - self.last_index_ms <= get_draw_grace_ms(self)
        ):
            return "index_finger"

        return gesture

    def _is_index_curl_pose(self, debug_snapshot: Optional[dict]) -> bool:
        if not debug_snapshot:
            return False
        index_curl = float(debug_snapshot.get("index_curl", 100.0))
        middle_curl = float(debug_snapshot.get("middle_curl", 0.0))
        ring_curl = float(debug_snapshot.get("ring_curl", 0.0))
        pinky_curl = float(debug_snapshot.get("pinky_curl", 0.0))
        finger_count = float(debug_snapshot.get("finger_count", 0.0))
        return (
            index_curl <= 50.0
            and middle_curl >= 55.0
            and ring_curl >= 55.0
            and pinky_curl >= 55.0
            and finger_count <= 2.0
        )

    def _is_index_pose(self, landmarks: Optional[List[Tuple[int, int]]]) -> bool:
        if not landmarks or len(landmarks) < 21:
            return False
        idx_tip_y, idx_pip_y = landmarks[8][1], landmarks[6][1]
        mid_tip_y, mid_pip_y = landmarks[12][1], landmarks[10][1]
        ring_tip_y, ring_pip_y = landmarks[16][1], landmarks[14][1]
        pink_tip_y, pink_pip_y = landmarks[20][1], landmarks[18][1]
        return (
            idx_tip_y < idx_pip_y
            and mid_tip_y >= (mid_pip_y - 5)
            and ring_tip_y >= (ring_pip_y - 5)
            and pink_tip_y >= (pink_pip_y - 5)
        )

    def _has_intent_to_act(self, action, expected_gesture, current_gesture, hold_key=None) -> bool:
        now_ms = int(time.time() * 1000)
        hold_id = hold_key or action
        if current_gesture != expected_gesture:
            self.intent_hold_start_ms.pop(hold_id, None)
            return False
        threshold = int(self.intent_thresholds_ms.get(action, 0))
        start_ms = self.intent_hold_start_ms.get(hold_id)
        if start_ms is None:
            self.intent_hold_start_ms[hold_id] = now_ms
            return threshold == 0
        return (now_ms - int(start_ms)) >= threshold

    def _can_navigate_page(self, gesture: str, pos: Tuple[int, int]) -> bool:
        now_ms = int(time.time() * 1000)
        if self.thumb_hold_gesture != gesture:
            self.thumb_hold_gesture = gesture
            self.thumb_hold_start_ms = now_ms
            return False
        if now_ms - self.thumb_hold_start_ms < self.thumb_hold_ms:
            return False
        last_nav_ms = self.state.get("last_nav_ms") or 0
        if now_ms - last_nav_ms < self.page_nav_cooldown_ms:
            return False
        if self.prev_pos is not None and distance(self.prev_pos, pos) > self.nav_stationary_px:
            return False
        self.state.set("last_nav_ms", now_ms)
        return True

    def _can_start_grab(self, pos: Tuple[int, int]) -> bool:
        now_ms = int(time.time() * 1000)
        if self.fist_hold_start_ms == 0 or self.fist_hold_pos is None:
            self.fist_hold_start_ms = now_ms
            self.fist_hold_pos = pos
            return False
        if distance(self.fist_hold_pos, pos) > self.nav_stationary_px:
            self.fist_hold_start_ms = now_ms
            self.fist_hold_pos = pos
            return False
        return (now_ms - self.fist_hold_start_ms) >= self.fist_hold_ms

    def _start_grab(self, pos: Tuple[int, int]) -> bool:
        item = self.canvas.get_item_at(pos)
        if item is None:
            return False
        item_x, item_y = item.get("pos", pos)
        item["is_grabbed"] = True
        self.state.set("grabbing", True)
        self.state.set("active_grab", item)
        self.state.set("grab_offset", (pos[0] - item_x, pos[1] - item_y))
        self.state.set("grab_path", [pos])
        self._set_mode("moving")
        return True

    def _move_active_grab(self, pos: Tuple[int, int]) -> None:
        item = self.state.get("active_grab")
        if item is None:
            return
        path = list(self.state.get("grab_path") or [])
        path.append(pos)
        path = path[-6:]
        smoothed = smooth_points(path, factor=0.65)
        smooth_pos = smoothed[-1] if smoothed else pos
        self.state.set("grab_path", path)
        offset_x, offset_y = self.state.get("grab_offset") or (0, 0)
        item["pos"] = (smooth_pos[0] - offset_x, smooth_pos[1] - offset_y)

    def _copy_active_grab(self) -> None:
        now_ms = int(time.time() * 1000)
        last_copy_ms = self.state.get("last_copy_ms") or 0
        if now_ms - last_copy_ms < self.copy_cooldown_ms:
            return
        item = self.state.get("active_grab")
        if item is None:
            return
        snapshot = self.canvas.snapshot_item(item)
        if snapshot is not None:
            self.clipboard.add_item("image", snapshot)
            self.state.set("last_copy_ms", now_ms)

    def _drop_grab(self, pos: Tuple[int, int], dropped_without_target: bool = False) -> None:
        item = self.state.get("active_grab")
        if item is None:
            self.state.set("grabbing", False)
            return
        item["is_grabbed"] = False
        if not dropped_without_target and self.ui.is_over_clipboard(pos[0], pos[1]):
            snapshot = self.canvas.snapshot_item(item)
            if snapshot is not None:
                self.clipboard.add_item("image", snapshot)
            self.canvas.remove_item(item)
        else:
            self.canvas.record_history()
        self.state.set("grabbing", False)
        self.state.set("active_grab", None)
        self.state.set("grab_offset", (0, 0))
        self.state.set("grab_path", [])
        self._set_mode("drawing")

    def _recognize_and_draw_char(self):
        stroke = self.stroke_processor.get_current_stroke() or self.stroke_points
        if not stroke:
            return
        shape = self.shape_detector.detect(stroke)
        if shape is not None:
            self._draw_detected_shape(shape, stroke)
            self.stroke_points = []
            self.stroke_processor.clear_current_stroke()
            return
        min_x = min(p[0] for p in stroke)
        max_y = max(p[1] for p in stroke)
        char = self.handwriting_recognizer.recognize_points(stroke)
        if char:
            upper_char = char.upper()
            self.text_buffer = (self.text_buffer + upper_char)[-24:]
            emoji_keyword = self.emoji_manager.match_keyword(self.text_buffer)
            if emoji_keyword:
                emoji_img = self.emoji_manager.render_keyword(emoji_keyword, size=96)
                self.canvas.add_item(
                    {
                        "type": "image",
                        "content": emoji_img,
                        "pos": (max(0, min_x), max(0, max_y - 96)),
                    }
                )
                self.text_buffer = self.text_buffer[: -len(emoji_keyword)]
            else:
                self.canvas.render_text(upper_char, (min_x, max_y), 2, get_current_color(self))
        self.stroke_points = []
        self.stroke_processor.clear_current_stroke()

    def _draw_detected_shape(self, shape: str, stroke: List[Tuple[int, int]]) -> None:
        if len(stroke) < 2:
            return
        xs, ys = [p[0] for p in stroke], [p[1] for p in stroke]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        color = get_current_color(self)
        if shape == "line":
            self.canvas.draw_line(stroke[0], stroke[-1], color, 3)
        elif shape == "circle":
            center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
            radius = max(4, int(max(max_x - min_x, max_y - min_y) * 0.5))
            cv2.circle(self.canvas.canvas, center, radius, color, 3)
        elif shape == "rectangle":
            cv2.rectangle(self.canvas.canvas, (min_x, min_y), (max_x, max_y), color, 3)

    def _paste_image_from_system_clipboard(self, cursor_pos: Tuple[int, int]) -> None:
        def _decode_image(raw: bytes) -> Optional[np.ndarray]:
            if not raw:
                return None
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return img

        for cmd in [
            ["wl-paste", "--no-newline", "--type", "image/png"],
            ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, check=False, timeout=1.5)
                if result.returncode == 0 and result.stdout:
                    image = _decode_image(result.stdout)
                    if image is not None:
                        h, w = image.shape[:2]
                        self.canvas.add_item(
                            {
                                "type": "image",
                                "content": image,
                                "pos": (cursor_pos[0] - w // 2, cursor_pos[1] - h // 2),
                            }
                        )
                        self.ui.show_clipboard()
                        return
            except Exception:
                continue

    def _confirm_clear(self):
        if getattr(self, "pending_clear", False):
            self.canvas.clear_canvas()
            self.pending_clear = False
        else:
            self.pending_clear = True

    def _render_display(self, camera_frame):
        canvas_img = self.canvas.get_canvas_with_items()

        if get_mode(self) == "writing" and len(self.stroke_points) > 1:
            cv2.polylines(canvas_img, [np.array(self.stroke_points, dtype=np.int32)], False, (100, 100, 100, 255), 2)

        if canvas_img.shape[0] != self.height or canvas_img.shape[1] != self.width:
            canvas_img = cv2.resize(canvas_img, (self.width, self.height))

        self._bg_buffer[:] = 0
        background = self._bg_buffer

        alpha = canvas_img[:, :, 3:4].astype(np.float32) / 255.0
        canvas_rgb = canvas_img[:, :, :3].astype(np.float32)
        background[:] = (alpha * canvas_rgb + (1.0 - alpha) * background.astype(np.float32)).astype(np.uint8)

        page_str = f"Page: {self.pages.current_index + 1}/{len(self.pages.pages)}"
        self.ui.draw_panels(
            background,
            get_mode(self),
            page_str,
            brush_size=self.state.get("brush_size", 8),
            opacity=self.state.get("brush_opacity", 1.0),
        )

        if get_mode(self) == "writing":
            cv2.putText(background, "WRITING MODE", (self.width - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (93, 202, 165), 2)

        if getattr(self, "pending_clear", False):
            cv2.putText(background, "PRESS 'C' AGAIN TO CLEAR", (self.width // 2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self.calibration_mode:
            background = self._draw_calibration_overlay(background)
        else:
            status = get_calibration_status()
            if status.get("is_calibrated"):
                cv2.putText(background, "Calibrated", (30, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (93, 202, 165), 2)

        if self.debug_overlay:
            self._draw_debug_overlay(background)

        return background

    def _draw_debug_overlay(self, frame: np.ndarray) -> None:
        panel_x, panel_y = 16, 64
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 500, panel_y + 210), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 500, panel_y + 210), (93, 202, 165), 1)
        lines = [
            f"raw: {self.debug_state.get('raw_gesture', '?')}",
            f"effective: {self.debug_state.get('effective_gesture', '?')}",
            f"confidence: {self.debug_state.get('confidence', 0.0):.2f}",
            f"mode: {self.debug_state.get('mode', '?')}",
            f"brush: {self.debug_state.get('brush_size', 0)}",
            f"fingers: {int(self.debug_state.get('finger_count', 0))} T/I/M/R/P: "
            f"{self.debug_state.get('thumb_curl', 0):.0f}/"
            f"{self.debug_state.get('index_curl', 0):.0f}/"
            f"{self.debug_state.get('middle_curl', 0):.0f}/"
            f"{self.debug_state.get('ring_curl', 0):.0f}/"
            f"{self.debug_state.get('pinky_curl', 0):.0f}",
            f"hand: {self.debug_state.get('hand_detected', False)} | processed: {self.debug_state.get('processed', False)}",
            "toggle debug: press 'g'",
        ]
        y = panel_y + 22
        for line in lines:
            cv2.putText(frame, line, (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            y += 20

    def _draw_calibration_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        blended = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        if self.calibration_step < len(self.calibration_corners):
            text = f"Point to {self.calibration_corners[self.calibration_step]} corner and press Enter"
        else:
            text = "Calibration complete"
        cv2.putText(blended, text, (max(20, w // 2 - 280), h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return blended

    def _draw_indicators(self, frame):
        pass

    def _draw_controls(self, frame):
        pass

    def cleanup(self):
        self.camera_thread.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import logging

    app = GestureWhiteboard()
    try:
        app.run()
    except Exception:
        logging.getLogger(__name__).exception("Startup failed:")
