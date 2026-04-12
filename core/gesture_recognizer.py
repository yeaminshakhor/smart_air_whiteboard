"""
GestureRecognizer - classifies hand poses from MediaPipe landmarks.

Fixes vs original:
  - TemporalGestureFilter window: 5->3  (less lag)
  - debounce_frames default: 4->2       (faster response)
  - min_frame_confidence: 0.4->0.55     (fewer false positives)
  - "four" gesture renamed to "four_fingers" to match _handle_gesture in main.py
  - Removed draw_grace false-positive workaround (handled in main._effective_gesture)
"""
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from .feature_extractor import FeatureExtractor


class TemporalGestureFilter:
    """Confidence-weighted temporal smoothing for flicker reduction."""

    def __init__(self, window_size: int = 3):
        self.history: Deque[Tuple[str, float]] = deque(maxlen=window_size)

    def update(self, gesture: str, confidence: float) -> None:
        self.history.append((gesture, float(max(0.0, min(1.0, confidence)))))

    def get_stable(self, min_confidence: float = 0.45) -> Tuple[str, float]:
        weighted: Dict[str, float] = {}
        total_weight = 0.0
        for gesture, confidence in self.history:
            if confidence < min_confidence:
                continue
            weighted[gesture] = weighted.get(gesture, 0.0) + confidence
            total_weight += confidence
        if not weighted or total_weight <= 0.0:
            return "unknown", 0.0
        best_gesture, best_weight = max(weighted.items(), key=lambda item: item[1])
        return best_gesture, float(best_weight / total_weight)


class GestureRecognizer:
    def __init__(
        self,
        debounce_frames: int = 2,
        smoothing_window: int = 3,
        scroll_threshold_px: int = 30,
        two_finger_distance_ratio: float = 0.08,
        min_frame_confidence: float = 0.55,
        smoothing_min_confidence: float = 0.5,
    ):
        self.debounce_frames = debounce_frames
        self._pending_gesture = "unknown"
        self._pending_count = 0
        self._stable_gesture = "unknown"
        self._temporal_filter = TemporalGestureFilter(window_size=smoothing_window)
        self._min_frame_confidence = float(max(0.0, min(1.0, min_frame_confidence)))
        self._smoothing_min_confidence = float(max(0.0, min(1.0, smoothing_min_confidence)))
        self.scroll_threshold_px = max(4, int(scroll_threshold_px))
        self.two_finger_distance_ratio = max(0.01, float(two_finger_distance_ratio))
        self.scroll_tracker: Dict = {
            "active": False,
            "start_y": 0.0,
            "accumulated_scroll": 0.0,
            "last_position": None,
        }
        self.last_debug: Dict[str, float] = {
            "raw_confidence": 0.0,
            "filtered_confidence": 0.0,
            "finger_count": 0.0,
            "index_curl": 100.0,
            "middle_curl": 100.0,
            "ring_curl": 100.0,
            "pinky_curl": 100.0,
            "thumb_curl": 100.0,
        }
        import logging

        logging.getLogger(__name__).info(
            "Initializing GestureRecognizer (debounce=%d, window=%d)", debounce_frames, smoothing_window
        )

    def recognize(self, landmarks: Optional[List[Tuple[int, int]]], handedness: Optional[str]) -> Tuple[str, float]:
        if not landmarks or len(landmarks) < 21:
            return "unknown", 0.0

        raw_gesture, raw_confidence = self._classify_raw(landmarks, handedness)
        self.last_debug["raw_confidence"] = float(raw_confidence)
        if raw_confidence < self._min_frame_confidence:
            raw_gesture = "unknown"

        self._temporal_filter.update(raw_gesture, raw_confidence)
        filtered_gesture, filtered_confidence = self._temporal_filter.get_stable(
            min_confidence=self._smoothing_min_confidence
        )

        if filtered_gesture == self._pending_gesture:
            self._pending_count += 1
        else:
            self._pending_gesture = filtered_gesture
            self._pending_count = 1

        if self._pending_count >= self.debounce_frames:
            self._stable_gesture = self._pending_gesture

        confidence = filtered_confidence if self._stable_gesture == filtered_gesture else filtered_confidence * 0.5
        self.last_debug["filtered_confidence"] = float(confidence)
        return self._stable_gesture, confidence

    def _classify_raw(self, landmarks: Optional[List[Tuple[int, int]]], handedness: Optional[str]) -> Tuple[str, float]:
        if not landmarks or len(landmarks) < 21:
            return "unknown", 0.0

        curls = FeatureExtractor.get_curl_percentage(landmarks)
        extended = {finger: curl <= 45.0 for finger, curl in curls.items()}
        extended["thumb"] = curls["thumb"] <= 55.0
        finger_count = sum(1 for state in extended.values() if state)

        self.last_debug["finger_count"] = float(finger_count)
        self.last_debug["thumb_curl"] = float(curls.get("thumb", 100.0))
        self.last_debug["index_curl"] = float(curls.get("index", 100.0))
        self.last_debug["middle_curl"] = float(curls.get("middle", 100.0))
        self.last_debug["ring_curl"] = float(curls.get("ring", 100.0))
        self.last_debug["pinky_curl"] = float(curls.get("pinky", 100.0))

        non_thumb_curl = [curls[name] for name in ("index", "middle", "ring", "pinky")]
        avg_non_thumb_curl = float(sum(non_thumb_curl) / len(non_thumb_curl))

        if finger_count == 0 or avg_non_thumb_curl > 70.0:
            confidence = min(0.98, avg_non_thumb_curl / 100.0)
            return "fist", confidence

        if finger_count == 1 and extended["index"] and not extended["thumb"]:
            confidence = 1.0 - min(1.0, curls["index"] / 100.0)
            return "index_finger", max(0.55, confidence)

        if finger_count == 1 and extended["thumb"]:
            thumb_direction = self._get_thumb_direction(landmarks)
            if thumb_direction is not None:
                direction_conf = 1.0 - min(1.0, curls["thumb"] / 100.0)
                return thumb_direction, max(0.55, direction_conf)

        if finger_count == 2 and extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
            if self._is_scroll_gesture(landmarks, handedness):
                scroll_dir = self._track_scroll(float(landmarks[8][1]), handedness)
                if scroll_dir is not None:
                    return scroll_dir, 0.88
            else:
                self._reset_scroll_tracker()
            confidence = 1.0 - min(1.0, (curls["index"] + curls["middle"]) / 200.0)
            return "peace", max(0.6, confidence)
        else:
            self._reset_scroll_tracker()

        if finger_count == 3 and extended["index"] and extended["middle"] and extended["ring"] and not extended["pinky"]:
            confidence = 1.0 - min(1.0, (curls["index"] + curls["middle"] + curls["ring"]) / 300.0)
            return "three", max(0.58, confidence)

        if (
            finger_count == 4
            and all(extended[name] for name in ("index", "middle", "ring", "pinky"))
            and not extended["thumb"]
        ):
            confidence = 1.0 - min(1.0, sum(curls[name] for name in ("index", "middle", "ring", "pinky")) / 400.0)
            return "four_fingers", max(0.58, confidence)

        if finger_count >= 4 and self._is_open_palm(landmarks, curls):
            palm_conf = 1.0 - min(1.0, avg_non_thumb_curl / 120.0)
            return "open_hand", max(0.6, palm_conf)

        return "unknown", 0.3

    @staticmethod
    def _joint_angle(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        ba_mag = math.hypot(ba[0], ba[1])
        bc_mag = math.hypot(bc[0], bc[1])
        if ba_mag == 0.0 or bc_mag == 0.0:
            return 0.0
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        cos_theta = max(-1.0, min(1.0, dot / (ba_mag * bc_mag)))
        return math.degrees(math.acos(cos_theta))

    def _get_thumb_direction(self, landmarks: List[Tuple[int, int]]) -> Optional[str]:
        thumb_tip = landmarks[4]
        wrist = landmarks[0]
        vec_x = float(thumb_tip[0] - wrist[0])
        vec_y = float(thumb_tip[1] - wrist[1])
        if vec_x == 0.0 and vec_y == 0.0:
            return None
        angle = math.degrees(math.atan2(-vec_y, vec_x))
        if 60.0 <= angle <= 120.0:
            return "thumbs_up"
        if -120.0 <= angle <= -60.0:
            return "thumbs_down"
        return None

    def _is_open_palm(self, landmarks: List[Tuple[int, int]], curls: Dict[str, float]) -> bool:
        non_thumb_extended = all(curls[name] <= 45.0 for name in ("index", "middle", "ring", "pinky"))
        if not non_thumb_extended:
            return False
        tip_indices = [4, 8, 12, 16, 20]
        tips = [landmarks[idx] for idx in tip_indices]
        palm_width = max(1.0, float(np.linalg.norm(np.array(landmarks[17]) - np.array(landmarks[5]))))
        separations = []
        for i in range(len(tips) - 1):
            delta = np.array(tips[i + 1]) - np.array(tips[i])
            separations.append(float(np.linalg.norm(delta) / palm_width))
        mean_sep = float(sum(separations) / max(1, len(separations)))
        return mean_sep >= 0.25

    def _is_scroll_gesture(self, landmarks: List[Tuple[int, int]], handedness: Optional[str]) -> bool:
        _ = handedness
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        finger_distance = math.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])
        palm_width = self.get_palm_width(landmarks)
        return (finger_distance / max(palm_width, 1.0)) < self.two_finger_distance_ratio

    def _track_scroll(self, current_y: float, handedness: Optional[str]) -> Optional[str]:
        _ = handedness
        if not self.scroll_tracker["active"]:
            self.scroll_tracker.update(
                {"active": True, "start_y": current_y, "accumulated_scroll": 0.0, "last_position": current_y}
            )
            return None
        last_pos = self.scroll_tracker["last_position"]
        if last_pos is None:
            self.scroll_tracker["last_position"] = current_y
            return None
        delta_y = current_y - float(last_pos)
        self.scroll_tracker["accumulated_scroll"] += delta_y
        self.scroll_tracker["last_position"] = current_y
        if abs(float(self.scroll_tracker["accumulated_scroll"])) >= self.scroll_threshold_px:
            direction = "scroll_down" if self.scroll_tracker["accumulated_scroll"] > 0 else "scroll_up"
            self.scroll_tracker["accumulated_scroll"] = 0.0
            return direction
        return None

    def _reset_scroll_tracker(self) -> None:
        self.scroll_tracker.update({"active": False, "start_y": 0.0, "accumulated_scroll": 0.0, "last_position": None})

    def get_palm_width(self, landmarks: List[Tuple[int, int]]) -> float:
        return FeatureExtractor.get_palm_width(landmarks)

    def get_brush_size_from_landmarks(self, landmarks: List[Tuple[int, int]]) -> int:
        palm_width = self.get_palm_width(landmarks)
        return int(np.clip(palm_width * 0.12, 2, 20))

    def get_debug_snapshot(self) -> Dict[str, float]:
        return dict(self.last_debug)
