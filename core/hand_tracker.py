import os
import sys

from .config_manager import ConfigManager
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import List, Tuple, Optional

class HandTracker:
    def __init__(self, model_asset_path: str, max_hands: int = 1, detection_confidence: float = 0.8, tracking_confidence: float = 0.5):
        import logging; logging.getLogger(__name__).info("Initializing HandTracker with new MediaPipe API")
        resolved_model_path = model_asset_path
        if not os.path.isabs(resolved_model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(project_root, resolved_model_path)
            if os.path.exists(candidate):
                resolved_model_path = candidate

        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(
                "Hand landmarker model not found. Expected at "
                f"'{resolved_model_path}'. Download with:\n"
                "mkdir -p data/models && cd data/models && "
                "wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=resolved_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], Optional[str]]:
        """Processes a single frame for hand landmarks and handedness using the new API."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = self.landmarker.detect(mp_image)
        
        landmarks: List[Tuple[int, int]] = []
        handedness: Optional[str] = None

        if detection_result.hand_landmarks:
            # Get handedness
            if detection_result.handedness:
                handedness = detection_result.handedness[0][0].category_name

            # Get landmarks
            for hand_landmarks_list in detection_result.hand_landmarks:
                for lm in hand_landmarks_list:
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))
        
        return landmarks if landmarks else None, handedness

    def close(self) -> None:
        # The new API doesn't have a close method on the landmarker instance
        pass


