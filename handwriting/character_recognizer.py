import os
import sys

try:
    import config
except ModuleNotFoundError:
    # Allow direct execution of this file (python handwriting/character_recognizer.py).
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
import cv2
import numpy as np
from typing import List, Optional, Tuple

try:
    from .simple_matcher import SimpleCharacterMatcher
    from .feature_based_recognizer import FeatureBasedRecognizer
except ImportError:
    from simple_matcher import SimpleCharacterMatcher
    from feature_based_recognizer import FeatureBasedRecognizer

class CharacterRecognizer:
    def __init__(self, model_path: str = 'data/models/character_model.h5'):
        self.fallback_matcher = SimpleCharacterMatcher()
        self.feature_recognizer = FeatureBasedRecognizer()
        self.char_map: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.model = None

        resolved_model_path = model_path
        if not os.path.isabs(resolved_model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(project_root, resolved_model_path)
            if os.path.exists(candidate):
                resolved_model_path = candidate

        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(resolved_model_path)
            import logging; logging.getLogger(__name__).info(f"Character recognition model loaded successfully from {resolved_model_path}.")
        except Exception as e:
            import logging; logging.getLogger(__name__).warning(f"Warning: character model unavailable ({e}). Using SimpleCharacterMatcher fallback.")


    def recognize(self, image: np.ndarray) -> Optional[str]:
        if self.model is None:
            return self.fallback_matcher.match(image)
        
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image, verbose=0)
        char_index = np.argmax(prediction)
        
        if char_index < len(self.char_map):
            return self.char_map[char_index]
        return None

    def preprocess_image(self, image: np.ndarray, size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        # Invert colors if necessary (model might be trained on white on black)
        if np.mean(image) > 127:
            image = 255 - image

        # Resize and normalize
        image = cv2.resize(image, size)
        image = image.astype('float32') / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        return image

    def points_to_image(self, points: List[Tuple[int, int]], size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """Render stroke points to a grayscale bitmap for model input."""
        if not points:
            return np.zeros(size, dtype=np.uint8)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(1, max_x - min_x)
        h = max(1, max_y - min_y)
        side = max(w, h)
        pad = 4

        temp = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
        x_off = ((side + 2 * pad) - w) // 2
        y_off = ((side + 2 * pad) - h) // 2
        shifted = [(p[0] - min_x + x_off, p[1] - min_y + y_off) for p in points]

        for i in range(1, len(shifted)):
            cv2.line(temp, shifted[i - 1], shifted[i], 255, 2)

        return cv2.resize(temp, size, interpolation=cv2.INTER_AREA)

    def recognize_points(self, points: List[Tuple[int, int]]) -> Optional[str]:
        """Recognize directly from a sequence of stroke points."""
        if self.feature_recognizer.enabled:
            features = self.feature_recognizer.extract_features(points)
            char = self.feature_recognizer.predict(features)
            if char:
                return char
        return self.recognize(self.points_to_image(points))

