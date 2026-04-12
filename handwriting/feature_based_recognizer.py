from typing import List, Optional, Tuple

import numpy as np


class FeatureBasedRecognizer:
    """Optional RandomForest recognizer for stroke-point features."""

    def __init__(self, labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        self.labels = labels
        self.model = None
        self.enabled = False
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.enabled = True
        except Exception:
            self.enabled = False

    def train(self, features_list: List[List[float]], labels: List[str]) -> bool:
        if not self.enabled or self.model is None or not features_list or not labels:
            return False

        self.model.fit(features_list, labels)
        return True

    def predict(self, features: List[float]) -> Optional[str]:
        if not self.enabled or self.model is None or not features:
            return None

        pred = self.model.predict([features])[0]
        return str(pred)

    @staticmethod
    def extract_features(points: List[Tuple[int, int]]) -> List[float]:
        if not points or len(points) < 3:
            return []

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)

        cx = float(xs.mean())
        cy = float(ys.mean())
        centered = np.stack([xs - cx, ys - cy], axis=1)

        max_dist = float(np.linalg.norm(centered, axis=1).max())
        if max_dist > 0:
            centered = centered / max_dist

        deltas = np.diff(centered, axis=0)
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=False)
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()

        angle_deltas = np.diff(angles)
        angle_deltas = (angle_deltas + np.pi) % (2 * np.pi) - np.pi
        avg_curvature = float(np.mean(np.abs(angle_deltas))) if len(angle_deltas) else 0.0

        width = float(xs.max() - xs.min())
        height = float(ys.max() - ys.min())
        aspect = width / max(1.0, height)

        start = centered[0]
        end = centered[-1]
        start_dir = float(np.arctan2(start[1], start[0]))
        end_dir = float(np.arctan2(end[1], end[0]))

        return [aspect, avg_curvature, float(np.cos(start_dir)), float(np.sin(start_dir)), float(np.cos(end_dir)), float(np.sin(end_dir))] + hist.tolist()
