import cv2
import numpy as np
from typing import Optional


class SimpleCharacterMatcher:
    """Fallback matcher based on template correlation and stroke-shape features."""

    def __init__(self, size=(28, 28), threshold: float = 0.25):
        self.size = size
        self.threshold = threshold
        self.characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.templates = self._build_templates()
        self.template_features = {
            key: self._extract_features(template) for key, template in self.templates.items()
        }

    def _build_templates(self):
        templates = {}
        # Add common handwritten shapes using multiple fonts
        for char in self.characters:
            for font in [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX]:
                img = np.zeros(self.size, dtype=np.uint8)
                scale = 0.9
                thickness = 2
                (w, h), baseline = cv2.getTextSize(char, font, scale, thickness)
                x = max(0, (self.size[1] - w) // 2)
                y = max(h, (self.size[0] + h) // 2 - baseline)
                cv2.putText(img, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                templates[f"{char}_{font}"] = img
        return templates

    def _prepare(self, stroke_image: np.ndarray) -> np.ndarray:
        if stroke_image.ndim == 3:
            stroke_image = cv2.cvtColor(stroke_image, cv2.COLOR_BGR2GRAY)
        if stroke_image.shape[:2] != self.size:
            stroke_image = cv2.resize(stroke_image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        return stroke_image

    def match(self, stroke_image: np.ndarray) -> Optional[str]:
        candidate = self._prepare(stroke_image)
        candidate_features = self._extract_features(candidate)
        best_per_char = {}

        for char, template in self.templates.items():
            raw_corr = float(cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)[0][0])
            template_score = (raw_corr + 1.0) * 0.5

            feature_score = 0.0
            feature_template = self.template_features.get(char)
            if candidate_features is not None and feature_template is not None:
                feature_score = self._feature_similarity(candidate_features, feature_template)

            combined_score = 0.65 * template_score + 0.35 * feature_score
            base_char = char.split("_")[0]
            best_per_char[base_char] = max(best_per_char.get(base_char, 0.0), combined_score)

        if not best_per_char:
            return None

        best_char, best_score = max(best_per_char.items(), key=lambda item: item[1])

        if best_score >= self.threshold:
            return best_char
        return None

    def _extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        _, binary = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        pts = contour[:, 0, :].astype(np.float32)
        if len(pts) < 8:
            return None

        sampled = self._resample_points(pts, 32)
        if sampled is None or len(sampled) < 8:
            return None

        deltas = np.diff(sampled, axis=0)
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])

        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=False)
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()

        angle_deltas = np.diff(angles)
        angle_deltas = (angle_deltas + np.pi) % (2 * np.pi) - np.pi
        curvature = float(np.mean(np.abs(angle_deltas))) if len(angle_deltas) else 0.0

        x, y, w, h = cv2.boundingRect(sampled.astype(np.int32))
        aspect = float(w) / float(max(1, h))

        hole_count = 0
        if hierarchy is not None:
            for hrow in hierarchy[0]:
                if hrow[3] >= 0:
                    hole_count += 1
        loop_feature = min(2.0, float(hole_count))

        center = np.mean(sampled, axis=0)
        start_dir = np.arctan2(sampled[0][1] - center[1], sampled[0][0] - center[0])
        end_dir = np.arctan2(sampled[-1][1] - center[1], sampled[-1][0] - center[0])

        feature_vector = np.concatenate(
            [
                np.array([aspect, curvature, loop_feature, np.cos(start_dir), np.sin(start_dir), np.cos(end_dir), np.sin(end_dir)], dtype=np.float32),
                hist,
            ]
        )
        return feature_vector

    def _resample_points(self, points: np.ndarray, target_count: int) -> Optional[np.ndarray]:
        if len(points) < 2:
            return None

        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0.0:
            return points[:target_count]

        cumulative = np.concatenate(([0.0], np.cumsum(seg)))
        samples = np.linspace(0.0, total, target_count)

        out = []
        j = 0
        for s in samples:
            while j < len(cumulative) - 2 and cumulative[j + 1] < s:
                j += 1

            span = cumulative[j + 1] - cumulative[j]
            if span <= 0.0:
                out.append(points[j])
                continue

            t = (s - cumulative[j]) / span
            interp = points[j] + t * (points[j + 1] - points[j])
            out.append(interp)

        return np.array(out, dtype=np.float32)

    def _feature_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        diff = float(np.linalg.norm(f1 - f2))
        return 1.0 / (1.0 + diff)